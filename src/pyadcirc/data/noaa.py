"""
noaa - Utilities for pulling NOAA Tide station data

See NOAA Websites for more information:
    - https://tidesandcurrents.noaa.gov
    - https://tidesandcurrents.noaa.gov/stations.html?type=Water+Levels
"""
import concurrent.futures
import pdb
import subprocess
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from alive_progress import alive_bar
from pyadcirc.viz import asciichart as ac

import noaa_coops as nc
import numpy as np
import pandas as pd
import requests
from pandas.errors import EmptyDataError

from pyadcirc.utils import get_bbox

# information from https://api.tidesandcurrents.noaa.gov/api/prod/#station


DATE_TIME = {
    "begin_date": "Use with either end_date to express an explicit range, or with range to express a range (in hours) of time starting from a certain date.",
    "end_date": "Use with eitehr begin_date to express an explicity range, or with range to express a range (in hours) of time ending on a certan date.",
    "range": "Specify a number of hours to to back from now and retrieve data for that period\nNote!\n• If used alone, only available for preliminary water level data, meteorological data\n• If used with a historical begin or end date, may be used with verified data",
    "date": "Data from today’s date.\nNote! Only available for preliminary water level data, meteorological data and predictions.\nValid options for the date parameter are:\n• Today (24 hours starting at midnight)\n• Latest (last data point available within the last 18 min)\n• Recent (last 72 hours)",
}

PRODUCTS = {
    "metadata": {
        "group": "metadata",
        "desc": "Station metadata, like location and station name.",
    },
    "water_level": {
        "group": "tide",
        "desc": "Preliminary or verified 6-minute interval water levels, depending on data availability.",
    },
    "hourly_height": {
        "group": "tide",
        "desc": "Verified hourly height water level data for the station.",
    },
    "high_low": {
        "group": "tide",
        "desc": "Verified high tide / low tide water level data for the station.",
    },
    "daily_mean": {
        "group": "tide",
        "desc": "Verified daily mean water level data for the station.\nNote!Great Lakes stations only. Only available with time_zone=LST",
    },
    "monthly_mean": {
        "group": "tide",
        "desc": "Verified monthly mean water level data for the station.",
    },
    "one_minute_water_level": {
        "group": "tide",
        "desc": "Preliminary 1-minute interval water level data for the station.",
    },
    "predictions": {
        "group": "tide",
        "desc": "Water level / tide prediction data for the station.\nNote!See Interval for available data interval options and data length limitations.",
    },
    "datums": {
        "group": "tide",
        "desc": "Observed tidal datum values at the station for the present National Tidal Datum Epoch (NTDE).",
    },
    "air_gap": {
        "group": "tide",
        "desc": "Air Gap (distance between a bridge and the water's surface) at the station.",
    },
    "air_temperature": {
        "group": "met",
        "desc": "Air temperature as measured at the station.",
    },
    "water_temperature": {
        "group": "met",
        "desc": "Water temperature as measured at the station.",
    },
    "wind": {
        "group": "met",
        "desc": "Wind speed, direction, and gusts as measured at the station.",
    },
    "air_pressure": {
        "group": "met",
        "desc": "Barometric pressure as measured at the station.",
    },
    "conductivity": {
        "group": "met",
        "desc": "The water's conductivity as measured at the station.",
    },
    "visibility": {
        "group": "met",
        "desc": "Visibility (atmospheric clarity) as measured at the station. (Units of Nautical Miles or Kilometers)",
    },
    "humidity": {
        "group": "met",
        "desc": "Relative humidity as measured at the station.",
    },
    "salinity": {
        "group": "met",
        "desc": "Salinity and specific gravity data for the station.",
    },
    "currents": {
        "group": "cur",
        "desc": "Currents data for the station. Note! Default data interval is 6-minute interval data.Use with “interval=h” for hourly data",
    },
    "currents_predictions": {
        "group": "cur",
        "desc": "Currents prediction data for the stations. Note! See Interval for options available and data length limitations.",
    },
    "ofs_water_level": {
        "group": "ofs",
        "desc": "Currents data for the station. Note! Default data interval is 6-minute interval data.Use with “interval=h” for hourly data",
    },
}

DATUMS = {
    "CRD": {
        "desc": "Columbia River Datum. Note!Only available for certain stations on the Columbia River, Washington/Oregon"
    },
    "IGLD": {
        "desc": "International Great Lakes Datum Note! Only available for Great Lakes stations."
    },
    "LWD": {
        "desc": "Great Lakes Low Water Datum (Nautical Chart Datum for the Great Lakes). Note! Only available for Great Lakes Stations"
    },
    "MHHW": {"desc": "Mean Higher High Water"},
    "MHW": {"desc": "Mean High Water"},
    "MTL": {"desc": "Mean Tide Level"},
    "MSL": {"desc": "Mean Sea Level"},
    "MLW": {"desc": "Mean Low Water"},
    "MLLW": {
        "desc": "Mean Lower Low Water (Nautical Chart Datum for all U.S. coastal waters). Note! Subordinate tide prediction stations must use “datum=MLLW”"
    },
    "NAVD": {
        "desc": "North American Vertical Datum Note! This datum is not available for all stations."
    },
    "STND": {
        "desc": "Station Datum - original reference that all data is collected to, uniquely defined for each station."
    },
}

UNITS = {
    "metric": "Metric units (Celsius, meters, cm/s appropriate for the data)\nNote!Visibility data is kilometers (km)",
    "english": "English units (fahrenheit, feet, knots appropriate for the data)\nNote!Visibility data is Nautical Miles (nm)",
}

TIME_ZONES = {
    "gmt": "Greenwich Mean Time",
    "lst": "Local Standard Time, not corrected for Daylight Saving Time, local to the requested station.",
    "lst_ldt": "Local Standard Time, corrected for Daylight Saving Time when appropriate, local to the requested station",
}

INTERVALS = ["h", "hilo", "max_slack"] + [str(x) for x in [1, 5, 6, 10, 15, 30, 60]]

FORMATS = {
    "xml": "Extensible Markup Language. This format is an industry standard for data.",
    "json": "Javascript Object Notation. This format is useful for direct import to a javascript plotting library. Parsers are available for other languages such as Java and Perl.",
    "csv": "Comma Separated Values. This format is suitable for import into Microsoft Excel or other spreadsheet programs. ",
}


def check_date_range(begin_date, end_date, interval):
    begin_date = datetime.strptime(begin_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    interval = interval.lower()
    if interval == "1-minute":
        if (end_date - begin_date) > timedelta(days=4):
            raise ValueError(
                "1-minute interval data - Data length is limited to 4 days"
            )
    elif interval == "6-minute":
        if (end_date - begin_date) > timedelta(days=30):
            raise ValueError(
                "6-minute interval data - Data length is limited to 1 month"
            )
    elif interval == "hourly":
        if (end_date - begin_date) > timedelta(days=365):
            raise ValueError("Hourly interval data - Data length is limited to 1 year")
    elif interval == "high/low":
        if (end_date - begin_date) > timedelta(days=365):
            raise ValueError("High/Low data - Data length is limited to 1 year")
    elif interval == "daily means":
        if (end_date - begin_date) > timedelta(days=3650):
            raise ValueError("Daily Means data - Data length is limited to 10 years")
    elif interval == "monthly means":
        if (end_date - begin_date) > timedelta(days=365 * 200):
            raise ValueError("Monthly Means data - Data length is limited to 200 years")
    else:
        raise ValueError("Invalid interval")


# Path to json file with noaa station data pulled from website
NOAA_STATIONS = pd.read_json(
    Path(Path(__file__).parents[1] / "configs/noaa_stations.json")
)

PRODUCTS = {"water_level": "waterlevels", "predictions": "noaatidepredictions"}


def parse_f15_station_list(region: str):
    """
    Parse fort.15 Station List

    Given a region of NOAA Tide Stations, parses list of long,lat values to be
    copied to fort.15 file NSTAE,NSTAV,NSTAM sections.

    Parameters
    ----------
    region : str
      Valid NOAA Tides region.

    Returns
    ------
    stations_list : str
      String that when passed to print() method should produce stations list to
      be copied into fort.15 file.

    Notes
    -----
    See https://tidesandcurrents.noaa.gov/stations.html?type=Water+Levels for a
    list of all tide stations/regions available from NOAA.

    Examples
    --------
    >>> noaa.get_f15_stations("Georgia")
    '279.0983 32.0367               ! Georgia FortPulaski 8670870.0'
    """
    region_idxs = NOAA_STATIONS["Region"] == region
    region_ids = NOAA_STATIONS[region_idxs]["ID"].values
    names = NOAA_STATIONS[region_idxs]["Name"].values

    station_list = []
    for idx, sid in enumerate(region_ids):
        print(f"Getting NOAA station {sid} - {names[idx]}")
        station = get_station_metadata(int(sid))
        line = f"{station['coords'][0]} {station['coords'][0]}"
        line = line.ljust(30, " ")
        name = station["name"].replace(" ", "")
        line = f"{line} ! {region} {name} {sid}"
        station_list.append(line)

    # Return final string to insert into f15 file
    return "\n".join(station_list)


def get_station_metadata(station_id: int):
    """
    Get NOAA Tides Station Metadata

    Parameters
    ----------
    station_id : int
      Seven digit unique identifier for station.

    Returns
    -------
    station : dict
      Dictionary containing the name, id, and coordinate location of the NOAA
      Tide station.

    Examples
    --------
    >>> get_station_metadata(8670870)
    {'name': 'Fort Pulaski', 'id': 8670870, 'coords': [279.0983, 32.0367]}

    Note how positive longitude coordinates always returned.

    >>> get_station_metadata(9468756)
    {'name': 'Nome, Norton Sound', 'id': 9468756,
        'coords': [194.560361, 64.494611]}
    """

    url = (
        "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/"
        + f"{station_id}.json?expand=details?units=metric"
    )

    # Get response from Metadata API
    response = requests.get(url, timeout=60)
    json_dict = response.json()
    station_metadata = json_dict["stations"][0]

    if station_metadata["lng"] < 0:
        station_metadata["lng"] = 360 + station_metadata["lng"]

    station = {
        "name": station_metadata["name"],
        "id": station_id,
        "coords": [station_metadata["lng"], station_metadata["lat"]],
    }

    return station


def check_time_window(
    params: dict, begin_date: datetime, end_date: datetime, date: datetime, range: int
) -> dict:
    """
    Check time window

    Parameters
    ----------
    params : dict
        A dictionary containing the parameters to update.
    begin_date : datetime
        The start date of the time window.
    end_date : datetime
        The end date of the time window.
    date : datetime
        A specific date within the time window.
    range : int
        The number of days in the time window.

    Returns
    -------
    dict
        The updated parameters dictionary.

    Raises
    ------
    ValueError
        If no valid date range is specified.

    Example
    -------
    >>> params = {}
    >>> begin_date = datetime(2022, 1, 1)
    >>> end_date = datetime(2022, 2, 1)
    >>> date = None
    >>> range = None
    >>> check_time_window(params, begin_date, end_date, date, range)
    >>> print(params)
    {'begin_date': '2022-01-01', 'end_date': '2022-02-01'}
    """
    if begin_date is not None and rnge is not None:
        params["begin_date"] = dt_date(begin_date).strftime("%Y%m%d")
        params["range"] = rnge
    elif begin_date is not None and end_date is not None:
        params["begin_date"] = dt_date(begin_date).strftime("%Y%m%d")
        params["end_date"] = dt_date(end_date).strftime("%Y%m%d")
    elif end_date is not None and rnge is not None:
        params["end_date"] = dt_date(end_date).strftime("%Y%m%d")
        params["range"] = int(rnge)
    elif date is not None:
        params["date"] = dt_date(date).strftime("%Y%m%d")
    elif rnge is not None:
        params["range"] = int(rnge)
    else:
        raise ValueError("No valid date range specified")

    return params


def get_tide_data(
    station_id,
    product,
    begin_date=None,
    end_date=None,
    date=None,
    range=None,
    output_format="csv",
    time_zone="lst_ldt",
    units="metric",
    application="pyadcirc",
    interval=6,
    workers=6,
):
    """
    Get station data

    Parameters
    ----------
    station_id : int
      Seven digit unique identifier for station.
    start_date : str
      String start date in either yyyyMMdd, yyyyMMdd HH:mm, MM/dd/yyyy,
      or MM/dd/yyyy HH:mm) format.
    end_date : str
      String end date in either yyyyMMdd, yyyyMMdd HH:mm, MM/dd/yyyy,
      or MM/dd/yyyy HH:mm) format.

    Returns
    -------
    water_level : xarray.DataArray
      xarray.DataArray with a `water_level` values over time for the given date
      range, in meters.

    Examples
    --------
    To use this function to request the "water_level" product in CSV format,
    you would call it with the desired begin_date, end_date, station_id,
    product, and output_format as arguments. For example:
    >>> response = get_tide_data("20210101", "20210105", "9447130", "water_level", "csv")
    """
    params = {}
    params = check_time_window({}, begin_date, end_date, date, rnge)
    begin_date = dt_date(begin_date)
    end_date = dt_date(end_date)
    if (end_date - begin_date).days < 30:
        # Build the URL for the API request
        url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
        params = {
            "begin_date": begin_date.strftime("%Y%m%d"),
            "end_date": end_date.strftime("%Y%m%d"),
            "station": station_id,
            "product": product,
            "datum": datum,
            "units": units,
            "time_zone": time_zone,
            "application": application,
            "format": output_format,
        }
        if product == "predictions":
            params["interval"] = interval

        response = requests.get(url, params=params, timeout=60)

        # Check if the request was successful
        if response.status_code != 200:
            # If the request was not successful, raise an error
            msg = f"Request returned status code {response.status_code}"
            msg = f"Response:{response.text}"
            msg += f"URL: {response.url}"
            raise ValueError(msg)

        if "Error: No data was found" in response.text:
            raise EmptyDataError(response.text.split("\n")[1])

        # If the request was successful, read the content into an xarray dataarray
        if response.headers["Content-Type"] in [
            "text/csv",
            "text/comma-separated-values",
        ]:
            data = pd.read_csv(StringIO(response.text))
        elif response.headers["Content-Type"] == "application/json":
            data = pd.read_json(response.text)
        else:
            raise ValueError(
                f"Unrecognized Content-Type: {response.headers['Content-Type']}"
            )
        data.rename(lambda x: x.strip(), axis=1, inplace=True)
        data["Date Time"] = pd.to_datetime(data["Date Time"])
        data.set_index("Date Time", inplace=True)
        data = data.dropna()
    else:
        data = process_date_range(
            begin_date, end_date, station_id, product=product, workers=workers
        )

    if data is not None:
        data = data[data.index >= begin_date]
        data = data[data.index < end_date]

    return data


def process_date_range(
    start_date, end_date, station_id, product="water_level", workers=8
):
    # Divide the date range into chunks of one month intervals
    intervals = []
    start_date = dt_date(start_date)
    end_date = dt_date(end_date)
    current_date = start_date
    while current_date < end_date:
        next_date = current_date + timedelta(days=29)
        intervals.append((current_date, next_date))
        current_date = next_date

    # Call the command line tool in parallel on each interval
    df_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_input = {
            executor.submit(
                subprocess.run,
                [
                    "noaa_data",
                    "get",
                    str(station_id),
                    "-p",
                    product,
                    "-b",
                    str_date(interval[0]),
                    "-e",
                    str_date(interval[1] - timedelta(seconds=1)),
                ],
                capture_output=True,
            ): interval
            for interval in intervals
        }
        with alive_bar(
            len(future_to_input),
            unknown="waves",
            bar="bubbles",
            spinner="dots_waves",
            receipt=False,
        ) as bar:
            for future in concurrent.futures.as_completed(future_to_input):
                bar()  # update the progress bar
                input = future_to_input[future]
                try:
                    output = future.result().stdout
                    df = pd.read_csv(
                        StringIO(output.decode("utf-8")),
                        delimiter=",",
                    )
                    df.rename(lambda x: x.strip(), axis=1, inplace=True)
                    df["Date Time"] = pd.to_datetime(df["Date Time"])
                    df.set_index("Date Time", inplace=True)
                    df = df.dropna()
                    df_list.append(df)
                except KeyError:
                    print(f"{input} generated no data")
                    pass
                except EmptyDataError:
                    print(f"{input} generated no data")
                    pass
                except Exception as exc:
                    print(f"{input} generated an exception: {exc}")
                    raise exc

    # Concatenate the data
    full_df = None
    if len(df_list) > 0:
        full_df = pd.concat(df_list, sort=True).dropna()

    return full_df


def find_stations_in_bbox(bbox):
    """Return list of NOAA stations that are within a given bounding box"""
    pass


def noaa_url(
    station_id: int, start_date: str, end_date: str, product: str, params: dict = None
):

    if product not in PRODUCTS.keys():
        raise ValueError(f"Unsupported product {product}")

    web_url = f"https://tidesandcurrents.noaa.gov/{PRODUCTS[product]}.html"
    web_url += f"?id={station_id}&units=metric"
    web_url += f"&bdate={str_date(start_date)}"
    web_url += f"&edate={str_date(end_date)}"
    web_url += "&timezone=GMT&datum=msl"

    return web_url


def str_date(date, fmt="%Y%m%d"):
    """Force date to a string format"""
    date = date.strftime(fmt) if isinstance(date, datetime) else date
    return date


def dt_date(date, fmt="%Y%m%d"):
    """Force date to a string format"""
    date = date if isinstance(date, datetime) else pd.to_datetime(date)
    return date


def pull_dataset(
    station_id, start_date, end_date, products=["water_level", "predictions"], **kwargs
):
    """
    Pull Dataset

    Method to compile groups of datasets into one. Encode logic on how to merge
    different NOAA datasets accross different intervals here.
    """
    data = get_tide_data(
        start_date, end_date, station_id, "water_level", "csv", **kwargs
    )
    preds = get_tide_data(
        start_date, end_date, station_id, "predictions", "csv", **kwargs
    )
    if data is None or preds is None:
        raise EmptyDataError("No data found")
    data["Prediction"] = preds["Prediction"][data.index]

    return data


def wicks_2017_algo(
    data,
    trigger_threshold=1.0,
    continuity_threshold=0.9,
    lull_duration=21600,
    shoulder_period=43200,
    chute_rule=9,
    interactive=True,
):
    """
    Wicks 2017 Algorithm

    TODO: Document
    """
    continuity_threshold = continuity_threshold * trigger_threshold
    data["Difference"] = abs(data["Prediction"] - data["Water Level"])
    data["TriggerThreshold"] = data["Difference"].apply(lambda x: x > trigger_threshold)
    data["ContinuityThreshold"] = data["Difference"].apply(
        lambda x: x > continuity_threshold
    )
    data["Group"] = (
        data["TriggerThreshold"].ne(data["TriggerThreshold"].shift()).cumsum()
    )

    found_events = []

    # Merge groups < lull_duration or that don't go below ContinuityThreshold
    index = 2 if data["TriggerThreshold"][0] else 3
    groups = data["Group"].values
    unique_groups = list(set(groups))
    datetimes = data["Date Time"].values
    while index <= unique_groups[-1]:
        group_idxs = np.where(groups == index)[0]
        times = pd.to_datetime(datetimes[group_idxs])
        duration = (times.max() - times.min()).seconds
        # Plot the data using asciichartpy
        if (group_idxs[-1] + 1) >= len(groups):
            next_index = None
        else:
            next_index = groups[group_idxs[-1] + 1]
        previous_index = groups[group_idxs[0] - 1]
        continuity_condition = (
            data["ContinuityThreshold"].iloc[group_idxs].eq(True).all()
        )
        merge = False
        reason = f"Distinct group {previous_index} found"
        if continuity_condition:
            reason = "Continuity condition satisfied"
            merge = True
        elif duration <= lull_duration:
            reason = "Period less than lull period"
            merge = True
        if merge:
            # merging this index, with the next two, and setting the next
            # current index to the third from the current one
            unique_groups = list(set(groups))
            unique_idx = unique_groups.index(index)
            ng = len(unique_groups)

            previous_index = unique_groups[unique_idx - 1]
            gt = groups >= previous_index

            new_index = None
            next_index = None
            if (unique_idx + 2) < ng:
                new_index = unique_groups[unique_idx + 2]
                next_index = unique_groups[unique_idx + 1]
            elif (unique_idx + 1) < ng:
                next_index = unique_groups[unique_idx + 1]

            if next_index is not None:
                lt = groups <= next_index
                new_group_idxs = np.logical_and(lt, gt)
            else:
                new_group_idxs = lt
            groups[new_group_idxs] = previous_index
            if interactive:
                if next_index is not None:
                    int_text = f"[{previous_index}, {index}, {next_index}]"
                else:
                    int_text = f"[{previous_index}, {index}]"
                ac.text_line_plot(
                    data["Date Time"][new_group_idxs].values,
                    data["Difference"][new_group_idxs].values,
                    threshold=[trigger_threshold, continuity_threshold],
                    title=f"{reason} - Merging {int_text}",
                    clear=True,
                    fmt="{: 3.3f}m",
                    hold_end=False,
                    scale_to_fit=True,
                )
            index = new_index
        else:
            index += 2
            found_idxs = np.where(groups == previous_index)[0]

            # Add shoulder period
            shoulder_timesteps = int(
                pd.to_timedelta(shoulder_period, "S") / pd.to_timedelta(6, "m")
            )
            found_idxs = np.hstack(
                [
                    np.arange(found_idxs[0] - shoulder_timesteps, found_idxs[0]),
                    found_idxs,
                    np.arange(found_idxs[-1], found_idxs[-1] + shoulder_timesteps),
                ]
            )

            # Apply Chute rule
            found_idxs = np.hstack(
                [
                    np.arange(found_idxs[0] - chute_rule, found_idxs[0]),
                    found_idxs,
                    np.arange(found_idxs[-1], found_idxs[-1] + chute_rule),
                ]
            )

            groups[new_group_idxs] = previous_index

            # Add event to found events
            event = [pd.to_datetime(datetimes[found_idxs]), found_idxs]
            if interactive:
                hrs = pd.to_timedelta(6 * len(found_idxs), "m").seconds / (60 * 60)
                response = ac.text_line_plot(
                    data["Date Time"][found_idxs].values,
                    data["Difference"][found_idxs].values,
                    threshold=[continuity_threshold, trigger_threshold],
                    title=f"Storm Surge Event Found ({hrs}H) : ",
                    clear=False,
                    fmt="{: 3.3f}m",
                    hold_end=True,
                    scale_to_fit=True,
                    prompt="\n ==== Keep event? (Y/N) ==== \n",
                )
                if response == "Y":
                    found_events.append(event)
            else:
                found_events.append(event)

    return found_events


# def process_tide_data(raw_data):
#     """
#     Process tide data
#
#     """
#     dfs = [raw_data] if not isinstance(raw_data, list) else raw_data
#     data = []
#     for df in dfs:
#         df.rename(lambda x: x.strip(), axis=1, inplace=True)
#         df["Date Time"] = pd.to_datetime(df["Date Time"])
#         df.set_index("Date Time", inplace=True)
#         df = df.dropna()
#
#         if "Water Level" in df.columns:
#             # Convert the "Water Level" column to an xarray data array
#             data.append(
#                 xr.DataArray(df["Water Level"], name="Water Level", dims=("Date Time"))
#             )
#             # Convert the "Sigma" column to an xarray data array
#             data.append(xr.DataArray(df["Sigma"], name="Sigma", dims=("Date Time")))
#         if "Prediction" in df.columns:
#             # Convert the "Water Level" column to an xarray data array
#             data.append(
#                 xr.DataArray(df["Prediction"], name="Prediction", dims=("Date Time"))
#             )
#
#     # Merge data found
#     ds = xr.merge(data)
#
#     return ds
#
#
# def get_station_data(station_id, start_date, end_date, concat=True):
#     """
#     Scan Date Range
#
#     Scan a date range for timestampes when water levels exceeded
#     a certain threshold at a certain station
#     """
#
#     def _pull_data(st, et):
#         """Should always pull"""
#         print(f"Getting chunk {st} - {et}")
#         chunk = []
#         chunk.append(get_tide_data(st, et, station_id, "water_level", "csv"))
#         chunk.append(get_tide_data(st, et, station_id, "predictions", "csv"))
#         try:
#             chunk = process_tide_data(chunk)
#         except dateutil.parser._parser.ParserError:
#             chunk = xr.Dataset(
#                 coords={
#                     "Date Time": xr.DataArray(
#                         np.array([], dtype="datetime64[ns]"), dims=["Date Time"]
#                     )
#                 },
#                 data_vars={
#                     "Water Level": (["Date Time"], []),
#                     "Sigma": (["Date Time"], []),
#                     "Prediction": (["Date Time"], []),
#                 },
#             )
#         chunk["Residual"] = chunk["Water Level"] - chunk["Prediction"]
#         print(f"\tPulled {len(chunk['Date Time'])} records")
#         return chunk
#
#     def _recurse_help(sd, ed):
#         d = (ed - sd).days
#         print(f"Scanning date range {sd} - {ed} = {d} Days")
#         if d < 31:
#             return [_pull_data(sd, ed)]
#         else:
#             chunk_end = sd + pd.to_timedelta(31, "D")
#             return [_pull_data(sd, chunk_end)] + _recurse_help(chunk_end, ed)
#
#     data = _recurse_help(dt_date(start_date), dt_date(end_date))
#     if concat:
#         data = xr.concat(data, dim="Date Time")
#
#     return data
#


def noaa_coops_get_data(
    station_id,
    begin_date,
    end_date,
    product,
    datum="msl",
    units="metric",
    interval=6,
    time_zone="gmt",
    application="pyadcirc",
):
    """
    Get data using noaa-coops python api

    """
    begin_date_str = (dt_date(begin_date).strftime("%Y%m%d"),)
    end_date_str = (dt_date(end_date).strftime("%Y%m%d"),)

    df_product = nc.Station(station_id).get_data(
        begin_date=begin_date_str,
        end_date=end_date_str,
        product=product,
        datum=datum,
        units=units,
        time_zone=time_zone,
        interval=interval,
    )

    return df_product


begin_date and end_date
begin_date and range
end_date and range
date
range
