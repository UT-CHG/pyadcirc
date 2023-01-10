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

import dateutil
import numpy as np
import pandas as pd
import requests
import xarray as xr
from pandas.errors import EmptyDataError

from pyadcirc.utils import get_bbox

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


def get_tide_data(
    begin_date,
    end_date,
    station_id,
    product,
    output_format,
    datum="msl",
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
            "time_zone": "lst_ldt",
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
):
    """
    Wicks 2017 Algorithm
    """
    data["Difference"] = abs(data["Prediction"] - data["Water Level"])
    data["TriggerThreshold"] = data["Difference"].apply(lambda x: x > trigger_threshold)
    data["ContinuityThreshold"] = data["Difference"].apply(
        lambda x: x > 0.9 * trigger_threshold
    )
    data["Group"] = (
        data["TriggerThreshold"].ne(data["TriggerThreshold"].shift()).cumsum()
    )

    # Merge groups < lull_duration or that don't go below ContinuityThreshold
    index = 2 if data["TriggerThreshold"][0] else 3
    groups = data["Group"].values
    datetimes = data["Date Time"].values
    while index <= len(groups):
        group_idxs = np.where(groups == index)[0]
        times = pd.to_datetime(datetimes[group_idxs])
        duration = (times.max() - times.min()).seconds
        # Plot the data using asciichartpy
        next_index = groups[group_idxs[-1] + 1]
        previous_index = groups[group_idxs[0] - 1]
        # print(f"Checking gap {index} ({duration}s) between {datetimes[previous_index]} and {datetimes[next_index]}")
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
            next_index = unique_groups[unique_idx + 1]
            previous_index = unique_groups[unique_idx - 1]
            new_index = unique_groups[unique_idx + 2]
            lt = groups <= next_index
            gt = groups >= previous_index
            new_group_idxs = np.logical_and(lt, gt)
            groups[new_group_idxs] = previous_index
            ac.text_line_plot(
                data["Date Time"][new_group_idxs].values,
                data["Difference"][new_group_idxs].values,
                title=f"{reason} - Merging [{previous_index}, {index}, {next_index}]",
                clear=True,
                fmt="{: 3.3f}m",
                hold_end=False,
                scale_to_fit=True,
            )
            index = new_index
        else:
            index += 2
            plot_idxs = np.where(groups == previous_index)[0]
            ac.text_line_plot(
                data["Date Time"][plot_idxs].values,
                data["Difference"][plot_idxs].values,
                title="Possible storm Surge event found",
                clear=False,
                fmt="{: 3.3f}m",
                hold_end=True,
                scale_to_fit=True,
            )

    pdb.set_trace()

    return data


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
