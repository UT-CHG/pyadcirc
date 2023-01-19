"""
noaa - Utilities for pulling NOAA Tide station data

See NOAA Websites for more information:
    - https://tidesandcurrents.noaa.gov
    - https://tidesandcurrents.noaa.gov/stations.html?type=Water+Levels
"""
import concurrent.futures
import pdb
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from alive_progress import alive_bar
from pyadcirc.viz import asciichart as ac
from pyadcirc.data.utils import PRODUCTS, DATUMS, TIME_ZONES, \
                                UNITS, INTERVALS, FORMATS, DATE_TIME, \
                                STATION_IDS, NOAA_STATIONS, REGIONS
import numpy as np
import pandas as pd
import requests
from pandas.errors import EmptyDataError


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
        "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/" +
        f"{station_id}.json?expand=details?units=metric")

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


def get_time_window(
    params: dict,
    begin_date: datetime,
    end_date: datetime,
    date: datetime,
    date_range: float,
) -> dict:
    """
    Get time window
    """
    b_dt = None if begin_date is None else dt_date(begin_date)
    b_dt_str = None if begin_date is None else b_dt.strftime("%Y%m%d")
    e_dt = None if end_date is None else dt_date(end_date)
    e_dt_str = None if end_date is None else e_dt.strftime("%Y%m%d")

    # Check valid combination of parameters is sepcified
    if begin_date is not None and date_range is not None:
        params["begin_date"] = b_dt_str
        params["range"] = int(date_range)
        e_dt = b_dt + timedelta(hours=date_range)
    elif begin_date is not None and end_date is not None:
        params["begin_date"] = b_dt_str
        params["end_date"] = e_dt_str
    elif end_date is not None and date_range is not None:
        params["end_date"] = dt_date(end_date).strftime("%Y%m%d")
        params["range"] = int(date_range)
        b_dt = e_dt - timedelta(hours=date_range)
    elif date is not None:
        if (params["product"]
                not in ["water_level", "one_minut_water_level", "predictions"]
                or PRODUCTS[params["product"]]["group"] == "met"):
            date = date.lower().capitalize()
            e_dt = datetime.now()
            if date == 'Today':
                b_dt = e_dt - timedelta(hours=24)
            elif date == 'Latest':
                b_dt = e_dt - timedelta(hours=72)
            elif date == 'Recent':
                b_dt = e_dt - timedelta(hours=0.3)
            else:
                raise ValueError(f"Invalid date specified. Valid: {DATE_TIME.keys()}")
            params["date"] = date
        else:
            raise ValueError(
                '"Date" Param only available for preliminary water level data nd met products'
            )
    elif date_range is not None:
        # TODO: Code the following logic:
        # • If used alone, only available for preliminary water level data, meteorological data
        # • If used with a historical begin or end date, may be used with verified data
        params["range"] = int(date_range)
        e_dt = datetime.now()
        b_dt = e_dt - timedelta(hours=date_range)
    else:
        raise ValueError("No valid date range specified")

    return b_dt, e_dt, params


def divide_date_range(params: dict, begin_date: datetime,
                      end_date: datetime) -> dict:

    config = PRODUCTS[params['product']]
    if 'interval' in params.keys():
        interval = params['interval']
    else:
        interval = "6"

    # If max interval key in config, then pre-configured interval for product
    max_interval = config.get("max_interval")
    if max_interval is None:
        if config["group"] == "met":
            if interval == "h":
                max_interval = timedelta(days=364)
            elif interval == "6":
                max_interval = timedelta(days=29)
            else:
                raise ValueError("Met products can only have intervals [6, h]")
        elif params['product'] == 'predictions':
            max_interval = timedelta(days=364)
        else:
            max_interval = timedelta(days=29)

    if end_date - begin_date >= max_interval:
        # Divide the date range into chunks of one month intervals
        params_list = []
        current_date = begin_date
        for k in ["begin_date", "end_date", "date", "range"]:
            params.pop(k, None)
        while current_date < end_date:
            next_date = current_date + max_interval

            p = params.copy()
            p["begin_date"] = current_date.strftime("%Y%m%d")
            p["end_date"] = next_date.strftime("%Y%m%d")
            params_list.append(p)

            current_date = next_date

        return params_list
    else:
        return [params]


def check_station_id(params: dict, station_id: int):
    """
    Check Staion ID
    """
    if station_id not in STATION_IDS:
        url = "Find available statiosn at http://tidesandcurrents.noaa.gov/map/"
        raise ValueError(f"Invalid station id {station_id}. {url}")

    params['station'] = station_id

    return params


def check_datum(params: dict, datum: str):
    """
    Check datum

    TODO: add logic checking according to
    https://api.tidesandcurrents.noaa.gov/api/prod/#station to process args
    """
    avail = DATUMS.keys()
    if datum.upper() not in avail:
        raise ValueError(f"Invalid datum {datum}. Possible values: {avail}")

    params["datum"] = datum

    return params


def check_units(params: dict, units: str):
    """
    Check units

    TODO: add logic checking according to
    https://api.tidesandcurrents.noaa.gov/api/prod/#station to process args
    """
    avail = UNITS.keys()
    if units.lower() not in avail:
        raise ValueError(f"Invalid units {units}. Possible values: {avail}")

    params["units"] = units

    return params


def check_tz(params: dict, time_zone: str):
    """
    Check Time Zone

    TODO: add logic checking according to
    https://api.tidesandcurrents.noaa.gov/api/prod/#station to process args
    """
    avail = TIME_ZONES.keys()
    if time_zone.lower() not in avail:
        raise ValueError(
            f"Invalid time_zone {time_zone}. Possible values: {avail}")

    params["time_zone"] = time_zone

    return params


def check_interval(params: dict, interval: str):
    """
    Check Interval

    TODO: add logic checking according to
    https://api.tidesandcurrents.noaa.gov/api/prod/#station to process args
    """
    avail = INTERVALS
    if str(interval).lower() not in avail:
        raise ValueError(
            f"Invalid interval {interval}. Possible values: {avail}")

    config = PRODUCTS[params['product']]
    if params["product"] == "predictions" or config['group'] == 'met':
        params["interval"] = interval

    return params


def check_format(params: dict, output_format: str):
    """
    Check Output Format

    TODO: add logic checking according to
    https://api.tidesandcurrents.noaa.gov/api/prod/#station to process args
    """
    avail = FORMATS.keys()
    if output_format.lower() not in avail:
        raise ValueError(
            f"Invalid output format {output_format}. Possible values: {avail}")

    params["format"] = output_format

    return params


def check_product(params: dict, product: str):
    """
    Check Output Format

    TODO: add logic checking according to
    https://api.tidesandcurrents.noaa.gov/api/prod/#station to process args
    """
    avail = PRODUCTS.keys()
    product = product.lower()
    if product not in avail:
        raise ValueError(
            f"Invalid output format {product}. Possible values: {avail}")

    params["product"] = product

    return params


def get_tide_data(
    station_id,
    product='predictions',
    begin_date=None,
    end_date=None,
    date='RECENT',
    date_range=None,
    output_format="csv",
    datum="msl",
    time_zone="lst_ldt",
    units="metric",
    application="pyadcirc",
    interval=6,
    workers=6,
):
    """
    Get tide data
    """
    params = check_station_id({}, station_id)
    params = check_product(params, product)
    params = check_datum(params, datum)
    params = check_units(params, units)
    params = check_tz(params, time_zone)
    params = check_format(params, output_format)
    params = check_interval(params, interval)
    params["application"] = application

    b_dt, e_dt, params = get_time_window(params, begin_date, end_date, date,
                                         date_range)
    params_list = divide_date_range(params, b_dt, e_dt)
    data = process_date_range(params_list)

    if data is not None:
        data = data[data.index >= dt_date(begin_date)]
        data = data[data.index < dt_date(end_date)]

    return data


def _make_request(params):

    # Build the URL for the API request
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
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
            f"Unrecognized Content-Type: {response.headers['Content-Type']}")

    return data


def process_date_range(params, workers=8):
    # Call the command line tool in parallel on each interval
    df_list = []
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers) as executor:
        future_to_input = {
            executor.submit(_make_request, p): p
            for p in params
        }
        with alive_bar(
                len(future_to_input),
                unknown="waves",
                bar="bubbles",
                spinner="dots_waves",
                receipt=False,
                force_tty=True,
        ) as bar:
            for future in concurrent.futures.as_completed(future_to_input):
                bar()  # update the progress bar
                input = future_to_input[future]
                try:
                    df = future.result()
                    df.rename(lambda x: x.strip(), axis=1, inplace=True)
                    df["Date Time"] = pd.to_datetime(df["Date Time"])
                    df.set_index("Date Time", inplace=True)
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
        full_df = pd.concat(df_list, sort=True)
    full_df = full_df.sort_index()

    return full_df


def find_stations_in_bbox(bbox):
    """Return list of NOAA stations that are within a given bounding box"""
    pass


def noaa_url(station_id: int,
             start_date: str,
             end_date: str,
             product: str,
             params: dict = None):

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


def wicks_2017_algo(
    data,
    trigger_threshold=1.0,
    continuity_threshold=0.9,
    lull_duration=21600,
    shoulder_period=43200,
    chute_rule=9,
    interactive=True,
    debug=False
):
    """
    Wicks 2017 Algorithm

    TODO: Document
    """
    data.sort_index()
    continuity_threshold = continuity_threshold * trigger_threshold
    data["Difference"] = abs(data["Prediction"] - data["Water Level"])
    data["TriggerThreshold"] = data["Difference"].apply(
        lambda x: x > trigger_threshold)
    data["ContinuityThreshold"] = data["Difference"].apply(
        lambda x: x > continuity_threshold)
    data["Group"] = (data["TriggerThreshold"].ne(
        data["TriggerThreshold"].shift()).cumsum())

    found_events = []

    # Merge groups < lull_duration or that don't go below ContinuityThreshold
    index = 2 if data["TriggerThreshold"][0] else 3
    groups = data["Group"].values
    datetimes = data.index
    event_idx = 0
    while index is not None:
        # Indices in group column equal to current index
        group_idxs = np.where(groups == index)[0]

        # Calculate the duration of this period
        times = pd.to_datetime(datetimes[group_idxs])
        duration = (times.max() - times.min()).seconds

        # Get previous and next indices (adjacent periods above trigger or < lull)
        previous_index = groups[group_idxs[0] - 1]
        next_index = None
        next_next_index = None
        if group_idxs[-1] + 1 < len(groups):
            next_index = groups[group_idxs[-1] + 1]
            next_group_idxs = np.where(groups == next_index)[0]
            if len(next_group_idxs) > 0:
                if next_group_idxs[-1] + 1 < len(groups):
                    next_next_index = groups[next_group_idxs[-1] + 1]

        # Check continuity condition and Lull Duration conditions
        continuity_condition = (
            data["ContinuityThreshold"].iloc[group_idxs].eq(True).all())
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
            new_group_idxs = groups >= previous_index
            if next_index is not None:
                new_group_idxs = np.logical_and(new_group_idxs,
                                                groups <= next_index)

            groups[new_group_idxs] = previous_index
            if debug:
                if next_index is not None:
                    int_text = f"[{previous_index}, {index}, {next_index}]"
                else:
                    int_text = f"[{previous_index}, {index}]"
                ac.text_line_plot(
                    data.index[new_group_idxs].values,
                    data["Difference"][new_group_idxs].values,
                    threshold=[trigger_threshold, continuity_threshold],
                    title=f"{reason} - Merging {int_text}",
                    clear=True,
                    fmt="{: 3.3f}m",
                    hold_end=True,
                    scale_to_fit=True,
                )
        else:
            found_idxs = np.where(groups == previous_index)[0]

                pd.to_timedelta(shoulder_period, "S") /
                pd.to_timedelta(6, "m"))
            found_idxs = np.hstack([
                np.arange(found_idxs[0] - shoulder_timesteps, found_idxs[0]),
                found_idxs,
                np.arange(found_idxs[-1], found_idxs[-1] + shoulder_timesteps),
            ])

            # Apply Chute rule
            found_idxs = np.hstack([
                np.arange(found_idxs[0] - chute_rule, found_idxs[0]),
                found_idxs,
                np.arange(found_idxs[-1], found_idxs[-1] + chute_rule),
            ])

            # Add event to found events
            event = data.iloc[found_idxs].copy()
            event["Event Number"] = event_idx
            if interactive:
                hrs = pd.to_timedelta(6 * len(found_idxs),
                                      "m").seconds / (60 * 60)
                response = ac.text_line_plot(
                    data.index[found_idxs].values,
                    data["Difference"][found_idxs].values,
                    threshold=[continuity_threshold, trigger_threshold],
                    title=f"Storm Surge Event Found ({hrs}H) : ",
                    clear=False,
                    fmt="{: 3.3f}m",
                    hold_end=True,
                    scale_to_fit=True,
                    prompt="\n ==== Keep event? (Y/N) ==== \n",
                )
                if response.strip().upper() == "Y":
                    found_events.append(event)
                    event_idx += 1
            else:
                found_events.append(event)
                event_idx += 1

        index = next_next_index
    # if len(found_events) > 0:
    #     found_events = pd.concat(found_events)
    # else:
    #     found_events = None

    return found_events


def pull_dataset(station_id,
                 **kwargs):
    """
    Pull Dataset
    Method to compile groups of datasets into one. Encode logic on how to merge
    different NOAA datasets accross different intervals here.
    """
    data = get_tide_data(station_id, product="water_level", **kwargs)
    preds = get_tide_data(station_id, product="predictions", **kwargs)

    if data is None or preds is None:
        raise EmptyDataError("No data found")

    data = data.merge(preds, on='Date Time', how='left')
    data = data.sort_index()
    data = data[['Sigma', 'Water Level', 'Prediction']]

    return data

