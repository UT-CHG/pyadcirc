"""
noaa - Utilities for pulling NOAA Tide station data

See NOAA Websites for more information:
    - https://tidesandcurrents.noaa.gov
    - https://tidesandcurrents.noaa.gov/stations.html?type=Water+Levels
"""
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import xarray as xa
from pyadcirc.utils import get_bbox

# Path to json file with noaa station data pulled from website
NOAA_STATIONS = pd.read_json(Path(Path(__file__).parents[1] / "configs/noaa_stations.json"))


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
    -------
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
    response = requests.get(url)
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


def get_station_data(station_id: int, start_date: str, end_date: str):
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
    Pull a single data point:

    >>> data = get_station_data(9468756, '20200101 00:00', '20200101 00:01')
    >>> data.values
    array([-1.209])

    Bad station ID should return no data

    >>> data = get_station_data(1, '20200101 00:00', '20200101 00:01')
    >>> len(data)==0
    True
    """
    # Make api request to get data in csv form
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?"
    params = {
        "begin_date": start_date,
        "end_date": end_date,
        "station": station_id,
        "product": "water_level",
        "datum": "msl",
        "units": "metric",
        "time_zone": "gmt",
        "application": "utaustin_chg",
        "format": "csv",
    }

    page = requests.get(url, params=params)

    # Convert data to xarray dataset, indexed by time
    df = pd.read_csv(StringIO(str(page.content, "utf-8")))
    if " Water Level" in df.columns:
        ds = xa.DataArray(
            df[" Water Level"],
            name="water_levels",
            coords={"time": pd.to_datetime(df["Date Time"])},
            dims={"time": pd.to_datetime(df["Date Time"])},
        )
        ds.set_index(time="time")
    else:
        print("Warning: Request returned no data")
        ds = xa.DataArray([], coords={"time": []}, dims={"time": pd.to_datetime([])})

    # Store details about request made in attrs
    web_url = "https://tidesandcurrents.noaa.gov/waterlevels.html"
    web_url += f"?id={station_id}&units=metric"
    web_url += f"&bdate={start_date.strftime('%Y%m%d')}"
    web_url += f"&edate={end_date.strftime('%Y%m%d')}"
    web_url += "&timezone=GMT&datum=msl"
    params.update({'api_url': url,
                   'web_url': web_url,
                   'description': 'NOAA/NOS/CO-OPS Observed Water Levels',
                   'units': 'meters'})
    ds.attrs = params

    return ds

def find_stations_in_bbox(bbox):
    """Return list of NOAA stations that are within a given bounding box"""
    pass
