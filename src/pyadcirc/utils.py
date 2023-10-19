"""
utils - Utility functions to use throughout ADCIRC suite.

"""

import argparse
import glob
import io
import json
import linecache as lc
import logging
import os
import pdb
import pprint
import re
import subprocess
import sys
import urllib.request
from contextlib import contextmanager
from functools import reduce
from pathlib import Path
from time import perf_counter, sleep
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from geopy.distance import distance
from rich.console import Console
from rich.table import Table
from termcolor import colored

from pyadcirc import __version__

__author__ = "Carlos del-Castillo-Negrete"
__copyright__ = "Carlos del-Castillo-Negrete"
__license__ = "MIT"
_logger = logging.getLogger(__name__)


logger = logging.getLogger()

# Read in ADCIRC Parameter Configuration Dictioanry
with open(
    str(Path(__file__).resolve().parent / "configs/adcirc_configs.json"), "r"
) as ac:
    ADCIRC_PARAM_DEFS = json.load(ac)


@contextmanager
def timing(label: str):
    t0 = perf_counter()
    yield lambda: (label, perf_counter() - t0)
    t1 = perf_counter()


def get_param_def(param: str):
    try:
        desc = ADCIRC_PARAM_DEFS[param]
        pprint.pp(f"{param} = {desc}")
    except:
        print(f"Did not find parameter {param}'")
        pass


def deploy_tapis_app():
    pass


def get_bbox(f14: xr.Dataset, scale_x: float = 0.1, scale_y: float = 0.1):
    """
    Get Long/Lat bounding box containing grid in f14_file.
    Computes bounding box using scale parameters where each bound
    is determined as follows:

        max_bound = max + scale * range
        min_bound = min - scale * range

    Parameters
    ----------
    f14_file : str
        Path to fort.14 ADCIRC grid file.
    scale_x : float, default=0.1
        What percent of total longitude range to add to ends
        of longitude min/max for determining bounding box limits.
    scale_y : float, default=0.1
        What percent of total latitude range to add to ends
        of latitude min/max for determining bounding box limits.


    Returns
    -------
    bbox : List[List[float]]
        Long/lat bounding box list in the form `[west,east,south,north]`.

    """

    bounds = [
        [f14["X"].values.min(), f14["X"].values.max()],
        [f14["Y"].values.min(), f14["Y"].values.max()],
    ]
    buffs = [
        (bounds[0][1] - bounds[0][0]) * scale_x,
        (bounds[1][1] - bounds[1][0]) * scale_y,
    ]
    bbox = [
        bounds[0][0] - buffs[0],
        bounds[0][1] + buffs[0],
        bounds[1][0] - buffs[1],
        bounds[1][1] + buffs[1],
    ]

    return bounds, bbox


def regrid(arr, old_lat, old_lon, new_lat, new_lon):
    """
    Regrid an array from one regular lat-lon grid to another

    Assumes that the new grid is finer-scale

    Credit: Benjamin Pachev
    """
    print("newgrid", type(old_lat), type(new_lat))
    lat_inds = np.searchsorted(old_lat, new_lat)
    lon_inds = np.searchsorted(old_lon, new_lon)
    nlat, nlon = len(old_lat), len(old_lon)
    lat_inds = np.clip(lat_inds, 1, nlat - 1)
    lon_inds = np.clip(lon_inds, 1, nlon - 1)
    lat_inds_lower = lat_inds - 1
    lon_inds_lower = lon_inds - 1

    # now determine interpolation weights

    lat_weights = (new_lat - old_lat[lat_inds_lower]) / (
        old_lat[lat_inds] - old_lat[lat_inds_lower]
    )
    lon_weights = (new_lon - old_lon[lon_inds_lower]) / (
        old_lon[lon_inds] - old_lon[lon_inds_lower]
    )
    print(type(lat_weights), type(lon_weights))
    print(arr.shape, len(lat_weights), len(lon_weights))
    lat_weights = lat_weights.reshape((1, 1, len(lat_weights), 1))
    lon_weights = lon_weights.reshape((1, 1, 1, len(lon_weights)))

    out = (
        lat_weights * arr[..., lat_inds_lower, :]
        + (1 - lat_weights) * arr[..., lat_inds, :]
    )
    out = (
        lon_weights * out[..., lon_inds_lower] + (1 - lon_weights) * out[..., lon_inds]
    )
    print(f"Old mean {arr.mean()}, new mean {out.mean()}")
    return out


# TODO: Move to this and check_file status to utils?
def sizeof_fmt(num, suffix="B"):
    """
    Formats number representing bytes to string with appropriate size unit.

    Parameters
    ----------
    num : int,float
        Number to convert to string with bytes unit.
    suffix : str, default=B
        Suffix to use for measurement. Kilobytes will be KiB with default.

    Returns
    -------
    fmt_str : str
        Formatted string with bytes units.

    Notes
    -----
    Taken from
    stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
    """
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Yi{suffix}"


def check_file_status(filepath, filesize):
    """
    Check and print a file status by computing current size / filesize.
    """
    sys.stdout.write("\r")
    sys.stdout.flush()
    size = int(os.stat(filepath).st_size)
    percent_complete = (size / filesize) * 100
    sys.stdout.write("%.3f %s" % (percent_complete, "% Completed"))
    sys.stdout.flush()


def make_rich_table(
    data: Union[pd.DataFrame, pl.DataFrame, dict],
    fields: List[str] = None,
    title: Optional[str] = None,
    search: Optional[str] = None,
    match: str = r".",
    filter_fun: Optional[Callable[[pd.Series], pd.Series]] = None,
    formats: Optional[List[dict]] = None,
    to_str: bool = True,
) -> str:
    """
    Makes a table using the rich library.

    Parameters
    ----------
    data : Union[pd.DataFrame, pl.DataFrame, dict]
        DataFrame containing response data.
    fields : List[str]
        List of strings containing names of fields to extract for each element.
    title : Optional[str], default=None
        Title for the table.
    search : Optional[str], default=None
        String containing column to perform string pattern matching on to filter results.
    match : str, default='.'
        Regular expression to match strings in the search column.
    filter_fun : Optional[Callable[[pd.Series], pd.Series]], default=None
        Function to filter each row in the DataFrame.
    formats : Optional[List[dict]], default=None
        List of dictionaries containing formatting options for each column. Must be the same length as fields.
    to_str : bool, default = True
        Whether to return the table as a string or print it.

    Returns
    -------
    str
        String representation of the rich table.
    """
    # Initialize console and table
    console = Console()
    table = Table(show_header=True, header_style="bold blue")

    # Handle dict input
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data

    if fields is None:
        # Only take first 10
        fields = list(df.columns)[:10]

    # Handle title
    if title is not None:
        if title in df:
            unique_vals = df[title].unique()
            title_text = ", ".join(map(str, unique_vals))[:30]
            if len(title_text) >= 30:
                title_text += "..."
            df = df.drop(columns=[title])
            if title in fields:
                fields.remove(title)
        else:
            title_text = title[:30]
        table.title = title_text

    # Validate formats
    if formats is None:
        formats = len(fields) * [{"style": "blue"}]
    elif len(formats) != len(fields):
        raise ValueError("Formats must be the same length as fields")

    # Add columns to table with formatting options
    for i, field in enumerate(fields):
        table.add_column(field, **formats[i])

    # Build table from DataFrame
    for _, r in df.dropna().iterrows():
        if filter_fun is not None:
            r = filter_fun(r)

        if search is not None:
            if re.search(match, r[search]) is not None:
                table.add_row(*[str(r[f]) for f in fields])
        else:
            table.add_row(*[str(r[f]) for f in fields])

    if to_str:
        # Create a console object that writes to a string
        console = Console(file=io.StringIO(), force_terminal=True)

        # Capture the table as a string
        console.print(table)
        table_str = console.file.getvalue()

        return table_str

    return table


# Example usage
# formats = [{"style": "bold red"}, {"style": "italic green"}]
# result = make_rich_table(df, fields=["column1", "column2"], title="My Table", formats=formats)
# print(result)
