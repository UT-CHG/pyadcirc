"""
utils - Utility functions to use throughout ADCIRC suite.

"""

import argparse
import logging
import sys

from pyadcirc import __version__

import re
import os
import pdb
import glob
import pprint
import json
from pathlib import Path
import logging
import subprocess
import numpy as np
import xarray as xr
import linecache as lc
import urllib.request
from functools import reduce
from geopy.distance import distance
from time import perf_counter, sleep
from contextlib import contextmanager

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
    yield lambda: (label, t1 - t0)
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

