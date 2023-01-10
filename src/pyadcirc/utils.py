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
