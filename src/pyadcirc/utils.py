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
with open(str(Path(__file__).resolve().parent / 'adcirc_configs.json'),
          'r') as ac:
    ADCIRC_PARAM_DEFS = json.load(ac)


@contextmanager
def timing(label: str):
  t0 = perf_counter()
  yield lambda: (label, t1 - t0)
  t1 = perf_counter()


def get_param_def(param:str):
  try:
    desc = ADCIRC_PARAM_DEFS[param]
    pprint.pp(f'{param} = {desc}')
  except:
    print(f"Did not find parameter {param}'")
    pass


def deploy_tapis_app():
    pass
