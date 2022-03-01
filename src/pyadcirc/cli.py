"""

adcirc_utils - A bunch of adcirc python functions.

Note:
    This skeleton file can be safely removed if not needed!

References:
    - https://setuptools.readthedocs.io/en/latest/userguide/entry_point.html
    - https://pip.pypa.io/en/stable/reference/pip_install
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



# ---- Python API ----
# The functions defined in this section can be imported by users in their
# Python scripts/interactive interpreter, e.g. via
# `from pyadcirc.skeleton import fib`,
# when using this Python module as a library.


def adcirc_to_xarray(path, filt='fort*', met_times=[0], out_path='./adc.nc'):
    """Convert adcric inputs (fort.*) to xarray netcdf file

    Args:
      path (int): Path to folder containing adcirc files
      filt (str): String to filter files to process on, defaults to 'fort.*'
      met_times (str): CSL list of time idices to pull meterological data at. n
      out_path (str): Output file path

    Returns:
      xarray: n-th Fibonacci number
    """

    res = adcirc_to_xarray(path, filt='fort.*', met_times=[])
    res.to_netcdf(out_path)

    return res


# ---- CLI ----
# The functions defined in this section are wrappers around the main Python
# API allowing them to be called directly from the terminal as a CLI
# executable/script.


def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Process ADCIRC Configs")
    parser.add_argument(
        "--version",
        action="version",
        version="pyadcirc {ver}".format(ver=__version__),
    )
    parser.add_argument(dest="path", help="Path to ADCIRC files", type=str, metavar="STR")
    parser.add_argument(
        "-f",
        "--filter",
        dest="filt",
        help="String filter for adcirc files to process.",
        default="fort.*"
    )
    parser.add_argument(
        "-m",
        "--met_idxs",
        dest="met_idxs",
        help="Indices to pull meterological data for",
        default="0"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    """Wrapper allowing :func:`fib` to be called with string arguments in a CLI fashion

    Instead of returning the value from :func:`fib`, it prints the result to the
    ``stdout`` in a nicely formated message.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Processing adcirc configs")

    res = adcirc_to_xarray(args.path, filt=args.filt, met_idxs=args.met_idxs)
    pdb.set_trace()
    print("Done Processing ADCIRC Configs")
    _logger.info("Script ends here")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    https://docs.python.org/3/library/__main__.html

    # After installing your project with pip, users can also run your Python
    # modules as scripts via the ``-m`` flag, as defined in PEP 338::
    #
    #     python -m pyadcirc.skeleton 42
    #
    run()
