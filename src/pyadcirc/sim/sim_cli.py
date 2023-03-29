"""
pyadcirc sim - CLI for ADCRIC simulations

"""

import argparse
import logging

from pyadcirc.io import merge_output
from pyadcirc import __version__

__author__ = "Carlos del-Castillo-Negrete"
__copyright__ = "Carlos del-Castillo-Negrete"
__license__ = "MIT"
_logger = logging.getLogger(__name__)


def merge_output_entry(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    # Parse command line options
    parser = argparse.ArgumentParser(descripton="Merge ADCIRC Outputs")
    parser.add_argument("output_dir", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument(
        "--stations", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--globs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--minmax", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nodals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--partmesh", action=argparse.BooleanOptionalAction, default=True
    )

    parsed_args = parser.parse_args(args)

    _logger.info(f"Starting to merge output at {args.output_dir}")
    output = merge_output(
        parsed_args.output_dir,
        parsed_args.stations,
        parsed_args.globs,
        parsed_args.minmax,
        parsed_args.nodals,
        parsed_args.partmesh,
    )
    _logger.info(f"Done merging output")
    _logger.info(f"Writing output netcdf file")
    output.to_netcdf(args.output_file)
    _logger.info(f"Done writing output file")


def main(args):
    """Wrapper allowing :func:`fib` to be called with string arguments in a CLI fashion

    Instead of returning the value from :func:`fib`, it prints the result to the
    ``stdout`` in a nicely formated message.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    sub_command = args[1]

    if sub_command == "merge_output":
        merge_output_entry(args[2:])
    else:
        raise ValueError("Unknown sub command :{sub_command}")
