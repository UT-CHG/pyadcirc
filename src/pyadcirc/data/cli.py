"""
data CLI - Pyadcirc data utilities
"""
import pdb
import sys
from pathlib import Path

import click
import pandas as pd
from pandas.errors import EmptyDataError
from termcolor import colored

from pyadcirc.data import noaa
from pyadcirc.data.utils import (get_banner_text, get_help_text,
                                 make_pretty_table)
from pyadcirc.viz import asciichart as ac


def _save_output(data, output_file, output_format):
    """
    Save Output
    """
    output_file = sys.stdout if output_file is None else output_file
    if output_format == "csv":
        data.to_csv(output_file, index=True)
    elif output_format == "json":
        data.to_json(output_file, index=True)


@click.group()
def data():
    """Commands for interacting with the NOAA API"""
    print(get_banner_text(title="NOAA CO-OPS"))
    pass


@data.command()
@click.option(
    "--region",
    "-r",
    default=None,
    type=click.Choice(noaa.REGIONS),
    help="Filter station list by region",
)
@click.option(
    "--name",
    "-n",
    default=r".",
    type=str,
    help="Filter station list by region (regular expression match)",
)
def stations(region=None, name="."):
    """Info on available products"""
    colored_url = colored(
        "https://tidesandcurrents.noaa.gov/", "red", attrs=["underline"]
    )
    print(f"See {colored_url} for more on stations and available products at each.")
    search = "Region" if region is not None else "Name"
    match = region if region is not None else name
    table = make_pretty_table(
        noaa.NOAA_STATIONS,
        ["ID", "Name", "Region"],
        search=search,
        match=match,
        colors=["yellow", "blue", "blue"],
    )
    print(table)


@data.command()
def info():
    """Info on available products"""
    colored_url = colored(
        "https://api.tidesandcurrents.noaa.gov/api/prod/", "red", attrs=["underline"]
    )
    print(f"See {colored_url} for more on products")
    print(get_help_text(noaa.PRODUCTS))


@data.command()
@click.argument("station_id", type=int)
@click.option(
    "--product",
    "-p",
    default="metadata",
    type=click.Choice(list(noaa.PRODUCTS.keys())),
    help=get_help_text(noaa.PRODUCTS),
)
@click.option(
    "--begin_date",
    "-b",
    type=str,
    help=colored(noaa.DATE_TIME["begin_date"], color="blue"),
)
@click.option(
    "--end_date", "-e", type=str, help=colored(noaa.DATE_TIME["end_date"], color="blue")
)
@click.option(
    "--date",
    "-t",
    type=click.Choice(["Today", "Latest", "Recent"], case_sensitive=False),
    help=noaa.DATE_TIME["date"],
)
@click.option(
    "--date_range",
    "-r",
    type=float,
    help=noaa.DATE_TIME["range"],
)
@click.option(
    "--interval",
    "-n",
    default="6",
    type=click.Choice(noaa.INTERVALS, case_sensitive=False),
)
@click.option(
    "--output_format",
    "-f",
    type=click.Choice(noaa.FORMATS.keys(), case_sensitive=False),
    default="csv",
    help=get_help_text(noaa.FORMATS),
)
@click.option(
    "--datum",
    "-d",
    default="MSL",
    type=click.Choice(noaa.DATUMS.keys(), case_sensitive=False),
    help=get_help_text(noaa.DATUMS),
)
@click.option(
    "--time_zone",
    "-z",
    default="lst_ldt",
    type=click.Choice(noaa.TIME_ZONES.keys(), case_sensitive=False),
    help=get_help_text(noaa.TIME_ZONES),
)
@click.option(
    "--units",
    "-u",
    default="metric",
    type=click.Choice(noaa.UNITS.keys(), case_sensitive=False),
    help=get_help_text(noaa.UNITS),
)
@click.option(
    "--application",
    "-a",
    type=str,
    default="pyadcirc-cli",
    help=colored(
        "".join(
            [
                "Provides an “identifier” in automated activity / error logs",
                " that allows us to identify your query from others.)",
            ]
        ),
        color="blue",
    ),
)
@click.option(
    "--output_file",
    "-f",
    type=str,
    default=None,
    help=colored(
        "".join(
            [
                "Name of file to write data to. Extension determined by the ",
                "`output_format` parameter. If no output file is specified, ",
                "result are printed",
            ]
        ),
        color="blue",
    ),
)
@click.option(
    "--workers",
    "-w",
    type=int,
    default=4,
    show_default=True,
    help=colored(
        "".join(
            [
                "Number of parallel workers to use to get data. each paralle",
                "l worker will submit individual http requests for chunks of",
                " data and process them.",
            ]
        ),
        color="blue",
    ),
)
@click.option(
    "--graph/--no-graph",
    "-g/-ng",
    is_flag=True,
    default=False,
    show_default=True,
    help=colored("Flag to print to stdout an ascii graph of data.", color="blue"),
)
def get(
    station_id,
    product,
    begin_date=None,
    end_date=None,
    date=None,
    date_range=None,
    interval=6,
    output_format="csv",
    datum="msl",
    time_zone="gmt",
    units="metric",
    application="pyadcirc-cli",
    output_file=None,
    workers=4,
    graph=False,
):
    """
    Get a product over a date range at particular station:

    noaa_data get -p water_level

    For list of available stations see:

    """
    data = None
    if product == "metadata":
        station = noaa.get_station_metadata(station_id)
        station["Longitude"] = station["coords"][0]
        station["Latitude"] = station["coords"][1]
        _ = station.pop("coords")
        data = pd.DataFrame([station])
        data.set_index("id", inplace=True)
    else:
        try:
            data = noaa.get_tide_data(
                station_id,
                product=product,
                begin_date=begin_date,
                end_date=end_date,
                date=date,
                date_range=date_range,
                output_format=output_format,
                datum=datum,
                time_zone=time_zone,
                units=units,
                interval=interval,
                application=application,
                workers=workers,
            )
        except EmptyDataError as e:
            print(e)
            return None

    if graph:
        ac.text_line_plot(
            data.index,
            data["Water Level"].values,
            scale_to_fit=True,
        )

    _save_output(data, output_file, output_format)

    return data


@data.command()
@click.argument("station_id", type=int)
@click.option("--begin_date", "-b", type=str)
@click.option("--end_date", "-e", type=str)
@click.option(
    "--date",
    "-d",
    type=click.Choice(["Today", "Latest", "Recent"], case_sensitive=False),
)
@click.option("--date_range", "-r", type=float)
@click.option("--input_file", "-i", type=str)
@click.option("--output_file", "-o", type=str)
@click.option(
    "--output_format",
    "-f",
    type=click.Choice(noaa.FORMATS.keys(), case_sensitive=False),
    default="csv",
)
@click.option(
    "--datum",
    "-d",
    default="MSL",
    type=click.Choice(noaa.DATUMS.keys(), case_sensitive=False),
)
@click.option(
    "--time_zone",
    "-z",
    default="lst_ldt",
    help="Time zone",
    type=click.Choice(noaa.TIME_ZONES.keys(), case_sensitive=False),
)
@click.option(
    "--units",
    "-u",
    type=click.Choice(noaa.UNITS.keys(), case_sensitive=False),
    default="metric",
)
@click.option(
    "--interval",
    "-n",
    default="6",
    type=click.Choice(noaa.INTERVALS, case_sensitive=False),
)
@click.option("--threshold", "-t", type=float, default=1.0)
@click.option("--workers", "-w", type=int, default=4)
@click.option("--interactive/--no-interactive", "-v", is_flag=True, default=True)
@click.option("--save-raw/--no-save-raw", is_flag=True, default=True)
def find_events(
    station_id,
    begin_date=None,
    end_date=None,
    date="Recent",
    date_range=None,
    input_file=None,
    output_file=None,
    output_format="csv",
    datum="msl",
    time_zone="lst_ldt",
    units="metric",
    interval=6,
    application="pyadcirc",
    threshold=1.0,
    workers=4,
    interactive=False,
    save_raw=True,
):
    """
    Find storm surge events
    """
    if input_file is None:
        data = noaa.pull_dataset(
            station_id,
            begin_date=begin_date,
            end_date=end_date,
            date=date,
            date_range=date_range,
            output_format="csv",
            datum=datum,
            time_zone=time_zone,
            units=units,
            interval="6",
            application=application,
            workers=workers,
        )
    else:
        data = pd.read_csv(input_file)
        data = data.set_index("Date Time")

    if output_file is not None and input_file is None and save_raw:
        raw_data_path = f"{Path(output_file).with_suffix('')}-raw.{output_format}"
        _save_output(data, raw_data_path, output_format)

    data = noaa.wicks_2017_algo(
        data, trigger_threshold=threshold, interactive=interactive
    )

    if data is not None:
        _save_output(data, output_file, output_format)

    return data
