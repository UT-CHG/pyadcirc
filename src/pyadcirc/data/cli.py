"""
data CLI - Pyadcirc data utilities
"""
import sys
import pdb
import click

import pandas as pd
from pathlib import Path
from pyadcirc.data import noaa
from pyadcirc.viz import asciichart as ac
from pandas.errors import EmptyDataError


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
    pass


@data.command()
@click.argument("station_id", type=int)
@click.option(
    "--product",
    "-p",
    default="metadata",
    type=click.Choice(list(noaa.PRODUCTS.keys())),
)
@click.option(
    "--begin_date",
    "-b",
    type=str)
@click.option(
    "--end_date",
    "-e",
    type=str)
@click.option(
    "--date",
    "-d",
    type=click.Choice(["Today", "Latest", "Recent"], case_sensitive=False),
    )
@click.option(
    "--date_range",
    "-r",
    type=float
    )
@click.option(
    "--output_file",
    "-o",
    type=str
)
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
    type=click.Choice(noaa.INTERVALS, case_sensitive=False)
)
@click.option("--application", "-a", type=str, default="pyadcirc-cli")
@click.option("--output_file", "-f", type=str, default=None)
@click.option("--workers", "-w", type=int, default=4)
@click.option("--print_graph", "-g", is_flag=True, default=True)
def get(
    station_id,
    product,
    begin_date=None,
    end_date=None,
    date=None,
    date_range=None,
    output_file=None,
    output_format="csv",
    datum="msl",
    time_zone="gmt",
    units="metric",
    application="pyadcirc-cli",
    interval=6,
    threshold=1.0,
    workers=4,
    print_graph=False,
):

    if product == "metadata":
        station = noaa.get_station_metadata(station_id)
        station["Longitude"] = station["coords"][0]
        station["Latitude"] = station["coords"][1]
        _ = station.pop("coords")
        data = pd.DataFrame([station])
        data.set_index("id", inplace=True)
    elif product in ["water_level", "predictions"]:
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

    if print_graph:
        ac.text_line_plot(
            data.index,
            data["Water Level"].values,
            threshold=threshold,
            scale_to_fit=True,
        )

    _save_output(data, output_file, output_format)

    return data


@data.command()
@click.argument("station_id", type=int)
@click.option(
    "--begin_date",
    "-b",
    type=str)
@click.option(
    "--end_date",
    "-e",
    type=str)
@click.option(
    "--date",
    "-d",
    type=click.Choice(["Today", "Latest", "Recent"], case_sensitive=False),
    )
@click.option(
    "--date_range",
    "-r",
    type=float
    )
@click.option(
    "--input_file",
    "-i",
    type=str
)
@click.option(
    "--output_file",
    "-o",
    type=str
)
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
    type=click.Choice(noaa.INTERVALS, case_sensitive=False)
)
@click.option("--threshold", "-t", type=float, default=1.0)
@click.option("--workers", "-w", type=int, default=4)
@click.option("--interactive/--no-interactive", "-v", is_flag=True, default=True)
def find_events(
    station_id,
    begin_date=None,
    end_date=None,
    date='Recent',
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

    if output_file is not None:
        raw_data_path = f"{Path(output_file).with_suffix('')}-raw.{output_format}"
        _save_output(data, raw_data_path, output_format)

    data = noaa.wicks_2017_algo(
        data, trigger_threshold=threshold, interactive=interactive
    )

    if data is not None:
        _save_output(data, output_file, output_format)

    return data


# @data.command()
# @click.argument("station_id", type=int)
# @click.option(
#     "--product",
#     "-p",
#     default="metadata",
#     type=click.Choice(["storm_surge_events", "metadata", "water_level", "predictions"]),
# )
# @click.option('--begin_date', '-b', type=str, default='20000101', help='Begin date in YYYYMMDD format')
# @click.option('--end_date', '-e', type=str, default='20000102', help='End date in YYYYMMDD format')
# @click.option('--datum', '-d', default='msl', type=str, help='Datum for water level data')
# @click.option('--units', '-u', default='metric', type=str, help='Units for data')
# @click.option('--interval', '-n', default=6, type=int, help='Interval between data points')
# @click.option('--time_zone', '-z', default='gmt', type=str, help='Time zone')
# @click.option('--application', default='pyadcirc', type=str, help='Application using the data')
# @click.option('--output_file', '-f', type=str, default=None)
# @click.option('--output_format', '-o', type=str, default='csv')
# def nc_get(station_id,
#            product,
#            begin_date,
#            end_date,
#            datum="msl",
#            units="metric",
#            interval=6,
#            time_zone="gmt",
#            application="pyadcirc",
#            output_file=None,
#            output_format="csv",
#            ):
#     """
#     Get station data using noaa-coops python api
#     """
#
#     data = noaa.noaa_coops_get_data(station_id,
#                                     begin_date,
#                                     end_date,
#                                     product,
#                                     datum=datum,
#                                     units=units,
#                                     interval=interval,
#                                     time_zone=time_zone,
#                                     application=application)
#
#     pdb.set_trace()
#
#     _save_output(data, output_file, output_format)
#
#     return data
#
#
# if __name__ == "__main__":
#     get.main()



