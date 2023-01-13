"""
data CLI - Pyadcirc data utilities
"""
import sys
import pdb
import click

import pandas as pd
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
@click.option("--begin_date", "-b", type=str, default="20000101")
@click.option("--end_date", "-e", type=str, default="20000102")
@click.option(
    "--output_format", "-o", type=click.Choice(["csv", "json", "xml"]), default="csv"
)
@click.option("--datum", "-d", type=str, default="msl")
@click.option(
    "--time_zone",
    "-z",
    default="csv",
    help="Time zone",
    type=click.Choice(["gmt", "lst", "lst_ldt"]),
)
@click.option(
    "--units",
    "-u",
    type=str,
    type=click.Choice(["metric", "english"]),
    default="metric",
)
@click.option(
    "--interval", "-n", type=str, default="6", type=click.Choice(noaa.INTERVALS)
)
@click.option("--application", "-a", type=str, default="pyadcirc")
@click.option("--output_file", "-f", type=str, default=None)
@click.option("--workers", "-w", type=int, default=4)
@click.option("--print_graph", "-g", is_flag=True, default=True)
def get(
    station_id,
    product,
    begin_date=None,
    end_date=None,
    output_file=None,
    output_format="csv",
    datum="msl",
    units="metric",
    application="pyadcirc",
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
                units=metric,
                interval=interval,
                application="pyadcirc-cli",
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
@click.option("--begin_date", "-b", type=str, default="20000101")
@click.option("--end_date", "-e", type=str, default="20000102")
@click.option("--input_file", "-i", type=str, default=None)
@click.option("--output_file", "-f", type=str, default=None)
@click.option("--output_format", "-o", type=str, default="csv")
@click.option("--datum", "-d", type=str, default="msl")
@click.option("--units", "-u", type=str, default="metric")
@click.option("--application", "-a", type=str, default="pyadcirc")
@click.option("--interval", "-n", type=int, default=6)
@click.option("--threshold", "-t", type=float, default=1.0)
@click.option("--workers", "-w", type=int, default=4)
@click.option("--interactive/--no-interactive", "-v", is_flag=True, default=True)
def find_events(
    station_id,
    begin_date=None,
    end_date=None,
    input_file=None,
    output_file=None,
    output_format="csv",
    datum="msl",
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
        data = noaa.get_storm_surge_events(
            station_id,
            begin_date,
            end_date,
            threshold=threshold,
            workers=workers,
        )
    else:
        data = pd.read_csv(input_file)

    data = noaa.wicks_2017_algo(
        data, trigger_threshold=threshold, interactive=interactive
    )

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
