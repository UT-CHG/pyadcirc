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
    type=click.Choice(["storm_surge_events", "metadata", "water_level", "predictions"]),
)
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
@click.option("--print_graph", "-g", type=bool, default=False)
def get(
    station_id,
    product,
    begin_date=None,
    end_date=None,
    input_file=None,
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
                begin_date,
                end_date,
                station_id,
                product,
                output_format,
                datum=datum,
                units=units,
                application=application,
                interval=interval,
            )
        except EmptyDataError as e:
            print(e)
            return None
    elif product == "storm_surge_events":
        try:
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

            data = noaa.wicks_2017_algo(data, trigger_threshold=threshold)
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

    output_file = sys.stdout if output_file is None else output_file
    if product != "all":
        if output_format == "csv":
            data.to_csv(output_file, index=True)
        elif output_format == "json":
            data.to_json(output_file, index=True)
    else:
        data.to_netcdf(output_file)

    return data


if __name__ == "__main__":
    get.main()
