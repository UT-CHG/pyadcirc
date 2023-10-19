"""
Io utilites CLI for pyADCIRC packacge

TODO: Update trim function ehre to use new mesh class and trim function in utils

"""
from typing import Optional, Tuple, Union

import geopandas as gpd
import rich_click as click
from rich.console import Console
from rich.table import Table
from rich_click.cli import patch

from pyadcirc.io.read_shape import (create_nc_file, save_element_gdf,
                                    save_geodataframe_with_metadata,
                                    trim_f14_grid)

click.rich_click.USE_RICH_MARKUP = True
patch()


@click.group()
def io():
    """Commands for interacting with the NOAA API"""
    pass


@io.command()
@click.option(
    "--f14_path", default="fort.14", type=str, help="Path to the fort.14 file."
)
@click.option(
    "--shape_path",
    default="boundary",
    type=str,
    help="Path to the boundary shape file to trim fort.14 file according to.",
)
@click.option(
    "--center", type=str, help="Comma-separated x,y coordinates of the center point."
)
@click.option(
    "--size", type=float, help="The size of the box around the center point to select."
)
@click.option(
    "--neighbors", default=0, type=int, help="Depth of neighboring elements to include."
)
@click.option(
    "--within_box",
    default=True,
    type=bool,
    help="If True, selects only elements within the bounding box.",
)
@click.option("--save_to", type=str, help="Path to save the GeoDataFrame.")
@click.option(
    "--format",
    type=click.Choice(["netcdf", "shape", "csv"]),
    default="netcdf",
    help="File format to save the GeoDataFrame.",
)
def trim_f14(
    f14_path: Optional[str],
    shape_path: Optional[str],
    center: Optional[Tuple[float, float]],
    size: Optional[float],
    neighbors: Optional[int],
    within_box: Optional[bool],
    save_to: Optional[str],
    format: Optional[str],
):
    """
    CLI entry point for the trim_f14_grid function. This function enhances the
    output using the rich library and provides a summary of the GeoDataFrame
    returned by trim_f14_grid.
    """
    console = Console()

    # Call the trim_f14_grid function
    if center:
        center = tuple(map(int, center.split(",")))

    elements_gdf = trim_f14_grid(
        f14_path=f14_path,
        shape_path=shape_path,
        center=center,
        size=size,
        neighbors=neighbors,
        within_box=within_box,
    )

    # Print summary using rich
    table = Table(title="Element GeoDataFrame Summary")
    table.add_column("Column Name", style="cyan")
    table.add_column("First 5 Values", style="magenta")

    for col in elements_gdf.columns:
        table.add_row(col, str(elements_gdf[col].head(5).values))

    console.print(table)

    # Save the GeoDataFrame if the save_to option is provided
    if save_to:
        if format == "netcdf":
            create_nc_file(elements_gdf, save_to)
        elif format == "shape":
            attrs = {"Description": "Generated by trim_f14_grid CLI"}
            save_geodataframe_with_metadata(elements_gdf, save_to, attrs=attrs)
        elif format == "csv":
            save_element_gdf(elements_gdf, save_to)
