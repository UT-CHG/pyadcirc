"""
viz.py

High level vizualization functions for ADCIRC data.

"""
from pathlib import Path
from typing import List

import imageio
import imageio as iio
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import xarray as xr
from cartopy import crs, feature
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from pygifsicle import optimize


def pyplot_mesh(
    data: xr.Dataset,
    var: str,
    save_path: str = None,
    timestep: int = 0,
    time: str = None,
    ax: plt.axes = None,
    projection=crs.PlateCarree(),
    vrange: List[float] = None,
    features: List[str] = [
        ("coastlines", {"resolution": "50m", "color": "black"}),
        ("land", {"facecolor": "wheat"}),
        ("ocean", {}),
        ("states", {"linestyle": "-", "lw": 1.0, "edgecolor": "black"}),
        ("borders", {"linestyle": "-", "lw": 1.0, "edgecolor": "black"}),
        ("rivers", {}),
    ],
    title: str = None,
    title_style={"fontsize": 18, "weight": "bold"},
    xlabel_loc: List[float] = None,
    ylabel_loc: List[float] = None,
    label_style: dict = {"size": 12, "color": "black"},
    gridline_style: dict = {
        "color": "black",
        "alpha": 0.2,
        "linestyle": "--",
        "linewidth": 2,
    },
):
    """
    PyPlot Mesh

    Plots meshed data stored in xarray using matplotlib.pyplot and cartopy.
    Used primarly for plotting meteorological forcing data that is in netcdf
    format (fort.22*.nc) used to run ADCIRC.

    Parameters
    ----------


    Returns
    -------

    """
    # Create axis with appropriate coordiante system
    ax = plt.axes(projection=projection,
                  facecolor="gray") if ax is None else ax

    # Plot data and cartopy features
    if time is not None:
        d = data[var].sel(time=time)
    else:
        d = data[var].isel(time=timestep)

    # Plot main data - Note adcirc met netcdf data should always be in long/lat
    # PlateCarree projection, but the axis plot may be on a different
    # projection, so must specify the transform argument
    if vrange is not None:
        p = d.plot(ax=ax, transform=crs.PlateCarree(),
                   vmin=vrange[0], vmax=vrange[1])
    else:
        p = d.plot(ax=ax, transform=crs.PlateCarree())

    # Add desired features
    for f in features:
        if f[0] == "coastlines":
            ax.coastlines(**f[1])
        elif f[0] == "land":
            ax.add_feature(feature.LAND, **f[1])
        elif f[0] == "ocean":
            ax.add_feature(feature.OCEAN, **f[1])
        elif f[0] == "states":
            ax.add_feature(feature.STATES, **f[1])
        elif f[0] == "borders":
            ax.add_feature(feature.BORDERS, **f[1])
        elif f[0] == "rivers":
            ax.add_feature(feature.RIVERS, **f[1])
        else:
            raise ValueError(f"Unrecognized feature type {f[0]}")

    # Set grid line properties - Gridliner classes
    if gridline_style is not None and type(projection) == crs.PlateCarree:
        gl = ax.gridlines(crs=projection, draw_labels=True, **gridline_style)
        gl.left_labels = True
        gl.xlines = True

        if xlabel_loc is not None:
            gl.xlocator = mticker.FixedLocator(xlabel_loc)
        if ylabel_loc is not None:
            gl.ylocator = mticker.FixedLocator(ylabel_loc)

        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = label_style
        gl.ylabel_style = label_style
        gl.top_labels = False

    # Set colorbar properties. Note we get the colorbar from the dataset object
    p.colorbar.ax.tick_params(labelsize=12)

    ts = pd.to_datetime(d['time'].item(0))
    title = f"{var} at {ts}" if title is None else title
    ax.set_title(title, **title_style)

    if save_path is not None:
        plt.savefig(save_path)
