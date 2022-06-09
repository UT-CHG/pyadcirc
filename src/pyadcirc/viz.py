"""
viz.py

High level vizualization functions for ADCIRC data.

"""
import pdb
from pathlib import Path
from typing import AnyStr, Callable, List, Tuple, Union

import imageio as iio
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
from cartopy import crs, feature
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from pygifsicle import optimize

def pyplot_mesh(
    data: xr.Dataset,
    var: str,
    data_coords=crs.PlateCarree(),
    time_var:str = 'time',
    data_vars: List[str]=['longitude', 'latitude'],
    vec_data: xr.Dataset=None,
    vec_vars: List[str]=['longitude', 'latitude', 'u10', 'v10'],
    num_vecs: int = 10,
    vec_coords=crs.PlateCarree(),
    save_path: str = None,
    bounding_box: List[float]=None,
    timestep: int = 0,
    time: str = None,
    ax: plt.axes = None,
    projection=crs.PlateCarree(),
    vrange: List[float] = None,
    features: List[str] = [
        ("coastlines", {"resolution": "50m", "color": "black"}),
        ("land", {"facecolor": "wheat"}),
        ("ocean", {}),
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
    vec_params: dict = {
        'scale': 300,
        'linewidth': 0.2,
        'headlength': 4,
        'headwidth': 4,
        'headaxislength': 4,
        'width': 0.003,
    },
    cmap: str = 'viridis',
    colorbar_opts: dict = {
        'shrink': 0.8,
    }
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

    # Select timestep we want to plot if specified
    if time is not None:
        d = data[var].sel({time_var:time}, method='nearest')
        if vec_data is not None:
            vec_data = vec_data.sel({time_var:time}, method='nearest')
    else:
        d = data[var].isel({time_var:timestep})
        if vec_data is not None:
            vec_data = vec_data.isel({time_var:timestep})

    if bounding_box is not None:
        d = d.sel({data_vars[0]:slice(bounding_box[0], bounding_box[1]),
                   data_vars[1]:slice(bounding_box[3], bounding_box[2])})
        if vec_data is not None:
            vec_data = vec_data.sel(
                    {vec_vars[0]:slice(bounding_box[0], bounding_box[1]),
                     vec_vars[1]:slice(bounding_box[3], bounding_box[2])})

    # Plot main data - Note adcirc met netcdf data should always be in long/lat
    # PlateCarree projection, but the axis plot may be on a different
    # projection, so must specify the transform argument
    if vrange is not None:
        p = d.plot(ax=ax, transform=data_coords,
                   vmin=vrange[0], vmax=vrange[1],
                   cmap=cmap, cbar_kwargs=colorbar_opts)
    else:
        p = d.plot(ax=ax, transform=data_coords,
                   cbar_kwargs=colorbar_opts, cmap=cmap)

    # Plot vector data
    if vec_data is not None:
        num_lon = len(vec_data[vec_vars[0]])
        num_lat = len(vec_data[vec_vars[1]])
        long_idxs = np.arange(0, num_lon, int(num_lon / num_vecs) + 1)
        lat_idxs = np.arange(0, num_lat, int(num_lat / num_vecs) + 1)

        vec_data.isel(longitude=long_idxs,
                      latitude=lat_idxs).plot.quiver(
            transform=vec_coords,
            x=vec_vars[0], y=vec_vars[1], u=vec_vars[2], v=vec_vars[3],
            **vec_params)

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


def generate_gif(name:str, gen_image: Callable,
                 args: List,
                 repeat: Union[List, int]=None,
                 figsize: Tuple=(12,6),
                 build_dir: str=None,
                 hold_end: float=0.25):
    """
    Generate GIF

    Given a callable and a list of arguments to pass to the callable, build a
    gif using sequence of images produced by `gen_image` callabe function.

    Parameters
    ----------

    Returns
    -------


    """

    build_dir = Path.cwd() if build_dir is None else Path(build_dir)

    gif_images_path = build_dir / ".gif_images"
    gif_images_path.mkdir(exist_ok=True)
    gif_path = build_dir / name

    images = []
    gif_images = []

    if repeat is None:
        repeat = np.array([1]).repeat(len(args))
    elif type(repeat) == int:
        repeat = np.array([repeat]).repeat(len(args))

    if args is None:
        args = range(len(data["time"]))
    for i, t in enumerate(args):
        filename = str(gif_images_path / f"{t}")
        filename = gen_image(t, filename)
        images.append(filename)
        for j in range(repeat[i]):
            gif_images.append(filename)

    if hold_end>0:
        num_extra = int(hold_end*len(gif_images))
        for i in range(num_extra):
            gif_images.append(filename)

    with iio.get_writer(str(gif_path), mode="I") as writer:
        for i in gif_images:
            writer.append_data(iio.imread(i))

    optimize(gif_path)

    for i in images:
        Path(i).unlink()

    return gif_path


