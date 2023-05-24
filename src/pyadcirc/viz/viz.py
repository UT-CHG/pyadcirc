"""
viz.py

High level vizualization functions for ADCIRC data.

"""
import pdb
from pathlib import Path
from typing import AnyStr, Callable, List, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
from cartopy import crs, feature
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER


def pyplot_mesh(
    data: xr.Dataset,
    var: str,
    data_coords=crs.PlateCarree(),
    time_var: str = "time",
    data_vars: List[str] = ["longitude", "latitude"],
    vec_data: xr.Dataset = None,
    vec_vars: List[str] = ["longitude", "latitude", "u10", "v10"],
    num_vecs: int = 10,
    vec_coords=crs.PlateCarree(),
    save_path: str = None,
    bounding_box: List[float] = None,
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
        "scale": 300,
        "linewidth": 0.2,
        "headlength": 4,
        "headwidth": 4,
        "headaxislength": 4,
        "width": 0.003,
    },
    quiver_key_args: dict = {
        'X': 0.85, 'Y': 0.85, 'U': 10,
        'label': '10 m/s',
        'labelpos': 'N',
        'fontproperties': {'size': 16},
    },
    cmap: str = "viridis",
    colorbar_opts: dict = {
        "shrink": 0.8,
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
    ax = plt.axes(projection=projection, facecolor="gray") if ax is None else ax

    # Select timestep we want to plot if specified
    d = data[var]
    if time is not None:
        d = d.sel({time_var: time}, method="nearest")
        if vec_data is not None:
            vec_data = vec_data.sel({time_var: time}, method="nearest")
    else:
        if 'time' in d.coords:
            d = d.isel({time_var: timestep})
            if vec_data is not None:
                vec_data = vec_data.isel({time_var: timestep})

    if bounding_box is not None:
        d = d.sel(
            {
                data_vars[0]: slice(bounding_box[0], bounding_box[1]),
                data_vars[1]: slice(bounding_box[3], bounding_box[2]),
            }
        )
        if vec_data is not None:
            vec_data = vec_data.sel(
                {
                    vec_vars[0]: slice(bounding_box[0], bounding_box[1]),
                    vec_vars[1]: slice(bounding_box[3], bounding_box[2]),
                }
            )

    # Plot main data - Note adcirc met netcdf data should always be in long/lat
    # PlateCarree projection, but the axis plot may be on a different
    # projection, so must specify the transform argument
    if vrange is not None:
        p = d.plot(
            ax=ax,
            transform=data_coords,
            vmin=vrange[0],
            vmax=vrange[1],
            cmap=cmap,
            cbar_kwargs=colorbar_opts,
        )
    else:
        p = d.plot(ax=ax, transform=data_coords, cbar_kwargs=colorbar_opts, cmap=cmap)

    # Plot vector data
    quiv = None
    if vec_data is not None:
        num_lon = len(vec_data[vec_vars[0]])
        num_lat = len(vec_data[vec_vars[1]])
        long_idxs = np.arange(0, num_lon, int(num_lon / num_vecs) + 1)
        lat_idxs = np.arange(0, num_lat, int(num_lat / num_vecs) + 1)

        quiv = vec_data.isel(longitude=long_idxs, latitude=lat_idxs).plot.quiver(
            transform=vec_coords,
            x=vec_vars[0],
            y=vec_vars[1],
            u=vec_vars[2],
            v=vec_vars[3],
            **vec_params,
        )
        plt.quiverkey(quiv, **quiver_key_args)

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

    title = f"{var}" if title is None else title
    if 'time' in d.coords:
        title = f"{title} at {pd.to_datetime(d['time'].item(0))}"
    ax.set_title(title, **title_style)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    return ax


if __name__ == "__main__":
    import pyadcirc.io.io as pyio
    from cartopy import crs
    import matplotlib.pyplot as plt

    start_date = '1993-01-07 01:36:00'
    end_date = '1993-01-13 18:18:00'
    root_path = Path('/Users/carlos/repos/pyADCIRC/notebooks/')
    wind_data = xr.open_dataset(str(root_path / 'data/fort.222.nc')) 
    wind_data['speed'] = np.sqrt(wind_data['u10']**2 + wind_data['v10']**2)
    wind_data['speed'].attrs['units'] = 'm/s'
    plt.figure(figsize=(12, 8))
    res = pyplot_mesh(
        wind_data,
        "speed",
        projection=crs.Orthographic(-170, 45),
        timestep=0,
        features=[("coastlines", {"resolution": "10m", "color": "black"})],
        vec_data=wind_data,
        num_vecs=20,
        save_path="test.png",
        quiver_key_args={
        'X': 0.87, 'Y': 0.75, 'U': 10,
        'label': '10 m/s',
        'labelpos': 'N',
        'fontproperties': {'size': 16},
        })