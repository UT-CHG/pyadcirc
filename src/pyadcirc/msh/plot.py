"""
Plotting functions for ADCIRC meshes.

"""
import heapq
import os
import pdb
import shutil
import warnings
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple, Union

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import netCDF4 as nc
import networkx as nx
import numpy as np
import pandas as pd
from alive_progress import alive_bar
from geopandas import GeoDataFrame
from rich.traceback import install
from shapely.geometry import LineString, Point, Polygon
from shapely.wkt import loads

from pyadcirc.io.io import read_fort14_element_map, read_fort14_node_map


def plot_folium_map(
    gdf_or_path: Union[gpd.GeoDataFrame, str],
    crs_epsg: int = 32616,
    map_filename: str = "map.html",
) -> None:
    """
    Plot the shapefile using Folium.

    Parameters
    ----------
    gdf_or_path : Union[gpd.GeoDataFrame, str]
        Either a GeoPandas DataFrame or a file path to load the DataFrame.
    crs_epsg : int, optional
        The EPSG code for the CRS transformation. Default is 32616 (UTM zone 16N).
    map_filename : str, optional
        The filename for the saved Folium map. Default is 'map.html'.
    """
    if isinstance(gdf_or_path, str):
        gdf = gpd.read_file(gdf_or_path)
    else:
        gdf = gdf_or_path

    # Project the data to the specified CRS for accurate centroid calculation
    gdf = set_crs_for_gdf(gdf, crs_epg)

    # Calculate the center of the map
    center = gdf.unary_union.centroid

    # Convert centroid back to geographic coordinates
    center_geographic = gpd.GeoSeries([center], crs=crs_epsg).to_crs(epsg=4326)
    avg_latitude, avg_longitude = (
        center_geographic.geometry[0].y,
        center_geographic.geometry[0].x,
    )

    # Create a Map instance
    m = folium.Map(
        location=[avg_latitude, avg_longitude], zoom_start=12, control_scale=True
    )

    # Add the shapefile to the map
    folium.GeoJson(gdf).add_to(m)

    # Show or save the map
    m.save(map_filename)


def plot_mesh(
    node_map: gpd.GeoDataFrame,
    element_map: gpd.GeoDataFrame,
    plot: Optional[bool] = True,
    plot_filename: Optional[str] = None,
    map_filename: Optional[str] = "map.html",
    crs_epsg: Optional[int] = 32616,
) -> None:
    """
    Plot mesh elements and nodes using GeoPandas and Folium.

    Parameters
    ----------
    node_map : gpd.GeoDataFrame
        GeoDataFrame containing the node map with columns 'X', 'Y', 'JN'.
    element_map : gpd.GeoDataFrame
        GeoDataFrame containing the element map with columns 'NM_1', 'NM_2', 'NM_3'.
    plot : bool, optional
        If True, plot the shapefile using GeoPandas. Default is True.
    plot_filename : str, optional
        The filename for the saved GeoPandas plot. If None, the plot is shown. Default is None.
    map_filename : str, optional
        The filename for the saved Folium map. Default is 'map.html'.
    crs_epsg : int, optional
        The EPSG code for the CRS transformation. Default is 32616.

    Returns
    -------
    None
        This function does not return a value but saves or displays the plots.
    """
    # Convert the node map to a GeoDataFrame
    nodes = gpd.GeoDataFrame(
        node_map, geometry=gpd.points_from_xy(node_map.X, node_map.Y)
    ).set_crs(epsg=4326)
    nodes.set_index("JN", inplace=True)  # set 'JN' as the index for direct lookup

    if "DP" not in element_map.columns:
        element_map["DP"] = element_map.apply(
            lambda row: nodes.loc[[row["NM_1"], row["NM_2"], row["NM_3"]]].DP.mean(),
            axis=1,
        )
    polygons = element_map.apply(
        lambda row: Polygon(
            nodes.loc[[row["NM_1"], row["NM_2"], row["NM_3"]]].geometry.values
        ),
        axis=1,
    )
    elements = gpd.GeoDataFrame(element_map, geometry=polygons).set_crs(epsg=4326)

    # Plot the elements using GeoPandas
    if plot:
        fig, ax = plt.subplots(1, 1)
        elements.plot(column="DP", ax=ax, legend=True)
        if plot_filename is not None:
            plt.savefig(plot_filename)
        else:
            plt.show()

    # Create a map with Folium
    m = folium.Map(location=[nodes.Y.mean(), nodes.X.mean()], zoom_start=12)
    folium.GeoJson(elements).add_to(m)
    m.save(map_filename)


def plot_elements_with_node(
    node_map: pd.DataFrame,
    element_map: pd.DataFrame,
    node_index: int,
    neighbors: Optional[int] = 0,
    plot: Optional[bool] = True,
    plot_filename: Optional[str] = None,
    map_filename: Optional[str] = "map.html",
) -> None:
    """
    Plot mesh elements containing a specific node, its neighbors, and associated nodes.

    Parameters
    ----------
    node_map : pd.DataFrame
        DataFrame containing the node map with columns 'X', 'Y', 'JN'.
    element_map : pd.DataFrame
        DataFrame containing the element map with columns 'NM_1', 'NM_2', 'NM_3'.
    node_index : int
        The index of the node to highlight.
    neighbors : int, optional
        The number of neighboring elements to include in the plot. Default is 0.
    plot : bool, optional
        If True, plot the shapefile using GeoPandas. Default is True.
    plot_filename : str, optional
        The filename for the saved GeoPandas plot. If None, the plot is shown. Default is None.
    map_filename : str, optional
        The filename for the saved Folium map. Default is 'map.html'.

    Returns
    -------
    None
        This function does not return a value but saves or displays the plots.
    """

    # Filter element_map to find elements that contain the given node_index
    elements_with_node = element_map[
        (element_map["NM_1"] == node_index)
        | (element_map["NM_2"] == node_index)
        | (element_map["NM_3"] == node_index)
    ]
    elements_with_node = add_neighbors_to_element_map(
        elements_with_node, element_map, depth=neighbors
    )

    # Extract the node indices from these elements
    node_indices = pd.unique(
        elements_with_node[["NM_1", "NM_2", "NM_3"]].values.ravel("K")
    )

    # Filter node_map to get these nodes
    nodes = node_map[node_map["JN"].isin(node_indices)]

    # Convert the node map to a GeoDataFrame
    nodes_gdf = gpd.GeoDataFrame(
        nodes, geometry=gpd.points_from_xy(nodes.X, nodes.Y)
    ).set_crs(epsg=4326)

    # Create polygons for the elements
    elements_with_node["DP"] = elements_with_node.apply(
        lambda row: nodes.loc[
            [row["NM_1"] - 1, row["NM_2"] - 1, row["NM_3"] - 1]
        ].DP.mean(),
        axis=1,
    )
    polygons = elements_with_node.apply(
        lambda row: Polygon(
            nodes_gdf.loc[
                [row["NM_1"] - 1, row["NM_2"] - 1, row["NM_3"] - 1]
            ].geometry.values
        ),
        axis=1,
    )
    elements_gdf = gpd.GeoDataFrame(elements_with_node, geometry=polygons).set_crs(
        epsg=4326
    )

    # Plot the elements using GeoPandas
    if plot:
        fig, ax = plt.subplots(1, 1)
        elements_gdf.plot(column="DP", ax=ax, legend=True)
        # Plot a distinct black point for the original node
        original_node = nodes_gdf.loc[node_index - 1]
        # original_node.geometry.plot(ax=ax, color='black', markersize=50)
        ax.scatter(
            original_node.geometry.x, original_node.geometry.y, color="black", s=50
        )

        if plot_filename is not None:
            plt.savefig(plot_filename)
        else:
            plt.show()

    # Create a map with Folium
    m = folium.Map(location=[nodes.Y.mean(), nodes.X.mean()], zoom_start=12)
    folium.GeoJson(elements_gdf).add_to(m)
    m.save(map_filename)
