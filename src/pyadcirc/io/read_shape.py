"""
Old file accumulating lots of functions. NEeds to be removed and each function put in appopriate location.

SEe bottome. not sure where to put load and read from csv for element maps. Within mesh class itself? jsut a one line call, centroid and geoemetry are necessary?

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

install(show_locals=True)


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


def read_xml_metadata(path: str) -> dict:
    """
    Read XML metadata for the shapefile.

    Parameters
    ----------
    path : str
        Either a path to the XML metadata file or a directory containing it.

    Returns
    -------
    metadata : dict
        A dictionary containing the XML metadata.
    """
    # Initialize empty metadata dictionary
    metadata = {}

    # Check if the path is a directory
    path = str(path)
    if os.path.isdir(path):
        # Search for .shp.xml files in the directory
        xml_files = glob(os.path.join(path, "*.shp.xml"))

        # Warning and selection if multiple .shp.xml files are found
        if len(xml_files) > 1:
            print(
                f"Warning: Multiple .shp.xml files found. Using the first one: {xml_files[0]}"
            )

        if xml_files:
            xml_file = xml_files[0]
        else:
            print("No .shp.xml file found in directory.")
            return metadata
    else:
        # Modify the file path based on the extension
        if path.endswith(".shp"):
            xml_file = path.replace(".shp", ".shp.xml")
        elif not path.endswith(".xml"):
            xml_file = f"{path}.xml"
        else:
            xml_file = path

    try:
        # Parse the XML file
        tree = ET.parse(xml_file)

        # Get the root element
        root = tree.getroot()

        # Populate metadata
        for elem in root.iter():
            metadata[elem.tag] = elem.text
    except FileNotFoundError:
        print("Metadata file not found.")

    return metadata


def write_xml_metadata(attrs: dict, shapefile_path: str, output_dir: str) -> None:
    """
    Write XML metadata for a shapefile.

    Parameters
    ----------
    attrs : dict
        A dictionary containing the XML attributes.
    shapefile_path : str
        The path to the shapefile for which the metadata is being written.
    output_dir : str
        The directory where the XML file will be saved.

    """
    # Create the root XML element
    root = ET.Element("root")

    # Create child elements based on the attrs dictionary
    for key, value in attrs.items():
        elem = ET.SubElement(root, key)
        elem.text = str(value)

    # Create the ElementTree object
    tree = ET.ElementTree(root)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create the output file path
    shapefile_name = os.path.basename(shapefile_path)
    xml_filename = shapefile_name.replace(".shp", ".shp.xml")
    output_path = os.path.join(output_dir, xml_filename)

    # Write the XML file
    tree.write(output_path)

    print(f"XML metadata written to {output_path}")


def process_shapefiles(
    path: str = None,
    gdf: Optional[gpd.GeoDataFrame] = None,
    plot: Optional[bool] = True,
    plot_kwargs: Optional[dict] = None,
    plot_filename: Optional[str] = None,
    map_filename: Optional[str] = None,
    metadata: Optional[bool] = True,
    crs_epsg: Optional[int] = 4326,
) -> gpd.GeoDataFrame:
    """
    Given a path to a .shp file and supporting files, perform various
    operations including reading, plotting, and extracting metadata.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame, optional
        A GeoDataFrame representing the shapefile if it has already been read in.
    dir : str, optional
        The path to the directory containing the .shp file and supporting files.
    plot : bool, optional
        If True, plot the shapefile using GeoPandas. Default is True.
    plot_kwargs : dict, optional
        Additional keyword arguments to pass to the GeoPandas plot function.
    plot_filename : str, optional
        The filename for the saved GeoPandas plot.
    map_filename : str, optional
        The filename for the saved Folium map. Default is 'map.html'.
    metadata : bool, optional
        If True, read in the XML metadata. Default is False.
    crs_epsg : int, optional
        The EPSG code for the CRS transformation. Default is 32616 (UTM zone 16N).

    Returns
    -------
    gdf : gpd.GeoDataFrame
        A GeoDataFrame representing the shapefile.
    """
    # Load shapefile
    if gdf is None:
        path = str(Path.cwd()) if path is None else path
        gdf = gpd.read_file(str(path))

        if metadata:
            gdf.attrs = read_xml_metadata(path)

    # Ensure the original GeoDataFrame has a geographic (latitude/longitude) CRS
    gdf = set_crs_for_gdf(gdf, crs_epsg)

    # Plot with GeoPandas
    if plot:
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        gdf.plot(**plot_kwargs)
        # plt.show()
        if plot_filename is not None:
            plt.savefig(plot_filename)

    if map_filename is not None:
        plot_folium_map(gdf, crs_epsg, map_filename)

    return gdf


def trim_f14_grid(
    f14_path: Optional[str] = "fort.14",
    shape_path: Optional[str] = "boundary",
    bounding_box: Optional[gpd.GeoDataFrame] = None,
    center: Optional[Tuple[float, float]] = None,
    size: Optional[float] = None,
    neighbors: Optional[int] = 0,
    within_box: Optional[bool] = True,
) -> gpd.GeoDataFrame:
    """
    Read an ADCIRC Grid and Boundary Information File (fort.14), select a subset of
    the grid based on the specified center point and size, and return the elements
    in that subset.

    Parameters
    ----------
    f14_path : str, optional
        Path to the fort.14 file. Default is 'fort.14'.
    shape_path : str, optional
        Path to the boundary shape file. Default is 'boundary'.
    bounding_box : gpd.GeoDataFrame, optional
        GeoDataFrame defining the bounding box. If None, it is calculated.
    center : Tuple[float, float], optional
        The (x, y) coordinates of the center point of the area to select.
    size : float, optional
        The size of the box around the center point to select.
    neighbors : int, optional
        Depth of neighboring elements to include. Default is 0.
    within_box : bool, optional
        If True, selects only elements within the bounding box. Default is True.

    Returns
    -------
    elements_gdf : gpd.GeoDataFrame
        A GeoDataFrame containing the elements in the selected area.
    """
    # Calculate bounding box from shape file
    if bounding_box is None:
        bounding_box = create_bbox_polygon(
            shape_path=shape_path, center=center, size=size
        )
    if not isinstance(bounding_box, gpd.GeoDataFrame):
        bounding_box = create_bbox_polygon(bounding_box=bounding_box)

    print(f"Reading in node map from {f14_path}")
    node_map = read_fort14_node_map(f14_path)

    print(f"Converting to GeoDataFrame")
    nodes_gdf = gpd.GeoDataFrame(
        node_map, geometry=gpd.points_from_xy(node_map.X, node_map.Y)
    ).set_crs(epsg=4326)

    print(f"Filtering node by boundary")
    nodes_within_bounding_box = gpd.sjoin(nodes_gdf, bounding_box, predicate="within")

    print(f"Reading in element map from {f14_path}")
    element_map = read_fort14_element_map(f14_path)

    # Select rows where NM_1, NM_2 or NM_3 is in nodes_within_bounding_box['JN']
    mask = (
        element_map["NM_1"].isin(nodes_within_bounding_box["JN"])
        | element_map["NM_2"].isin(nodes_within_bounding_box["JN"])
        | element_map["NM_3"].isin(nodes_within_bounding_box["JN"])
    )

    elements_with_filtered_nodes = element_map[mask].copy()

    if not within_box:
        elements_with_filtered_nodes = add_neighbors_to_element_map(
            elements_with_filtered_nodes, element_map, depth=neighbors
        )
        ndf = nodes_gdf
    else:
        ndf = nodes_within_bounding_box

    elements_with_filtered_nodes.loc[:, "DP"] = np.nan
    elements_with_filtered_nodes.loc[:, "X"] = np.nan
    elements_with_filtered_nodes.loc[:, "Y"] = np.nan

    # * Merge in X, Y and depth data from nodes dataframe by linking on 'NM_1', 'NM_2', and 'NM_3' cols
    cols = ["JN", "X", "Y", "DP"]
    element_map = elements_with_filtered_nodes.merge(
        ndf[cols], left_on="NM_1", right_on="JN", suffixes=("", "_1")
    )
    element_map = element_map.merge(
        ndf[cols], left_on="NM_2", right_on="JN", suffixes=("", "_2")
    )
    element_map = element_map.merge(
        ndf[cols], left_on="NM_3", right_on="JN", suffixes=("", "_3")
    )
    element_map["DP"] = element_map[["DP_1", "DP_2", "DP_3"]].mean(axis=1)
    element_map["geometry"] = element_map.apply(
        lambda row: Polygon(
            [
                (row["X_1"], row["Y_1"]),
                (row["X_2"], row["Y_2"]),
                (row["X_3"], row["Y_3"]),
            ]
        ),
        axis=1,
    )
    elements_gdf = gpd.GeoDataFrame(element_map, geometry="geometry")
    elements_gdf["centroid"] = elements_gdf["geometry"].centroid

    return elements_gdf


def add_neighbors_to_element_map(
    sub_map: pd.DataFrame, full_map: pd.DataFrame, depth: int = 1
) -> pd.DataFrame:
    """
    Add neighboring elements to a subset of elements from the full element map based on a given depth.

    Parameters
    ----------
    sub_map : pd.DataFrame
        A DataFrame containing the subset of elements for which the neighbors are to be found.
    full_map : pd.DataFrame
        A DataFrame containing the full element map, including all elements.
    depth : int, optional
        The depth of neighboring elements to include. For depth = 1, only immediate neighbors will be included;
        for depth = 2, neighbors of neighbors will be included, and so on. Default is 1.

    Returns
    -------
    sub_map : pd.DataFrame
        A DataFrame containing the original subset of elements, plus the neighboring elements up to the specified depth.

    Example
    -------
    sub_map:
        JE  NM_1  NM_2  NM_3
        1    100   101   102
        2    101   102   103
    full_map:
        JE  NM_1  NM_2  NM_3
        1    100   101   102
        2    101   102   103
        3    102   103   104
        4    103   104   105

    Result (with depth=1):
        JE  NM_1  NM_2  NM_3
        1    100   101   102
        2    101   102   103
        3    102   103   104
    """
    for n in range(depth):
        # Now, find all the nodes in the selected elements
        selected_nodes = pd.concat(
            [sub_map["NM_1"], sub_map["NM_2"], sub_map["NM_3"]]
        ).unique()

        # Create a mask to select elements that have at least one node in selected_nodes
        neighbor_mask = (
            full_map["NM_1"].isin(selected_nodes)
            | full_map["NM_2"].isin(selected_nodes)
            | full_map["NM_3"].isin(selected_nodes)
        )

        # Append the neighboring elements to the selected elements
        sub_map = pd.concat([sub_map, full_map[neighbor_mask]]).drop_duplicates()

    return sub_map


def get_lines_from_element_map(element_map):
    """ """

    lines = pd.DataFrame(
        [
            {
                "geometry": LineString(
                    [(row["X_1"], row["Y_1"]), (row["X_2"], row["Y_2"])]
                ),
                "nodes": frozenset([row["NM_1"], row["NM_2"]]),
            }
            for _, row in element_map.iterrows()
        ]
        + [
            {
                "geometry": LineString(
                    [(row["X_2"], row["Y_2"]), (row["X_3"], row["Y_3"])]
                ),
                "nodes": frozenset([row["NM_2"], row["NM_3"]]),
            }
            for _, row in element_map.iterrows()
        ]
        + [
            {
                "geometry": LineString(
                    [(row["X_3"], row["Y_3"]), (row["X_1"], row["Y_1"])]
                ),
                "nodes": frozenset([row["NM_3"], row["NM_1"]]),
            }
            for _, row in element_map.iterrows()
        ]
    )
    # remove duplicates
    lines = lines.drop_duplicates(subset="nodes")
    lines_gdf = gpd.GeoDataFrame(lines, geometry="geometry")

    return lines_gdf


def calculate_bbox(
    gdf: Optional[gpd.GeoDataFrame] = None, dir: Optional[str] = None
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Calculate the bounding box of a shapefile.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame, optional
        A GeoDataFrame containing the data. If not provided, the `dir` parameter must be specified.
    dir : str, optional
        The path to the .shp file. Required if `gdf` is not provided.

    Returns
    -------
    bounding_box : Tuple[Tuple[float, float], Tuple[float, float]]
        The bounding box of the shapefile as a tuple of tuples, where each tuple contains the x and y coordinates.
        The bounding box is represented as ((minx, miny), (maxx, maxy)).

    Raises
    ------
    ValueError
        If both `gdf` and `dir` are None.
    """
    if gdf is None:
        # Load the shapefile
        if dir is None:
            raise ValueError("Either gdf or dir must be specified.")
        gdf = gpd.read_file(dir)

    # Ensure the GeoDataFrame has a geographic (latitude/longitude) CRS
    gdf = set_crs_for_gdf(gdf, 4326)

    # Calculate the bounding box
    minx, miny, maxx, maxy = gdf.total_bounds

    return ((minx, miny), (maxx, maxy))


def create_bbox_polygon(
    bounding_box: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    gdf: Optional[gpd.GeoDataFrame] = None,
    shape_path: Optional[str] = None,
    center: Optional[Tuple[float, float]] = None,
    size: Optional[float] = None,
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame containing a polygon representing the bounding box.

    Parameters
    ----------
    bounding_box : Tuple[Tuple[float, float], Tuple[float, float]], optional
        The bounding box as ((minx, miny), (maxx, maxy)). If not provided, it will be calculated based on the other parameters.
    gdf : geopandas.GeoDataFrame, optional
        A GeoDataFrame containing the data.
    shape_path : str, optional
        The path to the .shp file.
    center : Tuple[float, float], optional
        The (x, y) coordinates of the center point of the bounding box.
    size : float, optional
        The size of the bounding box.
    crs : str, default 'EPSG:4326'
        The coordinate reference system of the resulting GeoDataFrame.

    Returns
    -------
    bounding_box_gdf : geopandas.GeoDataFrame
        A GeoDataFrame containing the bounding box polygon.

    Raises
    ------
    ValueError
        If the required parameters are not provided.
    """

    if bounding_box is None:
        if center and size is not None:
            bounding_box = [
                [center[0] - size / 2, center[1] - size / 2],
                [center[0] + size / 2, center[1] + size / 2],
            ]
        elif shape_path or gdf is not None:
            bounding_box = calculate_bbox(gdf=gdf, dir=shape_path)
        else:
            raise ValueError(
                "Either center/size, bounding_box, shapefile_dir, or gdf must be specified."
            )

    # Create a polygon from the bounding box
    bounding_box_polygon = Polygon(
        [
            (bounding_box[0][0], bounding_box[0][1]),
            (bounding_box[0][0], bounding_box[1][1]),
            (bounding_box[1][0], bounding_box[1][1]),
            (bounding_box[1][0], bounding_box[0][1]),
        ]
    )

    # Create a GeoDataFrame from the bounding box polygon
    bounding_box_gdf = gpd.GeoDataFrame(geometry=[bounding_box_polygon], crs=crs)

    return bounding_box_gdf


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


def score_path(node_map, point1, point2, segments):
    """
    Calculates the score of a path of segments between two nodes.

    Parameters
    ----------
    node_map : pd.DataFrame
        DataFrame containing the nodes with columns 'JN', 'X', and 'Y'.
    point1 : Tuple[float, float]
        The (x, y) coordinates of the first point.
    point2 : Tuple[float, float]
        The (x, y) coordinates of the second point.
    segments : List[shapely.geometry.LineString]
        The line segments connecting the two closest nodal locations of the points.

    Returns
    -------
    float
        The ratio of the true distance between the points to the total distance of the segments.
    """

    # Convert points to shapely geometry
    point1 = Point(point1)
    point2 = Point(point2)

    # Calculate the true distance between the points
    true_distance = point1.distance(point2)

    # Calculate the total distance of all the line segments
    segment_distance = np.sum([segment.length for segment in segments.geometry.values])

    # Calculate the ratio
    ratio = true_distance / segment_distance if segment_distance != 0 else 0

    return ratio


def greedy_shortest_path(
    element_gdf: gpd.GeoDataFrame, start_point: tuple, end_point: tuple
) -> gpd.GeoDataFrame:
    """
    Find the shortest path between two points using a greedy search algorithm.

    Parameters
    ----------
    element_gdf : GeoDataFrame
        GeoPandas DataFrame containing element map data.
    start_point : tuple
        Coordinates (x, y) of the starting point.
    end_point : tuple
        Coordinates (x, y) of the ending point.

    Returns
    -------
    GeoDataFrame
        GeoPandas DataFrame containing the path as a series of elements.
    """
    # Convert start and end points to Points
    start_point = Point(start_point)
    end_point = Point(end_point)

    # Find the nearest element to the starting point
    # start_element = element_gdf.loc[element_gdf['centroid'].distance(start_point).idxmin()]
    start_element = element_gdf.loc[
        element_gdf["centroid"].apply(lambda x: x.distance(start_point)).idxmin()
    ]

    # Initialize the path with the starting element
    path_elements = [start_element]

    # Greedy search
    current_element = start_element
    while True:
        # Find neighboring elements
        neighbors = element_gdf[
            (element_gdf["NM_1"].isin(current_element[["NM_1", "NM_2", "NM_3"]]))
            | (element_gdf["NM_2"].isin(current_element[["NM_1", "NM_2", "NM_3"]]))
            | (element_gdf["NM_3"].isin(current_element[["NM_1", "NM_2", "NM_3"]]))
        ]

        # Exclude elements already in the path
        neighbors = neighbors[~neighbors["JE"].isin([e["JE"] for e in path_elements])]

        # Find the neighbor that minimizes the distance to the end point
        # next_element = neighbors.loc[neighbors["centroid"].distance(end_point).idxmin()]
        next_element = element_gdf.loc[
            neighbors["centroid"].apply(lambda x: x.distance(start_point)).idxmin()
        ]

        # Add the next element to the path
        path_elements.append(next_element)

        # Update the current element
        current_element = next_element

        # Check if the current element's centroid is close enough to the end point
        if current_element["centroid"].distance(end_point) < np.finfo(float).eps:
            break

    return GeoDataFrame(path_elements)


def build_graph(element_map):
    # Create a graph from the elements
    G = nx.Graph()

    # Extract unique nodes and their coordinates, adjusting for 1-based indexing
    nodes = {}
    for _, row in element_map.iterrows():
        nodes[row["NM_1"] - 1] = (row["X_1"], row["Y_1"])
        nodes[row["NM_2"] - 1] = (row["X_2"], row["Y_2"])
        nodes[row["NM_3"] - 1] = (row["X_3"], row["Y_3"])

    # Add nodes to the graph
    for node, coords in nodes.items():
        G.add_node(node, pos=coords)

    # Add edges to the graph, adjusting for 1-based indexing
    for _, row in element_map.iterrows():
        G.add_edge(
            row["NM_1"] - 1,
            row["NM_2"] - 1,
            weight=Point(nodes[row["NM_1"] - 1]).distance(
                Point(nodes[row["NM_2"] - 1])
            ),
        )
        G.add_edge(
            row["NM_2"] - 1,
            row["NM_3"] - 1,
            weight=Point(nodes[row["NM_2"] - 1]).distance(
                Point(nodes[row["NM_3"] - 1])
            ),
        )
        G.add_edge(
            row["NM_1"] - 1,
            row["NM_3"] - 1,
            weight=Point(nodes[row["NM_1"] - 1]).distance(
                Point(nodes[row["NM_3"] - 1])
            ),
        )

    return nodes, G


def dijkstra_shortest_path(
    nodes, G: GeoDataFrame, point1: Tuple[float, float], point2: Tuple[float, float]
) -> GeoDataFrame:
    # Find the nearest nodes to the given points
    start_node = min(nodes, key=lambda node: Point(nodes[node]).distance(Point(point1)))
    end_node = min(nodes, key=lambda node: Point(nodes[node]).distance(Point(point2)))

    # Apply Dijkstra's algorithm
    path_nodes = nx.shortest_path(
        G, source=start_node, target=end_node, weight="weight"
    )

    # Construct the path as a list of LineString objects
    path_segments = [
        LineString([nodes[path_nodes[i]], nodes[path_nodes[i + 1]]])
        for i in range(len(path_nodes) - 1)
    ]

    # Convert to a GeoDataFrame
    path_gdf = GeoDataFrame(geometry=path_segments)

    return path_gdf


def approximate_shapefile_boundary(
    shapefile: Union[str, GeoDataFrame],
    element_map: GeoDataFrame,
    nodes,
    algorithm: str = "greedy",
) -> Tuple[List[LineString], float]:
    """
    Approximates the boundary of a shapefile using the given element map.

    Parameters
    ----------
    shapefile : Union[str, GeoDataFrame]
        Path to the shapefile or a loaded GeoDataFrame.
    element_map : GeoDataFrame
        Element map representing the mesh.
    algorithm : str, optional
        Algorithm to use for approximation ('greedy' or 'dijkstra'). Defaults to 'greedy'.

    Returns
    -------
    list
        List of approximate line segments in the element map.
    float
        Total score for the approximation.
    """
    # If the shapefile is given as a path, load it
    if isinstance(shapefile, str):
        shapefile = gpd.read_file(shapefile)

    # Ensure the original GeoDataFrame has a geographic (latitude/longitude) CRS
    shapefile = shapefile.to_crs(epsg=4326)

    # Extract the coordinates of the shapefile's geometry
    coords = (
        shapefile.geometry.iloc[0].coords
        if shapefile.geometry.iloc[0].is_ring
        else shapefile.geometry.iloc[0].coords[:-1]
    )

    # Initialize the resulting path and score
    total_score = 0

    # Define the number of segments for the progress bar
    num_segments = len(coords) - 1

    # Iterate through the line segments in the shapefile with alive-progress bar
    with alive_bar(num_segments, title="Approximating boundary", bar="smooth") as bar:
        approx_paths = []
        true_paths = []
        scores = []
        # TODO: If can't find segment in consecutive, skip and try between start and end of next segment
        # and keep going -> For when the lien to snap is sampled at a higher rate than mesh resolution
        # TODO: Compute segment length in relation to mesh resolution?
        for start, end in zip(coords[:-1], coords[1:]):
            bar.text(f"Going from {start} to {end}")
            # Approximate the line segment using the selected search algorithm
            if algorithm == "greedy":
                approx_path = greedy_shortest_path(element_map, start, end)
            elif algorithm == "dijkstra":
                approx_path = dijkstra_shortest_path(nodes, element_map, start, end)
            else:
                raise ValueError(
                    "Invalid algorithm specified. Choose 'greedy' or 'dijkstra'."
                )

            # Calculate the score for the approximation
            if len(approx_path) != 0:
                approx_score = score_path(element_map, start, end, approx_path)

                # Append the results
                scores.append(approx_score)

                approx_paths.append(approx_path)
                true_paths.append(LineString([start, end]))
            else:
                print(f"No path found from {start} to {end}")

            # Update the progress bar with the current score
            bar.text(f"Current Score: {total_score:.2f}")
            bar()

    return true_paths, approx_paths, scores


def save_element_gdf(element_gdf: gpd.GeoDataFrame, filepath: str) -> None:
    """
    Save a GeoPandas DataFrame containing element data to a CSV file.

    Parameters
    ----------
    element_gdf : GeoDataFrame
        GeoPandas DataFrame containing element data with geometry and centroid columns.
    filepath : str
        Path to the CSV file where the data will be saved.

    Returns
    -------
    None
    """
    # Convert geometry and centroid columns to WKT
    element_gdf["geometry"] = element_gdf["geometry"].apply(lambda x: x.wkt)
    element_gdf["centroid"] = element_gdf["centroid"].apply(lambda x: x.wkt)

    # Save to CSV without index
    element_gdf.to_csv(filepath, index=False)


def load_element_gdf(filepath: str) -> gpd.GeoDataFrame:
    """
    Load a GeoPandas DataFrame containing element data from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing the element data.

    Returns
    -------
    GeoDataFrame
        GeoPandas DataFrame containing element data with reconstructed geometry and centroid columns.
    """
    # Read CSV file
    element_gdf = pd.read_csv(filepath)

    # Reconstruct geometry and centroid columns from WKT
    element_gdf["geometry"] = element_gdf["geometry"].apply(lambda x: loads(x))
    element_gdf["centroid"] = element_gdf["centroid"].apply(lambda x: loads(x))

    # Convert to GeoDa4taFrame
    element_gdf = gpd.GeoDataFrame(element_gdf, geometry="geometry")

    return element_gdf


def split_linestring_by_height(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Splits LineString Z objects in a GeoDataFrame into separate LineStrings
    based on unique Z-values (heights).

    Parameters
    ----------
    df : gpd.GeoDataFrame
        The input GeoDataFrame containing LineString Z objects in the
        'geometry' column.

    Returns
    -------
    gpd.GeoDataFrame
        A new GeoDataFrame containing separate LineStrings for each unique
        height (Z-value), with the heights stored in a separate 'height' column.

    """
    # Create an empty dataframe to store the results
    result = []
    orig_crs = df.crs

    for _, row in df.iterrows():
        geometry = row["geometry"]
        # Extract (X, Y, Z) values
        coords = list(geometry.coords)

        # Group coordinates by unique Z values
        groups = {}
        for x, y, z in coords:
            if z not in groups:
                groups[z] = []
            groups[z].append((x, y))

        # For each unique Z value, create a LINESTRING and store in result
        for z, xy_coords in groups.items():
            linestring = LineString(xy_coords)
            z = np.abs(z)
            result.append({"geometry": linestring, "height": z})

    result = gpd.GeoDataFrame(result, columns=["geometry", "height"])
    result.crs = orig_crs

    return result


def remove_segment(
    orig: gpd.GeoDataFrame, to_remove: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Removes a segment defined in `to_remove` from the original
    linestring(s) in `orig`.

    Parameters
    ----------
    orig : gpd.GeoDataFrame
        The original GeoDataFrame containing linestrings.

    to_remove : gpd.GeoDataFrame
        The GeoDataFrame containing the segment(s) to be removed.

    Returns
    -------
    gpd.GeoDataFrame
        The modified GeoDataFrame with the segment removed.
    """
    # Determine start and end points from `to_remove`
    start_point = np.array(to_remove.iloc[0]["geometry"].coords[0])
    end_point = np.array(to_remove.iloc[-1]["geometry"].coords[-1])

    # Initialize empty list to store new rows
    new_rows = []

    # Iterate through each row in `orig` to find closest points
    for _, row in orig.iterrows():
        geometry = row["geometry"]
        coords = np.array(geometry.coords)

        # Calculate distances to start_point and end_point
        dist_to_start = np.sum((coords - start_point) ** 2, axis=1)
        dist_to_end = np.sum((coords - end_point) ** 2, axis=1)

        # Find closest points
        closest_start = np.argmin(dist_to_start)
        closest_end = np.argmin(dist_to_end)

        # Handle special case where both `orig` and `to_remove` have single rows
        if len(orig) == 1 and len(to_remove) == 1:
            if closest_start < closest_end:
                part1 = LineString(coords[: closest_start + 1])
                part2 = LineString(coords[closest_end:])
            else:
                part1 = LineString(coords[: closest_end + 1])
                part2 = LineString(coords[closest_start:])

            new_rows.append({"geometry": part1})
            new_rows.append({"geometry": part2})
            continue

        # Cut segments for general case
        if closest_start < closest_end:
            new_coords = np.concatenate(
                [coords[:closest_start], coords[closest_end + 1 :]]
            )
        else:
            new_coords = coords[closest_end + 1 : closest_start]

        if new_coords.shape[0] > 1:
            new_linestring = LineString(new_coords)
            new_row = {"geometry": new_linestring}
            new_rows.append(new_row)

    # Create new GeoDataFrame
    new_df = gpd.GeoDataFrame(new_rows)

    return new_df


def create_nc_file(
    element_gdf_or_path: Union[gpd.GeoDataFrame, str],
    output_path: str,
    base_crs: int = 4326,
    target_crs: int = 4326,
) -> None:
    """
    Create a NetCDF file from a given GeoPandas DataFrame or a file path.

    Parameters
    ----------
    element_gdf_or_path : Union[gpd.GeoDataFrame, str]
        Either a GeoPandas DataFrame or a file path to load the DataFrame.
    output_path : str
        The path where the .nc file will be saved.
    base_crs : int, optional
        The base Coordinate Reference System to set if not already set.
    target_crs : int, optional
        The target Coordinate Reference System to transform to.

    Returns
    -------
    None
    """

    # Load the GeoPandas DataFrame if a file path is given
    if isinstance(element_gdf_or_path, str):
        element_gdf = gpd.read_file(element_gdf_or_path)
    else:
        element_gdf = element_gdf_or_path

    # Set the base CRS if not already set or if base_crs is explicitly provided
    if element_gdf.crs is None or base_crs:
        element_gdf.crs = f"EPSG:{base_crs}"

    # Transform to the target CRS if provided
    if target_crs:
        element_gdf = element_gdf.to_crs(f"EPSG:{target_crs}")

    # 1. Extract nodes from the element_gdf
    nodes = []
    node_indices = {}  # Dictionary to keep track of nodes and avoid duplicates
    count = 0
    for idx, row in element_gdf.iterrows():
        for i in [1, 2, 3]:
            x, y = row[f"X_{i}"], row[f"Y_{i}"]
            if (x, y) not in node_indices:
                nodes.append((x, y))
                node_indices[(x, y)] = count
                count += 1

    nodes = np.array(nodes)

    # 2. Define faces (triangles in this case)
    faces = []
    for idx, row in element_gdf.iterrows():
        face = [node_indices[(row[f"X_{i}"], row[f"Y_{i}"])] for i in [1, 2, 3]]
        faces.append(face)

    faces = np.array(faces)

    # 3. Assign data values to nodes
    node_data = np.empty(len(nodes))
    for idx, row in element_gdf.iterrows():
        for i in [1, 2, 3]:
            node_idx = node_indices[(row[f"X_{i}"], row[f"Y_{i}"])]
            node_data[node_idx] = row[f"DP_{i}"]

    node_data = np.array(node_data)

    # 4. Assign data values to faces
    face_data = element_gdf["DP"].to_numpy()

    # Save the NetCDF file to the specified output path
    rootgrp = nc.Dataset(output_path, "w", format="NETCDF4")

    # 1. Define dimensions
    nNodes_dim = rootgrp.createDimension("nMesh2_node", len(nodes))
    nFaces_dim = rootgrp.createDimension("nMesh2_face", len(faces))
    Two_dim = rootgrp.createDimension("Two", 2)
    Three_dim = rootgrp.createDimension("Three", 3)

    # 2. Mesh topology variable
    Mesh2 = rootgrp.createVariable("Mesh", "i4")
    Mesh2.cf_role = "mesh_topology"
    Mesh2.topology_dimension = 2
    Mesh2.node_coordinates = "Mesh2_node_x Mesh2_node_y"
    Mesh2.face_node_connectivity = "Mesh2_face_nodes"
    # Add other attributes as needed, e.g., edge_node_connectivity, face_coordinates, etc.

    # 3. Node coordinates
    Mesh2_node_x = rootgrp.createVariable("Mesh2_node_x", "f4", "nMesh2_node")
    Mesh2_node_x.standard_name = "longitude"
    Mesh2_node_x.long_name = "Longitude of 2D mesh nodes."
    Mesh2_node_x.units = "degrees_east"
    Mesh2_node_x[:] = nodes[
        :, 0
    ]  # Assuming nodes is a Nx2 array with [longitude, latitude]

    Mesh2_node_y = rootgrp.createVariable("Mesh2_node_y", "f4", "nMesh2_node")
    Mesh2_node_y.standard_name = "latitude"
    Mesh2_node_y.long_name = "Latitude of 2D mesh nodes."
    Mesh2_node_y.units = "degrees_north"
    Mesh2_node_y[:] = nodes[:, 1]

    # 4. Face-to-node connectivity
    Mesh2_face_nodes = rootgrp.createVariable(
        "Mesh2_face_nodes", "i4", ("nMesh2_face", "Three")
    )
    Mesh2_face_nodes.cf_role = "face_node_connectivity"
    Mesh2_face_nodes.long_name = "Maps every triangular face to its three corner nodes."
    Mesh2_face_nodes.start_index = 0  # Using 0-based indexing
    Mesh2_face_nodes[:] = faces  # Assuming faces is an Mx3 array with node indices

    # Create a variable for node_data
    node_data_var = rootgrp.createVariable("node_data_var", "f4", "nMesh2_node")
    node_data_var.long_name = "Data values at each node"
    node_data_var.units = "Your_units_here"  # Replace with appropriate units, e.g., "meters", "degrees", etc.
    node_data_var.mesh = "Mesh2"  # Associating with the mesh topology
    node_data_var[:] = node_data

    # Create a variable for face_data
    face_data_var = rootgrp.createVariable("face_data_var", "f4", "nMesh2_face")
    face_data_var.long_name = "Data values at each face"
    face_data_var.units = "Your_units_here"  # Replace with appropriate units, e.g., "meters", "degrees", etc.
    face_data_var.mesh = "Mesh2"  # Associating with the mesh topology
    face_data_var[:] = face_data

    # Close the NetCDF file
    rootgrp.close()


def save_geodataframe_with_metadata(
    gdf: gpd.GeoDataFrame, name: str, attrs: dict = None, directory: str = "."
) -> None:
    """
    Save a GeoDataFrame along with its metadata.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The GeoDataFrame to save.
    name : str
        The name to give to the saved shapefile and metadata.
    attrs : dict
        A dictionary containing the metadata attributes.
    directory : str, optional
        The directory where to save the files. Defaults to the current working directory.

    Example
    -------
    >>> attrs = {
        "Author": "John Doe",
        "Description": "This is a sample shapefile.",
        "Version": "1.0"
    }
    >>> gdf = gpd.GeoDataFrame({
        'geometry': [gpd.points_from_xy([1, 2, 3], [4, 5, 6])],
        'attribute': ['A', 'B', 'C']
    })
    >>> save_geodataframe_with_metadata(gdf, "my_shapefile", attrs, "/path/to/directory")
    """
    # Create full path
    full_path = os.path.join(directory, name)

    # Check if directory exists
    if os.path.exists(full_path):
        print(f"Warning: Directory {full_path} already exists.")
        print("Contents:", os.listdir(full_path))
        user_input = input("Do you want to remove these files and overwrite? (Y/N): ")

        if user_input.lower() == "y":
            # Remove existing directory and its contents
            shutil.rmtree(full_path)
        else:
            print("Operation cancelled.")
            return

    # Create directory
    os.makedirs(full_path)

    # Save GeoDataFrame as a shapefile
    shapefile_path = os.path.join(full_path, f"{name}.shp")
    gdf.to_file(shapefile_path)

    # Write XML metadata
    meta_dict = gdf.attrs
    meta_dict.update(attrs if attrs is not None else {})
    write_xml_metadata(meta_dict, shapefile_path, full_path)

    print(f"GeoDataFrame and metadata saved in {full_path}")


def set_crs_for_gdf(gdf: GeoDataFrame, crs_epsg: int) -> GeoDataFrame:
    """
    Set or convert the CRS (Coordinate Reference System) for a given GeoDataFrame.

    Parameters
    ----------
    gdf : GeoDataFrame
        The GeoDataFrame whose CRS needs to be set or converted.
    crs_epsg : int
        The EPSG code for the desired CRS.

    Returns
    -------
    GeoDataFrame
        The GeoDataFrame with the CRS set or converted to the specified EPSG code.
    """
    # Project the data to the specified CRS for accurate centroid calculation
    if gdf.crs is None:
        gdf.crs = f"EPSG:{crs_epsg}"
    else:
        gdf = gdf.to_crs(epsg=crs_epsg)

    return gdf


def read_and_convert_to_line(shapefile_path: str, crs_epsg: int = None) -> GeoDataFrame:
    """
    Read a shapefile and convert all geometries to lines, removing Z-coordinates if present.

    Parameters
    ----------
    shapefile_path : str
        The file path to the shapefile.
    crs_epsg : int, optional
        The EPSG code for the desired CRS (Coordinate Reference System).
        If None, the CRS will not be changed.

    Returns
    -------
    GeoDataFrame
        The GeoDataFrame containing only LineString geometries without Z-coordinates.

    Notes
    -----
    This function assumes that the shapefile contains 'Polygon Z', 'LineString Z',
    'Polygon', or 'LineString' geometries. It will convert all 'Polygon Z' and
    'LineString Z' geometries to 'LineString' by taking their exterior coordinates
    and removing Z-coordinates.
    """
    # Read the shapefile into a GeoDataFrame
    gdf = process_shapefiles(shapefile_path, crs_epsg=crs_epsg, plot=False)

    # Optionally, set or convert CRS
    if crs_epsg is not None:
        if gdf.crs is None:
            gdf.crs = f"EPSG:{crs_epsg}"
        else:
            gdf = gdf.to_crs(epsg=crs_epsg)

    # Convert all geometries to LineString and remove Z-coordinates
    new_geometries = []
    for geom in gdf["geometry"]:
        if geom.geom_type == "Polygon":
            new_geometries.append(LineString(geom.exterior.coords))
        elif geom.geom_type == "LineString":
            if geom.has_z:
                new_geometries.append(LineString([(x, y) for x, y, z in geom.coords]))
            else:
                new_geometries.append(geom)
        else:
            raise ValueError(f"Unknown geometry {geom.geom_type} found.")

    gdf["geometry"] = new_geometries
    gdf = split_linestring_by_height(gdf)

    return gdf


def read_and_convert_to_polygon(
    shapefile_path: str, crs_epsg: int = None
) -> GeoDataFrame:
    """
    Read a shapefile and convert all Z-Polygons to regular Polygons,
    setting their height to the average of the Z values.

    Parameters
    ----------
    shapefile_path : str
        The file path to the shapefile.
    crs_epsg : int, optional
        The EPSG code for the desired CRS (Coordinate Reference System).
        If None, the CRS will not be changed.

    Returns
    -------
    GeoDataFrame
        The GeoDataFrame containing only Polygon geometries without Z-coordinates
        and a new column 'avg_height' for the average height of Z-Polygons.

    Notes
    -----
    This function assumes that the shapefile contains 'Polygon Z', 'Polygon', or
    'LineString' geometries. It will convert all 'Polygon Z' geometries to 'Polygon'
    by taking their exterior coordinates and removing Z-coordinates.
    """
    # Read the shapefile into a GeoDataFrame
    gdf = process_shapefiles(shapefile_path, crs_epsg=crs_epsg, plot=False)

    # Prepare lists to store new geometries and their average heights
    new_geometries = []
    avg_heights = []

    for geom in gdf["geometry"]:
        if geom.geom_type == "Polygon":
            if geom.has_z:
                coords = np.array(geom.exterior.coords)
                avg_z = np.mean(coords[:, 2])
                new_geom = Polygon(coords[:, :2])
                new_geometries.append(new_geom)
                avg_heights.append(avg_z)
            else:
                new_geometries.append(geom)
                avg_heights.append(None)
        elif geom.geom_type == "LineString":
            new_geometries.append(geom)
            avg_heights.append(None)
        else:
            new_geometries.append(None)  # Unknown geometry type
            avg_heights.append(None)

    gdf["geometry"] = new_geometries
    gdf["height"] = avg_heights

    return gdf


#### NOT SURE WHERE TO PUT THESE:
