import heapq
import pdb
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple, Union

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from alive_progress import alive_bar
from geopandas import GeoDataFrame
from pyadcirc.io.io import read_fort14_element_map, read_fort14_node_map
from rich.traceback import install
from shapely.geometry import LineString, Point, Polygon
from shapely.wkt import loads
import networkx as nx
from shapely.geometry import LineString, Point

install(show_locals=True)


def plot_shapes(
    gdf: Optional[gpd.GeoDataFrame] = None,
    dir: str = "/Users/carlos/Downloads/Galveston_Ring_Barrier_System_Polyline",
    plot: Optional[bool] = True,
    plot_kwargs: Optional[dict] = None,
    plot_filename: Optional[str] = None,
    map_filename: Optional[str] = "map.html",
    metadata: Optional[bool] = False,
    crs_epsg: Optional[int] = 32616,
) -> gpd.GeoDataFrame:
    """
    Given a path to a .shp file and supporting files, perform various operations
    including reading, plotting, and extracting metadata.

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
        gdf = gpd.read_file(dir)

    # Ensure the original GeoDataFrame has a geographic (latitude/longitude) CRS
    gdf = gdf.to_crs(epsg=4326)

    # Plot with GeoPandas
    if plot:
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        gdf.plot(**plot_kwargs)
        plt.show()
        if plot_filename is not None:
            plt.savefig(plot_filename)

    # Plot with Folium
    if map:
        # Project the data to the specified CRS for accurate centroid calculation
        gdf_projected = gdf.to_crs(epsg=crs_epsg)

        # Calculate the center of the map
        center = gdf_projected.unary_union.centroid

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

    # Read XML metadata
    if metadata:
        xml_file = dir.replace(".shp", ".shp.xml")
        try:
            # Parse the XML file
            tree = ET.parse(xml_file)

            # Get the root element
            root = tree.getroot()

            # Print all elements in the XML
            metadata = {}
            for elem in root.iter():
                metadata[elem.tag] = elem.text
        except FileNotFoundError:
            print("Metadata file not found.")

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
    gdf = gdf.to_crs(epsg=4326)

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
    result_path = []
    total_score = 0

    # Define the number of segments for the progress bar
    num_segments = len(coords) - 1

    # Iterate through the line segments in the shapefile with alive-progress bar
    with alive_bar(num_segments, title="Approximating boundary", bar="smooth") as bar:
        approx_paths = []
        true_paths = []
        scores = []
        for start, end in zip(coords[:-1], coords[1:]):
            bar.text(f'Going from {start} to {end}')
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
                print(f'No path found from {start} to {end}')

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

    # Convert to GeoDataFrame
    element_gdf = gpd.GeoDataFrame(element_gdf, geometry="geometry")

    return element_gdf


# Main script execution entrypoint
if __name__ == "__main__":
    ring_barrier_path = "/Users/carlos/Downloads/Galveston_Ring_Barrier_System_Polyline"
    f14 = "/Users/carlos/Documents/UT Austin/Research/Data/adcirc_data/grids/si/fort.14"
    f14_large = "/Users/carlos/repos/pyDCI/fort.14"

    boundary = gpd.read_file(ring_barrier_path).to_crs(epsg=4326)
    bbox = create_bbox_polygon(shape_path=ring_barrier_path)
    # element_gdf = trim_f14_grid(f14_path=f14_large, bounding_box=bbox, within_box=False, neighbors=2)
    # element_gdf = trim_f14_grid(f14_path=f14_large, bounding_box=bbox, within_box=False, neighbors=2)


    element_gdf = load_element_gdf("ring_barrier_cutout.csv")
    nodes, Graph = build_graph(element_gdf)
    true, approx, score = approximate_shapefile_boundary(
        ring_barrier_path, G, n, algorithm="dijkstra"
    )
    # try:
    # except Exception as e:
    #     print(e)
    # pdb.set_trace()

    fig, ax = plt.subplots(1, 1)
    element_gdf.plot(column="DP", cmap="Blues", legend=True, ax=ax)
    bbox.boundary.plot(ax=ax)
    plot_shapes(gdf=boundary, plot=True, plot_kwargs=dict(ax=ax))
    plt.show()

    dissolved = pd.concat([a.dissolve() for a in approx])
    true_gdf = gpd.GeoDataFrame(geometry=true)
    boundary = boundary.to_crs(epsg=4326)
    fig, ax = plt.subplots(1, 1)
    ax = element_gdf.plot(column="DP", cmap="Blues", legend=True, ax=ax)
    ax = dissolved[13:16].plot(ax=ax)
    ax = true_gdf[13:15].plot(ax=ax, color='green')
    plt.show()

    pdb.set_trace()

    # bbx = create_bbox_polygon(dir=ring_barrier_path)
    # bbox = create_bbox_polygon([[-72.4874359638,40.854701761],[-72.4689653685,40.8640148532]])
    # bbox = create_bbox_polygon(bounding_box=[[-72.4874359638,40.854701761],[-72.4689653685,40.8640148532]])
    # node_map = read_fort14_node_map(f14)
    # element_map = read_fort14_element_map(f14)
    # plot_elements_with_node(node_map, element_map, 4, neighbors=3)
    # pdb.set_trace()
    # plot_mesh(node_map, element_map)

    # node_map, element_map = trim_f14_grid(f14_path=f14, bounding_box=bbox)
    # plot_elements_with_node(node_map, element_map, 1)
    # pdb.set_trace()
    # # node_map, element_map = trim_f14_grid(f14_path=f14, center=(-72.05, 40.99), size=0.05)
    # # node_map, element_map = trim_f14_grid(f14_path=f14_2, shape_path=ring_barrier_path)
    # plot_mesh(node_map, element_map)

# TODO:
# 1. Plot cut-out grid and shapefile on top of one another.
# 2. attempt "snapping" algorithm.



# line_strings = approx_boundary['geometry'].tolist()
# coords = [coord for line in line_strings for coord in line.coords[:-1]] + [line_strings[-1].coords[-1]]
# collapsed_line = LineString(coords)
# gdf_collapsed = gpd.GeoDataFrame(geometry=[collapsed_line])
# 
# # test = gpd.GeoDataFrame(geometry=approx[155:-1])
# test = gpd.GeoDataFrame(geometry=approx[:175])
# boundary = boundary.to_crs(epsg=4326)
# ax = approx_boundary.plot()
# ax = boundary.plot(ax=ax, color='green')
# test.plot(ax=ax, color='red')
# plt.show()
# 


# idx = 10
# test = gpd.GeoDataFrame(geometry=approx[idx])
# test.plot()
# plt.show()
# 
# # line_strings = approx_boundary['geometry'].tolist()
# # coords = [coord for line in line_strings for coord in line.coords[:-1]] + [line_strings[-1].coords[-1]]
# # collapsed_line = LineString(coords)
# # gdf_collapsed = gpd.GeoDataFrame(geometry=[collapsed_line])
# # 
# dissolved = pd.concat([a.dissolve() for a in approx])
# true_gdf = gpd.GeoDataFrame(geometry=true)
# boundary = boundary.to_crs(epsg=4326)
# fig, ax = plt.subplots(1, 1)
# ax = element_gdf.plot(column="DP", cmap="Blues", legend=True, ax=ax)
# ax = dissolved[13:16].plot(ax=ax)
# ax = true_gdf[13:15].plot(ax=ax, color='green')
# plt.show()