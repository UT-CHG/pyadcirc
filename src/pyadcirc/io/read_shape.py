
import pdb
import numpy as np
import geopandas as gpd
import folium
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from typing import Optional
from shapely.geometry import Polygon
from typing import Tuple, Optional
from pyadcirc.io.io import read_fort14_element_map, read_fort14_node_map
from rich.traceback import install
install(show_locals=True)


def plot_shapes(
    gdf: Optional[gpd.GeoDataFrame] = None,
    dir: str = '/Users/carlos/Downloads/Galveston_Ring_Barrier_System_Polyline',
    plot: Optional[bool] = True,
    plot_kwargs: Optional[dict] = None,
    plot_filename: Optional[str] = None,
    map_filename: Optional[str] = 'map.html',
    metadata: Optional[bool] = False,
    crs_epsg: Optional[int] = 32616
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
        avg_latitude, avg_longitude = center_geographic.geometry[0].y, center_geographic.geometry[0].x

        # Create a Map instance
        m = folium.Map(
            location=[avg_latitude, avg_longitude],
            zoom_start=12,
            control_scale=True
        )

        # Add the shapefile to the map
        folium.GeoJson(gdf).add_to(m)

        # Show or save the map
        m.save(map_filename)

    # Read XML metadata
    if metadata:
        xml_file = dir.replace('.shp', '.shp.xml')
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
    f14_path: Optional[str] = 'fort.14',
    shape_path: Optional[str] = 'boundary',
    bounding_box: Optional[gpd.GeoDataFrame] = None,
    center: Optional[Tuple[float, float]] = None,
    size: Optional[float] = None,
    neighbors: Optional[int] = 0,
    within_box: Optional[bool] = True
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
            shape_path=shape_path, center=center, size=size)
    if not isinstance(bounding_box, gpd.GeoDataFrame):
        bounding_box = create_bbox_polygon(bounding_box=bounding_box)

    print(f'Reading in node map from {f14_path}')
    node_map = read_fort14_node_map(f14_path)

    print(f'Converting to GeoDataFrame')
    nodes_gdf = gpd.GeoDataFrame(node_map, geometry=gpd.points_from_xy(node_map.X, node_map.Y)).set_crs(epsg=4326)

    print(f'Filtering node by boundary')
    nodes_within_bounding_box = gpd.sjoin(nodes_gdf, bounding_box, predicate='within')

    print(f'Reading in element map from {f14_path}')
    element_map = read_fort14_element_map(f14_path)

    # Select rows where NM_1, NM_2 or NM_3 is in nodes_within_bounding_box['JN']
    mask = (element_map['NM_1'].isin(nodes_within_bounding_box['JN']) |
            element_map['NM_2'].isin(nodes_within_bounding_box['JN']) |
            element_map['NM_3'].isin(nodes_within_bounding_box['JN']))

    elements_with_filtered_nodes = element_map[mask].copy()

    if not within_box:
        elements_with_filtered_nodes = add_neighbors_to_element_map(
            elements_with_filtered_nodes, element_map, depth=neighbors) 
        ndf = nodes_gdf
    else:
        ndf = nodes_within_bounding_box

    elements_with_filtered_nodes.loc[:, 'DP'] = np.nan
    elements_with_filtered_nodes.loc[:, 'X'] = np.nan
    elements_with_filtered_nodes.loc[:, 'Y'] = np.nan

    # * Merge in X, Y and depth data from nodes dataframe by linking on 'NM_1', 'NM_2', and 'NM_3' cols
    cols = ['JN', 'X', 'Y', 'DP']
    element_map = elements_with_filtered_nodes.merge(ndf[cols], left_on='NM_1', right_on='JN', suffixes=('', '_1'))
    element_map = element_map.merge(ndf[cols], left_on='NM_2', right_on='JN', suffixes=('', '_2'))
    element_map = element_map.merge(ndf[cols], left_on='NM_3', right_on='JN', suffixes=('', '_3'))
    element_map['DP'] = element_map[['DP_1', 'DP_2', 'DP_3']].mean(axis=1)
    element_map["geometry"] = element_map.apply(
        lambda row: Polygon([(row['X_1'], row['Y_1']), (row['X_2'], row['Y_2']), (row['X_3'], row['Y_3'])]), axis=1)
    elements_gdf = gpd.GeoDataFrame(element_map, geometry='geometry')
    elements_gdf["centroid"] = elements_gdf["geometry"].centroid

    return elements_gdf


def add_neighbors_to_element_map(
    sub_map: pd.DataFrame,
    full_map: pd.DataFrame,
    depth: int = 1
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
            [sub_map['NM_1'], sub_map['NM_2'], sub_map['NM_3']]).unique()

        # Create a mask to select elements that have at least one node in selected_nodes
        neighbor_mask = (
            full_map['NM_1'].isin(selected_nodes) |
            full_map['NM_2'].isin(selected_nodes) |
            full_map['NM_3'].isin(selected_nodes))

        # Append the neighboring elements to the selected elements
        sub_map = pd.concat(
            [sub_map, full_map[neighbor_mask]]).drop_duplicates()

    return sub_map


def get_lines_from_element_map(element_map):
    """
    """

    lines = pd.DataFrame([
        {"geometry": LineString([(row['X_1'], row['Y_1']), (row['X_2'], row['Y_2'])]), "nodes": frozenset([row['NM_1'], row['NM_2']])} for _, row in element_map.iterrows()] +
        [{"geometry": LineString([(row['X_2'], row['Y_2']), (row['X_3'], row['Y_3'])]), "nodes": frozenset([row['NM_2'], row['NM_3']])} for _, row in element_map.iterrows()] +
        [{"geometry": LineString([(row['X_3'], row['Y_3']), (row['X_1'], row['Y_1'])]), "nodes": frozenset([row['NM_3'], row['NM_1']])} for _, row in element_map.iterrows()]
    )
    # remove duplicates
    lines = lines.drop_duplicates(subset="nodes")
    lines_gdf = gpd.GeoDataFrame(lines, geometry='geometry')

    return lines_gdf


def calculate_bbox(
    gdf: Optional[gpd.GeoDataFrame] = None,
    dir: Optional[str] = None
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
            raise ValueError('Either gdf or dir must be specified.')
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
    crs: str = 'EPSG:4326'
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
                [center[0] - size / 2,
                 center[1] - size / 2],
                [center[0] + size / 2,
                 center[1] + size / 2]
            ]
        elif shape_path or gdf is not None:
            bounding_box = calculate_bbox(gdf=gdf, dir=shape_path)
        else:
            raise ValueError('Either center/size, bounding_box, shapefile_dir, or gdf must be specified.')

    # Create a polygon from the bounding box
    bounding_box_polygon = Polygon([
        (bounding_box[0][0], bounding_box[0][1]),
        (bounding_box[0][0], bounding_box[1][1]),
        (bounding_box[1][0], bounding_box[1][1]),
        (bounding_box[1][0], bounding_box[0][1])
    ])

    # Create a GeoDataFrame from the bounding box polygon
    bounding_box_gdf = gpd.GeoDataFrame(geometry=[bounding_box_polygon], crs=crs)

    return bounding_box_gdf


def plot_mesh(
    node_map: gpd.GeoDataFrame,
    element_map: gpd.GeoDataFrame,
    plot: Optional[bool] = True,
    plot_filename: Optional[str] = None,
    map_filename: Optional[str] = 'map.html',
    crs_epsg: Optional[int] = 32616
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
    nodes = gpd.GeoDataFrame(node_map, geometry=gpd.points_from_xy(node_map.X, node_map.Y)).set_crs(epsg=4326)
    nodes.set_index('JN', inplace=True)  # set 'JN' as the index for direct lookup

    if 'DP' not in element_map.columns:
        element_map['DP'] = element_map.apply(lambda row: nodes.loc[[row['NM_1'], row['NM_2'], row['NM_3']]].DP.mean(), axis=1)
    polygons = element_map.apply(lambda row: Polygon(nodes.loc[[row['NM_1'], row['NM_2'], row['NM_3']]].geometry.values), axis=1)
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
    map_filename: Optional[str] = 'map.html'
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
        (element_map['NM_1'] == node_index) |
        (element_map['NM_2'] == node_index) |
        (element_map['NM_3'] == node_index)
    ]
    elements_with_node = add_neighbors_to_element_map(elements_with_node, element_map, depth=neighbors)

    # Extract the node indices from these elements
    node_indices = pd.unique(elements_with_node[['NM_1', 'NM_2', 'NM_3']].values.ravel('K'))

    # Filter node_map to get these nodes
    nodes = node_map[node_map['JN'].isin(node_indices)]

    # Convert the node map to a GeoDataFrame
    nodes_gdf = gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes.X, nodes.Y)).set_crs(epsg=4326)

    # Create polygons for the elements
    elements_with_node['DP'] = elements_with_node.apply(lambda row: nodes.loc[[row['NM_1'] - 1, row['NM_2'] - 1, row['NM_3'] - 1]].DP.mean(), axis=1)
    polygons = elements_with_node.apply(lambda row: Polygon(nodes_gdf.loc[[row['NM_1'] - 1, row['NM_2'] - 1, row['NM_3'] - 1]].geometry.values), axis=1)
    elements_gdf = gpd.GeoDataFrame(elements_with_node, geometry=polygons).set_crs(epsg=4326)

    # Plot the elements using GeoPandas
    if plot:
        fig, ax = plt.subplots(1, 1)
        elements_gdf.plot(column='DP', ax=ax, legend=True)
        # Plot a distinct black point for the original node
        original_node = nodes_gdf.loc[node_index - 1]
        # original_node.geometry.plot(ax=ax, color='black', markersize=50)
        ax.scatter(original_node.geometry.x, original_node.geometry.y, color='black', s=50)

        if plot_filename is not None:
            plt.savefig(plot_filename)
        else:
            plt.show()

    # Create a map with Folium
    m = folium.Map(location=[nodes.Y.mean(), nodes.X.mean()], zoom_start=12)
    folium.GeoJson(elements_gdf).add_to(m)
    m.save(map_filename)


# Main script execution entrypoint
if __name__ == '__main__':
    ring_barrier_path = '/Users/carlos/Downloads/Galveston_Ring_Barrier_System_Polyline'
    f14 = '/Users/carlos/Documents/UT Austin/Research/Data/adcirc_data/grids/si/fort.14'
    f14_large = '/Users/carlos/repos/pyDCI/fort.14'

    boundary = gpd.read_file(ring_barrier_path)
    bbox = create_bbox_polygon(shape_path=ring_barrier_path)
    element_gdf = trim_f14_grid(f14_path=f14_large, bounding_box=bbox, within_box=False, neighbors=2)

    fig, ax = plt.subplots(1, 1)
    element_gdf.plot(column='DP', cmap='Blues', legend=True, ax=ax)
    bbox.boundary.plot(ax=ax)
    plot_shapes(gdf=boundary, plot=True, plot_kwargs=dict(ax=ax))
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