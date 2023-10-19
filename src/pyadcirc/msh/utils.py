from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.wkt import loads as wkt_loads

from pyadcirc.io.shapefiles import create_bbox_polygon, set_crs_for_gdf
from pyadcirc.msh.ADCIRCMesh import ADCIRCMesh


def trim_f14_grid(
    msh: ADCIRCMesh = None,
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

    if msh is None:
        msh = ADCIRCMesh(f14_path)

    # Select rows where NM_1, NM_2 or NM_3 is in nodes_within_bounding_box['JN']
    nodes_within_bounding_box = gpd.sjoin(
        msh.node_gdf, bounding_box, predicate="within"
    )
    mask = (
        msh.element_map["NM_1"].isin(nodes_within_bounding_box["JN"])
        | msh.element_map["NM_2"].isin(nodes_within_bounding_box["JN"])
        | msh.element_map["NM_3"].isin(nodes_within_bounding_box["JN"])
    )

    elements_with_filtered_nodes = msh.element_map[mask].copy()

    if not within_box:
        elements_with_filtered_nodes = add_neighbors_to_element_map(
            elements_with_filtered_nodes, msh.element_map, depth=neighbors
        )
        ndf = msh.node_gdf
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
    element_gdf["geometry-wkt"] = element_gdf["geometry"].apply(lambda x: x.wkt)
    element_gdf["centroid"] = element_gdf["centroid"].apply(lambda x: x.wkt)

    # Save to CSV without index
    element_gdf.to_csv(filepath, index=False)


def load_element_gdf(
    filepath: str,
    crs_epsg: int = 4326,
) -> gpd.GeoDataFrame:
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
    element_gdf["geometry"] = element_gdf["geometry-wkt"].apply(lambda x: wkt_loads(x))
    element_gdf["centroid"] = element_gdf["centroid"].apply(lambda x: wkt_loads(x))

    # Convert to GeoDa4taFrame
    element_gdf = gpd.GeoDataFrame(element_gdf, geometry="geometry")
    element_gdf = set_crs_for_gdf(element_gdf, crs_epsg=crs_epsg)

    return element_gdf
