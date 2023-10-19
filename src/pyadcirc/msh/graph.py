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


# TODO: Improve build_graph function
def build_graph(node_map, element_map):
    if type(element_map) == pd.DataFrame:
        df1 = (
            element_map[["NM_1", "NM_2"]]
            .rename(columns={"NM_1": "X", "NM_2": "Y"})
            .astype({"X": "Int64", "Y": "Int64"})
        )
        df2 = (
            element_map[["NM_2", "NM_3"]]
            .rename(columns={"NM_2": "X", "NM_3": "Y"})
            .astype({"X": "Int64", "Y": "Int64"})
        )
        df3 = (
            element_map[["NM_3", "NM_1"]]
            .rename(columns={"NM_3": "X", "NM_1": "Y"})
            .astype({"X": "Int64", "Y": "Int64"})
        )
        edge_list_df = pd.concat([df1, df2, df3], ignore_index=True)
    elif type(element_map) == pl.DataFrame:
        edge_list_df = pd.DataFrame(
            pl.concat(
                [
                    element_map["element_map"].select(
                        [
                            pl.col("NM_1").alias("X").cast(pl.Int64),
                            pl.col("NM_2").alias("Y").cast(pl.Int64),
                        ]
                    ),
                    element_map["element_map"].select(
                        [
                            pl.col("NM_2").alias("X").cast(pl.Int64),
                            pl.col("NM_3").alias("Y").cast(pl.Int64),
                        ]
                    ),
                    element_map["element_map"].select(
                        [
                            pl.col("NM_3").alias("X").cast(pl.Int64),
                            pl.col("NM_1").alias("Y").cast(pl.Int64),
                        ]
                    ),
                ]
            ),
            columns=["X", "Y"],
        )
    else:
        raise ValueError(f"Invalid type for element_map: {type(element_map)}")

    graph = nx.from_pandas_edgelist(edge_list_df, source="X", target="Y")

    return graph


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


def find_closest_element_min_length(
    segment: LineString, nodes: dict, graph: nx.Graph
) -> float:
    """
    Finds the minimum length of the closest mesh element to a given segment.

    Parameters
    ----------
    segment : LineString
        The line segment for which to find the closest mesh element.
    nodes : dict
        Dictionary mapping node indices to their coordinates.
    graph : nx.Graph
        NetworkX graph object representing the mesh.

    Returns
    -------
    float
        The minimum length of the closest mesh element to the given segment.
    """

    # Initialize variables to keep track of closest node and its distance
    closest_node = None
    min_distance = float("inf")

    # Iterate through all nodes to find the closest node to the segment
    for node_idx, coords in nodes.items():
        point = Point(coords)
        distance = segment.distance(point)

        if distance < min_distance:
            min_distance = distance
            closest_node = node_idx

    # Now find the minimum length among the edges connected to the closest node
    min_length = float("inf")
    for neighbor in graph.neighbors(closest_node):
        edge_length = Point(nodes[closest_node]).distance(Point(nodes[neighbor]))

        if edge_length < min_length:
            min_length = edge_length

    return min_length
