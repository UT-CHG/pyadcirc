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

from pyadcirc.io.shapefiles import load_shapefile

install(show_locals=True)


def write_nc_qgis_mesh(
    element_gdf_or_path: Union[gpd.GeoDataFrame, str],
    output_path: str,
    target_crs: int = 4326,
    data_col: str = "DP",
) -> None:
    """
    Create a NetCDF file from a given GeoPandas DataFrame or a file path.
    The output netcdf file should be loadable by qgis.

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
        # TODO: Load using different functions depending on if .shp or .f14
        element_gdf = load_shapefile(
            element_gdf_or_path,
            metatdata=True,
            crs_epsg=target_crs,
        )
    else:
        element_gdf = element_gdf_or_path

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
            node_data[node_idx] = row[f"{data_col}_{i}"]

    node_data = np.array(node_data)

    # 4. Assign data values to faces
    face_data = element_gdf[f"{data_col}"].to_numpy()

    # Save the NetCDF file to the specified output path
    rootgrp = nc.Dataset(output_path, "w", format="NETCDF4")

    # 1. Define dimensions
    rootgrp.createDimension("nMesh2_node", len(nodes))
    rootgrp.createDimension("nMesh2_face", len(faces))
    rootgrp.createDimension("Two", 2)
    rootgrp.createDimension("Three", 3)

    # 2. Mesh topology variable
    Mesh2 = rootgrp.createVariable(f"Mesh-{data_col}", "i4")
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


def load_nc_qgis_mesh(nc_file_path: str) -> gpd.GeoDataFrame:
    """
    Load a triangular mesh from a NetCDF file and return it as a GeoDataFrame.

    Parameters
    ----------
    nc_file_path : str
        The path to the NetCDF file.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame representing the triangular mesh.
    """
    # Read the NetCDF file
    rootgrp = nc.Dataset(nc_file_path, "r")

    # Extract nodes and faces
    nodes = np.stack(
        [rootgrp.variables["Mesh2_node_x"][:], rootgrp.variables["Mesh2_node_y"][:]],
        axis=1,
    )
    faces = rootgrp.variables["Mesh2_face_nodes"][:]

    # Initialize a list to hold the rows
    rows = []

    # Populate the DataFrame
    for i, face in enumerate(faces):
        coords = nodes[face]
        polygon = Polygon(coords)
        centroid = polygon.centroid

        element_data = {
            "JE": i,  # Just using the index as JE for this example
            "NHY": None,  # Placeholder, fill as needed
            "NM_1": face[0],
            "NM_2": face[1],
            "NM_3": face[2],
            "DP": None,  # Placeholder, fill as needed
            "X": coords[0][0],
            "Y": coords[0][1],
            "JN": face[0],  # Assuming JN corresponds to NM_1
            "X_1": coords[0][0],
            "Y_1": coords[0][1],
            "DP_1": None,  # Placeholder, fill as needed
            "JN_2": face[1],  # Assuming JN_2 corresponds to NM_2
            "X_2": coords[1][0],
            "Y_2": coords[1][1],
            "DP_2": None,  # Placeholder, fill as needed
            "JN_3": face[2],  # Assuming JN_3 corresponds to NM_3
            "X_3": coords[2][0],
            "Y_3": coords[2][1],
            "DP_3": None,  # Placeholder, fill as needed
            "geometry": polygon,
            "centroid": centroid,
        }

        rows.append(element_data)

    # Create a GeoDataFrame from the list of rows
    element_gdf = gpd.GeoDataFrame(rows, geometry="geometry")

    return element_gdf
