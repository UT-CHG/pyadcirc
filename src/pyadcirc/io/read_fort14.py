"""
TODO: Shift all functions to polars
"""

import glob
import linecache as lc
import logging
import os
import pdb
import re
import subprocess
from contextlib import contextmanager
from functools import reduce
from pathlib import Path
from time import perf_counter, sleep
from typing import List, Optional, Tuple

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from geopandas import GeoDataFrame
from shapely import Polygon, wkt

from pyadcirc.io.shapefiles import get_bbox_poly
from pyadcirc.log import logger


def clean_f14(file_path: str, in_place: bool = False) -> None:
    """
    Modify an ADCIRC fort.14 file to make it readable by polars engine by:

    1. Replace all continuous groups of whitespace characters with a single space.
    2. Remove all leading whitespaces at the beginning of each line.

    These modifications should not affect how the file is read in by ADCIRC.

    Parameters
    ----------
    file_path : str
        Path to the file to be modified.
    in_place : bool, default = False
        Whether to modify the file in place or save it as a new file.

    Returns
    -------
    None
    """
    # Generate the output file path
    path_obj = Path(file_path)
    if in_place:
        output_file_path = file_path
    else:
        output_file_path = f"{path_obj.stem}_modified{path_obj.suffix}"

    # Replace all continuous groups of whitespace characters with a single space
    cmd1 = f"sed -E 's/[[:space:]]+/ /g' {file_path}"
    # Remove all leading whitespaces at the beginning of each line
    cmd2 = f"sed 's/^[[:space:]]*//'"

    full_cmd = f"{cmd1} | {cmd2} > {output_file_path}"

    if in_place:
        # If in-place modification is required, use a temporary file to hold the changes
        temp_file_path = f"{path_obj.stem}_temp{path_obj.suffix}"
        full_cmd = f"{cmd1} | {cmd2} > {temp_file_path} && mv {temp_file_path} {output_file_path}"

    # Perform the sed operations
    try:
        subprocess.run(full_cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while modifying the file: {e}")

    return output_file_path


def get_base_params(mesh_path):
    """ """
    params = {}
    params["AGRID"] = lc.getline(mesh_path, 1).strip()
    params["NE"], params["NP"] = map(int, lc.getline(mesh_path, 2).split()[:2])

    return params


def get_raw_data_q(mesh_path: str = "fort.14"):
    """ """
    raw_data_query = pl.scan_csv(
        mesh_path,
        has_header=False,
        separator=" ",
        skip_rows=2,
        schema=dict([(f"column_{i}", pl.Float64) for i in range(8)]),
        comment_char="=",
        truncate_ragged_lines=True,
        ignore_errors=True,
    )

    return raw_data_query


def get_node_map_q(
    mesh_path="fort.14",
    params=None,
    raw_data=None,
    node_map=None,
    bbox=None,
    bbox_factor=1.1,
    neighbors=0,
    crs="EPSG:4326",
):
    """ """
    if type(node_map) == pl.DataFrame:
        node_map_q = node_map.lazy()
    elif node_map is None:
        raw_data_q = raw_data
        if raw_data_q is None:
            raw_data_q = get_raw_data_q(mesh_path)
        elif type(raw_data_q) == pl.DataFrame:
            raw_data_q = raw_data_q.lazy()
        if params is None:
            if not Path(mesh_path).exists():
                raise ValueError(
                    f"Mesh path {mesh_path} does not exist and params not specified. Use get_base_params to get params from a mesh file."
                )
            params = get_base_params(mesh_path)

        node_map_q = raw_data_q.slice(0, params["NP"]).select(
            [
                pl.col("column_1").alias("JN").cast(pl.Int64),
                pl.col("column_2").alias("X").cast(pl.Float64),
                pl.col("column_3").alias("Y").cast(pl.Float64),
                pl.col("column_4").alias("DP").cast(pl.Float64),
            ]
        )
    else:
        node_map_q = node_map

    if bbox is not None:
        minx, miny, maxx, maxy = get_bbox_poly(
            bbox, factor=bbox_factor, crs=crs
        ).total_bounds
        node_map_q = (
            node_map_q.filter(pl.col("X") >= minx)
            .filter(pl.col("X") <= maxx)
            .filter(pl.col("Y") >= miny)
            .filter(pl.col("Y") <= maxy)
        )

    return node_map_q


def get_element_map_q(
    mesh_path="fort.14",
    raw_data=None,
    node_map=None,
    params=None,
    bbox=None,
    bbox_factor=1.1,
    crs="EPSG:4326",
):
    """ """
    if raw_data is None:
        raw_data_q = get_raw_data_q(mesh_path)
    elif type(raw_data) == pl.DataFrame:
        raw_data_q = raw_data.lazy()
    elif type(raw_data) == pl.LazyFrame:
        raw_data_q = raw_data
    else:
        raise ValueError(f"Invalid type for raw_data: {type(raw_data)}")

    if params is None:
        params = get_base_params(mesh_path)

    if node_map is None:
        node_map_q = get_node_map_q(
            raw_data=raw_data_q,
            params=params,
            bbox=bbox,
            bbox_factor=bbox_factor,
            crs=crs,
        )
    elif type(node_map) == pl.DataFrame:
        node_map_q = node_map.lazy()
    elif type(node_map) == pl.LazyFrame:
        node_map_q = node_map
    else:
        raise ValueError(f"Invalid type for node_map: {type(node_map)}")

    cols = ["JN", "X", "Y", "DP"]
    element_map_q = (
        raw_data_q.slice(params["NP"], params["NE"])
        .select(
            [
                pl.col("column_1").alias("JE").cast(pl.Int64),
                pl.col("column_2").alias("NHY").cast(pl.Int64),
                pl.col("column_3").alias("NM_1").cast(pl.Int64),
                pl.col("column_4").alias("NM_2").cast(pl.Int64),
                pl.col("column_5").alias("NM_3").cast(pl.Int64),
                pl.lit(np.nan).alias("X").cast(pl.Float64),
                pl.lit(np.nan).alias("Y").cast(pl.Float64),
                pl.lit(np.nan).alias("DP").cast(pl.Float64),
            ]
        )
        .join(
            node_map_q.select(cols),
            left_on="NM_1",
            right_on="JN",
            suffix="_1",
        )
        .join(
            node_map_q.select(cols),
            left_on="NM_2",
            right_on="JN",
            suffix="_2",
        )
        .join(
            node_map_q.select(cols),
            left_on="NM_3",
            right_on="JN",
            suffix="_3",
        )
    )

    return element_map_q


def get_boundaries(params, raw_data):
    """
    Reads boundary information from fort.14.

    Parameters
    ----------
    f14_file : str, default = 'fort.14'
        Path to fort.14 file.
    params : dict, optional
        Dictioanry containing already loaded in f14 parameters. If None, load in the parameters first from the fort.14 file.

    Returns
    -------
    elev_boundary : pd.DataFrame
        DataFrame indexed by boundary segment, and with columns 'JN' and 'IBTYPEE', with 'IBTYPEE' being the boundary type for the elevation specified boundary segment (always 0 for elevation boundaries) and 'JN' being the node number of the boundary node for the elevation specified boundary segment.
    """
    nope = int(raw_data.slice(params["NE"] + params["NP"], 1).to_numpy()[0][0])
    idx = 1 + params["NE"] + params["NP"]
    open_bnds = []
    for i in range(nope):
        vals = raw_data.slice(idx, 1).to_numpy()
        logger.info(f"Raw data for open boundary {i}: {vals}")
        nvdll = int(vals[0][0])
        ibtypee = 0 if np.isnan(vals[0][1]) else int(vals[0][1])
        bnd_seg = raw_data.slice(idx + 1, nvdll).select(
            pl.lit(i).alias("nope_idx"),
            pl.lit(i).alias("ibtypee"),
            pl.col("column_1").alias("JN").cast(pl.Int64),
        )
        logger.info(
            f"Read in open boundary segment {i} of length {nvdll} with type {ibtypee}"
        )
        open_bnds.append(bnd_seg)
        idx = idx + nvdll

    idx = idx + 2
    nvel = int(raw_data.slice(idx, 1).to_numpy()[0][0])
    idx = idx + 2
    flow_bnds = []
    for i in range(nvel):
        vals = raw_data.slice(idx, 1).to_numpy()
        logger.info(f"Raw data for flow boundary {i}: {vals}")
        nvell = int(vals[0][0])
        ibtype = 0 if np.isnan(vals[0][1]) else int(vals[0][1])
        if ibtype in [0, 1, 2, 10, 11, 12, 20, 21, 22, 30]:
            bnd_seg = raw_data.slice(idx + 2, nvell).select(
                pl.lit(i).alias("nvel_idx"),
                pl.lit(ibtype).alias("ibtype"),
                pl.col("column_1").alias("JN").cast(pl.Int64),
            )
        elif ibtype in [3, 13, 23]:
            bnd_seg = raw_data.slice(idx + 2, nvell).select(
                pl.lit(i).alias("nvel_idx"),
                pl.lit(ibtype).alias("ibtype"),
                pl.col("column_1").alias("JN").cast(pl.Int64),
                pl.col("column_2").alias("barlanht").cast(pl.Float64),
                pl.col("column_3").alias("barlancfsp").cast(pl.Float64),
                pl.lit(int(vals[0][2])).alias("barlanht"),
                pl.lit(int(vals[0][3])).alias("barlancfsp"),
            )
        elif ibtype in [4, 24]:
            bnd_seg = raw_data.slice(idx + 2, nvell).select(
                pl.lit(i).alias("nvel_idx"),
                pl.lit(ibtype).alias("ibtype"),
                pl.col("column_1").alias("JN").cast(pl.Int64),
                pl.col("column_2").alias("IBCONN").cast(pl.Int64),
                pl.col("column_3").alias("barinht").cast(pl.Float64),
                pl.col("column_4").alias("barincfsb").cast(pl.Float64),
                pl.col("column_5").alias("barincfsp").cast(pl.Float64),
            )
        elif ibtype in [5, 25]:
            bnd_seg = raw_data.slice(idx + 2, nvell).select(
                pl.lit(i).alias("nvel_idx"),
                pl.lit(ibtype).alias("ibtype"),
                pl.col("column_1").alias("JN").cast(pl.Int64),
                pl.col("column_2").alias("IBCONN").cast(pl.Int64),
                pl.col("column_3").alias("barinht").cast(pl.Float64),
                pl.col("column_4").alias("barincfsb").cast(pl.Float64),
                pl.col("column_5").alias("barincfsp").cast(pl.Float64),
                pl.col("column_6").alias("pipeht").cast(pl.Float64),
                pl.col("column_7").alias("pipecoef").cast(pl.Float64),
                pl.col("column_8").alias("pipediam").cast(pl.Float64),
            )
        logger.info(
            f"Read in open boundary segment {i} of length {nvell} with type {ibtype}"
        )

        flow_bnds.append(bnd_seg)
        idx = idx + nvell + 1

    open_bnds = pl.concat(open_bnds)
    flow_bnds = pl.concat(flow_bnds, how="diagonal")

    return open_bnds, flow_bnds


def load_f14(
    f14_file: str = "fort.14",
    crs: str = "EPSG:4326",
):
    """
    Reads node map from fort.14.

    Parameters
    ----------
    f14_file : str, default = 'fort.14'
        Path to fort.14 file.
    params : dict, optional
        Dataset containing already loaded in f14 parameters. If None, load in the parameters first from the fort.14 file.
    comment_char : str, default = '='
        Character to use as a comment character in the fort.14 file.
    engine : str, default = 'polars'
        Engine to use for reading the file ('polars' or 'pandas').

    Returns
    -------
    node_map : pd.DataFrame or pl.DataFrame
        DataFrame indexed by node number, and with columns 'X', 'Y', and 'DP', with 'X' and 'Y' being the x and y coordinates of the node, and 'DP' being the bathymetric depth at the node.
    """
    logger.info(f"Getting AGRID, NP, NE for fort.14 file at {f14_file}...")
    params = get_base_params(f14_file)
    logger.info(f"Mesh params: {params}")

    logger.info(f"Building raw data query...")
    raw_data_q = get_raw_data_q(f14_file)

    logger.info(f"Validating fort.14 file...")
    if np.isnan(raw_data_q.select(pl.col("column_1")).fetch(1).to_numpy()[0][0]):
        msg = "Multiple spaces detected. Cleaning file first using clean_file()"
        logger.error(msg)
        raise ValueError(msg)

    logger.info(f"raw_data query: {raw_data_q}\nLoading...")
    raw_data = raw_data_q.collect()
    logger.info("Done loading raw data: {raw_data}")

    logger.info(f'Loading node map with {params["NP"]} nodes...')
    node_map_q = get_node_map_q(
        params=params,
        raw_data=raw_data,
        crs=crs,
    )
    logger.info(f"Node map query: {node_map_q}\nLoading...")
    node_map = node_map_q.collect()
    logger.info(f"Done loading node map: {node_map}. Converting to GeoDataFrame...")
    node_gdf = build_node_gdf(node_map, crs)
    logger.info(f"Done converting node map to GeoDataFrame: {node_gdf}")

    logger.info(f'Building query for map with {params["NE"]} element...')
    element_map_q = get_element_map_q(
        raw_data=raw_data,
        node_map=node_map,
        params=params,
        crs=crs,
    )
    logger.info(f"Element map query: {element_map_q}\nLoading...")
    element_map = element_map_q.collect()
    logger.info(f"Done loading element map: {element_map}")

    logger.info("Parsing boundary information...")
    open_bnds, flow_bnds = get_boundaries(params, raw_data)
    logger.info(
        f"Boundary data read:\nOpen Boundaries:{open_bnds}"
        + f"\nNormal Flow Boundaries:\n{flow_bnds}"
    )

    f14 = {
        "params": params,
        "gdf": {
            "node_gdf": node_gdf,
            "element_gdf": element_gdf,
            "open_bnds_gdf": open_bnds_gdf,
            "flow_bnds_gdf": flow_bnds_gdf,
        },
        "polars": {
            "queries": {
                "raw_data_q": raw_data_q,
                "node_map_q": node_map_q,
                "element_map_q": element_map_q,
            },
            "dfs": {
                "node_map": node_map,
                "element_map": element_map,
                "open_bnds": open_bnds,
                "flow_bnds": flow_bnds,
            },
        },
    }

    return f14


def build_node_gdf(node_map, crs: Optional[str] = "EPSG:4326") -> GeoDataFrame:
    """
    Build node GeoDataFrame from node map.

    Parameters:
        node_map: DataFrame containing node information.
        crs (str, optional): Coordinate Reference System. Defaults to "EPSG:4326".

    Returns:
        GeoDataFrame: GeoDataFrame containing node geometries.
    """

    # Extract X and Y columns to Pandas DataFrame for compatibility with gpd.points_from_xy
    x_col = node_map.select(pl.col("X")).to_pandas().X
    y_col = node_map.select(pl.col("Y")).to_pandas().Y

    # Create GeoDataFrame
    node_gdf = GeoDataFrame(
        node_map.to_pandas(),  # Convert to Pandas DataFrame
        geometry=gpd.points_from_xy(x_col, y_col),
    )

    # Set CRS
    node_gdf.set_crs(crs, inplace=True)

    # Add 'Depth' field if 'DP' exists in node_map
    if "DP" in node_map.columns:
        node_gdf["Depth"] = node_map.select(pl.col("DP")).to_pandas().DP

    return node_gdf


def build_element_gdf(element_map, crs: Optional[str] = "EPSG:4326") -> GeoDataFrame:
    """
    Convert polars dataframe to element GeoDataFrame with geometries and appropriate CRS.

    Parameters:
        element_map: DataFrame containing element information.
        crs (str, optional): Coordinate Reference System. Defaults to "EPSG:4326".

    Returns:
        GeoDataFrame: GeoDataFrame containing element geometries.
    """

    element_gdf = element_map.to_pandas()
    element_gdf["geometry"] = element_gdf.apply(
        lambda row: Polygon(
            [
                (row["X_1"], row["Y_1"]),
                (row["X_2"], row["Y_2"]),
                (row["X_3"], row["Y_3"]),
            ]
        ),
        axis=1,
    )

    element_gdf = gpd.GeoDataFrame(element_gdf, geometry="geometry")

    # Set CRS
    element_gdf.set_crs(crs, inplace=True)

    element_gdf["centroid"] = element_gdf["geometry"].centroid

    # Add point geometry for each node
    for i in range(1, 4):
        col_name = f"n_{i}"
        element_gdf[col_name] = element_gdf.apply(
            lambda row: Point(row[f"X_{i}"], row[f"Y_{i}"]), axis=1
        )

    # Add line geometry for each edge
    for i, j in [(1, 2), (2, 3), (3, 1)]:
        col_name = f"e_{i}{j}"
        element_gdf[col_name] = element_gdf.apply(
            lambda row: LineString(
                [(row[f"X_{i}"], row[f"Y_{i}"]), (row[f"X_{j}"], row[f"Y_{j}"])]
            ),
            axis=1,
        )

    # Calculate lengths for each edge
    for i, j in [(1, 2), (2, 3), (3, 1)]:
        col_name = f"l_{i}{j}"
        element_gdf[col_name] = element_gdf.apply(
            lambda row: row[f"e_{i}{j}"].length, axis=1
        )

    return element_gdf


def read_fort14_params(f14_file: str = "fort.14"):
    """
    Reads only in parameters and no grind/boundary information from fort.14 file.

    These pareamters include:
        - AGRID: alpha-numeric grid identification (<=24 characters).
        - NE: number of elements in horizontal grid
        - NP: number of nodes in horizontal grid
        - NOPE: number of elevation specified boundary forcing segments
        - NETA: total number of elevation specified boundary nodes
        - NBOU: number of normal flow (discharge) specified boundary segments
        - NVEL: total number of normal flow specified boundary nodes

    Read in assuming the following format:

    | Line # | Value |
    | --- | ----- |
    | 1 | AGRID  |
    | 2 | NE, NP  |
    | 3 + NE + NP | NOPE |
    | 4 + NE + NP | NETA |
    | 5 + NE + NP + (NETA + NOPE)| NBOU |
    | 6 + NE + NP + (NETA + NOPE)| NVEL |

    Parameters
    ----------
    f14_file : str
        Path to fort.14 file

    Returns
    -------
    params : dict
        Dictionary containing the fort.14 parameters
    """
    params = {}  # Initialize empty dictionary to store parameters

    # Directly access the required lines using linecache
    # This avoids having to read through the file line-by-line
    params["AGRID"] = lc.getline(f14_file, 1).strip()
    params["NE"], params["NP"] = map(int, lc.getline(f14_file, 2).split()[:2])
    params["NOPE"] = int(
        lc.getline(f14_file, 3 + params["NE"] + params["NP"]).split()[0]
    )
    params["NETA"] = int(
        lc.getline(f14_file, 4 + params["NE"] + params["NP"]).split()[0]
    )
    params["NBOU"] = int(
        lc.getline(
            f14_file, 5 + params["NE"] + params["NP"] + params["NETA"] + params["NOPE"]
        ).split()[0]
    )
    params["NVEL"] = int(
        lc.getline(
            f14_file, 6 + params["NE"] + params["NP"] + params["NETA"] + params["NOPE"]
        ).split()[0]
    )

    return params


def _build_edge_list(element_map):
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

    nx.from_pandas_edgelist(
        df,
        source="source",
        target="target",
        edge_attr=None,
        create_using=None,
        edge_key=None,
    )

    return edge_list_df
