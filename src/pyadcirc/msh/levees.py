import pdb
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from alive_progress import alive_bar
from geopandas import GeoDataFrame
from networkx import Graph
from shapely.geometry import LineString, Point, Polygon

from pyadcirc.io.netcdf import load_nc_qgis_mesh, write_nc_qgis_mesh
from pyadcirc.io.shapefiles import (get_combined_bounding_box, load_shapefile,
                                    set_crs_for_gdf, write_shapefile)
from pyadcirc.log import logger
from pyadcirc.msh.ADCIRCMesh import ADCIRCMesh
from pyadcirc.msh.graph import (build_graph, dijkstra_shortest_path,
                                find_closest_element_min_length,
                                greedy_shortest_path, score_path)
from pyadcirc.msh.utils import (load_element_gdf, save_element_gdf,
                                trim_f14_grid)


def convert_to_levee(df: gpd.GeoDataFrame, name: str) -> gpd.GeoDataFrame:
    """
    Reads a GeoDataFrame containing various geometry types and converts them to levees.

    Parameters
    ----------
    df : gpd.GeoDataFrame
        The input GeoDataFrame containing geometry objects.

    name : str
        The name to be assigned to the levees.

    Returns
    -------
    gpd.GeoDataFrame
        A new GeoDataFrame containing levee features with the columns:
        ['Geometry', 'BoundingBox', 'X', 'Y', 'Name', 'height'].

    """
    result = []
    orig_crs = df.crs

    for _, row in df.iterrows():
        geometry = row["geometry"]
        geom_type = geometry.geom_type

        if geom_type == "LineString":
            coords = list(geometry.coords)
            x, y = zip(*coords)
            result.append(
                {
                    "geometry": geometry,
                    "BoundingBox": geometry.bounds,
                    "X": x,
                    "Y": y,
                    "Name": name,
                    "height": None,
                }
            )

        elif geom_type == "LineStringZ":
            coords = list(geometry.coords)
            groups = {}
            for x, y, z in coords:
                if z not in groups:
                    groups[z] = []
                groups[z].append((x, y))
            for z, xy_coords in groups.items():
                linestring = LineString(xy_coords)
                x, y = zip(*xy_coords)
                result.append(
                    {
                        "geometry": linestring,
                        "BoundingBox": linestring.bounds,
                        "X": x,
                        "Y": y,
                        "Name": name,
                        "height": np.abs(z),
                    }
                )

        elif geom_type == "Polygon":
            coords = list(geometry.exterior.coords)[:-1]
            linestring = LineString(coords)
            x, y = zip(*coords)
            result.append(
                {
                    "geometry": linestring,
                    "BoundingBox": linestring.bounds,
                    "X": x,
                    "Y": y,
                    "Name": name,
                    "height": None,
                }
            )

        elif geom_type == "PolygonZ":
            coords = list(geometry.exterior.coords)
            groups = {}
            for x, y, z in coords:
                if z not in groups:
                    groups[z] = []
                groups[z].append((x, y))
            for z, xy_coords in groups.items():
                linestring = LineString(xy_coords)
                x, y = zip(*xy_coords)
                result.append(
                    {
                        "geometry": linestring,
                        "BoundingBox": linestring.bounds,
                        "X": x,
                        "Y": y,
                        "Name": name,
                        "height": np.abs(z),
                    }
                )

    result_df = gpd.GeoDataFrame(
        result, columns=["geometry", "BoundingBox", "X", "Y", "Name", "height"]
    )
    result_df = result_df.set_geometry("geometry")
    result_df = set_crs_for_gdf(result_df, crs=orig_crs)

    return result_df


def check_filter_levee_segments(
    levee: gpd.GeoDataFrame, snapped_levee: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Filters a GeoPandas DataFrame to return rows containing LineString geometries
    with fewer than 2 points. Also checks for LineStrings with mismatched .X and .Y
    coordinates and raises an error if found.

    Parameters
    ----------
    geodf : gpd.GeoDataFrame
        GeoPandas DataFrame containing LineString geometries to be filtered.

    Returns
    -------
    gpd.GeoDataFrame
        Filtered GeoPandas DataFrame containing only rows with LineStrings having fewer
        than 2 points.

    Raises
    ------
    ValueError
        If any LineString has mismatched lengths for .X and .Y coordinates.

    Example
    -------
    >>> geodf = gpd.read_file('lines.geojson')
    >>> filter_linestrings(geodf)
    (Filtered GeoDataFrame)
    """
    valid_idxs = []
    for idx, row in levee.iterrows():
        geometry = row["geometry"]

        if isinstance(geometry, LineString):
            coords = list(geometry.coords)
            x_values = [coord[0] for coord in coords]
            y_values = [coord[1] for coord in coords]

            if len(x_values) != len(y_values):
                raise ValueError(
                    f"LineString at index {idx} has mismatched .X and .Y coordinate lengths."
                )

            if len(x_values) < 2:
                print(
                    "Warning: LineString at index {idx} has fewer than 2 points -> Below mesh resolution. Removing"
                )
            else:
                valid_idxs.append(idx)
        else:
            raise ValueError(f"LineString at index {idx} is not a LineString.")

    levee = levee.iloc[valid_idxs]
    snapped_levee = snapped_levee.iloc[valid_idxs]

    return levee, snapped_levee


def load_or_make_mesh_cutout(msh, levee, levee_dir, name, overwrite=False, fmt="nc"):
    """
    Cutout msh around levee feature
    """

    # Create bounding box around all levees, and trim mesh to that box

    levee_dir = Path(levee_dir)

    if fmt == "csv":
        mesh_cutout_path = levee_dir / f"{name}-mesh-cutout.csv"
    else:
        mesh_cutout_path = levee_dir / f"{name}-mesh-cutout.nc"

    if not mesh_cutout_path.exists() or overwrite:
        bbox = get_combined_bounding_box(
            [levee] if isinstance(levee, GeoDataFrame) else levee
        )
        logger.info(f"Trimming mesh at {msh._f14} to bounding box {bbox}")
        element_map = trim_f14_grid(
            msh=msh,
            bounding_box=bbox,
            within_box=False,
            neighbors=5,
        )
        logger.info(f"Saving trimmed mesh to {mesh_cutout_path}")
        if fmt == "csv":
            save_element_gdf(element_map, str(mesh_cutout_path))
        else:
            write_nc_qgis_mesh(
                element_map,
                output_path=str(mesh_cutout_path),
                data_col="DP",
            )
    else:
        logger.info(f"Loading trimmed mesh from {mesh_cutout_path}")
        if fmt == "csv":
            element_map = load_element_gdf(str(mesh_cutout_path))
        else:
            element_map = load_nc_qgis_mesh(str(nc_file))

    return msh, element_map


def compile_levees(
    configs,
    shapefile_dir="levees",
    msh: ADCIRCMesh = None,
    f14_path="fort.14",
    output_dir="scenario",
    algorithm="dijkstra",
    overwrite=False,
    crs_epsg=4326,
):
    """
    Given a dictioanry of levee configurations, a directory containing them,
    loads the shapefiles for each levee, sets the appropriate heights in the
    shapefile linestrings for each levee, and saves them all to the same shapefile,
    with a name column indicating the name of each feature.

    This final levee file can be used to add a group of levees at once using
    the MATLAB OceanMesh2D library.
    """
    out_dir = Path(output_dir).resolve()
    if out_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory {output_dir} already exists and overwrite not set."
        )
    else:
        out_dir.mkdir(exist_ok=True)
    output_dir = str(output_dir)
    scenario_name = out_dir.name

    logger.info(
        f"Configuring levee scenario {scenario_name}\n",
        f"output_dir:f{output_dir}\nlevee_configs:{configs}",
    )

    # Load each levee according to configuration dictionary
    levees = []
    if msh is None:
        msh = ADCIRCMesh(f14_path)
    msh_cutouts = []
    for i, (feature_name, height) in enumerate(configs.items()):
        if height > 0:
            logger.debug(f"Loading {feature_name}")
            feature_path = feature_name.replace(" ", "_").lower()
            levee_dir = str(Path(shapefile_dir) / f"{feature_path}")
            shape_path = (
                Path(shapefile_dir) / f"{feature_path}" / "simplified"
            ).resolve()

            if not shape_path.exists():
                raise FileNotFoundError(
                    f"Could not find shapefile for {feature_name} at {shape_path}"
                )

            levee = load_shapefile(str(shape_path))
            levee = convert_to_levee(levee, feature_name)
            levee.loc[:, "height"] = height

            logger.info(f"Loaded {feature_name} with height {height}:\n{levee}")

            msh, mesh_cutout = load_or_make_mesh_cutout(
                msh, levee, levee_dir, feature_path, overwrite
            )
            msh_cutouts.append(mesh_cutout)

            levees.append(levee)
        else:
            logger.info(f"Skipping {feature_name}. {height} <= 0")

    if len(levees) == 0:
        raise ValueError("No levees set. Base mesh valid")

    # Create bounding box around all levees, and trim mesh to that box
    _, _ = load_or_make_mesh_cutout(
        msh,
        levees,
        output_dir,
        f"{scenario_name}-all_levees",
        overwrite=overwrite,
        fmt="nc",
    )

    snapped_levees = []
    for i, (feature_name, height) in enumerate(configs.items()):
        # Build graph from trimmed mesh
        logger.info(f"Building graph from trimmed mesh around {feature_name}")
        nodes, graph = build_graph(msh_cutouts[i])
        pdb.set_trace()

        logger.info(f"Snapping levee for {feature_name} to mesh")
        _, snapped_levee, scores = snap_levee(levee, nodes, graph, algorithm=algorithm)
        logger.info(f"Levee snapped: {scores}")

        logger.info(f"Filtering segments below mesh resolution and validating")
        levee, snapped_levee = check_filter_levee_segments(levees[i], snapped_levee)
        logger.info(
            f"{feature_name}:\nlevee: {levee.shape}\n",
            f"snapped_levee: {snapped_levee.shape}",
        )
        logger.info(f"Succesfully snapped levee {feature_name} to mesh")
        levees[i] = levee
        snapped_levees.append(snapped_levee)

    # Merge shapefiles
    logger.info(f"Filtering segments below mesh resolution")
    levees = set_crs_for_gdf(
        gpd.GeoDataFrame(pd.concat(levees)), crs_epsg=crs_epsg
    ).set_geometry("geometry")
    snapped_levees = set_crs_for_gdf(
        gpd.GeoDataFrame(pd.concat(snapped_levees)), crs_epsg=crs_epsg
    )

    write_shapefile(levees, "levees", attrs=None, directory=output_dir)
    write_shapefile(snapped_levees, "levees-snapped", attrs=None, directory=output_dir)

    # Return the arrays
    return msh, levees, snapped_levees


def snap_levee(
    levee: Union[str, GeoDataFrame],
    nodes: Dict[int, Tuple[float, float]],
    graph: Graph,
    algorithm: str = "greedy",
) -> Tuple[List[LineString], List["Any"], List[float]]:
    """
    Approximates the boundary of a levee shape using the given element map.

    Parameters
    ----------
    levee : Union[str, GeoDataFrame]
        Path to the shapefile or a loaded GeoDataFrame.
    nodes : Dict[int, Tuple[float, float]]
        Dictionary mapping node indices to their coordinates.
    graph : Graph
        NetworkX graph object representing the mesh.
    algorithm : str, optional
        Algorithm to use for approximation ('greedy' or 'dijkstra'). Defaults to 'greedy'.

    Returns
    -------
    Tuple[List[LineString], List[Any], List[float]]
        1. List of true line segments in the levee shape.
        2. List of approximate line segments in the element map.
        3. List of scores for each approximation.
    """

    # Ensure the original GeoDataFrame has a geographic (latitude/longitude) CRS
    levee = levee.to_crs(epsg=4326)

    # Extract the coordinates of the levee shape's geometry
    coords = (
        levee.geometry.iloc[0].coords
        if levee.geometry.iloc[0].is_ring
        else levee.geometry.iloc[0].coords[:-1]
    )

    # Initialize the resulting path and score
    total_score = 0

    # Define the number of segments for the progress bar
    num_segments = len(coords) - 1

    # Use alive-progress bar to show progress
    with alive_bar(
        num_segments,
        title="Snapping levee to mesh",
        bar="smooth",
        force_tty=True,
    ) as bar:
        approx_paths = []
        true_paths = []
        scores = []

        current_start = coords[0]

        def dijk_alg(current_start, end):
            return dijkstra_shortest_path(nodes, graph, current_start, end)

        def greedy_alg(current_start, end):
            return greedy_shortest_path(graph, current_start, end)

        # Approximate the line segment using the selected algorithm
        if algorithm == "greedy":
            algo = greedy_alg
        elif algorithm == "dijkstra":
            algo = dijk_alg
        else:
            raise ValueError(
                "Invalid algorithm specified. Choose 'greedy' or 'dijkstra'."
            )

        for end in coords[1:]:
            bar.text(f"Going from {current_start} to {end}")

            approx_path = algo(current_start, end)

            # Handle cases
            if len(approx_path) != 0:
                approx_score = score_path(graph, current_start, end, approx_path)
                scores.append(approx_score)
                approx_paths.append(approx_path)
                true_paths.append(LineString([current_start, end]))
                current_start = end  # Move to the next point
            else:
                # Handle case where no path is found
                segment = LineString([current_start, end])
                min_length_closest_element = find_closest_element_min_length(
                    segment, nodes, graph
                )

                seg_len = segment.length
                if seg_len < min_length_closest_element:
                    logger.debug(
                        f"len({segment}) = {seg_len} < {min_length_closest_element} -> Skipping point {end}."
                    )
                else:
                    logger.error(
                        f"len({segment}) = {seg_len} >= {min_length_closest_element}!! No path found"
                    )

            # Update the progress bar with the current score
            bar.text(f"Current Score: {total_score:.2f}")
            bar()

    return true_paths, approx_paths, scores


# TODO: OLD Method remove when new one is working
def snap_levee_2(
    levee: Union[str, GeoDataFrame],
    nodes,
    graph,
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
    # Ensure the original GeoDataFrame has a geographic (latitude/longitude) CRS
    levee = levee.to_crs(epsg=4326)

    # Extract the coordinates of the shapefile's geometry
    coords = (
        levee.geometry.iloc[0].coords
        if levee.geometry.iloc[0].is_ring
        else levee.geometry.iloc[0].coords[:-1]
    )

    # Initialize the resulting path and score
    total_score = 0

    # Define the number of segments for the progress bar
    num_segments = len(coords) - 1

    # Iterate through the line segments in the shapefile with alive-progress bar
    with alive_bar(
        num_segments,
        title="Snapping levee to mesh",
        bar="smooth",
        force_tty=True,
    ) as bar:
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
                approx_path = greedy_shortest_path(graph, start, end)
            elif algorithm == "dijkstra":
                approx_path = dijkstra_shortest_path(nodes, graph, start, end)
            else:
                raise ValueError(
                    "Invalid algorithm specified. Choose 'greedy' or 'dijkstra'."
                )

            # Calculate the score for the approximation
            if len(approx_path) != 0:
                approx_score = score_path(graph, start, end, approx_path)

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


def create_chunked_linestring_geodataframes(
    flow_boundary: dict,
    nodes_df: pd.DataFrame,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> List[gpd.GeoDataFrame]:
    """
    Create a list of GeoDataFrames for segments within a given bounding box (if specified),
    where each GeoDataFrame contains multiple LineString geometries for each segment chunk.

    Parameters
    ----------
    flow_boundary : dict
        Dictionary containing 'segments' and 'nodes' DataFrames.
    nodes_df : pd.DataFrame
        DataFrame containing nodal locations.
    bbox : tuple, optional
        Bounding box in the format (min_lat, max_lat, min_lon, max_lon).

    Returns
    -------
    List[gpd.GeoDataFrame]
        List of GeoDataFrames containing multiple LineString geometries for each segment chunk.
    """
    segments = find_segments_within_bbox(flow_boundary, nodes_df, bbox)
    segments_filtered = segments[segments["IBTYPE"] == 24]

    chunked_geodataframes = []

    for idx, segment in segments_filtered.iterrows():
        start_idx = segment["start_idx"]
        end_idx = segment["end_idx"]

        segment_nodes = flow_boundary["nodes"].iloc[start_idx:end_idx]
        nbvv_nodes = segment_nodes[segment_nodes.columns[0]].tolist()
        ibconn_nodes = segment_nodes[segment_nodes.columns[1]].tolist()

        line_strings = []

        for nbvv, ibconn in zip(nbvv_nodes, ibconn_nodes):
            nbvv_coords = nodes_df[nodes_df["JN"] == nbvv].iloc[0]
            ibconn_coords = nodes_df[nodes_df["JN"] == ibconn].iloc[0]

            line = LineString(
                [
                    (nbvv_coords["X"], nbvv_coords["Y"]),
                    (ibconn_coords["X"], ibconn_coords["Y"]),
                ]
            )
            line_strings.append(line)

        gdf = gpd.GeoDataFrame(
            {
                "ID": [idx] * len(line_strings),
                "Type": ["Chunk"] * len(line_strings),
                "geometry": line_strings,
            }
        )
        chunked_geodataframes.append(gdf)

    return chunked_geodataframes


def find_segments_within_bbox(
    flow_boundary: dict, nodes_df: pd.DataFrame, bbox: tuple
) -> pd.DataFrame:
    """
    Find all boundary segments within a given bounding box.

    Parameters
    ----------
    flow_boundary : dict
        Dictionary containing 'segments' and 'nodes' DataFrames.
    segments : pd.DataFrame
        DataFrame containing boundary segment information.
    nodes_df : pd.DataFrame
        DataFrame containing nodal locations.
    bbox : tuple
        Bounding box in the format (min_lat, max_lat, min_lon, max_lon).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the boundary segments that have at least one node within the bounding box.
    """

    # Extract the 'segments' and 'nodes' DataFrames from the flow_boundary dictionary
    segments = flow_boundary["segments"]
    nodes = flow_boundary["nodes"]

    # Convert the bounding box to the same coordinate system as the nodes
    # Note: This step may require a coordinate transformation depending on your actual data
    min_lat, max_lat, min_lon, max_lon = bbox

    # Filter the nodes within the bounding box
    nodes_within_bbox = nodes_df[
        (nodes_df["X"].astype(float) >= min_lon)
        & (nodes_df["X"].astype(float) <= max_lon)
        & (nodes_df["Y"].astype(float) >= min_lat)
        & (nodes_df["Y"].astype(float) <= max_lat)
    ]

    # Find boundary segments containing at least one node within the bounding box
    segments_within_bbox = []
    for idx, row in segments.iterrows():
        segment_nodes = nodes.iloc[row["start_idx"] : row["end_idx"]]
        if any(segment_nodes[0].isin(nodes_within_bbox["JN"])):
            segments_within_bbox.append(row)

    return pd.DataFrame(segments_within_bbox)


def create_geodataframes(
    flow_boundary: dict,
    nodes_df: pd.DataFrame,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> List[gpd.GeoDataFrame]:
    """
    Create a list of GeoDataFrames for segments within a given bounding box (if specified).

    Parameters
    ----------
    flow_boundary : dict
        Dictionary containing 'segments' and 'nodes' DataFrames.
    nodes_df : pd.DataFrame
        DataFrame containing nodal locations.
    bbox : tuple, optional
        Bounding box in the format (min_lat, max_lat, min_lon, max_lon).

    Returns
    -------
    List[gpd.GeoDataFrame]
        List of GeoDataFrames merged by the boundary index.
    """
    segments = find_segments_within_bbox(flow_boundary, nodes_df, bbox)
    segments_filtered = segments[segments["IBTYPE"] == 24]

    geodataframes = []

    for idx, segment in segments_filtered.iterrows():
        start_idx = segment["start_idx"]
        end_idx = segment["end_idx"]

        segment_nodes = flow_boundary["nodes"].iloc[start_idx:end_idx]
        nbvv_nodes = segment_nodes[segment_nodes.columns[0]]
        ibconn_nodes = segment_nodes[segment_nodes.columns[1]]

        nbvv_coords = nodes_df[nodes_df["JN"].isin(nbvv_nodes)]
        ibconn_coords = nodes_df[nodes_df["JN"].isin(ibconn_nodes)]

        nbvv_line = LineString(
            list(zip(nbvv_coords["X"].astype(float), nbvv_coords["Y"].astype(float)))
        )
        ibconn_line = LineString(
            list(
                zip(ibconn_coords["X"].astype(float), ibconn_coords["Y"].astype(float))
            )
        )

        gdf = gpd.GeoDataFrame(
            {
                "ID": [idx, idx],
                "Type": ["NBVV", "IBCONN"],
                "geometry": [nbvv_line, ibconn_line],
            }
        )
        geodataframes.append(gdf)

    return geodataframes


def process_levee_boundaries(
    flow_boundary: dict,
    nodes_df: pd.DataFrame,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> List[gpd.GeoDataFrame]:
    """
    Create a list of GeoDataFrames for segments within a given bounding box (if specified),
    where each GeoDataFrame contains a single merged LineString for each boundary segment of ibtype=24, corresponding to levees.

    Parameters
    ----------
    flow_boundary : dict
        Dictionary containing 'segments' and 'nodes' DataFrames.
    nodes_df : pd.DataFrame
        DataFrame containing nodal locations.
    bbox : tuple, optional
        Bounding box in the format (min_lat, max_lat, min_lon, max_lon).

    Returns
    -------
    List[gpd.GeoDataFrame]
        List of GeoDataFrames containing a single merged LineString for each segment.
    """
    if bbox is not None:
        segments = find_segments_within_bbox(flow_boundary, nodes_df, bbox)
    segments_filtered = segments[segments["IBTYPE"] == 24]

    merged_geodataframes = []

    for idx, segment in segments_filtered.iterrows():
        start_idx = segment["start_idx"]
        end_idx = segment["end_idx"]

        segment_nodes = flow_boundary["nodes"].iloc[start_idx:end_idx]
        nbvv_nodes = segment_nodes[segment_nodes.columns[0]].tolist()
        ibconn_nodes = segment_nodes[segment_nodes.columns[1]].tolist()

        coords_chain = []

        for nbvv, ibconn in zip(nbvv_nodes, ibconn_nodes):
            nbvv_coords = nodes_df[nodes_df["JN"] == nbvv].iloc[0]
            ibconn_coords = nodes_df[nodes_df["JN"] == ibconn].iloc[0]

            coords_chain.append((nbvv_coords["X"], nbvv_coords["Y"]))
            coords_chain.append((ibconn_coords["X"], ibconn_coords["Y"]))

        merged_line = LineString(coords_chain)

        gdf = gpd.GeoDataFrame(
            {"ID": [idx], "Type": ["Merged"], "geometry": [merged_line]}
        )
        merged_geodataframes.append(gdf)

    return merged_geodataframes


# Folllowing functions are for extracting boundary segments from fort.14 files, in particular flow boundary segments (not elevation boundary)
# TODO: Use io.shapefile functioncalls for actual reading and writing and keep logic for extracting segments in this file


def save_point_geodataframes_to_shapefiles(
    geodataframes: List[gpd.GeoDataFrame], shapefile_dir: str
) -> None:
    """
    Save a list of GeoDataFrames to separate shapefiles: one for NBVV and one for IBCONN for each segment,
    where each shapefile contains a list of point shapes.

    Parameters
    ----------
    geodataframes : List[gpd.GeoDataFrame]
        List of GeoDataFrames to save.
    shapefile_dir : str
        Directory where the shapefiles will be saved.
    """
    for gdf in geodataframes:
        idx = gdf.iloc[0]["ID"]

        for t in ["NBVV", "IBCONN"]:
            line = gdf[gdf["Type"] == t].iloc[0]["geometry"]
            points = [Point(x, y) for x, y in line.coords]

            point_gdf = gpd.GeoDataFrame(
                {
                    "ID": [idx] * len(points),
                    "Type": [t] * len(points),
                    "geometry": points,
                }
            )

            point_gdf.to_file(f"{shapefile_dir}/{t}_Segment_{idx}_Points.shp")


def save_geodataframes_to_shapefiles(
    geodataframes: List[gpd.GeoDataFrame], shapefile_dir: str
) -> None:
    """
    Save a list of GeoDataFrames to separate shapefiles: one for NBVV and one for IBCONN for each segment.

    Parameters
    ----------
    geodataframes : List[gpd.GeoDataFrame]
        List of GeoDataFrames to save.
    shapefile_dir : str
        Directory where the shapefiles will be saved.
    """
    for gdf in geodataframes:
        idx = gdf.iloc[0]["ID"]
        gdf.to_file(f"{shapefile_dir}/Segment_{idx}.shp")
