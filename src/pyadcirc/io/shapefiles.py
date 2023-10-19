import os
import pdb
import shutil
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from geopandas import GeoDataFrame
from shapely.geometry import LineString, Point, Polygon

from pyadcirc.log import logger


def set_crs_for_gdf(
    gdf: GeoDataFrame, crs_epsg: int = 4326, crs: str = None
) -> GeoDataFrame:
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
    if crs is None:
        if gdf.crs is None:
            gdf.crs = f"EPSG:{crs_epsg}"
        else:
            gdf = gdf.to_crs(epsg=crs_epsg)
    else:
        gdf.crs = crs

    return gdf


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
            logger.warning(
                f"Multiple .shp.xml files found. Using the first one: {xml_files[0]}"
            )

        if xml_files:
            xml_file = xml_files[0]
        else:
            logger.debug("No .shp.xml file found in directory.")
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
        logger.warning("Metadata file not found.")

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

    logger.info(f"XML metadata written to {output_path}")


def load_shapefile(
    path: str = None,
    gdf: Optional[gpd.GeoDataFrame] = None,
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

    return gdf


def write_shapefile(
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


def plot_shapefile(
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
        The filename for the saved Folium map. Default is 'map.html'. NOT IMPLEMENTED
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
        gdf = load_shapefile(path, metadata=metadata, crs_epsg=crs_epsg)

    gdf = set_crs_for_gdf(gdf, crs_epsg)

    # Plot with GeoPandas
    if plot:
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        gdf.plot(**plot_kwargs)
        if plot_filename is not None:
            plt.savefig(plot_filename)

    # if map_filename is not None:
    #     plot_folium_map(gdf, crs_epsg, map_filename)

    return gdf


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


def get_combined_bounding_box(
    geodf_list: List[gpd.GeoDataFrame],
) -> Tuple[float, float, float, float]:
    """
    Calculate the bounding box containing all geometry objects in a list of GeoPandas dataframes.

    Parameters
    ----------
    geodf_list : List[gpd.GeoDataFrame]
        List of GeoPandas dataframes to be considered for the bounding box.

    Returns
    -------
    Tuple[float, float, float, float]
        The bounding box containing all geometry objects in the format (minx, miny, maxx, maxy).

    Example
    -------
    >>> df1 = gpd.read_file("df1.geojson")
    >>> df2 = gpd.read_file("df2.geojson")
    >>> get_combined_bounding_box([df1, df2])
    (minx, miny, maxx, maxy)
    """
    # Initialize variables to store min and max coordinates for x and y
    minx, miny, maxx, maxy = float("inf"), float("inf"), float("-inf"), float("-inf")

    for geodf in geodf_list:
        # Get the bounding box for each dataframe
        current_minx, current_miny, current_maxx, current_maxy = geodf.total_bounds

        # Update the global min and max coordinates
        minx = min(minx, current_minx)
        miny = min(miny, current_miny)
        maxx = max(maxx, current_maxx)
        maxy = max(maxy, current_maxy)

    return ((minx, miny), (maxx, maxy))


def get_bbox_poly(
    bbox, crs: str = "EPSG:4326", factor: float = 1.0
) -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame containing a polygon representing the bounding box.

    Parameters
    ----------
    bbox:
        The bounding box as to make a polygon box from in one of the forms:
        bbox: Tuple[Tuple[float, float], Tuple[float, float]], optional
            The bounding box as ((minx, miny), (maxx, maxy)). If not provided, it will be calculated based on the other parameters.
        gdf : geopandas.GeoDataFrame, optional
            A GeoDataFrame containing the data.
        shape_path : str, optional
            The path to the .shp file.
        search_str: str, optional
            String of name to search open street map for and create bounding box around polygon containing feature.
        center, size : Tuple[float, float], optional
            Tupole of the (x, y) coordinates of the center point of the bounding box, and the size of the bounding box.
    crs : str, default 'EPSG:4326'
        The coordinate reference system of the resulting GeoDataFrame.
    factor : float, default 1.0
        The scaling factor to enlarge or shrink the bounding box. For example, a factor of 1.5 will make the bounding box 1.5 times larger.

    Returns
    -------
    bounding_box_gdf : geopandas.GeoDataFrame
        A GeoDataFrame containing the bounding box polygon square over region.

    Raises
    ------
    ValueError
        If the required parameters are not provided.
    """
    if type(bbox) == gpd.GeoDataFrame:
        minx, miny, maxx, maxy = bbox.total_bounds
    elif type(bbox) == Path:
        if not bbox.exists():
            raise ValueError(f"Invalid path for bbox shapefile: {bbox}")
        minx, miny, maxx, maxy = load_shapefile(
            dir=str(bbox), crs_epsg=crs
        ).total_bounds
    elif type(bbox) == str:
        if (bbox_path := Path(bbox).resolve()).exists():
            minx, miny, maxx, maxy = load_shapefile(
                dir=bbox_path, crs_epsg=crs
            ).total_bounds
        else:
            # Get the polygon geometries for the area
            minx, miny, maxx, maxy = ox.geocode_to_gdf(bbox).total_bounds
    elif type(bbox) == tuple:
        vm = "".join(
            [
                "Valid bbox tuples are [[x_min, y_min],",
                "[x_max, y_max]] or [[center_x, center_y], size]",
                f"bbox: {bbox}",
            ]
        )
        if len(bbox) != 2:
            raise ValueError(f"Invalid bbox tuple - {vm}")
        if len(bbox[0]) != 2:
            raise ValueError(f"Invalid bbox tuple - {vm}")
        if len(bbox[1]) == 1:
            center, size = bbox
            center_x, center_y = center
            bounding_box = (
                (center[0] - size / 2, center[1] - size / 2),
                (center[0] + size / 2, center[1] + size / 2),
            )
        minx, miny = bounding_box[0]
        maxx, maxy = bounding_box[1]
    else:
        raise ValueError(f"Invalid bbox type - {type(bbox)}")

    # Expand bounding box according to factor
    center_x = (maxx + minx) / 2
    center_y = (maxy + miny) / 2
    width = maxx - minx
    height = maxy - miny
    new_width = width * factor
    new_height = height * factor
    new_minx = center_x - (new_width / 2)
    new_miny = center_y - (new_height / 2)
    new_maxx = center_x + (new_width / 2)
    new_maxy = center_y + (new_height / 2)

    bounding_box_polygon = Polygon(
        [
            (new_minx, new_miny),
            (new_minx, new_maxy),
            (new_maxx, new_maxy),
            (new_maxx, new_miny),
        ]
    )

    # Create a GeoDataFrame from the bounding box polygon
    bounding_box_gdf = gpd.GeoDataFrame(geometry=[bounding_box_polygon], crs=crs)

    return bounding_box_gdf


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


def polygon_to_linestring(polygon_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a polygon object to a linestring object.

    Parameters
    ----------
    polygon_obj : Dict[str, Any]
        A dictionary containing a polygon's shapefile information.
        Fields should include 'X' and 'Y' for coordinates.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing a linestring's shapefile information.
        Fields will include 'X' and 'Y' for coordinates.

    Examples
    --------
    >>> polygon_obj = {'Geometry': 'Polygon', 'X': [-94.7154, -94.7160, ...], 'Y': [29.3923, 29.3928, ...]}
    >>> linestring_obj = polygon_to_linestring(polygon_obj)
    """

    # Validate the input
    if polygon_obj.get("Geometry") != "Polygon":
        raise ValueError("Input object must have Geometry set to 'Polygon'.")

    # Create the linestring object
    linestring_obj = {}
    linestring_obj["Geometry"] = "Line"

    # Extract the X and Y coordinates, breaking the loop at an arbitrary point
    # (Here, using all points except the last)
    linestring_obj["X"] = polygon_obj["X"][:-1]
    linestring_obj["Y"] = polygon_obj["Y"][:-1]

    return linestring_obj
