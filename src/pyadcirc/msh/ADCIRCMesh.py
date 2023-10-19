import asyncio
import pdb
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

import geopandas as gpd
import numpy as np
import pandas as pd  # Replace with your DataFrame library if not using pandas
import polars as pl
from shapely import Point, Polygon

from pyadcirc.io.read_fort14 import clean_f14, load_f14
from pyadcirc.log import logger
from pyadcirc.utils import make_rich_table


class ADCIRCMesh:
    """
    ADCIRC Mesh Class

    Defines an ADCIRC grid which consists of node map, element map, and boundary information.

    Attributes
    ----------
    node_map : pd.DataFrame
        DataFrame indexed by node number, and with columns 'X', 'Y', and 'DP', with 'X'
        and 'Y' being the x and y coordinates of the node, and 'DP' being the bathymetric
        depth at the node.
    element_map : pd.DataFrame
        DataFrame indexed by element number, and with columns 'NHY', 'NM_1', 'NM_2', and
        'NM_3', with 'NHY' being the number of nodes in the element (always 3 for
        triangular elements), and 'NM_1', 'NM_2', and 'NM_3' being the node numbers of the
        element.
    elev_boundary : pd.DataFrame
        DataFrame indexed by boundary segment, and with columns 'JN' and 'IBTYPEE', with
        'IBTYPEE' being the boundary type for the elevation specified boundary segment
        (always 0 for elevation boundaries) and 'JN' being the node number of the boundary
        node for the elevation specified boundary segment.
    flow_boundary : pd.DataFrame
        DataFrame indexed by boundary segment, and with columns 'JN' and 'IBTYPE', with
        'IBTYPE' being the boundary type for the normal flow specified boundary segment
        (always 20 or 21 for normal flow boundaries) and 'JN' being the node number of the
        boundary node for the normal flow specified boundary segment.
    """

    def __init__(
        self, f14_file: str = "fort.14", crs: str = "EPSG:4326", clean: bool = True
    ) -> None:
        self._f14 = f14_file
        self.crs = "EPSG:4326"

        # Futures for loading data asynchrounously - Execute queries in thread pool
        self._f = {}
        self._executor = ThreadPoolExecutor(max_workers=1)

        # if np.isnan(
        #     self._q['raw_data'].select(
        #         pl.col("column_1")).fetch(1).to_numpy()[0][0]):
        #     logger.warning(
        #         ''.join(
        #             ['Parsed NaN for first value -',
        #              ' f14 file should be strictly single space deliminated']))
        #     if clean:
        #         logger.info('Auto-clean seat. Cleaning f14 file')
        #         self._f['clean_f14'] = self._executor.submit(clean_f14, self._f14)

        self._f["data"] = self._executor.submit(load_f14, self._f14)

    def _print_executor_info(self, pf: bool = True):
        ex_str = "\n".join(
            [
                "\nExecutor Info:",
                f"  Max workers: {self._executor._max_workers}",
                f"  Current tasks in queue: {self._executor._work_queue.qsize()}",
                f"  Number of active threads: {len(self._executor._threads)}\n",
            ]
        )
        if pf:
            print(ex_str)
        else:
            return ex_str

    @property
    def params(self) -> dict:
        if self._f["data"].done():
            return self._f["data"].result()["params"]

    @property
    def node_map(self) -> pd.DataFrame:
        if self._f["data"].done():
            return self._f["data"].result()["node_map"]
        return None

    @property
    def element_map(self) -> pd.DataFrame:
        if self._f["data"].done():
            return self._f["data"].result()["element_map"]
        return None

    @property
    def elev_boundary(self) -> pd.DataFrame:
        if self._f["data"].done():
            return self._f["data"].result()["open_bnds"]
        return None

    @property
    def flow_boundary(self) -> pd.DataFrame:
        if self._f["data"].done():
            return self._load_future.result()["flow_bnds"]
        return None

    def get_data(self) -> None:
        """
        Waits for load to finish and returns mesh data
        """
        # This will block until result is ready
        res = self._f["data"].result()
        return res

    def get_node_gdf(self) -> gpd.GeoDataFrame:
        _ = self.get_data()

        return node_gdf

    def get_element_gdf(self) -> gpd.GeoDataFrame:
        _ = self.get_data()

        element_gdf = self.element_map.to_pandas()
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
        element_gdf["centroid"] = element_gdf["geometry"].centroid

        for i in range(1, 4):
            col_name = f"n_{i}"
            element_gdf[col_name] = element_gdf.apply(
                lambda row: Point(row[f"X_{i}"], row[f"Y_{i}"]), axis=1
            )

        return element_gdf
