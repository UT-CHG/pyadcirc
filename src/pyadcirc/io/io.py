"""
io.py - Utilities Reading/Writing local ADCIRC Files

"""
import glob
import linecache as lc
import logging
import os
import pdb
import re
from contextlib import contextmanager
from functools import reduce
from pathlib import Path
from time import perf_counter, sleep
from typing import List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import dask
from dask.diagnostics.progress import ProgressBar

from pyadcirc.utils import get_bbox, regrid
from pyadcirc import __version__

__author__ = "Carlos del-Castillo-Negrete"
__copyright__ = "Carlos del-Castillo-Negrete"
__license__ = "MIT"
_logger = logging.getLogger(__name__)

pd.options.display.float_format = "{:,.10f}".format
logger = logging.getLogger("adcirc_io")


dask.config.set ({"array.slicing.split_large_chunks": False})

@contextmanager
def timing(label: str):
    t0 = perf_counter()
    yield lambda: (label, t1 - t0)
    t1 = perf_counter()


def read_param_line(out, params, f, ln=1, dtypes=None):
    """
    Read a line containing parameters from a fort file and store them in a dictionary
    ADCIRC fort.* file parameter lines are the off the form:

    <param1> ... <paramN> ! <param1_name>,...,<paramN_name>

    Where the ! comes at the 80th column and the <paramN_name> are optional.

    Parameters
    ----------
    out : dict
        Dictionary to store the attributes.
    params : list of str
        Names of the attributes.
    f : file
        File to read from.
    ln : int, optional
        Line number to read from. If None, read from the next line in the file.
    dtypes : list of types, optional
        Data types to cast the parameter values. If None, use the default string values.

    Returns
    -------
    out : dictionary
        Dictioanry with parameter values from read in line in file.
    ln : int
        Line number of the next line to read.
    """
    ln = 1 if ln is None else ln
    line = lc.getline(f, ln)
    logger.info(",".join(params) + " : " + line)
    vals = [x for x in re.split("\\s+", line) if x != ""]
    for i in range(len(params)):
        try:
            if dtypes:
                out[params[i]] = dtypes[i](vals[i])
            else:
                out[params[i]] = vals[i]
        except ValueError:
            out[params[i]] = np.nan
        except IndexError:
            out[params[i]] = np.nan

    ln += 1
    return out, ln


def read_text_line(out, param, f, ln=1):
    """"
    Reads a line of text from a fort file and stores it in the out dataset as an attribute
    
    Parameters
    ----------
    out : dict
        Dictioanry to store the attribute
    param : str
        Name of the attribute
    f : file
        File to read from
    ln : int, optional
        Line number to read from. If None, read from the next line in the file
    
    Returns
    -------
    out : dict
        Dictionary with the new attribute
    ln : int
        Line number of the next line to read
    """
    ln = 1 if ln is None else ln
    line = lc.getline(f, ln).strip()
    logger.info(param + " : " + line)
    out[param] = line
    ln += 1
    return out, ln


def write_numeric_line(vals, f):
    """
    Write a numeric line to a fort file.

    Parameters
    ----------
    vals : List of numeric values
        Values to write to file in a single line, space separated.
    f : file
        File to write the values to.
        
    """
    line = " ".join([str(v) for v in vals])
    f.write(line + "\n")


def write_param_line(params, keys, f):
    """"
    Write a parameter line to a fort file.

    If the input dataset (`ds`) is of type `xr.Dataset`, it retrieves the attribute values corresponding to the provided `params` and writes them to the file. Otherwise, it assumes `ds` is an iterable and writes its values directly to the file.

    The parameter values are separated by spaces. If the resulting line is shorter than 80 characters, it pads the line with spaces to reach 80 characters and appends the parameter names (`params`) preceded by an exclamation point '!'. If the resulting line exceeds 80 characters, it logs a warning message.

    Parameters
    ----------
    ds : xr.Dataset or iterable
        Dataset or iterable containing the parameter values.
    params : list of str
        Names of the parameters.
    f : file
        File to write the parameter line to.

    Returns
    -------
    None
        This function writes the parameter line directly to the file.
    """
    if isinstance(params, dict):
        line = " ".join([str(params[k]) for k in keys])
    else:
        line = " ".join([str(x) for x in params])
    if len(line) < 80:
        line += ((80 - len(line)) * " ") + "! " + ",".join(keys)
    else:
        logger.warning("WARNING - fort config files shouldn't be wider than 80 cols!")
    logger.info("Writing param line for " + ",".join(keys) + " - " + line)
    f.write(line + "\n")


def write_text_line(ds : xr.Dataset,
                    param : str,
                    f : str,
                    comment_char : str = '!'):
    """"
    Write a text line to a file.
    
    If the input dataset (`ds`) is of type `xr.Dataset`, it retrieves the attribute value corresponding to the provided `param` and writes it to the file. Otherwise, it assumes `ds` is a string and writes it directly to the file.

    Parameters
    ----------
    ds : xr.Dataset or str
        Dataset or string containing the text line.
    param : str
        Name of the parameter.
    f : file
        File to write the text line to.
        
    Returns
    -------
    None
        This function writes the text line directly to the file.
    """
    if type(ds) == xr.Dataset:
        line = ds.attrs[param]
    else:
        line = ds
    logger.info("Writing text line for " + param + " - " + line)
    if len(line) < 80:
        if param != "":
            line += ((80 - len(line)) * " ") + f"{comment_char} {param}"
    else:
        logger.warning("WARNING - fort config files shouldn't be wider than 80 cols!")
    f.write(line + "\n")

def read_fort14_params(f14_file: str = 'fort.14'):
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

    Parameters
    ----------
    f14_file : str
        Path to fort.14 file
    
    Returns
    -------
    params : dict
        Dictionary containing the fort.14 parameters
    """
    # 1 : AGRID = alpha-numeric grid identification (<=24 characters).
    params, _ = read_text_line({}, "AGRID", f14_file, ln=1)

    # 2 : NE, NP = number of elements, nodes in horizontal grid
    params, _ = read_param_line(params, ["NE", "NP"], f14_file, ln=2, dtypes=2 * [int])

    # 3 + NE + NP : NOPE
    params, _ = read_param_line(
        params, ["NOPE"], f14_file,
        ln=3 + params['NE'] + params['NP'],
        dtypes=[int])

    # 4 + NE + NP : NETA
    params, _ = read_param_line(
        params, ["NETA"], f14_file,
        ln=4 + params['NE'] + params['NP'],
        dtypes=[int])

    # 5 + NE + NP + (NETA + NOPE) : NBOU
    params, _ = read_param_line(
        params, ["NBOU"], f14_file,
        ln=5 + params['NE'] + params['NP'] + \
            params['NETA'] + params['NOPE'],
        dtypes=[int])

    # 6 + NE + NP + (NETA + NOPE) : NVEL
    params, _ = read_param_line(
        params, ["NVEL"], f14_file,
        ln=6 + params['NE'] + params['NP'] + \
            params['NETA'] + params['NOPE'],
        dtypes=[int])
    
    return params


def read_fort14_node_map(f14_file : str = 'fort.14',
                         params : dict = None,
                         comment_char: str = '='):
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

    Returns
    -------
    node_map : pd.DataFrame
        DataFrame indexed by node number, and with columns 'X', 'Y', and 'DP', with 'X' and 'Y' being the x and y coordinates of the node, and 'DP' being the bathymetric depth at the node.
    """
    params = read_fort14_params(f14_file) if params is None else params

    node_map = pd.read_csv(
                f14_file,
                delim_whitespace=True,
                nrows=params["NP"],
                skiprows=2,
                header=None,
                names=["JN", "X", "Y", "DP"],
                comment=comment_char,
            )

    return node_map


def read_fort14_element_map(f14_file : str = 'fort.14',
                            params : dict = None,
                            comment_char: str = '='):
    """
    Reads in triangular element map from fort.14 as a pandas DataFrame.

    Parameters
    ----------
    f14_file : str, default = 'fort.14'
        Path to fort.14 file.
    params : dict, optional
        Dictioanry containing already loaded in f14 parameters. If None, load in the parameters first from the fort.14 file.
    comment_char : str, default = '='
        Character to use as a comment character in the fort.14 file.

    Returns
    -------
    element_map : pd.DataFrame
        DataFrame indexed by element number, and with columns 'NHY', 'NM_1', 'NM_2', and 'NM_3', with 'NHY' being the number of nodes in the element (always 3 for triangular elements), and 'NM_1', 'NM_2', and 'NM_3' being the node numbers of the element.
    """
    params = read_fort14_params(f14_file) if params is None else params

    element_map = pd.read_csv(
        f14_file,
        delim_whitespace=True,
        nrows=params['NE'],
        skiprows=2 + params['NP'],
        header=None,
        names=['JE', 'NHY', 'NM_1', 'NM_2', 'NM_3'], 
        comment=comment_char,
        )

    return element_map

def read_fort14_elev_boundary(f14_file : str = 'fort.14',
                              ds : xr.Dataset = None):
    """
    Reads boundary information from fort.14 for elevation specified boundaries.

    Parameters
    ----------
    f14_file : str, default = 'fort.14'
        Path to fort.14 file.
    ds : xr.Dataset, optional
        Dataset containing already loaded in f14 parameters. If None, load in the parameters first from the fort.14 file.

    
    Returns
    -------
    elev_boundary : pd.DataFrame
        DataFrame indexed by boundary segment, and with columns 'JN' and 'IBTYPEE', with 'IBTYPEE' being the boundary type for the elevation specified boundary segment (always 0 for elevation boundaries) and 'JN' being the node number of the boundary node for the elevation specified boundary segment.
    """
    ds = read_fort14_params(f14_file) if ds is None else ds
    start_line = 4 + ds.attrs['NE'] + ds.attrs['NP']

    elev_boundary = pd.read_csv(
        f14_file,
        delim_whitespace=True,
        header=None,
        skiprows=start_line,
        nrows=ds.attrs['NOPE'] + ds.attrs['NETA'],
        usecols=[0],
        names=['JN']
    )
    elev_boundary["IBTYPEE"] = 0
    elev_boundary["BOUNDARY"] = None

    # Select segments by those that ONLY have first two columns as non NA values
    
    # Get elevation sepcified boundary forcing segments
    start_idx = 0
    segments = ds.attrs['NOPE']*[None]
    for i in range(ds.attrs["NOPE"]):
        num_nodes = elev_boundary.loc[start_idx, 'JN']
        segments[i] = [i, num_nodes]
        start_idx += num_nodes + 1
    segments = pd.DataFrame(segments, columns=['BOUNDARY', 'num_nodes'])
    segments['counter'] = 1     # Extra line with ibytpe per segment
    segments['end_idx'] = segments['num_nodes'].cumsum()
    segments['start_idx'] = segments['end_idx'].shift(1).fillna(0).astype(int)
    segments['f14_start_idx'] = start_line + segments['start_idx'] + segments['counter'].cumsum() # 2 + segments['f14_end_idx'].shift(1).fillna(2 + start_line).astype(int)
    segments['f14_end_idx'] = segments['f14_start_idx'] + segments['num_nodes']
    segments['IBTYPEE'] = 0
    segments = segments[['BOUNDARY', 'IBTYPEE', 'start_idx', 'end_idx', 'f14_start_idx', 'f14_end_idx']].set_index('BOUNDARY')
    node_df = elev_boundary[~elev_boundary.index.isin(segments['f14_start_idx'] - start_line)][['JN']]

    return {'segments': segments, 'nodes': node_df}

def read_fort14_flow_boundary(f14_file : str = 'fort.14',
                              ds : xr.Dataset = None,
                              comment_char : str = '='):
    """
    Reads boundary information from fort.14 for normal flow specified boundaries.

    Parameters
    ----------
    f14_file : str, default = 'fort.14'
        Path to fort.14 file.
    ds : xr.Dataset, optional
        Dataset containing already loaded in f14 parameters. If None, load in the parameters first from the fort.14 file.

    
    Returns
    -------
    ds : xr.Dataset
        Dataset containing the element map information.
    """
    ds = read_fort14_params(f14_file) if ds is None else ds
    start_line = 6 + ds.attrs['NE'] + ds.attrs['NP'] + ds.attrs['NOPE'] + ds.attrs['NETA']

    max_cols = 7
    flow_boundary = pd.read_csv(
        f14_file,
        delim_whitespace=True,
        header=None,
        skiprows=start_line,
        nrows=ds.attrs['NBOU'] + ds.attrs['NVEL'],
        names=range(max_cols),
        comment='=',
    )
    flow_boundary[0] = flow_boundary[0].astype(int)
    flow_boundary["IBTYPEE"] = 0
    flow_boundary["BOUNDARY"] = None

    # Select segments by those that ONLY have first two columns as non NA values
    segments = flow_boundary[flow_boundary.iloc[:, :2].notna().all(axis=1)][[0, 1]].reset_index()
    segments['BOUNDARY'] = segments.index + 1
    segments['num_nodes'] = segments[0].astype(int)
    segments['counter'] = 1     # Extra line with ibytpe per segment
    segments['end_idx'] = segments['num_nodes'].cumsum()
    segments['start_idx'] = segments['end_idx'].shift(1).fillna(0).astype(int)
    segments['f14_start_idx'] = start_line + segments['start_idx'] + segments['counter'].cumsum() # 2 + segments['f14_end_idx'].shift(1).fillna(2 + start_line).astype(int)
    segments['f14_end_idx'] = segments['f14_start_idx'] + segments['num_nodes']
    segments['IBTYPE'] = segments[1].astype(int)
    segments = segments[['BOUNDARY', 'IBTYPE', 'start_idx', 'end_idx', 'f14_start_idx', 'f14_end_idx']].set_index('BOUNDARY')
    node_df = flow_boundary[~flow_boundary.iloc[:, :2].notna().all(axis=1)][range(max_cols)].rename(columns={0: 'JN'})

    return {'segments': segments, 'nodes': node_df}

class ADCIRCGrid(object):
    """
    ADCIRC Grid Class
    
    Defines an ADCIRC grid which consists of node map, element map, and boundary information. 

    Attributes
    ----------
    node_map : pd.DataFrame
        DataFrame indexed by node number, and with columns 'X', 'Y', and 'DP', with 'X' and 'Y' being the x and y coordinates of the node, and 'DP' being the bathymetric depth at the node.
    element_map : pd.DataFrame
        DataFrame indexed by element number, and with columns 'NHY', 'NM_1', 'NM_2', and 'NM_3', with 'NHY' being the number of nodes in the element (always 3 for triangular elements), and 'NM_1', 'NM_2', and 'NM_3' being the node numbers of the element.
    elev_boundary : pd.DataFrame
        DataFrame indexed by boundary segment, and with columns 'JN' and 'IBTYPEE', with 'IBTYPEE' being the boundary type for the elevation specified boundary segment (always 0 for elevation boundaries) and 'JN' being the node number of the boundary node for the elevation specified boundary segment.
    flow_boundary : pd.DataFrame
        DataFrame indexed by boundary segment, and with columns 'JN' and 'IBTYPE', with 'IBTYPE' being the boundary type for the normal flow specified boundary segment (always 20 or 21 for normal flow boundaries) and 'JN' being the node number of the boundary node for the normal flow specified boundary segment.
    """
    def __init__(self,
                 f14_file : str = 'fort.14',
                 load : bool = True):
       
        self.element_map = None 
        self.node_map = None
        self.elev_boundary = None
        self.flow_boundary = None
        if load:
            if not Path(f14_file).exists():
                raise FileNotFoundError(f"fort.14 file at {f14_file} not found")
            self.f14 = f14_file

    def load(self,
                 comment_char : str = '=',
                 load_grid : bool = True):
        """
        Reads in ADCIRC fort.14 file, which conatins grid and boundary information.
        Grid information includes the number of elements, nodes, node map, triangular element
        map, bathymetry at each node, and boundary conditions.
        
        See documentation at https://adcirc.org/home/documentation/users-manual-v53/input-file-descriptions/adcirc-grid-and-boundary-information-file-fort-14/. For
        more information on the fort.14 file format.

        Parameters
        ----------
        f14_file : str
            Path to fort.14 file
        """
        ds = ds if type(ds) == xr.Dataset else xr.Dataset()

        ds = read_fort14_params(f14_file) if ds is None else ds
        self.element_map = read_fort14_node_map(f14_file, ds, comment_char=comment_char)
        self.node_map = read_fort14_element_map(f14_file, ds, comment_char=comment_char)
        self.elev_boundary = read_fort14_elev_boundary(f14_file, ds)
        self.flow_boundary = read_fort14_flow_boundary(f14_file, ds)

class ADCIRCSimulation(object):
    """
    ADCIRC Simulation Class

    Base class with minimal parameters for a 2D ADCIRC run

    Attributes
    ----------
    f14 : str 
        Path to fort.14 (Grid and Boundary conditions) file. Checks if file exists upon setting.
    f15 : str 
        Path to fort.15 (Control and Tide Configurations) file. Checks if file exists upon setting.

    """
    def __init__(self,
                 dir_path : str = None,
                 f14_path : str = 'fort.14',
                 f15_path : str = 'fort.15'):
        dir_path = Path(dir_path) if dir_path is not None else Path.cwd()
        self.f14 = dir_path / f14_path
        self.f15 = dir_path / f15_path
        self.grid = ADCIRCGrid(self.f14)
        self.control_params = {}
    
    def _get_param(self, param_name, line, type='text'):
        """
        """
        if param_name not in self.control_params:
            self.control_params[param_name] = read_text_line(
                self.control_params, param_name, self.f15, ln=line)
        return self.control_params[param_name]


    @property
    def f14(self):
        return self.source_files['14']

    @f14.setter
    def f14(self, f14_path):
        if not Path(f14_path).exists():
            raise FileNotFoundError(f"ADCIRC fort.14 (Grid and Boundary conditions) file at {str(f14_path)} not found")
        self.f14 = f14_path

    @property
    def f15(self):
        return self.source_files['15']

    @f14.setter
    def f15(self, f15_path):
        if not Path(f15_path).exists():
            raise FileNotFoundError(f"ADCIRC fort.14 (Model Parameter and Periodic Boundary Conditions) file at {str(f15_path)} not found")
        self.f15 = f15_path

    @property
    def RUNDES(self):
        """
        Run Description 

        Alpha-numeric string of up to 80 characters describing the run. 
        """
        return self._get_param('RUNDES', 1)

    @RUNDES.setter
    def RUNDES(self, value):
        """
        Set Run Description
        
        Ensures run description is < 80 characters before setting.
        """
        if not isinstance(value, str):
            raise TypeError(f"RUNDES must be of type str")
        if len(value) >= 80:
            raise ValueError(f"RUNDES must be 80 characters or less")
        self.control_params['RUNDES'] = value
    
    @property
    def RUNID(self):
        """
        Run ID

        Alpha-numeric string of up to 8 characters identifying the run.
        """
        return self._get_param('RUNID', 2)
    
    @RUNID.setter
    def RUNID(self, value):
        """
        Set Run ID
        
        Ensures run ID is < 80 characters before setting.

        TODO: Enfore ID requirement?
        """
        if not isinstance(value, str):
            raise TypeError(f"RUNID must be of type str")
        if len(value) >= 80:
            raise ValueError(f"RUNID must be 80 characters or less")
        self.control_params['RUNID'] = value

    @property
    def ICS(self):
        """
        Coordinate System Paremeter. 1 for Carteisan, 2 for Spherical.

        Specifies the initial conditions for the run. 
        """
        return self._get_param('ICS', 8)
    
    @ICS.setter
    def ICS(self, value):
        """
        Set ICS
        
        Ensures ICS is 1 or 2 before setting.
        """
        if not isinstance(value, int):
            raise TypeError(f"ICS must be of type int")
        if value not in [1, 2]:
            raise ValueError(f"ICS must be 1 (Cartesian) or 2 (Spherical)")
        # TODO: Does the value here have to match the fort.14 grid?
        self.control_params['ICS'] = value



def read_fort15(f15_file, ds):
    """
    Reads in ADCIRC fort.15 f15_file, which contains the model parameter and periodic boundary conditions. This file contains the majority of parameters that control the execution of an ADCIRC run, along with harmonic boundary conditions (either elevation or flux).
    
    Parameters
    ----------
    f15_file : str
        Path to fort.15 file
    ds : xr.Dataset, optional
        Dataset to store the information. If None, create a new Dataset.
    
    Returns
    -------
    ds : xr.Dataset
        Dataset containing the fort.15 information.
    """
    if not Path(f15_file).is_file():
        raise ValueError(f"fort.15 file at {f15_file} not found")

    if type(ds) != xr.Dataset:
        raise ValueError(f"ds must be of type xr.Dataset")
        if 'NETA' not in ds.attrs.keys():
            raise ValueError(f"ds must contain NETA attribute")

    ds, ln = read_text_line(ds, "RUNDES", f15_file, ln=1)
    ds, ln = read_text_line(ds, "RUNID", f15_file, ln=ln)

    for i, p in enumerate(["NFOVER", "NABOUT", "NSCREEN", "IHOT", "ICS", "IM"]):
        ds, ln = read_param_line(ds, [p], f15_file, ln=ln, dtypes=[int])

    if ds.attrs["IM"] in [21, 31]:
        ds, ln = read_param_line(ds, ["IDEN"], f15_file, ln=ln, dtypes=[int])

    for i, p in enumerate(["NOLIBF", "NOLIFA", "NOLICA", "NOLICAT", "NWP"]):
        ds, ln = read_param_line(ds, [p], f15_file, ln=ln, dtypes=[int])

    nodals = []
    # Nodal Attributes
    for i in range(ds.attrs["NWP"]):
        line = lc.getline(f15_file, ln).strip()
        logger.info("Nodal Attribute: " + str(i) + " - " + line)
        nodals.append(line)
        ln += 1
    ds = ds.assign({"NODAL_ATTRS": nodals})

    for i, p in enumerate(["NCOR", "NTIP", "NWS", "NRAMP"]):
        ds, ln = read_param_line(ds, [p], f15_file, ln=ln, dtypes=[int])

    for i, p in enumerate(["G", "TAU0"]):
        ds, ln = read_param_line(ds, [p], f15_file, ln=ln, dtypes=[float])

    if ds.attrs["TAU0"] == -5.0:
        ds, ln = read_param_line(
            ds,
            ["TAU0_FullDomain_Min", "TAU0_FullDomain_Max"],
            f15_file,
            ln=ln,
            dtypes=2 * [float],
        )

    for i, p in enumerate(["DTDP", "STATIM", "REFTIM"]):
        ds, ln = read_param_line(ds, [p], f15_file, ln=ln, dtypes=[int])

    if ds.attrs["NWS"] != 0:
        if ds.attrs["NWS"] > 100:
            tmp, ln = read_param_line(
                xr.Dataset(), ["W1", "W2"], f15_file, ln=ln, dtypes=[int, int]
            )
            ds.attrs["WTIMINC"] = [tmp.attrs["W1"], tmp.attrs["W2"]]
        else:
            ds, ln = read_param_line(ds, ["WTIMINC"], f15_file, ln=ln, dtypes=[int])

    ds, ln = read_param_line(ds, ["RNDAY"], f15_file, ln=ln, dtypes=[float])

    if ds.attrs["NRAMP"] in [0, 1]:
        ds, ln = read_param_line(ds, ["DRAMP"], f15_file, ln=ln, dtypes=[int])
    elif ds.attrs["NRAMP"] == 2:
        ds, ln = read_param_line(
            ds, ["DRAMP", "DRAMPExtFlux", "FluxSettlingTime"], f15_file, ln=ln
        )
    elif ds.attrs["NRAMP"] == 3:
        ds, ln = read_param_line(
            ds,
            ["DRAMP", "DRAMPExtFlux", "FluxSettlingTime", "DRAMPIntFlux"],
            f15_file,
            ln=ln,
        )
    elif ds.attrs["NRAMP"] == 4:
        ds, ln = read_param_line(
            ds,
            ["DRAMP", "DRAMPExtFlux", "FluxSettlingTime", "DRAMPIntFlux", "DRAMPElev"],
            f15_file,
            ln=ln,
        )
    elif ds.attrs["NRAMP"] == 5:
        ds, ln = read_param_line(
            ds,
            [
                "DRAMP",
                "DRAMPExtFlux",
                "FluxSettlingTime",
                "DRAMPIntFlux",
                "DRAMPElev",
                "DRAMPTip",
            ],
            f15_file,
            ln=ln,
        )
    elif ds.attrs["NRAMP"] == 6:
        ds, ln = read_param_line(
            ds,
            [
                "DRAMP",
                "DRAMPExtFlux",
                "FluxSettlingTime",
                "DRAMPIntFlux",
                "DRAMPElev",
                "DRAMPTip",
                "DRAMPMete",
            ],
            f15_file,
            ln=ln,
        )
    elif ds.attrs["NRAMP"] == 7:
        ds, ln = read_param_line(
            ds,
            [
                "DRAMP",
                "DRAMPExtFlux",
                "FluxSettlingTime",
                "DRAMPIntFlux",
                "DRAMPElev",
                "DRAMPTip",
                "DRAMPMete",
                "DRAMPWRad",
            ],
            f15_file,
            ln=ln,
        )
    elif ds.attrs["NRAMP"] == 8:
        ds, ln = read_param_line(
            ds,
            [
                "DRAMP",
                "DRAMPExtFlux",
                "FluxSettlingTime",
                "DRAMPIntFlux",
                "DRAMPElev",
                "DRAMPTip",
                "DRAMPMete",
                "DRAMPWRad",
                "DUnRampMete",
            ],
            f15_file,
            ln=ln,
        )

    ds, ln = read_param_line(
        ds, ["A00", "B00", "C00"], f15_file, ln=ln, dtypes=3 * [float]
    )

    nolifa = int(ds.attrs["NOLIFA"])
    if nolifa in [0, 1]:
        ds, ln = read_param_line(ds, ["H0"], f15_file, ln=ln, dtypes=[float])
    elif nolifa in [2, 3]:
        ds, ln = read_param_line(
            ds,
            ["H0", "NODEDRYMIN", "NODEWETMIN", "VELMIN"],
            f15_file,
            ln=ln,
            dtypes=4 * [float],
        )

    ds, ln = read_param_line(
        ds, ["SLAM0", "SFEA0"], f15_file, ln=ln, dtypes=2 * [float]
    )

    nolibf = int(ds.attrs["NOLIBF"])
    if nolibf == 0:
        ds, ln = read_param_line(ds, ["TAU"], f15_file, ln=ln, dtypes=[float])
    elif nolibf == 1:
        ds, ln = read_param_line(ds, ["CF"], f15_file, ln=ln, dtypes=[float])
    elif nolibf == 2:
        ds, ln = read_param_line(
            ds,
            ["CF", "HBREAK", "FTHETA", "FGAMMA"],
            f15_file,
            ln=ln,
            dtypes=4 * [float],
        )
    elif nolibf == 3:
        ds, ln = read_param_line(
            ds, ["CF", "HBREAK", "FTHETA"], f15_file, ln=ln, dtypes=3 * [float]
        )

    if ds.attrs["IM"] != "10":
        ds, ln = read_param_line(ds, ["ESLM"], f15_file, ln=ln, dtypes=[float])
    else:
        ds, ln = read_param_line(
            ds, ["ESLM", "ESLC"], f15_file, ln=ln, dtypes=2 * [float]
        )

    ds, ln = read_param_line(ds, ["CORI"], f15_file, ln=ln, dtypes=[float])
    ds, ln = read_param_line(ds, ["NTIF"], f15_file, ln=ln, dtypes=[int])

    tides = []
    for i in range(ds.attrs["NTIF"]):
        tc = xr.Dataset()
        tc, ln = read_param_line(tc, ["TIPOTAG"], f15_file, ln=ln)
        tc, ln = read_param_line(
            tc,
            ["TPK", "AMIGT", "ETRF", "FFT", "FACET"],
            f15_file,
            ln=ln,
            dtypes=5 * [float],
        )
        tides.append(tc.attrs)
    ds = xr.merge(
        [ds, pd.DataFrame(tides).set_index("TIPOTAG").to_xarray()],
        combine_attrs="override",
    )

    ds, ln = read_param_line(ds, ["NBFR"], f15_file, ln=ln, dtypes=[int])

    # Tidal forcing frequencies on elevation specified boundaries
    tides_elev = []
    for i in range(ds.attrs["NBFR"]):
        temp = xr.Dataset()
        temp, ln = read_param_line(temp, ["BOUNTAG"], f15_file, ln=ln)
        temp, ln = read_param_line(
            temp, ["AMIG", "FF", "FACE"], f15_file, ln=ln, dtypes=3 * [float]
        )
        tides_elev.append(temp.attrs)
    ds = xr.merge(
        [ds, pd.DataFrame(tides_elev).set_index("BOUNTAG").to_xarray()],
        combine_attrs="override",
    )

    # Harmonic forcing function specification at elevation sepcified boundaries
    force_elevs = []
    for i in range(ds.attrs["NBFR"]):
        temp, _ = read_param_line(temp, ["ALPHA"], f15_file, ln=ln)
        temp_df = pd.read_csv(
            f15_file,
            skiprows=ln,
            nrows=ds.attrs["NETA"],
            names=["EMO", "EFA"],
            delim_whitespace=True,
            header=None,
            low_memory=False,
        )
        temp_df["ALPHA"] = temp.attrs["ALPHA"]
        force_elevs.append(temp_df)
        ln += ds.attrs["NETA"] + 1
    force_elev = pd.concat(force_elevs)
    ds = xr.merge(
        [ds, force_elev.set_index("ALPHA").to_xarray()], combine_attrs="override"
    )

    # ANGINN
    ds, ln = read_param_line(ds, ["ANGINN"], f15_file, ln=ln, dtypes=[float])

    # TODO: Handle cases with tidal forcing frequencies on normal flow external boundaries
    # if not set([int(x['IBTYPE']) for x in info['NBOU_BOUNDS']]).isdisjoint([2, 12, 22, 32, 52]):
    #   msg = "Tidal forcing frequencies on normal flow extrnal boundaries not implemented yet"
    #   logger.error(msg)
    #   raise Exception(msg)
    # info = read_param_line(info, ['NFFR'], f15_file)

    # Tidal forcing frequencies on normal flow  external boundar condition
    # info['TIDES_NORMAL'] = []
    # for i in range(int(info['NBFR'])):
    #   temp = {}
    #   temp = read_param_line(temp, ['FBOUNTAG'], f15_file)
    #   temp = read_param_line(temp, ['FAMIG', 'FFF', 'FFACE'], f15_file)
    #   info['TIDES_NORMAL'].append(temp)

    # info['FORCE_NORMAL'] = []
    # for i in range(int(info['NBFR'])):
    #   temp = {}
    #   temp = read_param_line(temp, ['ALPHA'], f15_file)
    #   in_set = not set([int(x['IBTYPE']) for x in info['NBOU_BOUNDS']]).isdisjoint(
    #           [2, 12, 22, 32])
    #   sz = 2 if in_set else 4
    #   temp['VALS'] =  np.empty((int(info['NVEL']), sz), dtype=float)
    #   for j in range(int(info['NVEL'])):
    #     temp['VALS'][j] = read_numeric_line(f15_file, float)
    #   info['FORCE_NORMAL'].append(temp)

    ds, ln = read_param_line(
        ds,
        ["NOUTE", "TOUTSE", "TOUTFE", "NSPOOLE"],
        f15_file,
        ln=ln,
        dtypes=4 * [float],
    )
    ds, ln = read_param_line(ds, ["NSTAE"], f15_file, ln=ln, dtypes=[float])
    df = pd.read_csv(
        f15_file,
        skiprows=ln - 1,
        nrows=int(ds.attrs["NSTAE"]),
        delim_whitespace=True,
        header=None,
        names=["XEL", "YEL"],
        usecols=["XEL", "YEL"],
    )
    df.index.name = "STATIONS"
    df["XEL"] = df["XEL"].astype(float)
    df["YEL"] = df["YEL"].astype(float)
    ds = xr.merge([ds, df.to_xarray()], combine_attrs="override")
    ln += int(ds.attrs["NSTAE"])

    ds, ln = read_param_line(
        ds,
        ["NOUTV", "TOUTSV", "TOUTFV", "NSPOOLV"],
        f15_file,
        ln=ln,
        dtypes=4 * [float],
    )
    ds, ln = read_param_line(ds, ["NSTAV"], f15_file, ln=ln, dtypes=[float])
    df = pd.read_csv(
        f15_file,
        skiprows=ln - 1,
        nrows=int(ds.attrs["NSTAV"]),
        delim_whitespace=True,
        header=None,
        names=["XEV", "YEV"],
        usecols=["XEV", "YEV"],
    )
    df.index.name = "STATIONS_VEL"
    df["XEV"] = df["XEV"].astype(float)
    df["YEV"] = df["YEV"].astype(float)
    ds = xr.merge([ds, df.to_xarray()], combine_attrs="override")
    ln += int(ds.attrs["NSTAV"])

    if ds.attrs["IM"] == 10:
        ds, ln = read_param_line(
            ds,
            ["NOUTC", "TOUTSC", "TOUTFC", "NSPOOLC"],
            f15_file,
            ln=ln,
            dtypes=4 * [float],
        )
        ds, ln = read_param_line(ds, ["NSTAC"], f15_file, ln=ln, dtypes=[float])
        df = pd.read_csv(
            f15_file,
            skiprows=ln - 1,
            nrows=int(ds.attrs["NSTAC"]),
            delim_whitespace=True,
            header=None,
            names=["XEC", "YEC"],
            usecols=["XEC", "YEC"],
        )
        df.index.name = "STATIONS_CONC"
        df["XEV"] = df["XEV"].astype(float)
        df["YEV"] = df["YEV"].astype(float)
        ds = xr.merge([ds, df.to_xarray()], combine_attrs="override")
        ln += int(ds.attrs["NSTAC"])

    if ds.attrs["NWS"] != 0:
        ds, ln = read_param_line(
            ds,
            ["NOUTM", "TOUTSM", "TOUTFM", "NSPOOLM"],
            f15_file,
            ln=ln,
            dtypes=4 * [float],
        )
        ds, ln = read_param_line(ds, ["NSTAM"], f15_file, ln=ln, dtypes=[float])
        df = pd.read_csv(
            f15_file,
            skiprows=ln - 1,
            nrows=int(ds.attrs["NSTAM"]),
            delim_whitespace=True,
            header=None,
            names=["XEM", "YEM"],
            usecols=["XEM", "YEM"],
        )
        df.index.name = "STATIONS_MET"
        df["XEM"] = df["XEM"].astype(float)
        df["YEM"] = df["YEM"].astype(float)
        ds = xr.merge([ds, df.to_xarray()], combine_attrs="override")
        ln += int(ds.attrs["NSTAM"])

    ds, ln = read_param_line(
        ds,
        ["NOUTGE", "TOUTSGE", "TOUTFGE", "NSPOOLGE"],
        f15_file,
        ln=ln,
        dtypes=4 * [float],
    )
    ds, ln = read_param_line(
        ds,
        ["NOUTGV", "TOUTSGV", "TOUTFGV", "NSPOOLGV"],
        f15_file,
        ln=ln,
        dtypes=4 * [float],
    )
    if ds.attrs["IM"] == "10":
        ds, ln = read_param_line(
            ds,
            ["NOUTGC", "TOUTSGC", "TOUTFGC", "NSPOOLGC"],
            f15_file,
            ln=ln,
            dtypes=4 * [float],
        )
    if float(ds.attrs["NWS"]) != 0:
        ds, ln = read_param_line(
            ds,
            ["NOUTGW", "TOUTSGW", "TOUTFGW", "NSPOOLGW"],
            f15_file,
            ln=ln,
            dtypes=4 * [float],
        )

    harmonic_analysis = []
    ds, ln = read_param_line(ds, ["NFREQ"], f15_file, ln=ln, dtypes=[int])
    if ds.attrs["NFREQ"] > 0:
        for i in range(ds.attrs["NFREQ"]):
            tmp = xr.Dataset()
            tmp, ln = read_param_line(tmp, ["NAMEFR"], f15_file, ln=ln)
            tmp, ln = read_param_line(
                tmp, ["HAFREQ", "HAFF", "HAFACE"], f15_file, ln=ln, dtypes=3 * [float]
            )
            harmonic_analysis.append(tmp.attrs)
        ds = xr.merge(
            [ds, pd.DataFrame(harmonic_analysis).set_index("NAMEFR").to_xarray()],
            combine_attrs="override",
        )

    ds, ln = read_param_line(
        ds, ["THAS", "THAF", "NHAINC", "FMV"], f15_file, ln=ln, dtypes=4 * [float]
    )
    ds, ln = read_param_line(
        ds, ["NHASE", "NHASV", "NHAGE", "NHAGV"], f15_file, ln=ln, dtypes=4 * [float]
    )
    ds, ln = read_param_line(
        ds, ["NHSTAR", "NHSINC"], f15_file, ln=ln, dtypes=2 * [float]
    )
    ds, ln = read_param_line(
        ds, ["ITITER", "ISLDIA", "CONVCR", "ITMAX"], f15_file, ln=ln, dtypes=4 * [float]
    )

    # Note - fort.15 files configured for 3D runs not supported yet
    if int(ds.attrs["IM"]) in [1, 2, 11, 21, 31]:
        msg = "fort.15 files configured for 3D runs not supported yet."
        logger.error(msg)
        raise Exception(msg)
    elif ds.attrs["IM"] == 6:
        if ds.attrs["IM"][0] == 1:
            msg = "fort.15 files configured for 3D runs not supported yet."
            logger.error(msg)
            raise Exception(msg)

    # Last 10 fields before control list is netcdf params
    for p in [
        "NCPROJ",
        "NCINST",
        "NCSOUR",
        "NCHIST",
        "NCREF",
        "NCCOM",
        "NCHOST",
        "NCCONV",
        "NCCONT",
        "NCDATE",
    ]:
        ds, ln = read_text_line(ds, p, f15_file, ln=ln)

    # At the very bottom is the control list. Don't parse this as of now
    ds.attrs["CONTROL_LIST"] = []
    line = lc.getline(f15_file, ln)
    while line != "":
        ds.attrs["CONTROL_LIST"].append(line.strip())
        ln += 1
        line = lc.getline(f15_file, ln)

    return ds


def read_fort13(f13_file, ds=None):
    if type(ds) != xr.Dataset:
        ds = xr.Dataset()

    ds, ln = read_param_line(ds, ["AGRID"], f13_file, ln=1)

    # Note this must match NP
    ds, ln = read_param_line(ds, ["NumOfNodes"], f13_file, ln=ln, dtypes=[int])

    # Note this must be >= NWP
    ds, ln = read_param_line(ds, ["NAttr"], f13_file, ln=ln, dtypes=[int])

    # Read Nodal Attribute info
    nodals = []
    for i in range(int(ds.attrs["NAttr"])):
        tmp, ln = read_param_line(xr.Dataset(), ["AttrName"], f13_file, ln=ln)
        tmp, ln = read_param_line(tmp, ["Units"], f13_file, ln=ln)
        tmp, ln = read_param_line(tmp, ["ValuesPerNode"], f13_file, ln=ln, dtypes=[int])
        tmp, ln = read_param_line(
            tmp,
            ["v" + str(i) for i in range(tmp.attrs["ValuesPerNode"])],
            f13_file,
            ln=ln,
            dtypes=tmp.attrs["ValuesPerNode"] * [float],
        )
        nodals.append(tmp.attrs)
    ds = xr.merge(
        [ds, pd.DataFrame(nodals).set_index("AttrName").to_xarray()],
        combine_attrs="override",
    )

    # Read Non Default Nodal Attribute Values
    non_default = []
    line = lc.getline(f13_file, ln)
    while line != "":
        tmp, ln = read_param_line(tmp, ["AttrName"], f13_file, ln=ln)
        tmp, ln = read_param_line(tmp, ["NumND"], f13_file, ln=ln, dtypes=[int])

        num_vals = ds["ValuesPerNode"][ds["AttrName"] == tmp.attrs["AttrName"]].values[
            0
        ]
        cols = ["JN"] + [
            "_".join([tmp.attrs["AttrName"], str(x)]) for x in range(num_vals)
        ]
        tmp_df = pd.read_csv(
            f13_file,
            skiprows=ln - 1,
            nrows=tmp.attrs["NumND"],
            delim_whitespace=True,
            names=cols,
        )
        non_default.append(tmp_df)
        ln += tmp.attrs["NumND"]
        line = lc.getline(f13_file, ln)
    ds = xr.merge(
        [
            ds,
            reduce(lambda x, y: x.merge(y, how="outer"), non_default)
            .set_index("JN")
            .to_xarray(),
        ],
        combine_attrs="override",
    )

    return ds


def read_fort22(f22_file, NWS=12, ds=None):
    if type(ds) == xr.Dataset:
        if "NWS" in ds.attrs.keys():
            NWS = ds.attrs["NWS"]
    else:
        ds = xr.Dataset()

    if NWS in [12, 12012]:
        ds, _ = read_param_line(ds, ["NWSET"], f22_file, ln=1, dtypes=[float])
        ds, _ = read_param_line(ds, ["NWBS"], f22_file, ln=2, dtypes=[float])
        ds, _ = read_param_line(ds, ["DWM"], f22_file, ln=3, dtypes=[float])
    else:
        msg = f"NWS {NWS} Not yet implemented!"
        logger.error(msg)
        raise Exception(msg)

    return ds


def read_fort24(f22_file, ds=None):
    if type(ds) != xr.Dataset:
        ds = xr.Dataset()

    data = pd.read_csv(
        f22_file,
        delim_whitespace=True,
        names=["JN", "SALTAMP", "SALTPHA"],
        low_memory=False,
        header=None,
    )
    tides = data[data["SALTPHA"].isna()]
    all_tmp = []
    for i in range(int(tides.shape[0] / 4)):
        stop = (tides.index[(i + 1) * 4] - 1) if i != 7 else data.index[-1]
        tmp = data.loc[(tides.index[i * 4 + 3] + 1) : stop][
            ["JN", "SALTAMP", "SALTPHA"]
        ].copy()
        tmp["JN"] = tmp["JN"].astype(int)
        tmp["SALTAMP"] = tmp["SALTAMP"].astype(float)
        tmp["SALTPHA"] = tmp["SALTPHA"].astype(float)
        tmp["SALTFREQ"] = float(tides["JN"].iloc[i * 4 + 1])
        tmp = tmp.set_index("JN").to_xarray()
        tmp = tmp.expand_dims(dim={"SALTNAMEFR": [tides["JN"].iloc[i * 4 + 3]]})
        all_tmp.append(tmp)

    ds = xr.merge([ds, xr.concat(all_tmp, "SALTNAMEFR")], combine_attrs="override")

    return ds


def read_fort25(f25_file, NWS=12, ds=None):
    if ds != None:
        if "NWS" in ds.attrs.keys():
            NWS = ds.attrs["NWS"]
    else:
        ds = xr.Dataset()

    if NWS in [12, 12012]:
        ds, _ = read_param_line(ds, ["NUM_ICE_FIELDS"], f25_file, ln=1, dtypes=[float])
        ds, _ = read_param_line(
            ds, ["NUM_BLANK_ICE_SNAPS"], f25_file, ln=2, dtypes=[float]
        )
    else:
        msg = f"NWS {NWS} Not yet implemented!"
        logger.error(msg)
        raise Exception(msg)

    return ds


def read_fort221(f221_file, NWS=12, times=[], ds=None):
    if ds != None:
        if "NWS" in ds.attrs.keys():
            NWS = ds.attrs["NWS"]
    else:
        ds = xr.Dataset()

    if NWS in [12, 12012]:
        pressure_data = read_owi_met(f221_file, vals=["press"], times=times)
    else:
        msg = f"NWS {NWS} Not yet implemented!"
        logger.error(msg)
        raise Exception(msg)

    attrs = {"press_" + str(key): val for key, val in pressure_data.attrs.items()}
    pressure_data.attrs = attrs
    return xr.merge([ds, pressure_data], combine_attrs="no_conflicts")


def read_fort222(f222_file, NWS=12, times=[], ds=None):
    if ds != None:
        if "NWS" in ds.attrs.keys():
            NWS = ds.attrs["NWS"]
    else:
        ds = xr.Dataset()

    if NWS in [12, 12012]:
        wind_data = read_owi_met(f222_file, vals=["u_wind", "v_wind"], times=times)
    else:
        msg = f"NWS {NWS} Not yet implemented!"
        logger.error(msg)
        raise Exception(msg)

    attrs = {"wind_" + str(key): val for key, val in wind_data.attrs.items()}
    wind_data.attrs = attrs
    return xr.merge([ds, wind_data], combine_attrs="no_conflicts")


def read_fort225(f225_file, NWS=12, times=[], ds=None):
    if ds != None:
        if "NWS" in ds.attrs.keys():
            NWS = ds.attrs["NWS"]
    else:
        ds = xr.Dataset()

    if NWS in [12, 12012]:
        ice_data = read_owi_met(f225_file, vals=["ice_cov"], times=times)
    else:
        msg = f"NWS {NWS} Not yet implemented!"
        logger.error(msg)
        raise Exception(msg)

    attrs = {"ice_" + str(key): val for key, val in ice_data.attrs.items()}
    ice_data.attrs = attrs
    return xr.merge([ds, ice_data], combine_attrs="no_conflicts")


def read_owi_met(path, vals=["v1"], times=[0]):
    # NWS 12 - Ocean Weather Inc (OWI) met data
    attrs = {}

    # Title line:
    # 10 format (t56,i10,t71,i10)
    # read (20,10) date1,date2
    line = lc.getline(path, 1)
    attrs["source"] = line[0:56]
    attrs["start_ts"] = pd.to_datetime(line[55:66].strip(), format="%Y%m%d%H")
    attrs["end_ts"] = pd.to_datetime(line[70:80].strip(), format="%Y%m%d%H")

    # TODO: Make sure times is sorted in increasing order, all positive values

    if len(lc.getline(path, 2)) > 79:
        tf = "%Y%m%d%H%M%S"
        ti_idx = 67
    else:
        tf = "%Y%m%d%H"
        ti_idx = 68

    i = 0  # Current idx in metereological data
    time_idx = 0  # Index in time array (array of indices we want)
    cur_line = 2
    all_data = []
    line = lc.getline(path, cur_line)
    while i < 1 + times[-1]:
        if line == "":
            break
        try:
            if times[time_idx] == i:
                # Grid Spec Line:
                # 11 format (t6,i4,t16,i4,t23,f6.0,t32,f6.0,t44,f8.0,t58,f8.0,t69,i10,i2)
                # read (20,11) iLat, iLong, dx, dy, swlat, swlong, lCYMDH, iMin
                grid_spec = re.sub(r"[^\-0-9=.]", "", line)[1:].split("=")
                ilat = int(grid_spec[0])
                ilon = int(grid_spec[1])
                dx = float(grid_spec[2])
                dy = float(grid_spec[3])
                swlat = float(grid_spec[4])
                swlon = float(grid_spec[5])
                ts = pd.to_datetime(grid_spec[6], format=tf)

                # swlon = float(line[57:65])
                # ts = pd.to_datetime(line[68:len(line)-1], format=tf)

                logger.info(f"Processing data at {ts}")

                data = {}
                line_count = int(ilat * ilon / 8.0)
                remainder = int(ilat * ilon - (line_count * 8))
                for v in vals:
                    vs = np.zeros(ilat * ilon)

                    with open(path, "r") as f:
                        vs[0 : (8 * line_count)] = pd.read_fwf(
                            f,
                            nrows=line_count,
                            skiprows=cur_line,
                            widths=8 * [10],
                            header=None,
                        ).values.flatten()
                    if remainder > 0:
                        with open(path, "r") as f:
                            vs[(8 * line_count) :] = pd.read_fwf(
                                f,
                                nrows=1,
                                skiprows=cur_line + line_count,
                                widths=remainder * [10],
                                header=None,
                            ).values.flatten()
                        cur_line = cur_line + line_count + 1
                    else:
                        cur_line = cur_line + line_count

                    vs = vs.reshape(1, ilat, ilon)
                    data[v] = (["time", "latitude", "longitude"], vs)

                # Convert swlon to positive longitude value
                if swlon < 0:
                    swlon = 360 + swlon
                lon = np.arange(start=swlon, stop=(swlon + (ilon * dx)), step=dx)
                # lon = np.where(lon<180.0,lon,lon-360)

                coords = {
                    "time": [ts],
                    "longitude": lon,
                    "latitude": np.arange(
                        start=swlat, stop=(swlat + (ilat * dy)), step=dy
                    ),
                }
                all_data.append(xr.Dataset(data, coords=coords))

                # Only increment index in time array if it matches current index of data
                time_idx = time_idx + 1
            else:
                grid_spec = re.sub(r"[^\-0-9=.]", "", line)[1:].split("=")
                ilat = int(grid_spec[0])
                ilon = int(grid_spec[1])
                ts = pd.to_datetime(grid_spec[6], format=tf)
                line_count = int(ilat * ilon / 8.0)
                remainder = int(ilat * ilon - (line_count * 8))

                logger.info(f"Skiping data at {ts}")
                for v in vals:
                    if remainder > 0:
                        cur_line = cur_line + line_count + 1
                    else:
                        cur_line = cur_line + line_count

            # Get next line corresponding to next datapoint, or empty if done
            i += 1
            cur_line += 1
            line = lc.getline(path, cur_line)
        except Exception as e:
            raise e

    if len(all_data) > 0:
        ret_ds = xr.concat(all_data, "time")
    else:
        ret_ds = xr.Dataset()

    ret_ds.attrs = attrs

    return ret_ds


def write_fort14(ds, f14_file):
    """write_fort14.
    Reads in ADCIRC fort.14 f14_file

    :param f14_file: Path to Python file.
    """
    with open(f14_file, "w") as f14:
        # 1 : AGRID = alpha-numeric grid identification (<=24 characters).
        write_text_line(ds, "AGRID", f14)

        # 2 : NE, NP = number of elements, nodes in horizontal grid
        write_param_line(ds, ["NE", "NP"], f14)

    # 3-NP : NODES
    # for k=1 to NP
    #    JN, X(JN), Y(JN), DP(JN)
    # end k loop
    logger.info("Wriing Node Map.")
    ds[["JN", "X", "Y", "DP"]].to_dataframe().to_csv(
        f14_file, sep=" ", mode="a", header=False
    )

    # (2+NP)-(2+NP+NE) : ELEMENTS
    # for k=1 to NE
    #    JE, NHY, NM(JE,1),NM(JE,2), NM(JE,3)
    # end k loop
    logger.info("Writing Element Map.")
    ds[["JE", "NHY", "NM_1", "NM_2", "NM_3"]].to_dataframe().to_csv(
        f14_file, sep=" ", mode="a", header=False
    )

    # Elevation specified boundaries
    with open(f14_file, "a") as f14:
        # (3+NP+NE) : NOPE = number of elevation specified boundary forcing segments.
        write_param_line(ds, ["NOPE"], f14)

        # (4+NP+NE) : NETA = total number of elevation specified boundary nodes
        write_param_line(ds, ["NETA"], f14)

    for (bnd_idx, bnd) in ds[["ELEV_BOUNDARY_NODES", "IBTYPEE"]].groupby(
        "ELEV_BOUNDARY"
    ):
        with open(f14_file, "a") as f14:
            # NVDLL(k), IBTYPEE(k) = number of nodes, and boundary type
            write_param_line(
                xr.Dataset(
                    attrs={
                        "NVDLL": bnd["ELEV_BOUNDARY_NODES"].shape[0],
                        "IBTYPEE": bnd["IBTYPEE"].item(0),
                    }
                ),
                ["NVDLL", "IBTYPEE"],
                f14,
            )
        bnd["ELEV_BOUNDARY_NODES"].to_dataframe().to_csv(
            f14_file, sep=" ", mode="a", header=False, index=False
        )

    # Normal flow specified boundaries
    with open(f14_file, "a") as f14:
        # NBOU = number of normal flow (discharge) specified boundary segments
        write_param_line(ds, ["NBOU"], f14)

        # NVEL = total number of normal flow specified boundary nodes
        write_param_line(ds, ["NVEL"], f14)

    for (bnd_idx, bnd) in ds[["NORMAL_BOUNDARY_NODES", "IBTYPE"]].groupby(
        "NORMAL_BOUNDARY"
    ):
        with open(f14_file, "a") as f14:
            # NVELL(k), IBTYPE(k) = number of nodes, and boundary type
            write_param_line(
                xr.Dataset(
                    attrs={
                        "NVELL": bnd["NORMAL_BOUNDARY_NODES"].shape[0],
                        "IBTYPE": bnd["IBTYPE"].item(0),
                    }
                ),
                ["NVELL", "IBTYPE"],
                f14,
            )
        bnd["NORMAL_BOUNDARY_NODES"].to_dataframe().to_csv(
            f14_file, sep=" ", mode="a", header=False, index=False
        )


def write_fort15(ds, f15_file):
    """write_fort15.
    Reads in ADCIRC fort.15 f15_file

    :param f15_file: Path to Python file.
    """
    with open(f15_file, "w") as f15:
        write_text_line(ds, "RUNDES", f15)
        write_text_line(ds, "RUNID", f15)

        for i, p in enumerate(["NFOVER", "NABOUT", "NSCREEN", "IHOT", "ICS", "IM"]):
            write_param_line(ds, [p], f15)

        if int(ds.attrs["IM"]) in [21, 31]:
            write_param_line(ds, ["IDEN"], f15)

        for i, p in enumerate(["NOLIBF", "NOLIFA", "NOLICA", "NOLICAT", "NWP"]):
            write_param_line(ds, [p], f15)

        # Nodal Attributes
        for nodal in ds["NODAL_ATTRS"].values:
            f15.write(nodal + "\n")

        for i, p in enumerate(["NCOR", "NTIP", "NWS", "NRAMP", "G", "TAU0"]):
            write_param_line(ds, [p], f15)

        if float(ds.attrs["TAU0"]) == -5.0:
            write_param_line(ds, ["TAU0_FullDomain_Min", "TAU0_FullDomain_Max"], f15)

        for i, p in enumerate(["DTDP", "STATIM", "REFTIM"]):
            write_param_line(ds, [p], f15)

        if ds.attrs["NWS"] != 0:
            if ds.attrs["NWS"] > 100:
                tmp = ds.attrs["WTIMINC"]
                ds.attrs["WTIMINC"] = f"{tmp[0]} {tmp[1]}"
                write_param_line(ds, ["WTIMINC"], f15)
                ds.attrs["WTIMINC"] = tmp
            else:
                write_param_line(ds, ["WTIMINC"], f15)

        write_param_line(ds, ["RNDAY"], f15)

        nramp = int(ds.attrs["NRAMP"])
        if nramp in [0, 1]:
            write_param_line(ds, ["DRAMP"], f15)
        elif nramp == 2:
            write_param_line(ds, ["DRAMP", "DRAMPExtFlux", "FluxSettlingTime"], f15)
        elif nramp == 3:
            write_param_line(
                ds, ["DRAMP", "DRAMPExtFlux", "FluxSettlingTime", "DRAMPIntFlux"], f15
            )
        elif nramp == 4:
            write_param_line(
                ds,
                [
                    "DRAMP",
                    "DRAMPExtFlux",
                    "FluxSettlingTime",
                    "DRAMPIntFlux",
                    "DRAMPElev",
                ],
                f15,
            )
        elif nramp == 5:
            write_param_line(
                ds,
                [
                    "DRAMP",
                    "DRAMPExtFlux",
                    "FluxSettlingTime",
                    "DRAMPIntFlux",
                    "DRAMPElev",
                    "DRAMPTip",
                ],
                f15,
            )
        elif nramp == 6:
            write_param_line(
                ds,
                [
                    "DRAMP",
                    "DRAMPExtFlux",
                    "FluxSettlingTime",
                    "DRAMPIntFlux",
                    "DRAMPElev",
                    "DRAMPTip",
                    "DRAMPMete",
                ],
                f15,
            )
        elif nramp == 7:
            write_param_line(
                ds,
                [
                    "DRAMP",
                    "DRAMPExtFlux",
                    "FluxSettlingTime",
                    "DRAMPIntFlux",
                    "DRAMPElev",
                    "DRAMPTip",
                    "DRAMPMete",
                    "DRAMPWRad",
                ],
                f15,
            )
        elif nramp == 8:
            write_param_line(
                ds,
                [
                    "DRAMP",
                    "DRAMPExtFlux",
                    "FluxSettlingTime",
                    "DRAMPIntFlux",
                    "DRAMPElev",
                    "DRAMPTip",
                    "DRAMPMete",
                    "DRAMPWRad",
                    "DUnRampMete",
                ],
                f15,
            )

        write_param_line(ds, ["A00", "B00", "C00"], f15)

        nolifa = ds.attrs["NOLIFA"]
        if nolifa in [0, 1]:
            write_param_line(ds, ["H0"], f15)
        elif nolifa in [2, 3]:
            write_param_line(ds, ["H0", "NODEDRYMIN", "NODEWETMIN", "VELMIN"], f15)

        write_param_line(ds, ["SLAM0", "SFEA0"], f15)

        if ds.attrs["NOLIBF"] == 0:
            write_param_line(ds, ["TAU"], f15)
        elif ds.attrs["NOLIBF"] == 1:
            write_param_line(ds, ["CF"], f15)
        elif ds.attrs["NOLIBF"] == 2:
            write_param_line(ds, ["CF", "HBREAK", "FTHETA", "FGAMMA"], f15)
        elif ds.attrs["NOLIBF"] == 3:
            write_param_line(ds, ["CF", "HBREAK", "FTHETA"], f15)

        if ds.attrs["IM"] != "10":
            write_param_line(ds, ["ESLM"], f15)
        else:
            write_param_line(ds, ["ESLM", "ESLC"], f15)

        for i, p in enumerate(["CORI", "NTIF"]):
            write_param_line(ds, [p], f15)

        params = ["TPK", "AMIGT", "ETRF", "FFT", "FACET"]
        for name in [x for x in ds["TIPOTAG"].values]:
            write_text_line(name, "TIPOTAG", f15)
            write_param_line(
                [ds[x].sel(TIPOTAG=name).item(0) for x in params], params, f15
            )

        write_param_line(ds, ["NBFR"], f15)

        # Tidal forcing frequencies at elevation specified boundaries
        params = ["AMIG", "FF", "FACE"]
        for name in [x for x in ds["BOUNTAG"].values]:
            write_text_line(name, "BOUNTAG", f15)
            write_param_line(
                [ds[x].sel(BOUNTAG=name).item(0) for x in params], params, f15
            )

    # Harmonic forcing function at elevation sepcified boundaries
    # Cant use groupby because need to preserve order
    params = ["EMO", "EFA"]

    def uniq(seq):
        seen = set()
        return [x for x in seq if not (x in seen or seen.add(x))]

    for name in uniq([x for x in ds["ALPHA"].values]):
        with open(f15_file, "a") as f15:
            write_text_line(name, "ALPHA", f15)
        ds[params].sel(ALPHA=name).to_dataframe().to_csv(
            f15_file, sep=" ", mode="a", header=False, index=False
        )

    with open(f15_file, "a") as f15:
        # ANGINN
        write_param_line(ds, ["ANGINN"], f15)

        # TODO: Implement case when NBFR!=0
        # if ds.attrs['IBTYPE'] in [2, 12, 22, 32, 52]:
        #   write_param_line(ds, ['NFFR'], f15)

        #   # Tidal forcing frequencies on normal flow  external boundar condition
        #   for i in range(ds.attrs['NBFR']):
        #     write_param_line(ds['TIDES_NORMAL'][i], ['FBOUNTAG'], f15)
        #     write_param_line(ds['TIDES_NORMAL'][i], ['FAMIG', 'FFF', 'FFACE'], f15)

        #   # Periodic normal flow/unit width amplitude
        #   info['FORCE_NORMAL'] = []
        #   for i in range(ds.attrs['NBFR']):
        #     write_param_line(ds['FORCE_NORMAL'][i], ['ALPHA'], f15)
        #     for j in range(ds.attrs['NVEL']):
        #       write_numeric_line(ds['FORCE_NORMAL'][i]['VALS'], f15)

        write_param_line(ds, ["NOUTE", "TOUTSE", "TOUTFE", "NSPOOLE"], f15)
        write_param_line(ds, ["NSTAE"], f15)
    if ds.attrs["NSTAE"] > 0:
        ds[["STATIONS", "XEL", "YEL"]].to_dataframe().to_csv(
            f15_file, sep=" ", mode="a", header=False, index=False
        )

    with open(f15_file, "a") as f15:
        write_param_line(ds, ["NOUTV", "TOUTSV", "TOUTFV", "NSPOOLV"], f15)
        write_param_line(ds, ["NSTAV"], f15)
    if ds.attrs["NSTAV"] > 0:
        ds[["STATIONS_VEL", "XEV", "YEV"]].to_dataframe().to_csv(
            f15_file, sep=" ", mode="a", header=False, index=False
        )

    if ds.attrs["IM"] == 10:
        with open(f15_file, "a") as f15:
            write_param_line(ds, ["NOUTC", "TOUTSC", "TOUTFC", "NSPOOLC"], f15)
            write_param_line(ds, ["NSTAC"], f15)
        if ds.attrs["NSTAC"] > 0:
            ds[["STATIONS_CONC", "XEC", "YEC"]].to_dataframe().to_csv(
                f15_file, sep=" ", mode="a", header=False, index=False
            )

    if ds.attrs["NWS"] != 0:
        with open(f15_file, "a") as f15:
            write_param_line(ds, ["NOUTM", "TOUTSM", "TOUTFM", "NSPOOLM"], f15)
            write_param_line(ds, ["NSTAM"], f15)
        if ds.attrs["NSTAM"] > 0:
            ds[["STATIONS_MET", "XEM", "YEM"]].to_dataframe().to_csv(
                f15_file, sep=" ", mode="a", header=False, index=False
            )

    with open(f15_file, "a") as f15:
        write_param_line(ds, ["NOUTGE", "TOUTSGE", "TOUTFGE", "NSPOOLGE"], f15)
        write_param_line(ds, ["NOUTGV", "TOUTSGV", "TOUTFGV", "NSPOOLGV"], f15)

        if ds.attrs["IM"] == 10:
            write_param_line(ds, ["NOUTGC", "TOUTSGC", "TOUTFGC", "NSPOOLGC"], f15)
        if ds.attrs["NWS"] != 0:
            write_param_line(ds, ["NOUTGW", "TOUTSGW", "TOUTFGW", "NSPOOLGW"], f15)

        write_param_line(ds, ["NFREQ"], f15)
        params = ["HAFREQ", "HAFF", "HAFACE"]
        for name in [x for x in ds["NAMEFR"].values]:
            write_text_line(name, "NAMEFR", f15)
            write_param_line(
                [ds[x].sel(NAMEFR=name).item(0) for x in params], params, f15
            )

        write_param_line(ds, ["THAS", "THAF", "NHAINC", "FMV"], f15)
        write_param_line(ds, ["NHASE", "NHASV", "NHAGE", "NHAGV"], f15)
        write_param_line(ds, ["NHSTAR", "NHSINC"], f15)
        write_param_line(ds, ["ITITER", "ISLDIA", "CONVCR", "ITMAX"], f15)

        # Note - fort.15 files configured for 3D runs not supported yet
        if ds.attrs["IM"] in [1, 2, 11, 21, 31]:
            msg = "fort.15 files configured for 3D runs not supported yet."
            logger.error()
            raise NotImplementedError(msg)
        elif len(str(ds.attrs["IM"])) == 6:
            # 1 in 6th digit indicates 3D run
            if int(ds.attrs["IM"] / 100000.0) == 1:
                msg = "fort.15 files configured for 3D runs not supported yet."
                logger.error(msg)
                raise NotImplementedError(msg)

        # Last 10 fields before control list is netcdf params
        nc_params = [
            "NCPROJ",
            "NCINST",
            "NCSOUR",
            "NCHIST",
            "NCREF",
            "NCCOM",
            "NCHOST",
            "NCCONV",
            "NCCONT",
            "NCDATE",
        ]
        for p in nc_params:
            write_text_line(ds.attrs[p], "", f15)

        # Add Control List to Bottom
        for line in ds.attrs["CONTROL_LIST"]:
            write_text_line(line, "", f15)


def create_nodal_att(name, units, default_vals, nodal_vals):
    str_vals = [f"v{str(x)}" for x in range(len(default_vals))]
    base_df = (
        pd.DataFrame(
            [[name, units, len(default_vals)] + list(default_vals)],
            columns=["AttrName", "Units", "ValuesPerNode"] + str_vals,
        )
        .set_index("AttrName")
        .to_xarray()
    )

    default_vals = (
        pd.DataFrame(
            nodal_vals,
            columns=["JN"]
            + ["_".join([name, str(x)]) for x in range(len(default_vals))],
        )
        .set_index("JN")
        .to_xarray()
    )

    return xr.merge([base_df, default_vals])


def add_nodal_attribute(f13, name, units, default_vals, nodal_vals):
    if type(f13) != xr.Dataset:
        f13 = read_fort13(f13)
    if name in f13["AttrName"]:
        raise Exception(f"Error - Nodal Attribute {name} already in f13 configs.")
    new_nodal = create_nodal_att(name, units, default_vals, nodal_vals)

    df = xr.merge([f13, new_nodal], combine_attrs="override")
    df.attrs["NAttr"] = len(df["AttrName"].values)

    return df


def write_fort13(ds, f13_file):

    with open(f13_file, "w") as f13:
        write_param_line(ds, ["AGRID"], f13)

        write_param_line(ds, ["NumOfNodes"], f13)

        write_param_line(ds, ["NAttr"], f13)

        # Write Nodal Attribute info
        for attr in ds["AttrName"].values:
            write_text_line(attr, "", f13)
            write_text_line(str(ds.sel(AttrName=attr)["Units"].item(0)), "", f13)
            write_text_line(
                str(ds.sel(AttrName=attr)["ValuesPerNode"].item(0)), "", f13
            )
            def_vs = [
                "v" + str(x)
                for x in range(int(ds.sel(AttrName=[attr])["ValuesPerNode"].item(0)))
            ]
            def_vals = [
                str(x.item(0)) for x in ds.sel(AttrName=[attr])[def_vs].values()
            ]
            write_text_line(" ".join(def_vals), "", f13)

    # Write non default values
    for attr in ds["AttrName"].values:
        with open(f13_file, "a") as f13:
            write_text_line(attr, "", f13)
            cols = [
                "_".join([attr, str(x)])
                for x in range(int(ds.sel(AttrName=attr)["ValuesPerNode"].item(0)))
            ]
            out_df = ds[cols].dropna("JN").to_dataframe()
            write_text_line(str(out_df.shape[0]), "", f13)
        out_df.to_csv(f13_file, sep=" ", mode="a", header=None)


def gen_uniform_beta_fort13(
    base_f13_path: str = "fort.13",
    targ_dir: str = None,
    name: str = "beta",
    num_samples: int = 10,
    domain: List[int] = [0.0, 2.0],
):
    """
    Generate fort.13 files w/beta vals from uniform distribution

    Parameters
    ----------
    base_f13_path : str, default='fort.13'
        Path to base fort.13 file to modify beta values for
    targ_dir : str, optional
        Path to output directory. Defaults to current working directory.
    name : str, default='beta'
        Name to give to output directory. Final name will be in the
        format {name}_{domain min}-{domain max}_u{num samples}
    num_samples : int, default=10
        Number of samples to take from a uniform distribution
    domain : List[int], default=[0.0, 2.0]
        Range for beta values.


    Returns
    ----------
    targ_path : str
        Path to directory containing all the seperate job directories
        with individual fort.13 files

    """

    targ_dir = Path.cwd() if targ_dir is None else targ_dir
    if not targ_dir.exists():
        raise ValueError(f"target directory {str(targ_dir)} does not exist")
    if not Path(base_f13_path).exists():
        raise ValueError(f"Unable to find base fort.13 file {base_f13_path}")

    targ_path = Path(
        f"{str(targ_dir)}/{name}_{domain[0]:.1f}-{domain[1]:.1f}_u{num_samples}"
    )
    targ_path.mkdir(exist_ok=True)

    beta_vals = np.random.uniform(domain[0], domain[1], size=num_samples)
    f13 = read_fort13(base_f13_path)

    for idx, b in enumerate(beta_vals):
        f13["v0"][0] = b
        job_name = f"beta-{idx}_{b:.2f}"
        job_dir = targ_path / job_name
        job_dir.mkdir(exist_ok=True)
        write_fort13(f13, str(job_dir / "fort.13"))

    return str(targ_path)


def process_adcirc_configs(path, filt="fort.*", met_times=[]):
    ds = xr.Dataset()

    # Always read fort.14 and fort.15
    adcirc_files = glob.glob(os.path.join(path, filt))
    for ff in adcirc_files:
        ftype = int(ff.split(".")[-1])
        with timing(f"Reading {ff}") as read_time:
            if ftype == 14:
                logger.info(f"Reading fort.14 file {ff}...")
                ds = read_fort14(ff, ds=ds)
            elif ftype == 15:
                logger.info(f"Reading fort.15 file {ff}...")
                ds = read_fort15(ff, ds=ds)
            elif ftype == 13:
                logger.info(f"Reading fort.13 file {ff}...")
                ds = read_fort13(ff, ds=ds)
            elif ftype == 22:
                logger.info(f"Reading fort.22 file {ff}...")
                ds = read_fort22(ff, ds=ds)
            elif ftype == 24:
                logger.info(f"Reading fort.24 file {ff}...")
                ds = read_fort24(ff, ds=ds)
            elif ftype == 25:
                logger.info(f"Reading fort.25 file {ff}...")
                ds = read_fort25(ff, ds=ds)
            elif ftype == 221:
                logger.info(f"Reading fort.221 file {ff}...")
                ds = read_fort221(ff, ds=ds, times=met_times)
            elif ftype == 222:
                logger.info(f"Reading fort.222 file {ff}...")
                ds = read_fort222(ff, ds=ds, times=met_times)
            elif ftype == 225:
                logger.info(f"Reading fort.225 file {ff}...")
                ds = read_fort225(ff, ds=ds, times=met_times)
            else:
                msg = f"Uncreognized file type = {ff}"
                logger.error(msg)
                raise Exception(msg)
        logger.info(f"Read {ff} successfully! - {read_time()[1]}")

    return ds


def modify_f15_attrs(
    fort_path: str = Path.cwd(),
    nc_path: str = None,
    output_format: str = "fort",
    output_dir: str = ".",
    overwrite: bool = False,
    **kwargs,
):
    """Read in f15 file, apply changes in kwargs, and write to output dir."""
    ds = None
    if nc_path is None:
        ds = read_fort14(str(fort_path / "fort.14"))
        ds = read_fort15(str(fort_path / "fort.15"), ds=ds)
    else:
        ds = xr.open_dataset(nc_path)

    to_overwrite = list(kwargs.keys())
    avail = list(ds.attrs.keys())
    wrong = [x for x in to_overwrite if x not in avail]
    if len(wrong) > 0:
        raise ValueError(f"Invalid configs to overwrite {wrong}")

    for attr in to_overwrite:
        ds[attr] = kwargs[attr]

    od_path = Path(output_dir)
    od_path.mkdir(exist_ok=overwrite)

    if output_format == "fort":
        output_path = str(od_path / "fort.15")
        write_fort15(ds, output_path)
    else:
        output_path = str(od_path / ".nc")
        ds.to_netcdf(path=output_path)


def merge_output(
    output_dir: str,
    stations: bool = True,
    globs: bool = False,
    minmax: bool = True,
    nodals: bool = True,
    partmesh: bool = True,
):
    """Merge ADCIRC output. Assumes local/global output on same frequency"""
    ds = xr.Dataset()

    if stations:
        station_idxs = [61, 62, 71, 72, 91]
        station_files = [f"{output_dir}/fort.{x}.nc" for x in station_idxs]
        for i, sf in enumerate(station_files):
            logger.info(f"Reading station data {sf}")
            if i != 0:
                station_data = xr.open_dataset(sf)
                ds = xr.merge([ds, station_data], compat="override")
            else:
                ds = xr.open_dataset(sf)

        d_vars = list(ds.data_vars.keys())
        new_names = [(x, f"{x}-station") for x in d_vars if x != "station_name"]
        ds = ds.rename(dict(new_names))

    if globs:
        glob_idxs = [63, 64, 73, 74, 93]
        global_files = [f"{output_dir}/fort.{x}.nc" for x in glob_idxs]
        for gf in global_files:
            logger.info(f"Reading global data {gf}")
            global_data = xr.open_dataset(gf)
            ds = xr.merge([ds, global_data])

    if minmax:
        minmax = ["maxele", "maxvel", "maxwvel", "minpr"]
        minmax_files = [f"{output_dir}/{x}.63.nc" for x in minmax]
        for mf in minmax_files:
            logger.info(f"Reading min/max data {mf}")
            minmax_data = xr.open_dataset(mf)
            minmax_data = minmax_data.drop("time")
            ds = xr.merge([ds, minmax_data])

    if nodals:
        # Load f13 nodal attribute data
        ds = read_fort13(f"{output_dir}/fort.13", ds)

    if partmesh:
        # Load partition mesh data
        ds["partition"] = (
            ["node"],
            pd.read_csv(f"{output_dir}/partmesh.txt", header=None)
            .to_numpy()
            .reshape(-1),
        )

    return ds


def gen_uniform_beta_fort13(
    base_f13_path: str = "fort.13",
    targ_dir: str = None,
    name: str = "beta",
    num_samples: int = 10,
    domain: List[int] = [0.0, 2.0],
):
    """
    Generate fort.13 files w/beta vals from uniform distribution

    Parameters
    ----------
    base_f13_path : str, default='fort.13'
        Path to base fort.13 file to modify beta values for
    targ_dir : str, optional
        Path to output directory. Defaults to current working directory.
    name : str, default='beta'
        Name to give to output directory. Final name will be in the
        format {name}_{domain min}-{domain max}_u{num samples}
    num_samples : int, default=10
        Number of samples to take from a uniform distribution
    domain : List[int], default=[0.0, 2.0]
        Range for beta values.


    Returns
    ----------
    targ_path : str
        Path to directory containing all the seperate job directories
        with individual fort.13 files

    """

    targ_dir = Path.cwd() if targ_dir is None else targ_dir
    if not targ_dir.exists():
        raise ValueError(f"target directory {str(targ_dir)} does not exist")
    if not Path(base_f13_path).exists():
        raise ValueError(f"Unable to find base fort.13 file {base_f13_path}")

    targ_path = Path(f"{str(targ_dir)}")
    targ_path.mkdir(exist_ok=True)

    beta_vals = np.random.uniform(domain[0], domain[1], size=num_samples)
    f13 = read_fort13(base_f13_path)

    files = []
    for idx, b in enumerate(beta_vals):
        f13["v0"][0] = b
        job_name = f"{name}_job-{idx}-{num_samples}_beta{b:.2f}"
        job_name += f"_u{domain[0]:.1f}-{domain[1]:.1f}"
        job_dir = targ_path / job_name
        job_dir.mkdir(exist_ok=True)
        fpath = str(job_dir / "fort.13")
        write_fort13(f13, fpath)
        files.append(fpath)

    return str(targ_path), files


def cfsv2_grib_to_adcirc_owi(
    files: List[str],
    outfile: str = None,
    newgrid=None,
    bounding_box: List[float] = None,
    date_range: Tuple[str] = None,
):
    """
    CFSv2 Grib Data to ADCIRC fort.22* metereological forcing files.
    Converts and sub-samples data in time and space from a set of grib files
    that have been downloaded from NCAR's CFSv2 data set (id='ds094.1')
    Parameters
    ----------
    files : List[str]
      List of grib files to open. Must all correspond to the same type of data.
    output_name : str, optional
      Name of output netcdf file to write. If none specified (default), then no
      output file will be written, just the read in xarray will be returned.
    newgrid : Tuple[np.ndarray], optional
      Tuple in (newlat, newlong) format
       Contains the new grid to mesh to.
    bounding_box : List[float], optional
      Bounding box list in `[long_min, long_max, lat_min, lat_max]` format. By
      default grib datastes from CFSv2 are global.
    Returns
    -------
    data : xarray.Dataset
      xarray.Dataset containing dimensions `(time, latitude, longitude)` with
      data variables corresponding to meteorological forcing data from CFSv2.
    """
    # Open and load data-set
    data = xr.open_mfdataset(files)
    data = data.load()

    # Filter according to passed in args
    if bounding_box is not None:
        data = data.sel(
            latitude=slice(bounding_box[3], bounding_box[2]),
            longitude=slice(bounding_box[0], bounding_box[1]),
        )

    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

    # Write only if necessary
    if outfile is None:
        return data
    if outfile.endswith("221"):
        variables = ["prmsl"]
        forcing_type = "pressure"
    elif outfile.endswith("222"):
        variables = ["u10", "v10"]
        forcing_type = "wind"
    elif outfile.endswith("225"):
        variables = ["siconc"]
        forcing_type = "ice"
    else:
        raise ValueError("Unsupported ADCIRC input file {outfile}")

    latitude = data["latitude"].to_numpy()
    longitude = data["longitude"].to_numpy()
    print(type(latitude), type(longitude))
    print(latitude.reshape((1, len(latitude))))
    time = pd.to_datetime(data["valid_time"].values)

    fmt = "%Y%m%d%H"
    start_date_str, end_date_str = start_date.strftime(fmt), end_date.strftime(fmt)
    owi_cols = 8
    arrs = {v: data[v].to_numpy() for v in variables}

    if latitude[1] < latitude[0]:
        for v in arrs:
            arrs[v] = arrs[v][:, :, ::-1, :]
        latitude = latitude[::-1]

    if "prmsl" in arrs:
        # divide by 100 to convert to mBar
        arrs["prmsl"] /= 100

    if newgrid is not None:
        for v in arrs:
            arrs[v] = regrid(arrs[v], latitude, longitude, *newgrid)
        latitude, longitude = newgrid

    dx, dy = latitude[1] - latitude[0], longitude[1] - longitude[0]
    snap_str = (
        f"iLat={str(len(latitude)).rjust(4)}iLong={str(len(longitude)).rjust(4)}"
        + f"DX={dx:.4f}DY={dy:.4f}"
    )

    snap_str += (
        "SWLAT="
        + f"{latitude[0]:.4f}".rjust(8)
        + "SWLon="
        + f"{longitude[0]:.4f}".rjust(8)
    )

    arrs["latitude"], arrs["longitude"] = latitude, longitude
    arrs["time"] = time

    with open(outfile, "w") as fp:
        fp.write(
            f"NCAR {forcing_type} data".ljust(56)
            + f"{start_date_str}      {end_date_str}\n"
        )
        for i in range(len(time)):
            for j in range(len(time[i])):
                curr_time = time[i][j]
                if curr_time > end_date:
                    return arrs
                elif curr_time < start_date:
                    continue

                time_str = curr_time.strftime("%Y%m%d%H%M")
                fp.write(snap_str + f"DT={time_str}\n")
                for v in variables:
                    arr = arrs[v][i, j]
                    str_vals = [f"{x:.4f}".rjust(10) for x in arr.flatten()]
                    for k in range(owi_cols - 1, len(str_vals), owi_cols):
                        str_vals[k] += "\n"
                    if len(str_vals) % owi_cols:
                        str_vals[-1] += "\n"
                    fp.write("".join(str_vals))

    return arrs


def cfsv2_grib_to_adcirc_netcdf(
    files: List[str],
    data_dir: str = None,
    output_name: str = None,
    bounding_box: List[float] = None,
    date_range: Tuple[str] = None,
):
    """
    CFSv2 Grib Data to ADCIRC netcdf fort.22* metereological forcing files.

    Converts and sub-samples data in time and space from a set of grib files
    that have been downloaded from NCAR's CFSv2 data set (id='ds094.1')

    Parameters
    ----------
    files : List[str]
      List of grib files to open. Must all correspond to the same type of data.
    data_dir : str, optional
      Directory where grib files are location. Defaults to current working
      directory.
    output_e : str, optional
      Name of output netcdf file to write. If none specified (default), then no
      output file will be written, just the read in xarray will be returned.
    bounding_box : List[float], optiontal
      Bounding box list in `[long_min, long_max, lat_min, lat_max]` format. By
      default grib datastes from CFSv2 are global.
    date_range : Tuple[str]
      Date tuple, (start_date, end_date), to be fed into
      `data.sel(time=slice(start_date, end_Date))` to sub-sample `data` along
      the time dimension.

    Returns
    -------
    data : xarray.Dataset
      xarray.Dataset containing dimensiosn `(time, latitude, longitude)` with
      data variables corresponding to meteorological forcing data from CFSv2.

    """
    # Open data-set
    # pdb.set_trace()
    # TODO: Group by ds type and then loop through each type
    # Deal with different ds formats for each type.
    # format for filename is <field>.<type>.<date>.grb2
    # Parse the first two, load and merge into main data set per field,
    # Field will sometimes be further divided with a '.l.'
    # as in 'prmsl.cdas1.201801.grb2' and 'prmsl.l.cdas1.201801.grb2'
    # .l. dataset is coarser data. pick it to start. if no data over grid user finer.
    # ad option to force using finer dataset.
    # data = xr.open_mfdataset(files, engine="cfgrib", lock=False)
    data = xr.open_mfdataset(files)

    # Filter according to passed in args
    if bounding_box is not None:
        long_range = np.array(bounding_box[0:2]) % 360
        data = data.sel(
            latitude=slice(bounding_box[2], bounding_box[3])
        ).sel(
            longitude=slice(long_range[0], long_range[1]),
        )
    if date_range is not None:
        data = data.sel(time=slice(date_range[0], date_range[1]))

    # Data is divided into steps within each time step. Select first
    data = data.isel(step=0)

    # Drop unecessary coordiantes
    coords = ["time", "latitude", "longitude"]
    drop = [x for x in list(data.coords.keys()) if x not in coords]
    for x in drop:
        data = data.drop(x)
    data.load()

    write_job = data.to_netcdf(output_name, compute=False)
    with ProgressBar():
        print(f"Writing to {output_name}:")
        write_job.compute()

    return data


def add_cfsv2_met_forcing(
    ds, files, met_data_dir, start_date=None, end_date=None, out_path=None
):
    """Modifies adcirc run config to add meterological forcing found in met_data"""

    _, bbox = get_bbox(ds)
    met_data = cfsv2_grib_to_adcirc_netcdf(
        files=files, bounding_box=bbox, date_range=[start_date, end_date]
    )

    ds.attrs["RNDAY"] = (
        met_data["time"][-1] - met_data["time"][0]
    ).values / np.timedelta64(1, "D")
    wtiminc = (met_data["time"][1] - met_data["time"][0]).values / np.timedelta64(
        1, "s"
    )

    has_ice = any([str(f).startswith("ice") for f in files])
    ds.attrs["NWS"] = 14014 if has_ice else 14
    ds.attrs["WTIMINC"] = f"{wtiminc}, {wtiminc}" if has_ice else f"{wtiminc}"
    ds.attrs["NOUTM"] = -3
    ds.attrs["TOUTSM"] = 0.0
    ds.attrs["TOUTSM"] = 0.0
    ds.attrs["TOUTFM"] = 0.0
    ds.attrs["NSPOOLM"] = 0.0
    ds.attrs["NSTAM"] = 0.0
    ds.attrs["NCDATE"] = str(pd.to_datetime(met_data["time"][0].values))
    ds["XEM"] = xr.DataArray(
        [], dims=["STATIONS_MET"], coords=dict(STATIONS_MET=(["STATIONS_MET"], []))
    )
    ds["YEM"] = xr.DataArray(
        [], dims=["STATIONS_MET"], coords=dict(STATIONS_MET=(["STATIONS_MET"], []))
    )
    ds = ds.merge(met_data)

    if out_path is not None:
        ds.to_netcdf(out_path)

    return ds


def read_fort24(f24_file, ds):
    """Read fort24 Files, Self Attraction/Earth Load Tide Forcing Files

    Based off documentation in
    https://adcirc.org/home/documentation/users-manual-v51/input-file-descriptions/self-attractionearth-load-tide-forcing-file-fort-24
    Requires a pre-read fort.14 and fort.15 file contained in ds.

    This file exists if NTIP==2

    """
    if type(ds) != xr.Dataset:
        ds = xr.Dataset()

    ln = 1
    salt_data = None
    for i, name in enumerate(ds["BOUNTAG"]):
        tmp, ln = read_param_line(xr.Dataset(), ["AlphaLine"], f24_file, ln=ln)
        tmp, ln = read_param_line(tmp, ["Freq"], f24_file, ln=ln)
        tmp, ln = read_param_line(tmp, ["Dummy"], f24_file, ln=ln, dtypes=[int])
        tmp, ln = read_param_line(tmp, ["Name"], f24_file, ln=ln)
        # TODO check Frequeny and name match BOUNTAG
        if tmp.attrs["Name"] != name.values:
            raise ValueError("Order of Tidal components in f24 does not match f15")
        cols = ["JN", "SALTAMP", "SALTPHA"]
        tmp_df = pd.read_csv(
            f24_file,
            skiprows=ln - 1,
            nrows=ds.attrs["NP"],
            delim_whitespace=True,
            names=cols,
        )
        ln += ds.attrs["NP"]
        tmp_df["NAMEFR"] = name.values
        if i > 0:
            salt_data = salt_data.append(tmp_df)
        else:
            salt_data = tmp_df

    salt_xa = pd.DataFrame(salt_data).set_index(["JN", "NAMEFR"]).to_xarray()
    ds = xr.merge([ds, salt_xa], combine_attrs="override")

    return ds


def write_fort24(ds, f24_file):
    """Write fort24 Files, Self Attraction/Earth Load Tide Forcing Files

    Based off documentation in
    https://adcirc.org/home/documentation/users-manual-v51/input-file-descriptions/self-attractionearth-load-tide-forcing-file-fort-24
    Requires a pre-read fort.14 and fort.15 file contained in ds.

    This file exists if NTIP==2

    """
    Path(f24_file).unlink(missing_ok=True)
    for i, name in enumerate(ds["BOUNTAG"]):
        with open(f24_file, "a") as f24:
            freq = ds["AMIGT"].sel(TIPOTAG=name.values).values
            write_text_line(f"{name.values} SAL", "", f24)
            write_text_line(f"{freq}", "", f24)
            write_text_line("1", "", f24)
            write_text_line(f"{name.values}", "", f24)
        cols = ["JN", "SALTAMP", "SALTPHA"]
        out_df = ds.sel(NAMEFR=name.values).drop_vars("NAMEFR")[cols].to_dataframe()
        out_df.to_csv(f24_file, sep=" ", mode="a", header=None)


def write_fort_22_netcdf_files(ds):
    """ """
    pass


def add_elevation_stations(
    ds,
    stations,
    append=True,
):
    """Add stations to ds configuration for outputting"""
    pass


def sync_output_params(
    ds,
    station_start=None,
    station_end=None,
    station_freq=None,
    global_start=None,
    global_end=None,
    global_freq=None,
):
    """Sets start/stop and timestep for station and global output files for all
    relevant outputs configured currently in `ds`.
    """
    pass
