"""_summary_

    Tests for the pyadcirc.io module
"""
import pdb
import debugpy
from pathlib import Path
import os
import numpy as np
import pytest

import xarray as xr
from pyadcirc.io import io as pyio
from dotenv import load_dotenv
from conftest import temp_file_stream

# Use dotenv to load path to adcirc-testsuite repository
load_dotenv()
ADCIRC_TEST_SUITE_PATH = os.environ.get("ADCIRC_TEST_SUITE_PATH")

tests = {'si': 'adcirc_shinnecock_inlet',
         'alaska': 'adcirc_alaska_ice-2d'}


def _get_f14_path(test):
    return os.path.join(
        ADCIRC_TEST_SUITE_PATH, 'adcirc', tests[test], 'fort.14'
)

def test_read_text_line():
    """
    Test the read_param_line function 
    """
    answers = {'si': 'Shinacock Inlet Coarse Grid', 'alaska': 'OceanMesh2D'}

    for test in tests:
        params, ln = pyio.read_text_line({}, "AGRID", _get_f14_path(test), ln=1)

        assert ln == 2
        assert params['AGRID'] == answers[test]


def test_read_param_line():
    """
    Test the read_param_line function 
    """

    answers = {'si': {'NE': 5780, 'NP': 3070}, 'alaska': {'NE': 27757, 'NP': 15876}}
    for test in tests:
        
        params, ln = pyio.read_param_line(
            {}, ["NE", "NP"],
            _get_f14_path(test), ln=2, dtypes=2 * [int])

        assert ln == 3
        assert params['NE'] == answers[test]['NE']
        assert params['NP'] == answers[test]['NP'] 


def test_write_param_line(temp_file_stream):
    # Prepare test data
    params = {'param1': 1.23, 'param2': 4.56}

    # Call the function to write the parameter line
    pyio.write_param_line(params, params.keys(), temp_file_stream)

    # Read the written line from the file stream
    temp_file_stream.seek(0)
    written_line = temp_file_stream.readline().strip()

    # Check if the written line matches the expected format
    assert written_line.startswith('1.23 4.56')
    assert len(written_line) >= 80

    # Check if the parameter names are present
    assert 'param1' in written_line
    assert 'param2' in written_line


def test_write_text_line(temp_file_stream):
    # Prepare test data
    ds = xr.Dataset()
    ds.attrs['param'] = 'Test line'
    param = 'param'

    # Call the function to write the text line
    pyio.write_text_line(ds, param, temp_file_stream)

    # Reset the file stream position to the beginning for reading
    temp_file_stream.seek(0)

    # Read the written line from the file stream
    written_line = temp_file_stream.readline().strip()

    # Check if the written line matches the expected format
    assert written_line.startswith('Test line')
    assert len(written_line) >= 80

    # Check if the parameter name is present
    assert 'param' in written_line


def test_read_fort14_params(temp_file_path):
    """
    Test the read_param_line function 
    """
    answers = {'si': {'AGRID': 'Shinacock Inlet Coarse Grid',
                        'NE': 5780,
                        'NP': 3070,
                        'NOPE': 1,
                        'NETA': 75,
                        'NBOU': 1,
                        'NVEL':285},
                'alaska': {'AGRID': 'OceanMesh2D',
                            'NE': 27757,
                            'NP': 15876,
                            'NOPE': 2,
                            'NETA': 122,
                            'NBOU': 68,
                            'NVEL': 4070}}
    for test in tests:
        params = pyio.read_fort14_params(_get_f14_path(test))

        assert params['AGRID'] == answers[test]['AGRID']
        assert params['NE'] == answers[test]['NE']
        assert params['NP'] == answers[test]['NP']
        assert params['NOPE'] == answers[test]['NOPE']
        assert params['NETA'] == answers[test]['NETA']
        assert params['NBOU'] == answers[test]['NBOU']
        assert params['NVEL'] == answers[test]['NVEL']


def test_read_fort14_node_map(temp_file_path):
    """
    Test the read_param_line function 
    """
    answers = {'si': {'NP': 3070},
                'alaska': {'NP': 15876}}

    for test in tests:
        node_map = pyio.read_fort14_node_map(_get_f14_path(test))

        for key in ['JN', 'X', 'Y', 'DP']:
            assert len(node_map[key]) == answers[test]['NP']
        assert type(node_map['DP'].values[0]) == np.float64

    
def test_read_fort14_element_map(temp_file_path):
    """
    Test the read_param_line function 
    """
    ds = pyio.read_fort14_element_map(test_f14_file)

    assert ds.attrs['AGRID'] == 'Shinacock Inlet Coarse Grid'
    assert ds.attrs['NE'] == 5780
    assert ds.attrs['NE'] == len(ds.NM_1)
    assert ds.attrs['NE'] == len(ds.NM_2)
    assert ds.attrs['NE'] == len(ds.NM_3)


def test_read_fort14_elev_boundary(temp_file_path):
    """
    Test the read_param_line function 
    """
    ds = pyio.read_fort14_params(test_f14_file)
    bounds = pyio.read_fort14_elev_boundary(test_f14_file)

    assert len(bounds['segments']) == ds.attrs['NOPE']
    assert len(bounds['nodes']) == ds.attrs['NETA']


def test_read_fort14_flow_boundary(temp_file_path):
    """
    Test the read_param_line function 
    """
    ds = pyio.read_fort14_params(test_f14_file)
    bounds = pyio.read_fort14_flow_boundary(test_f14_file)

    assert len(bounds['segments']) == ds.attrs['NBOU']
    assert len(bounds['nodes']) == ds.attrs['NVEL']
    assert all(bounds['segments']['IBTYPE'].unique() == [0])


def test_read_fort14(temp_file_path):
    """
    Test the read_param_line function 
    """
    ds = pyio.read_fort14(test_f14_file)

    assert ds.attrs['AGRID'] == 'Shinacock Inlet Coarse Grid'
    assert ds.attrs['NE'] == 5780
    assert ds.attrs['NP'] == 3070


def test_write_fort14(temp_file_path):
    """
    Test the read_param_line function 
    """
    ds = pyio.read_fort14(test_f14_file)
    pyio.write_fort14(ds, temp_file_path)
    ds2 = pyio.read_fort14(temp_file_path)

    assert ds.attrs['AGRID'] == 'Shinacock Inlet Coarse Grid'
    assert ds.attrs['NE'] == 5780
    assert ds.attrs['NP'] == 3070

    assert ds.attrs['AGRID'] == ds2.attrs['AGRID']
    assert ds.attrs['NE'] == ds2.attrs['NE']
    assert ds.attrs['NP'] == ds2.attrs['NP']


def test_read_fort15(temp_file_path):
    """
    Test the read_param_line function 
    """
    ds = xr.Dataset()
    ds.attrs['NETA'] = 75
    ds = pyio.read_fort15(test_f15_file, ds)
    pyio.write_fort15(ds, temp_file_path)
    ds = pyio.read_fort15(test_f15_file, ds)

    assert ds.attrs['AGRID'] == 'Shinacock Inlet Coarse Grid'
    assert ds.attrs['NE'] == 5780
    assert ds.attrs['NP'] == 3070