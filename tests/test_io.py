import pdb
from pathlib import Path
import os
import pytest

from pyadcirc.io import io as pyio

__author__ = "Carlos del-Castillo-Negrete"
__copyright__ = "Carlos del-Castillo-Negrete"
__license__ = "MIT"

ak_test_path = '/Users/carlos/repos/adcirc-testsuite/adcirc/adcirc_alaska_ice-2d/'

@pytest.fixture
def test_f14():
    f15_path = str(Path(__file__).parents[1] / 'data/adcirc/inputs/SI-test/fort.14')
    yield f15_path

@pytest.fixture
def test_f15():
    f15_path = str(Path(__file__).parents[1] / 'data/adcirc/inputs/SI-test/fort.15')
    yield f15_path

@pytest.fixture
def ak_f14():
    f14_path = str(Path(__file__).parents[1] / 'data/adcirc/inputs/AK/fort.14')
    yield f14_path

@pytest.fixture
def ak_f15():
    f15_path = str(Path(__file__).parents[1] / 'data/adcirc/inputs/AK/fort.15')
    yield f15_path

@pytest.fixture
def ak_f24():
    f24_path = str(Path(__file__).parents[1] / 'data/adcirc/inputs/AK/fort.24')
    yield f24_path

@pytest.fixture
def test_out_f24():
    test_out_f24 = Path(__file__).parents[1] / 'data/adcirc/inputs/AK/out.fort.24'
    yield test_out_f24
    test_out_f24.unlink(missing_ok=True)

@pytest.fixture
def test_out_f15():
    test_out_f15 = Path(__file__).parents[1] / 'data/adcirc/inputs/AK/out.fort.15'
    yield test_out_f15
    test_out_f15.unlink(missing_ok=True)

@pytest.fixture
def test_ncar_data():
    test_ncar_path = Path(__file__).parents[1] / 'data/adcirc/ncar_data'
    yield test_ncar_path
    for p in test_ncar_path.iterdir():
        if str(p).endswith('.idx'):
            p.unlink()

@pytest.fixture
def test_adcirc_nc():
    test_adcirc_path = Path(__file__).parents[1] / 'data/adcirc/inputs/test_adcirc_inputs.nc'
    yield test_adcirc_path
    test_adcirc_path.unlink(missing_ok=True)

def test_modify_f15(test_f14, test_f15):
    """Test Modifying an f15 file"""
    ds = pyio.read_fort14(test_f14)
    ds = pyio.read_fort15(test_f15, ds=ds)
    pdb.set_trace()


def test_f24(ak_f14, ak_f15, ak_f24, test_out_f24):
    """Test Modifying an f15 file"""
    ds = pyio.read_fort14(ak_f14)
    ds = pyio.read_fort15(ak_f15, ds=ds)
    ds = pyio.read_fort24(ak_f24, ds=ds)

    pyio.write_fort24(ds, str(test_out_f24))

    ds2 = pyio.read_fort14(ak_f14)
    ds2 = pyio.read_fort15(ak_f15, ds=ds)
    ds2 = pyio.read_fort24(str(test_out_f24), ds=ds)

    assert (ds['SALTAMP'] == ds2['SALTAMP']).all()
    assert (ds['SALTPHA'] == ds2['SALTPHA']).all()

def test_wtiminc(test_out_f15):
    """Test modifying WTIMINC. In particualr when pair value present"""
    ds = pyio.read_fort14(str(Path(ak_test_path ) / 'fort.14'))
    ds = pyio.read_fort15(str(Path(ak_test_path ) / 'fort.15'), ds)

    pyio.write_fort15(ds, str(test_out_f15))

    ds2 = pyio.read_fort14(str(Path(ak_test_path ) / 'fort.14'))
    ds2 = pyio.read_fort15(str(test_out_f15), ds2)
    pdb.set_trace()

def test_add_cfsv2(test_f14, test_f15, test_ncar_data, test_adcirc_nc):
    """Test adding met forcing data (wind and pressure) to a dataset"""
    ds = pyio.read_fort14(test_f14)
    ds = pyio.read_fort15(test_f15, ds=ds)

    files = list(test_ncar_data.iterdir())
    ds = pyio.add_cfsv2_met_forcing(ds, files, str(test_ncar_data),
                                    start_date='2018-01-01 00:00:00',
                                    end_date='2018-01-05 00:00:00',
                                    out_path=str(test_adcirc_nc))
    pdb.set_trace()

