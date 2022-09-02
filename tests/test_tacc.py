import pdb
import os
import pytest
from dotenv import load_dotenv

from pyadcirc.sim.tacc import init_system, ADCIRCSim, ADCIRCDB

load_dotenv()

global SYSTEM, USER, PW, SYSTEM, ALLOCATION
USER = os.environ.get("TACCJM_USER")
PW = os.environ.get("TACCJM_PW")
SYSTEM = os.environ.get("TACCJM_SYSTEM")
ALLOCATION = os.environ.get("TACCJM_ALLOCATION")
GLOBUS_ID = os.environ.get("GLOBUS_ID")

__author__ = "Carlos del-Castillo-Negrete"
__copyright__ = "Carlos del-Castillo-Negrete"
__license__ = "MIT"


# def test_sim():
#     """API Tests"""
#     sim = ADCIRCSim('test', 'test-sim', system=SYSTEM, user=USER, psw=PW)
#     pdb.set_trace()

def test_db_ncar_list_files():
    """API Tests"""
    init_system('test', SYSTEM, user=USER, psw=PW)
    db = ADCIRCDB('test', GLOBUS_ID)
    files = db.list_ncar_data("ds094.1", ["prmsl.*", "wnd10m.*"], "2018-01-01", "2018-01-31")
    assert 'prmsl.cdas1.201801.grb2' in [f['name'] for f in files]
    assert 'wnd10m.cdas1.201801.grb2' in [f['name'] for f in files]
    assert 'prmsl.l.cdas1.201801.grb2' in [f['name'] for f in files]
    assert 'wnd10m.l.cdas1.201801.grb2' in [f['name'] for f in files]

def test_db_ncar_data_transfer():
    """API Tests"""
    init_system('test', SYSTEM, user=USER, psw=PW)
    db = ADCIRCDB('test', GLOBUS_ID)
    db.init_ncar_data_transfer("ds094.1", ["prmsl.cdas1", "wnd10m.cdas1"], "2018-01-01", "2018-02-01")
    res = db.check_ncar_data_transfer()
    assert 'SUCCEEDED' not in [r['code'] for r in res]
    time.sleep(10)
    res = db.check_ncar_data_transfer()
    assert 'SUCCEEDED' in [r['code'] for r in res]


def test_db_compile():
    """API Tests"""
    init_system('test', SYSTEM, user=USER, psw=PW)
    db = ADCIRCDB('test')
    db.compile_ADCIRC()
    execs = db.list_ADCIRC_execs()
    pdb.set_trace()


