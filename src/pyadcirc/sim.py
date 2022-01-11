"""

"""
from taccjm import taccjm as tjm
from pyadcrc.adcirc_utils import *

class ADCIRCSimManager(object):

    """Docstring for ADCIRCSim. """

    def __init__(self,
            cid:str=None
            system:str=None,
            user:str=None):
        """TODO: to be defined. """

        if cid not in tjm.list_conns():
            conn = tjm.init_conn(cid, system, user)


    def deploy_configs(

            ):

