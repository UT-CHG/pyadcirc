"""

"""
from taccjm import taccjm as tjm
from pyadcrc.adcirc_utils import *

class ADCIRCSimManager(object):

    """Docstring for ADCIRCSim. """

    def __init__(self,
            jm_id:str=None,
            exec_system:str=None,
            user:str=None):
        """TODO: to be defined. """

        if jm_id not in tjm.list_jms():
            self.jm =  tjm.init_jm(cid, system, user)
        else :
            self.jm = tjm.get_jm(jm_id)


