"""

"""
from taccjm import taccjm as tjm
from pyadcrc.adcirc_utils import *

class ADCIRCSim(object):

    """Docstring for ADCIRCSim. """

    def __init__(self,
            name:str
            
            user:str=None):
        """TODO: to be defined. """

        if jm_id not in tjm.list_jms():
            self.jm =  tjm.init_jm(cid, system, user)
        else :
            self.jm = tjm.get_jm(jm_id)


class TACCADCIRCSim(object):

    """Class for running ADCIRC simulations on TACC resources"""

    def __init__(self, jm_id, system:str=None):
        """TODO: to be defined.

        Parameters
        ----------
        jm_id : str
            TACCJobManager ID to use to connect to TACC systems.
        :system: TODO

        """
        self._jm_id = jm_id
        self._system = system

        if jm_id not in tjm.list_jms():
            self.jm =  tjm.init_jm(cid, system, user)
        else :
            self.jm = tjm.get_jm(jm_id)


    def _get_root_dir(self):
        """Get root dir shared accross all TACC work systems"""



        
