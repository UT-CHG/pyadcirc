"""
database.py

Implements Database for ADCIRC Data to be stored and accessed from either:
    (1) TACC directory
    (2) Tapis storage system
    (3) Local storage
"""


from taccjm import taccjm as tjm


class ADCIRCDataBase(object):
    """Class defining ADCIRC DataBase on TACC System"""

    def __init__(self, jm_id:str, root:str, system='ls6'):
        """TODO: to be defined. """

        if jm not in [j["jm_id"] for j in tjm.list_jms()]:
            tjm.init_jm(jm, system=system)






