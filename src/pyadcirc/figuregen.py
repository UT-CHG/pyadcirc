import pyadcirc.adcirc_utils as au

class FigureGenConfig(object):


  def __init__(self, config_dict:dict, config_file:str=None):
    """Initialize from dictionary and/or file. Prompt for missing params"""
    self.config = config_dict
    self.config_file = config_file

    if self.config_file is not None:
      # Note we udpate the config file with what was passed in config dict
      config = self.read_config_file(self.config_file)
      config.update(self.config)
      self.confg = config

  @classmethod
  def read_config_file(cls, file:str):
    """Read a FigureGen config file (.inp)

    :file: TODO
    :returns: TODO

    """

    with open(file, 'r') as fp:
      for idx,line in enumerate(fp):
        if idx in [0,1,49]:
          continue
w fg_confi




