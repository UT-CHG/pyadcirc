import pdb
import sys
import six
import click
from pyfiglet import figlet_format
from typing import List
from pprint import pprint as pp
import pyadcirc.adcirc_utils as au
from pyadcirc.fg_config import FG_config
from PyInquirer import style_from_dict, Token, prompt, Separator
from pprint import pprint

# See if docker, tapis, or taccjm options are installed for use
try:
  import docker
  docker_flag = True
except ModuleNotFoundError:
  docker_flag = False
try:
  import tapis
  tapis_flag = True
except ModuleNotFoundError:
  tapis_flag = False
try:
  import taccjm
  taccjm_flag = True
except ModuleNotFoundError:
  taccjm_flag = False

# Link to documentation online about FigureGen
FG_DOC_LINK = 'https://ccht.ccee.ncsu.edu/figuregen-v-49/'

try:
    from termcolor import colored
except ImportError:
    colored = None


@click.group()
def cli():
  if colored:
    six.print_(colored(figlet_format('FIGUREGEN', font='slant'), 'blue'))
  else:
    six.print_(figlet_format('FIGUREGEN', font='slant'))
  pass


@click.command()
@click.argument('input_file', type=click.File('r'))
def read(input_file:str):
  """Read a FigureGen config file (.inp)

  Parameters
  ----------
  input_file : str
    Path to FigureGen .inp input file.

  Returns
  -------
  config : dict
    Configuration dictionary for FigureGen
  """

  config = {}
  config['contours'] = {}         # contour settings between lines (15,34)
  config['particles'] = {}        # particle settings between lines (34,38)
  config['vectors'] = {}          # vector settings between lines (38,48)

  for idx,line in enumerate(input_file):
    if idx in [0,1,15,34,38,48]:
      # Lines correspond to blank/comment lines
      continue
    else:
      val_type = FG_config[idx]['type']
      val = str(line[0:50]).strip()
      if val_type==int:
        val = int(val)
      elif val_type==float:
        val = float(val)
      elif val_type==bool:
        val = False if int(val)==0 else True
      elif type(val_type)==tuple:
        val = tuple([t(v) for t,v in zip(list(val_type),val.split(','))])
      else:
        pass

      if idx>15 and idx<34:
        # contour dictionary value
        config['contours'][FG_config[idx]['name']] = val
      elif idx>34 and idx<38:
        # contour dictionary value
        config['particles'][FG_config[idx]['name']] = val
      elif idx>38 and idx<48:
        # contour dictionary value
        config['vectors'][FG_config[idx]['name']] = val
      else:
        config[FG_config[idx]['name']] = val

  pp(config)
  return config


@click.command()
@click.argument('config', type=dict)
def write(config:dict):
  """Read a FigureGen config file (.inp)

  Parameters
  ----------
  input_file : str
    Path to FigureGen .inp input file.

  Returns
  -------
  config : dict
    Configuration dictionary for FigureGen
  """

  config = {}
  config['contours'] = {}         # contour settings between lines (15,34)
  config['particles'] = {}        # particle settings between lines (34,38)
  config['vectors'] = {}          # vector settings between lines (38,48)

  for idx,line in enumerate(input_file):
    if idx in [0,1,15,34,38,48]:
      # Lines correspond to blank/comment lines
      continue
    else:
      val_type = FG_config[idx]['type']
      val = str(line[0:50]).strip()
      if val_type==int:
        val = int(val)
      elif val_type==float:
        val = float(val)
      elif val_type==bool:
        val = False if int(val)==0 else True
      elif type(val_type)==tuple:
        val = tuple([t(v) for t,v in zip(list(val_type),val.split(','))])
      else:
        pass

      if idx>15 and idx<34:
        # contour dictionary value
        config['contours'][FG_config[idx]['name']] = val
      elif idx>34 and idx<38:
        # contour dictionary value
        config['particles'][FG_config[idx]['name']] = val
      elif idx>38 and idx<48:
        # contour dictionary value
        config['vectors'][FG_config[idx]['name']] = val
      else:
        config[FG_config[idx]['name']] = val

  pp(config)
  return config




@click.command()
def config():

  # Promp styles
  style = style_from_dict({
      Token.Separator: '#cc5454',
      Token.QuestionMark: '#673ab7 bold',
      Token.Selected: '#cc5454',  # default
      Token.Pointer: '#673ab7 bold',
      Token.Instruction: '',  # default
      Token.Answer: '#f44336 bold',
      Token.Question: '',
  })

  questions = [
       {
            'type': 'input',
            'name': 'output_filename',
            'message': 'Base name for output images:',
        },
       {
        'type': 'list',
        'name': 'output_format',
        'message': 'Choose format of output files',
        'choices': ['PNG', 'JPG', 'BMP', 'TIFF', 'EPS', 'PDF'],
        'default': 'PNG'
        },
       {
            'type': 'input',
            'name': 'f14_file',
            'message': 'Enter path to fort.14 ADCIRC grid file:',
            'default': 'fort.14'
        },
       {
            'type': 'confirm',
            'name': 'auto_config_bbox',
            'message': 'Read fort.14 file to auto-configure bounding box?',
            'default': 'True'
        }
  ]
  config = prompt(questions, style=style)
  auto_config_bbox = config.pop('auto_config_bbox')

  while auto_config_bbox:
    auto_config_questions = [
         {
              'type': 'input',
              'name': 'scale_x',
              'message': 'Bounding box longitude scale factor:',
              'default': '0.1',
              'filter': lambda val: float(val)
          },
         {
              'type': 'input',
              'name': 'scale_y',
              'message': 'Bounding box latitude scale factor:',
              'default': '0.1',
              'filter': lambda val: float(val)
          }
    ]
    auto_config = prompt(auto_config_questions, style=style)
    pprint("Reading fort.14 file...")
    f14 = au.read_fort14(config['f14_file'])
    bounds = [[f14['X'].values.min(), f14['X'].values.max()],
        [f14['Y'].values.min(), f14['Y'].values.max()]]
    buffs = [(bounds[0][1]-bounds[0][0])*auto_config['scale_x'],
             (bounds[1][1]-bounds[1][0])*auto_config['scale_y']]
    bbox = [bounds[0][0]-buffs[0],bounds[0][1]+buffs[0],
            bounds[1][0]-buffs[1],bounds[1][1]+buffs[1]]
    pprint("Configured bounding box:")
    pprint(bbox)
    pprint("For grid in bounds:")
    pprint(bounds)

    ok = [
         {
              'type': 'confirm',
              'name': 'continue',
              'message': 'OK with lat/lon box (no for retry or manual entry)',
              'default': 'Y',
          }
    ]
    ok_res = prompt(ok, style=style)
    if not ok_res['continue']:
      retry = [
         {'type': 'confirm',
          'name': 'retry',
          'message': 'Retry auto-config or manual entry?',
          'default': 'Y',
         }
      ]
      retry_res = prompt(retry, style=style)
      if not_retry_res['retry']:
        auto_config_bbox = False
    else:
      config['bbox'] = bbox
      auto_config_bbox = False

  if 'bbox' not in config.keys():
    bbox_questions = [
         {
              'type': 'input',
              'name': 'west_bound',
              'message': 'Enter western longitude bound',
              'filter': lambda val: float(val)
          },
         {
              'type': 'input',
              'name': 'east_bound',
              'message': 'Enter eastern longitude bound',
              'filter': lambda val: float(val)
          },
         {
              'type': 'input',
              'name': 'north_bound',
              'message': 'Enter northern latitude bound',
              'filter': lambda val: float(val)
          },
         {
              'type': 'input',
              'name': 'south_bound',
              'message': 'Enter southern latitude bound',
              'filter': lambda val: float(val)
          },
    ]
    config.update(prompt(auto_config_questions, style=style))


  options_prompt = {
   'type': 'checkbox',
   'message': 'Select plot options',
   'name': 'plot_opts',
   'choices': [
     Separator('= Plot Options ='),
     {'name': 'Title', 'checked': True},
     Separator('= Data ='),
     {'name': 'Mesh Grid','checked': True},
     {'name': 'Contours'},
     {'name': 'Particle Data'},
     {'name': 'Vector Data'},
     {'name': 'Hurricane Track'},
     Separator('= Map Elements ='),
     {'name': 'Boundaries'},
     {'name': 'Coastline'},
     Separator('= Other ='),
     {'name': 'Labels'},
     {'name': 'Logo'},
     {'name': 'Background Image'},
     Separator('= Output Options ='),
     {'name': 'Geo-Referencing'},
     {'name': 'Google KMZ'}
   ],
   'validate': lambda answer: True
  }
  selected_options = prompt(options_prompt, style=style)['plot_opts']

  pdb.set_trace()
  if 'Title' in selected_options:
    title_input_prompt = {
          'type': 'input',
          'name': 'title',
          'message': 'Title for plot:'
    }
    config.update(prompt(title_input_prompt, style=style))
  if 'Mesh Grid' in selected_options:
    config['plot_grid'] = True
  if 'Contours' in selected_options:
    contour_questions = [
        {
         'type': 'input',
         'name': 'file',
         'message': 'Contour data file path',
         'choices': [ 'ADCIRC-OUTPUT', 'GRID-BATH', 'SIZE', 'DECOMP-#',
           '13-MANNING','CANOPY','TAU0','EVIS','WIND-REDUCTION(-#)'],
         'default': 'GRID-BATH'
        },
        {
         'type': 'list',
         'name': 'file_format',
         'message': 'Choose format of contour data file',
         'choices': [ 'ADCIRC-OUTPUT', 'GRID-BATH', 'SIZE', 'DECOMP-#',
           '13-MANNING','CANOPY','TAU0','EVIS','WIND-REDUCTION(-#)'],
         'default': 'GRID-BATH'
        },
        {
          'type': 'confirm',
          'name': 'contour_lines',
          'message': 'Plot contour lines?',
          'default': 'Y',
        },
        {
          'type': 'confirm',
          'name': 'contour_fill',
          'message': 'Fill contour intervals?',
          'default': 'Y',
        }
     ]

    c_opts = prompt(contour_questions, style=style)
    config['contour']['file'] = c_opts['file']
    config['contour']['format'] = c_opts['file_format']

    if c_opts['contour_lines']:
      c_lines_question = [
            {'type': 'confirm',
             'name': 'contour_lines',
             'message': 'Plot colored lines?',
             'default': 'Y'}]
      c_lines_opts = prompt(c_lines_question, style=style)
      if c_lines_opts['contour_lines']:
        c_lines_opts_question = [
            {'type': 'list',
             'name': 'contour_lines',
             'message': 'colored lines?',
             'choices': ['DEFAULT', 'CONTOUR-LINES',
                 'GRID-BATH', 'SIZE', 'DECOMP-#'],
             'default': 'DEFAULT'}]
        c_lines_opts = prompt(c_lines_opts_question, style=style)
        config['contour']['color_lines'] = c_lines_opts['contour_lines']
    pdb.set_trace()
  if 'Particle Data' in selected_options:
    pass
  if 'Vector Data' in selected_options:
    pass
  if 'Hurricane Track' in selected_options:
    pass
  if 'Boundaries' in selected_options:
    pass
  if 'Coastline' in selected_options:
    pass
  if 'Labels' in selected_options:
    pass
  if 'Logo' in selected_options:
    pass
  if 'Background Image' in selected_options:
    pass
  if 'Geo-Referencing' in selected_options:
    pass
  if 'Google KMZ' in selected_options:
    pass

  pp(config)
  pdb.set_trace()
  pp("here")


@click.command()
@click.argument('input_file', type=click.File('r'))
def run(type:str, input_file:str):
  pass


cli.add_command(read)
cli.add_command(config)

