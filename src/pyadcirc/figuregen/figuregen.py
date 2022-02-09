import os
import linecache as lc
import pdb
from pathlib import Path
from pprint import pprint
import tempfile
from typing import Union, List

import click
import colorcet as cc
import matplotlib as mpl
import numpy as np
import pandas as pd
import pyadcirc.adcirc_utils as au
import six
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator
from InquirerPy.validator import EmptyInputValidator, PathValidator
from prettytable import PrettyTable
from pyadcirc.figuregen.fg_config import *
from pyadcirc.figuregen.palettes import *
from pyfiglet import figlet_format

# See if docker, tapis, or taccjm options are installed for use
try:
    import docker

    # Initialize client
    client = docker.from_env()

    images = [str(x) for x in client.images.list()]
    if "<Image: 'georgiastuart/figuregen-serial:latest'>" not in images:
        print("Downloading figuregen image:")
        docker.images.get("georgiastuart/figuregen-serial:latest")

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
FG_DOC_LINK = "https://ccht.ccee.ncsu.edu/figuregen-v-49/"

try:
    from termcolor import colored
except ImportError:
    colored = None


def color_mixer(
    c1, c2, mix=0
):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    """
    Color Mixer

    Returns a color that is a mix of c1 and c2 according to the mix
    ratio `mix` which is between [0,1]. Default `mix=0` returns `c1`,
    and `mix=1` returns `c2`. Mixed color computed according to:

        `(1-mix)*c1 + mix*c2`.

    Parametres
    ----------
    c1 : array-like
        3 element list containg RGB colors as ints from range [0,255].
    c2 : array-like
        3 element list containg RGB colors as ints from range [0,255].
    mix : float
        Value between 0 and 1 determining how much of each color to mix in.

    Return
    ------
    mix : array-like
        3 element list containg RGB colors of mixed color between `c1` and `c2`.

    """
    c1 = np.array(mpl.colors.to_rgb(np.array(c1) / 255.0))
    c2 = np.array(mpl.colors.to_rgb(np.array(c2) / 255.0))
    return [int(x) for x in 255.0 * ((1 - mix) * c1 + mix * c2)]


def get_bbox(f14_file: str, scale_x: float = 0.1, scale_y: float = 0.1):
    """
    Get Long/Lat bounding box containing grid in f14_file.
    Computes bounding box using scale parameters where each bound
    is determined as follows:

        max_bound = max + scale * range
        min_bound = min - scale * range

    Parameters
    ----------
    f14_file : str
        Path to fort.14 ADCIRC grid file.
    scale_x : float, default=0.1
        What percent of total longitude range to add to ends
        of longitude min/max for determining bounding box limits.
    scale_y : float, default=0.1
        What percent of total latitude range to add to ends
        of latitude min/max for determining bounding box limits.


    Returns
    -------
    bbox : List[List[float]]
        Long/lat bounding box list in the form `[west,east,south,north]`.

    """

    f14 = au.read_fort14(f14_file)

    bounds = [
        [f14["X"].values.min(), f14["X"].values.max()],
        [f14["Y"].values.min(), f14["Y"].values.max()],
    ]
    buffs = [
        (bounds[0][1] - bounds[0][0]) * scale_x,
        (bounds[1][1] - bounds[1][0]) * scale_y,
    ]
    bbox = [
        bounds[0][0] - buffs[0],
        bounds[0][1] + buffs[0],
        bounds[1][0] - buffs[1],
        bounds[1][1] + buffs[1],
    ]

    return bounds, bbox


def get_border_spacing(bbox: List[float], num_lat_ints: int = 4, num_lon_ints: int = 6):
    """
    Calculates reasonable values for border annotation spacing and box placement
    for FigureGen plots.

    Parameters
    ----------
    bbox : List[float]
        Bounding box for plot, as computed by `get_bbox()`.
    num_lat_ints : int, default=4
        Approximately the number of black/white intervals to create on
        latitude border edge.
    num_lon_ints: int, default=6
        Approximately the number of black/white intervals to create on
        longitude border edge.

    Returns
    -------
    spacing : tuple
        Tuple of (border_box_spacing, border_annotation_spacing) figuregen
        configuration values.
    """

    lat_deg_spacing = (bbox[3] - bbox[2]) / num_lat_ints
    lon_deg_spacing = (bbox[1] - bbox[0]) / num_lon_ints
    border_box_spacing = round((lat_deg_spacing + lon_deg_spacing)) / 2
    border_annotation_spacing = 2 * border_box_spacing

    return (border_box_spacing, border_annotation_spacing)


def read_inp(input_file: str):
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
    config["contours"] = {}  # contour settings between lines (15,34)
    config["particles"] = {}  # particle settings between lines (34,38)
    config["vectors"] = {}  # vector settings between lines (38,48)

    with open(input_file, "r") as in_f:
        for idx, line in enumerate(in_f):
            if idx > 65:
                break
            if idx in [0, 1, 15, 34, 38, 48]:
                # Lines correspond to blank/comment lines
                continue
            else:
                val_type = FG_config[idx]["type"]
                val = str(line[0:50]).strip()

                def convert(val, val_type):
                    if val_type == int:
                        val = int(val)
                    elif val_type == float:
                        val = float(val)
                    elif val_type == bool:
                        val = False if int(val) == 0 else True
                    elif type(val_type) == tuple:
                        val = tuple(
                            [t(v) for t, v in zip(list(val_type), val.split(","))]
                        )
                    elif type(val_type) == list:
                        # Recurse on each in list until we find one that works
                        found_valid = False
                        for vt in val_type:
                            try:
                                cval = convert(val, vt)
                                found_valid = True
                            except Exception as e:
                                pass
                        if found_valid == False:
                            raise ValueError(
                                f"No conversion found for {val} in {val_type}"
                            )
                    return val

                try:
                    val = convert(val, val_type)
                except Exception as e:
                    print(f"ERROR on line #{1+idx} : {FG_config[idx]['name']} : {val}")
                    print(f"{FG_config[idx]['name']} - {FG_config[idx]['desc']}")
                    raise e

                if idx > 15 and idx < 34:
                    # contour dictionary value
                    config["contours"][FG_config[idx]["name"]] = val
                elif idx > 34 and idx < 38:
                    # contour dictionary value
                    config["particles"][FG_config[idx]["name"]] = val
                elif idx > 38 and idx < 48:
                    # contour dictionary value
                    config["vectors"][FG_config[idx]["name"]] = val
                else:
                    config[FG_config[idx]["name"]] = val

    return config


def pal_from_cc(name: str):
    """
    Get Palette config from a colorcet palette.

    Parameters
    ----------
    name : str,
        Name of colorcet color map.

    Returns
    -------
    pal_config : dict
        Dictionary containing the name, description, and RGB color palette
        to be used for FigureGen runs.

    Notes
    -----
    See https://colorcet.holoviz.org/user_guide/index.html.

    """

    try:
        intervals = getattr(cc, name)
    except AttributeError as ae:
        raise ValueError(f"{name} is not a valid colorcet palette")

    fracs = [[x / float(len(intervals))] for x in list(range(len(intervals)))]
    intervals = [[int(255 * y) for y in mpl.colors.to_rgb(x)] for x in intervals]
    intervals = [x[0] + x[1] for x in list(zip(fracs, intervals))]
    pal_config = {
        "name": name,
        "desc": "Colorcet palette",
        "num_pals": 1,
        "pals": {f"{name}_full": {"range": [0.0, 1.0], "intervals": intervals}},
    }
    return pal_config


# TODO: Support CPT palettes
def read_pal(pal_file: str, int_file: str = None):
    """
    Read and View SMS Color Palette

    Parameters
    ----------
    pal_file : str
      Path to SMS palette file.
    int_file : str, optional
      Path to SMS intervals file.

    Returns
    --------
    config : dict
      Palette dictionary configuration. Prints in table to stdout.


    Notes
    -----
    See https://stackoverflow.com/questions/70519979/printing-with-rgb-background
    for info on printing colored text in python.
    """
    config = {}

    def get_line(f, idx, delim=None):
        line = lc.getline(f, idx).strip()
        if delim is not None:
            line = line.split(delim)
            line = list(filter(("").__ne__, line))
        return line

    config["name"] = get_line(pal_file, 1)
    config["num_pals"] = int(get_line(pal_file, 2))
    config["pals"] = {}
    idx = 3
    for p in range(config["num_pals"]):
        pal = {}
        line = get_line(pal_file, idx, delim=" ")
        name = line[0]
        num_intervals = int(line[1])
        range_line = get_line(pal_file, idx + 1, delim=" ")
        pal["range"] = [float(x) for x in range_line[1:]]
        pal["intervals"] = pd.read_csv(
            pal_file,
            delimiter=" ",
            skiprows=idx + 1,
            nrows=num_intervals,
            index_col=False,
            names=["Lower Bound", "R", "G", "B"],
            skipinitialspace=True,
        )
        pal["intervals"] = pal["intervals"].values.tolist()
        config["pals"][name] = pal
        idx += 2 + num_intervals

    if int_file is not None:
        first_line = get_line(int_file, 1, delim=" ")
        num_ints = int(first_line[0])
        config["intervals"] = pd.read_csv(
            int_file,
            delimiter=" ",
            skiprows=1,
            nrows=num_ints,
            index_col=False,
            names=["Contour Value", "White Flag"],
            skipinitialspace=True,
        )
        config["intervals"] = config["intervals"].values.tolist()

    return config


def view_pal_config(config: dict = None):
    """
    Read and View SMS Color Palette

    Parameters
    ----------
    config: dict
      Python SMS dictionary config

    Returns
    --------

    Notes
    -----
    See https://stackoverflow.com/questions/70519979/printing-with-rgb-background
    for info on printing colored text in python.
    """

    table = PrettyTable()
    if "intervals" not in config.keys():
        table.field_names = ["Palette", "Interval Start", "Color"]
        for pal in config["pals"].keys():
            for p in config["pals"][pal]["intervals"]:
                color = (
                    f"\033[48;2;{int(p[1])};{int(p[2])};{int(p[3])}m          \033[0m"
                )
                table.add_row([pal.ljust(15, " "), p[0], color])
    else:
        table.field_names = ["Palette Interval", "Contour Interval", "Color"]

        # Currently only support
        pals = config["pals"].keys()
        if len(pals) > 1:
            raise ValueError("Currently only support singular palettes")
        pal_name = list(pals)[0]

        # Color pallete defined intervals
        pal_intervals = config["pals"][pal_name]["intervals"]
        num_pal_ints = len(pal_intervals)

        # Contour value intervals
        intervals = config["intervals"]
        num_ints = len(intervals)

        for idx, interval in enumerate(intervals):
            i = 0
            f = float(idx) / float(num_ints)
            while i < (num_pal_ints - 1) and pal_intervals[i + 1][0] < f:
                i += 1
            if interval[1] == 0:
                color = color_mixer(
                    pal_intervals[i][1:4],
                    pal_intervals[i + 1][1:4],
                    mix=f / pal_intervals[i + 1][0],
                )
            else:
                color = [255, 255, 255]
            txt = "          "
            ctxt = f"\033[48;2;{color[0]};{color[1]};{color[2]}m{txt}\033[0m"
            table.add_row([pal_intervals[i][0], interval[0], ctxt])

    print(table)


def write_pal(config: dict, out_file: str):
    """
    Write SMS Color Palette from config to out file

    Parameters
    ----------
    config : dict
      Dictionary sms palette config.
    out_file : str
      File to write palette to.

    Returns
    --------
    """

    ret = {}
    with open(out_file, "w") as pf:
        # Write title line and num palettes
        pf.write(config["name"] + "\n")
        pf.write(str(len(config["pals"].keys())) + "\n")
        for name, pal in config["pals"].items():
            # Write name num_intervals
            num_ints = len(pal["intervals"])
            pf.write(" ".join([name, str(num_ints)]) + "\n")

            # Write range line (not sure why this is incldued but always there)
            pf.write(f"0 {pal['range'][0]} {pal['range'][1]}\n")

            for i in pal["intervals"]:
                pf.write(" ".join([str(x) for x in i]) + "\n")
    ret["palette_file"] = out_file

    if "intervals" in config.keys():
        int_file = Path(out_file).with_suffix(".txt")
        with open(int_file, "w") as int_f:
            # Write first line - Number of intervals
            int_f.write(str(len(config["intervals"])) + "\n")
            for i in config["intervals"]:
                int_f.write(f"{i[0]} {i[1]}\n")
        ret["intervals_file"] = int_file

    return ret


def write_inp(config: dict, out_file: str, **kwargs: dict):
    """Write a FigureGen config file (.inp)

    Parameters
    ----------
    config : dict
      FigureGen configuration dictioanry.
    out_file : str
      Path to FigureGen .inp input file.
    kwargs : dict
      Keyword args are interpreted as to be overrides to config to make.

    Returns
    -------
    config : dict
      Configuration dictionary for FigureGen
    """

    # Update dictionary with keyword arguments
    config.update(kwargs)

    with open(out_file, "w") as out_f:
        for idx, spec in enumerate(FG_config):
            name = spec["name"]
            desc = spec["desc"]

            # Search for config
            val = None
            if idx > 15 and idx < 34:
                if "contours" in config.keys():
                    val = config["contours"].get(name)
            elif idx > 34 and idx < 38:
                if "particles" in config.keys():
                    val = config["particles"].get(name)
            elif idx > 38 and idx < 48:
                if "vectors" in config.keys():
                    val = config["vectors"].get(name)
            else:
                val = config.get(name)

            def convert(val):
                if type(val) == tuple:
                    out_str = ",".join([str(v) for v in val])
                elif type(val) == bool:
                    out_str = "1" if val else "0"
                else:
                    out_str = str(val)
                return out_str

            if "skip" in spec.keys():
                out_str = convert(spec["default"]).ljust(50, " ")
                out_f.write(f"{out_str}\n")
            elif val is None:
                out_str = convert(spec["default"]).ljust(50, " ")
                out_f.write(f"{out_str}! {name} : {desc}\n")
            else:
                out_str = convert(val).ljust(50, " ")
                out_f.write(f"{out_str}! {name} : {desc}\n")

    # Return modified config as it was written
    return config


def bathymetry_plot(
    f14_file: str,
    output_name: str = "bathymetry",
    bounding_box: List[float] = None,
    palette: Union[str, dict] = TopoBlueGreen,
    intervals: List[List[float]] = None,
    **kwargs,
):
    """
    Generate mesh plot from f14 file

    Parameters
    ----------
    f14_file : str
        Path to f14 file
    output_name : str, default='bathymetry'
        Name for output file.
    bounding_box : List[float]
        List with long/lat bounding box. format should be
        [west, east, south, north] bounds. If non specified,
        will be inferred from the fort.14 file (requires f14
        file to be read).
    palette : str, dict
        Palette to use.
    intervals : List[List[float]]
        Contour intervals to use
    kwargs : dict
        Extra keyword arguments will override figuregen config.

    Returns
    -------
    res : dict
        Result of running FigureGen, including path of resulting input
        configurations and image files produced.
    """
    f14_path = Path(f14_file)
    data_dir = f14_path.parents[0].absolute()
    f14_filename = f14_path.name
    config = {"output_filename": output_name, "f14_file": f14_filename}

    # Configure bounding box
    if bounding_box is None:
        _, bounding_box = get_bbox(f14_file)
    config["west_bound"] = bounding_box[0]
    config["east_bound"] = bounding_box[1]
    config["south_bound"] = bounding_box[2]
    config["north_bound"] = bounding_box[3]

    # Configure border annotation spacing
    if (
        "border_box_spacing" not in config.keys()
        and "border_annotation_spacing" not in config.keys()
    ):
        box_s, ann_s = get_border_spacing(bounding_box)
        config["border_box_spacing"] = box_s
        config["border_annotation_spacing"] = ann_s

    # Configure contours
    contours = {}
    contours["fill"] = 1
    contours["lines"] = 0
    contours["file"] = f14_filename
    contours["contour_format"] = "GRID-BATH"
    contours["conversion_factor"] = -1.0
    contours["unit_label"] = "m"

    # Load palette configuration, modify with intervals if necessary
    # and then write to data_dir (where f14_file lives)
    pal_config = {}
    if type(palette) == str:
        if Path(palette).is_file():
            # If string is valid path, then load palette path
            pal_config = read_pal(palette)
        else:
            # Else assume its a cc plaete
            pal_config = pal_from_cc(palette)
    if type(palette) == tuple:
        # If tuple, then path to palette file and intervals file.
        pal_config = read_pal(palette[0], palette[1])
    else:
        pal_config = palette

    if intervals is not None:
        pal_config["intervals"] = intervals

    # Write palette config files
    ret = write_pal(pal_config, str(data_dir / (pal_config["name"] + ".pal")))
    if "intervals_file" in ret.keys():
        contours["palette_file"] = (
            Path(ret["palette_file"]).name,
            Path(ret["intervals_file"]).name,
        )
        contours["palette"] = "SMS+INTERVALS"
    else:
        contours["palette_file"] = Path(ret["palette_file"]).name
        contours["palette"] = "SMS"

    config["contours"] = contours

    # Update extra kwargs for plot
    if kwargs:
        extra_configs = kwargs.keys()
        valid = [e in [x["name"] for x in FG_config] for e in extra_configs]
        if not all(valid):
            invalid = extra_configs[valid]
            ValueError(f"Invalid config specified: {invalid}")

        # Update with optional configurations
        config.update(kwargs)

    try:
        ret = fg_run(config, str(data_dir))
    except Exception as e:
        # TODO: Catch specific exceptions here
        raise e

    return ret


def mesh_plot(
    f14_file: str,
    output_name: str = "mesh",
    bounding_box: List[float] = None,
    boundary_color: str = "Black",
    **kwargs,
):
    """
    Generate mesh plot from f14 file

    Parameters
    ----------
    f14_file : str
        Path to f14 file
    output_name : str, default='mesh'
        Name for output file.
    bounding_box : List[float]
        List with long/lat bounding box. format should be
        [west, east, south, north] bounds. If non specified,
        will be inferred from the fort.14 file (requires f14
        file to be read).
    boundary_color : str, default='Black'
        Color for the mesh boundary.
    kwargs : dict
        Extra keyword arguments will override figuregen config.

    Returns
    -------
    otuput_path : str
        Path to resulting output file
    """

    f14_path = Path(f14_file)
    data_dir = str(f14_path.parents[0].absolute())
    f14_filename = f14_path.name
    config = {
        "output_filename": output_name,
        "f14_file": f14_filename,
        "boundaries": (1, boundary_color),
        "plot_grid": True,
    }

    if bounding_box is None:
        _, bounding_box = get_bbox(f14_file)
    config["west_bound"] = bounding_box[0]
    config["east_bound"] = bounding_box[1]
    config["south_bound"] = bounding_box[2]
    config["north_bound"] = bounding_box[3]

    if (
        "border_box_spacing" not in config.keys()
        and "border_annotation_spacing" not in config.keys()
    ):
        box_s, ann_s = get_border_spacing(bounding_box)
        config["border_box_spacing"] = box_s
        config["border_annotation_spacing"] = ann_s

    if kwargs:
        extra_configs = kwargs.keys()
        valid = [e in [x["name"] for x in FG_config] for e in extra_configs]
        if not all(valid):
            invalid = extra_configs[valid]
            ValueError(f"Invalid config specified: {invalid}")

        # Update with optional configurations
        config.update(kwargs)

    try:
        ret = fg_run(config, data_dir)
    except Exception as e:
        # TODO: Catch specific exceptions here
        raise e

    return ret


def fg_run(
    config: Union[dict, str],
    data_dir: str = None,
    engine: str = "docker",
    save_path: str = None,
    **kwargs,
):
    """
    Run FigureGen

    Parameters
    ----------
    config : dict, str
        FigureGen configuration dictionary or path to figuregen input file.
    data_dir : str, optional
        Path to directory with supporting input/data input files. Defaults
        to current working directory.
    engine : str, default='docker'
        Name of engine to use to run FigureGen. Currently only docker supported.
    save_path : str, optional
        If specified, resulting configuration file used will be saved to this path.
        Pat is relative to the dadta_dir.
    kwargs : dict
        All other keyword arguments will be used to override configuration values.

    Returns
    -------
    output_path : str
        Path to generated figure.

    Notes
    -----
    More documentation on using Docker to run figuregen at
    https://github.com/georgiastuart/FigureGen/blob/containers/CONTAINER-README.md
    """

    cwd = Path(os.getcwd()).parent.resolve()
    data_dir = cwd if data_dir is None else Path(data_dir)

    # Read input configuration file if config is string
    if type(config) == str:
        config = read_inp(config)
        config.update(kwargs)

    # Modify accordingly those variables that when specified
    # Need to be actually tuples with (1, Value)
    if "title" in config.keys():
        if type(config["title"]) == str:
            config["title"] = (1, config["title"])

    # Write FigureGen File
    save_path = "mesh.inp" if save_path is None else save_path
    fg_config = write_inp(config, str(data_dir / save_path), **kwargs)

    # Write
    ret = {}
    ret["confing_path"] = data_dir / save_path
    ret["image_path"] = data_dir / f"{config['output_filename']}0001.jpg"
    ret["fg_config"] = fg_config
    if engine == "docker":
        if docker_flag:
            # Run code in container
            dock = {}
            dock["cmd"] = f'bash -c "cd /mnt/data && figuregen -I {save_path}"'
            dock["vols"] = {data_dir: {"bind": "/mnt/data", "mode": "rw"}}
            res = client.containers.run(
                "georgiastuart/figuregen-serial",
                command=dock["cmd"],
                volumes=dock["vols"],
            )

            dock["res"] = res.decode("utf-8")
            ret["docker"] = dock
        else:
            raise ValueError("Docker not found. Install docker.")

    else:
        raise NotImplementedError("Only docker engine supported.")

    return ret


@click.command()
def config():

    config = {}

    # TODO: if does not end in '_' add one.
    config["out_name"] = inquirer.text(message="Base name for output images:").execute()

    config["out_format"] = inquirer.select(
        message="Choose format of output files:",
        choices=["PNG", "JPG", "BMP", "TIFF", "EPS", "PDF"],
        default="JPG",
    ).execute()

    config["f14_file"] = inquirer.filepath(
        message="Enter path to fort.14 ADCIRC grid file:",
        default="fort.14",
        validate=PathValidator(is_file=True, message="Input is not a file"),
        only_files=True,
    ).execute()

    auto_config = inquirer.confirm(
        message="Read fort.14 file to auto-configure bounding box?", default=True
    ).execute()

    while auto_config:
        scale_x = inquirer.number(
            message="Bounding box longitude scale factor:",
            float_allowed=True,
            validate=EmptyInputValidator(),
            default=0.1,
        ).execute()
        scale_y = inquirer.number(
            message="Bounding box latitude scale factor:",
            float_allowed=True,
            validate=EmptyInputValidator(),
            default=0.1,
        ).execute()

        pprint("Reading fort.14 file...")
        bounds, bbox = get_bbox(config["f14_file"])
        pprint("Configured bounding box:")
        pprint(bbox)
        pprint("For grid in bounds:")
        pprint(bounds)

        ok = inquirer.confirm(
            message="OK with lat/lon box (no for retry or manual entry)", default=True
        ).execute()
        if not ok:
            retry = inquirer.confirm(
                message="Retry auto-config or manual entry?", default=True
            ).execute()
            if not retry:
                auto_config = False
        else:
            config["bbox"] = bbox
            auto_config = False

        if "bbox" not in config.keys():
            if bbox is None:
                bbox = [[0.0, 0.0], [0.0, 0.0]]
            bbox[0][0] = inquirer.number(
                message="Enter western longitude bound",
                float_allowed=True,
                validate=EmptyInputValidator(),
                default=bbox[0][0],
            ).execute()
            bbox[0][1] = inquirer.number(
                message="Enter eastern longitude bound",
                float_allowed=True,
                validate=EmptyInputValidator(),
                default=bbox[0][1],
            ).execute()
            bbox[1][0] = inquirer.number(
                message="Enter southern latitude bound",
                float_allowed=True,
                validate=EmptyInputValidator(),
                default=bbox[1][0],
            ).execute()
            bbox[1][1] = inquirer.number(
                message="Enter northern latitude bound",
                float_allowed=True,
                validate=EmptyInputValidator(),
                default=bbox[1][1],
            ).execute()
            config["bbox"] = bbox

    plot_choices = [
        Separator("= Plot Options ="),
        Choice("title", name="Title", enabled=True),
        Separator("= Data ="),
        Choice("mesh", name="Mesh Grid", enabled=True),
        Choice("contours", name="Contours", enabled=True),
        Choice("particles", name="Particle Data", enabled=False),
        Choice("vectors", name="Vector Data", enabled=False),
        Choice("hurricane", name="Hurricane Track", enabled=False),
        Separator("= Map Elements ="),
        Choice("boundaries", name="Boundaries", enabled=False),
        Choice("coastline", name="Coastlines", enabled=False),
        Choice("labels", name="Labels", enabled=False),
        Choice("logo", name="Logo", enabled=False),
        Choice("background_image", name="Background Image", enabled=False),
        Choice("geo_ref", name="Geo-Referencing", enabled=False),
        Choice("kmz", name="Google KMZ", enabled=False),
    ]

    plot_options = inquirer.checkbox(
        message="Select Plotting Options:",
        choices=plot_choices,
    ).execute()

    if "title" in plot_options:
        config["title"] = inquirer.text(message="Title for plot: ").execute()
        # TODO: Configure a default title?
    if "mesh" in plot_options:
        config["plot_grid"] = True
    if "contours" in plot_options:
        contours = {}
        # TODO: Validate that passed in file is valid for contour generating
        contours["input"] = inquirer.filepath(
            message="Contour data file path:",
            default="fort.14",
            validate=PathValidator(is_file=True, message="Input is not a file"),
            only_files=True,
        ).execute()
        # TODO: Infer this?
        contours["file_format"] = inquirer.select(
            message="Select contour format:",
            choices=[
                Choice("ADCIRC-OUTPUT", name="ADCIRC Output File"),
                Choice("GRID-BATH", name="ADCIRC Grid Bathymetry File (fort.14)"),
                Choice("SIZE", name="Size (Not Supported)"),
                Choice("DECOMP-#", name="Decomposition (Not Supported)"),
                Choice("NODAL-ATT", name="Nodal Attribute (fort.13)"),
            ],
            multiselect=False,
        ).execute()
        contours["contour_lines"] = inquirer.confirm(
            message="Plot contour lines?", default=True
        ).execute()
        if not config["plot_grid"]:
            # Only offer option to fill contours if grid isn't being plotted
            contours["contour_fill"] = inquirer.confirm(
                message="Fill contour intervals?", default=True
            ).execute()
        else:
            contours["contour_fill"] = False

        if not contours["contour_fill"] and not contours["contour_lines"]:
            contours["contour_mesh_lines"] = inquirer.confirm(
                message="Plot on mesh grid?", default=True
            )
            if contours["contour_mesh_lines"]:
                contours["contour_mesh_lines_format"] = inquirer.select(
                    message="Select data to use for coloring mesh lines:",
                    choices=[
                        Choice("DEFAULT", name="No Coloring", enabled=True),
                        Choice("CONTOUR-LINES", name="Contour Data File"),
                        Choice(
                            "GRID-BATH", name="ADCIRC Grid Bathymetry File (fort.14)"
                        ),
                        Choice("SIZE", name="Size (Not Supported)"),
                        Choice("DECOMP-#", name="Decomposition (Not Supported)"),
                    ],
                    multiselect=False,
                ).execute()
        config["contours"] = contours
    if "Particle Data" in plot_options:
        pass
    if "Vector Data" in plot_options:
        pass
    if "Hurricane Track" in plot_options:
        pass
    if "Boundaries" in plot_options:
        pass
    if "Coastline" in plot_options:
        pass
    if "Labels" in plot_options:
        pass
    if "Logo" in plot_options:
        pass
    if "Background Image" in plot_options:
        pass
    if "Geo-Referencing" in plot_options:
        pass
    if "Google KMZ" in plot_options:
        pass

    pprint(config)
    pdb.set_trace()
    pprint("Test")


@click.group()
def cli():
    if colored:
        six.print_(colored(figlet_format("FIGUREGEN", font="slant"), "blue"))
    else:
        six.print_(figlet_format("FIGUREGEN", font="slant"))
    pass


@click.command()
@click.argument("input_file", type=click.File("r"))
def run(type: str, input_file: str):
    pass


cli.add_command(config)
