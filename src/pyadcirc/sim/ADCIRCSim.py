# from mpi4py import MPI
# import h5py
import time

from pathlib import Path
import numpy as np
from taccjm.log import logger
from taccjm.sim.TACCSimulation import TACCSimulation


class LocalADCIRCSimulation(object):
    """
    ADCIRC Simulation Class

    Base class with minimal parameters for a 2D ADCIRC run

    Attributes
    ----------
    f14 : str
        Path to fort.14 (Grid and Boundary conditions) file. Checks if file exists upon setting.
    f15 : str
        Path to fort.15 (Control and Tide Configurations) file. Checks if file exists upon setting.

    """

    def __init__(
        self, dir_path: str = None, f14_path: str = "fort.14", f15_path: str = "fort.15"
    ):
        dir_path = Path(dir_path) if dir_path is not None else Path.cwd()
        self.f14 = dir_path / f14_path
        self.f15 = dir_path / f15_path
        self.grid = ADCIRCMesh(self.f14)
        self.control_params = {}

    def _get_param(self, param_name, line, type="text"):
        """ """
        if param_name not in self.control_params:
            self.control_params[param_name] = read_text_line(
                self.control_params, param_name, self.f15, ln=line
            )
        return self.control_params[param_name]

    @property
    def f14(self):
        return self.source_files["14"]

    @f14.setter
    def f14(self, f14_path):
        if not Path(f14_path).exists():
            raise FileNotFoundError(
                f"ADCIRC fort.14 (Grid and Boundary conditions) file at {str(f14_path)} not found"
            )
        self.f14 = f14_path

    @property
    def f15(self):
        return self.source_files["15"]

    @f15.setter
    def f15(self, f15_path):
        if not Path(f15_path).exists():
            raise FileNotFoundError(
                f"ADCIRC fort.14 (Model Parameter and Periodic Boundary Conditions) file at {str(f15_path)} not found"
            )
        self.f15 = f15_path

    @property
    def RUNDES(self):
        """
        Run Description

        Alpha-numeric string of up to 80 characters describing the run.
        """
        return self._get_param("RUNDES", 1)

    @RUNDES.setter
    def RUNDES(self, value):
        """
        Set Run Description

        Ensures run description is < 80 characters before setting.
        """
        if not isinstance(value, str):
            raise TypeError(f"RUNDES must be of type str")
        if len(value) >= 80:
            raise ValueError(f"RUNDES must be 80 characters or less")
        self.control_params["RUNDES"] = value

    @property
    def RUNID(self):
        """
        Run ID

        Alpha-numeric string of up to 8 characters identifying the run.
        """
        return self._get_param("RUNID", 2)

    @RUNID.setter
    def RUNID(self, value):
        """
        Set Run ID

        Ensures run ID is < 80 characters before setting.

        TODO: Enfore ID requirement?
        """
        if not isinstance(value, str):
            raise TypeError("RUNID must be of type str")
        if len(value) >= 80:
            raise ValueError("RUNID must be 80 characters or less")
        self.control_params["RUNID"] = value

    @property
    def ICS(self):
        """
        Coordinate System Paremeter. 1 for Carteisan, 2 for Spherical.

        Specifies the initial conditions for the run.
        """
        return self._get_param("ICS", 8)

    @ICS.setter
    def ICS(self, value):
        """
        Set ICS

        Ensures ICS is 1 or 2 before setting.
        """
        if not isinstance(value, int):
            raise TypeError("ICS must be of type int")
        if value not in [1, 2]:
            raise ValueError("ICS must be 1 (Cartesian) or 2 (Spherical)")
        # TODO: Does the value here have to match the fort.14 grid?
        self.control_params["ICS"] = value


class BaseADCIRCSimulation(TACCSimulation):
    """
    Base ADCIRC Simulation Class

    For running a singular ADCIRC Simulation on TACC systems.
    """
    JOB_DEFAULTS = {
        "allocation": None,
        "node_count": 1,
        "processors_per_node": 12,
        "max_run_time": 0.1,
        "queue": "development",
        "dependencies": [],
    }

    # These are file/folder inputs needed to run the simulation
    ARGUMENTS = [
        {
            "name": "input_dir",
            "type": "argument",
            "label": "N",
            "desc": "Path on compute system where input files are located.",
            "default": "/work/06307/clos21/pub/adcirc/inputs/ShinnecockInlet/mesh/def",
        },
        {
            "name": "exec_dir",
            "type": "argument",
            "label": "Channels",
            "desc": "Number of channels to use to write the array.",
            "default": "/work/06307/clos21/pub/adcirc/execs/ls6/v56_beta/",
        },
        {
            "name": "wp",
            "type": "argument",
            "label": "Channels",
            "desc": "Number of write processes to use.",
            "defaultf": 0,
        },
    ]

    # TODO: Base environment config? for TACC simulation
    BASE_ENV_CONFIG = {
        "modules": ["remora"],
        "conda_packages": "pip",
        "pip_packages": "git+https://github.com/cdelcastillo21/taccjm.git@0.0.5-improv",
    }

    ENV_CONFIG = {
        "conda_env": "pyadcirc",
        "modules": ["netcdf"],
        "conda_packages": ["mpi4p", "h5py", "cfgrib"],
        "pip_packages": ["pyadcirc"],
    }

    def stage(
        self,
        input_directory: str,
        execs_directory: str,
    ) -> None:
        """
        Stage ADCIRC simulation by adding linking executables and inputs.

        Parameters
        ----------
        input_directory : str
            The directory containing ADCIRC inputs.
        execs_directory : str
            The directory containing ADCIRC executables.

        Raises
        ------
        RuntimeError
            If any command fails to execute successfully.
        """

        self.logger.info("Staging ADCIRC Simulation")

        self.client.exec(''.join(
            f"ln -sf {input_directory}/* . && ",
            f"ln -sf {execs_directory}/adcprep . &&",
            f"ln -sf {execs_directory}/padcirc .")
        )

    def adcrprep(
        self,
        write_processes: int,
    ) -> None:
        """
        Run ADCIRC simulation using given parameters.

        Parameters
        ----------
        input_directory : str
            The directory containing ADCIRC inputs.
        execs_directory : str
            The directory containing ADCIRC executables.
        write_processes : int
            Number of write processes to use.
        remora : int
            Whether to load remora module (1 for yes, 0 for no).
        debug : bool, optional
            Enable debug mode, by default True.

        Raises
        ------
        RuntimeError
            If any command fails to execute successfully.
        """
        self.logger.info("Starting adcprep")

        # Compute core allocation
        cores = int(self.client.exec("echo $SLURM_TACC_CORES"))
        pcores = cores - write_processes

        # Generate the two prep files
        self.client.exec(f'printf "{pcores}\\n1\\nfort.14\\n" | adcprep > adcprep.log')
        self.client.exec(f'printf "{pcores}\\n2\\n" | adcprep >> adcprep.log')

    def run_simulation(
        self,
        cores: int,
        write_processes: int,
    ) -> None:
        """
        Run ADCIRC simulation using given parameters.

        Parameters
        ----------
        input_directory : str
            The directory containing ADCIRC inputs.
        execs_directory : str
            The directory containing ADCIRC executables.
        write_processes : int
            Number of write processes to use.
        remora : int
            Whether to load remora module (1 for yes, 0 for no).
        debug : bool, optional
            Enable debug mode, by default True.

        Raises
        ------
        RuntimeError
            If any command fails to execute successfully.
        """
        self.logger.info("Starting Simulation (padcirc)")
        out_f = f"adcirc_{int(time.time())}.out.txt"
        err_f = f"adcirc_{int(time.time())}.err.txt"
        self.client.exec(
            f"ibrun -np {cores} ./padcirc -W {write_processes} > {out_f} 2> {err_f}"
        )
        exit_code = self.client.exec("echo $?")
        if int(exit_code) != 0:
            self.logger.error("ADCIRC exited with an error status.")
            self.client.exec("${AGAVE_JOB_CALLBACK_FAILURE}")
            raise RuntimeError("ADCIRC exited with an error status.")

        self.logger.info("Simulation Done")

    def setup_job(self):
        """
        Command to set-up job directory.

        This is a skeleton method that should be over-written.
        """
        logger.info("Job set-up Start")
        self.stage(
            self.job_config["args"]['input_dir'],
            self.job_config["args"]['exec_dir']
        )
        self.adcprep(self.job_config["args"]['wp'])
        logger.info("Job set-up Done")

    def run_job(self):
        """
        Job run entrypoint

        This is a skeleton method that should be over-written.

        Note: ibrun command should be here somewhere.
        """
        logger.info("Starting Simulation")
        self.run_simulation(self.job_config["args"]["wp"])
        logger.info("Simulation Done")