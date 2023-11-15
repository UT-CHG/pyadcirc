# from mpi4py import MPI
# import h5py
import json
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
            "label": "Inputs Diretory",
            "desc": "Path to Inputs (fort.* files) for ADCIRC Simulation",
            "default": "/work/06307/clos21/pub/adcirc/inputs/ShinnecockInlet/mesh/def",
        },
        {
            "name": "exec_dir",
            "type": "argument",
            "label": "Executables Directory",
            "desc": "Path to Executables Directory on TACC systems",
            "default": "/work/06307/clos21/pub/adcirc/execs/ls6/v56_beta/",
        },
        {
            "name": "cp",
            "type": "argument",
            "label": "Compute Processes",
            "desc": "Number of processes to parallelize computation accorss.",
            "default": 4,
        },
        {
            "name": "wp",
            "type": "argument",
            "label": "Write Processes",
            "desc": "Number of write processes to use.",
            "default": 0,
        },
        {
            "name": "swan",
            "type": "argument",
            "label": "SWAN coupling",
            "desc": "Flag indicating whether to couple to SWAN",
            "default": False
        }
    ]

    ENV_CONFIG = {
        "conda_env": "pya-test",
        "modules": ["netcdf"],
        "conda_packages": ["mpi4py", "h5py", "cfgrib"],
        "pip_packages": ["pyadcirc"],
    }

    def __init__(
        self,
        name: str = None,
        system: str = None,
        log_config: dict = None,
    ):
        super().__init__(
            name=name,
            system=system,
            log_config=log_config,
            script_file=__file__,
            class_name=self.__class__.__name__,
        )

    @property
    def num_compute_tasks(self):
        """
        """
        # TODO: Implement
        return None

    @property
    def num_write_tasks(self):
        """
        """
        # TODO: Implement
        return None


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

        logger.info("Staging ADCIRC Simulation")
        exec_name = self._get_exec_name()
        input_directory = str((Path(input_directory) / '*').absolute())
        adcprep_path = str((Path(execs_directory) / 'adcprep').absolute())
        padcirc_path = str((Path(execs_directory) / exec_name).absolute())
        stage_res = self.client.exec(''.join([
            f"ln -sf {input_directory} . && ",
            f"ln -sf {adcprep_path} . &&",
            f"ln -sf {padcirc_path} .",
        ]))

        return stage_res

    def adcprep(
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
        logger.info("Starting adcprep")

        # Compute core allocation
        cores = int(self.client.exec("echo $SLURM_TACC_CORES")['stdout'])
        pcores = cores - write_processes

        adcprep_res_1 = self.client.exec(
            f'printf "{pcores}\\n1\\nfort.14\\n" | adcprep > adcprep.log',
            fail=False)
        logger.info(f'ADCPREP 1 res: {adcprep_res_1}')
        adcprep_res_2 = self.client.exec(
            f'printf "{pcores}\\n2\\n" | adcprep >> adcprep.log',
            fail=False)
        logger.info(f'ADCPREP 1 res: {adcprep_res_2}')

        return (adcprep_res_1, adcprep_res_2)

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
        logger.info("Starting Simulation (padcirc)")
        out_f = f"adcirc_{int(time.time())}.out.txt"
        err_f = f"adcirc_{int(time.time())}.err.txt"
        main_exec_cmnd = self.client.exec(
            f"ibrun -np {cores} ./padcirc -W {write_processes} > {out_f} 2> {err_f}"
        )
        logger.info(f"Main Command Done: {main_exec_cmnd}")
        logger.info("Simulation Done")

    def _get_exec_name(self):
        swan = self.job_config['args'].get('swan', False)
        return 'padcswan' if swan else 'padcirc'

    def setup_job(self):
        """
        Command to set-up job directory.

        This is a skeleton method that should be over-written.
        """
        logger.info(f"Job set-up Start: {self.job_config}")
        stage_res = self.stage(
            self.job_config["args"]['input_dir'],
            self.job_config["args"]['exec_dir']
        )
        logger.info(f'Stage command res: {stage_res}')
        _ = self.adcprep(self.job_config["args"]['wp'])
        logger.info("Job set-up Done")

    def run_job(self):
        """
        Job run entrypoint

        This is a skeleton method that should be over-written.

        Note: ibrun command should be here somewhere.
        """
        logger.info(f"Starting Simulation: {self.job_config}")
        wp = int(self.job_config['args']['wp'])
        max_cp = int(self.slurm_env['NTASKS']) - wp
        cp = self.job_config['args']['cp']
        cp = cp if cp <= max_cp else max_cp

        logger.info("Staging ADCIRC inputs")
        exec_name = self._get_exec_name() 
        input_directory = str((Path(
            self.job_config['args']['input_dir']) / '*').absolute())
        adcprep_path = str((Path(
            self.job_config['args']['exec_dir']) / 'adcprep').absolute())
        padcirc_path = str((Path(
            self.job_config['args']['exec_dir']) / exec_name).absolute())
        stage_res = self.client.exec(''.join([
            f"ln -sf {input_directory} . && ",
            f"ln -sf {adcprep_path} . &&",
            f"ln -sf {padcirc_path} .",
        ]))
        logger.info(f"Staging done: {stage_res}")

        logger.info("ADCPREP Start")
        adcprep_res_1 = self.client.exec(
            f'printf "{cp}\\n1\\nfort.14\\n" | adcprep > adcprep.log',
            fail=False)
        logger.info(f'ADCPREP 1 res: {adcprep_res_1}')
        adcprep_res_2 = self.client.exec(
            f'printf "{cp}\\n2\\n" | adcprep >> adcprep.log',
            fail=False)
        logger.info(f'ADCPREP 2 res: {adcprep_res_2}')

        logger.info(f"Starting Simulation ({exec_name})")
        out_f = f"adcirc_{int(time.time())}.out.txt"
        err_f = f"adcirc_{int(time.time())}.err.txt"
        main_exec_cmnd = self.client.exec(
            f"ibrun -np {cp} ./{exec_name} -W {wp} > {out_f} 2> {err_f}"
        )
        logger.info(f"Simulation Done {main_exec_cmnd}")
