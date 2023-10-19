# from mpi4py import MPI
# import h5py
import time

import numpy as np
from taccjm.log import logger
from taccjm.sim.TACCSimulation import TACCSimulation


class BaseADCIRCSimulation(TACCSimulation):
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
            "desc": "Number of channels to use to write the array.",
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

    def run_job(self):
        """
        Job run entrypoint

        This is a skeleton method that should be over-written.

        Note: ibrun command should be here somewhere.
        """
        n = self.job_config["args"]["n"]
        channels = self.job_config["args"]["channels"]
        logger.info("Starting Simulation")

        # Use the client.exec function to execute commands.
        self.client.exec(f"tail -n {param} {input_file} > out.txt; sleep 10")

        logger.info("Simulation Done")

        # num_processes = MPI.COMM_WORLD.size
        # rank = MPI.COMM_WORLD.rank

        # if rank == 0:
        #     start = time.time()

        # np.random.seed(746574366 + rank)

        # f = h5py.File('parallel_test.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)
        # dset = f.create_dataset('test', (channels, n), dtype='f')

        # for i in range(channels):
        #     if i % num_processes == rank:
        #         data = np.random.uniform(size=n)
        #         dset[i] = data

        # f.close()

        # if rank == 0:
        #     print('Wallclock Time (s) Elapsed: ' + time.time()-start)


class ADCIRCSimulation(object):
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
            raise TypeError(f"RUNID must be of type str")
        if len(value) >= 80:
            raise ValueError(f"RUNID must be 80 characters or less")
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
            raise TypeError(f"ICS must be of type int")
        if value not in [1, 2]:
            raise ValueError(f"ICS must be 1 (Cartesian) or 2 (Spherical)")
        # TODO: Does the value here have to match the fort.14 grid?
        self.control_params["ICS"] = value

        def run_job(self):
            """
            Job run entrypoint

            This is a skeleton method that should be over-written.

            Note: ibrun command should be here somewhere.
            """
            n = self.job_config["args"]["n"]
            channels = self.job_config["args"]["channels"]
            logger.info("Starting Simulation")

            # Use the client.exec function to execute commands.
            self.client.exec(f"tail -n {param} {input_file} > out.txt; sleep 10")

            logger.info("Simulation Done")

            # num_processes = MPI.COMM_WORLD.size
            # rank = MPI.COMM_WORLD.rank

            # if rank == 0:
            #     start = time.time()

            # np.random.seed(746574366 + rank)

            # f = h5py.File('parallel_test.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)
            # dset = f.create_dataset('test', (channels, n), dtype='f')

            # for i in range(channels):
            #     if i % num_processes == rank:
            #         data = np.random.uniform(size=n)
            #         dset[i] = data

            # f.close()

            # if rank == 0:
            #     print('Wallclock Time (s) Elapsed: ' + time.time()-start)

            import logging

    def run_simulation(
        self,
        input_directory: str,
        execs_directory: str,
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

        self.logger.info("Starting Simulation")

        try:
            # Move inputs to current (job) directory
            self.client.exec(f"ln -sf {input_directory}/* .")

            # Link symbolically the executables
            self.client.exec(f"ln -sf {execs_directory}/adcprep .")
            self.client.exec(f"ln -sf {execs_directory}/padcirc .")

            # Compute core allocation
            cores = int(self.client.exec("echo $SLURM_TACC_CORES"))
            pcores = cores - write_processes

            # Generate the two prep files
            self.client.exec(f'printf "{pcores}\\n1\\nfort.14\\n" | adcprep')
            self.client.exec(f'printf "{pcores}\\n2\\n" | adcprep')

            self.client.exec(
                f"ibrun -np {cores} ./padcirc -W {write_processes} >> output.eo.txt 2>&1"
            )

            # Check if the command was successful
            exit_code = self.client.exec("echo $?")
            if int(exit_code) != 0:
                self.logger.error("ADCIRC exited with an error status.")
                self.client.exec("${AGAVE_JOB_CALLBACK_FAILURE}")
                raise RuntimeError("ADCIRC exited with an error status.")

            self.logger.info("Simulation Done")

        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            raise
