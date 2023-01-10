"""

"""
import pdb
import json
from io import StringIO
from taccjm import taccjm as tjm
from pathlib import Path
import pandas as pd
import pyslurmtq.SLURMTaskQueue as stq
from pyslurmtq.utils import filter_res

from pyadcirc.data import ncar


TACC_SYSTEMS = {
    "ls6": {"queue": "normal", "queue_test": "development"},
    "frontera": {"queue": "normal", "queue_test": "development"},
    "stampede2": {"queue": "skx-normal", "queue_test": "skx-dev"},
}

APPS = {"adcirc": None}
LOCAL_ADCIRC = Path(__file__).parents

GLOBUS_ENDPOINT = "142d715e-8939-11e9-b807-0a37f382de32"

from pyadcirc import __version__


def init_system(jm_id, system, user=None, psw=None, mfa=None, test=True):
    if jm_id not in [j["jm_id"] for j in tjm.list_jms()]:
        jm = tjm.init_jm(jm_id, system, user, psw, mfa)
    else:
        jm = tjm.get_jm(jm_id)

    apps = tjm.list_apps(jm_id)
    for app_dir in (Path(__file__).parent / "hpc_apps").iterdir():
        if app_dir.name in list(APPS.keys()):
            with open(str(app_dir / "configs/taccjm/app.json")) as fp:
                app_config = json.load(fp)
            app_config["name"] = f"{system}-{str(app_dir.name)}--{__version__}"
            if test:
                app_config["default_queue"] = TACC_SYSTEMS[system]["queue"]
            else:
                app_config["default_queue"] = TACC_SYSTEMS[system]["test_queue"]

            APPS[app_dir.name] = app_config
            _ = tjm.deploy_app(jm_id, app_config=app_config, local_app_dir=str(app_dir))

    return jm


class ADCIRCDB(object):
    """
    Class defining ADCIRC data and files on TACC systems.
    """

    def __init__(self, jm_id, globus_client_id=None):

        if jm_id not in [j["jm_id"] for j in tjm.list_jms()]:
            err = "TACCJM {jm_id} does not exist. Initialize with `init_system`"
            raise ValueError(err)

        self._jm = tjm.get_jm(jm_id)
        dir_id = Path(self._jm["apps_dir"]).parts[2]
        user = Path(self._jm["apps_dir"]).parts[3]
        self.base_adcirc_dir = Path(f"/work/{dir_id}/{user}/adcirc")

        base_contents = tjm.list_files(
            self._jm["jm_id"], str(self.base_adcirc_dir.parent)
        )
        if "adcirc" not in [f["filename"] for f in base_contents]:
            local_adcirc_dir = Path(__file__).parents[3] / "data/adcirc"
            tjm.upload(self._jm["jm_id"], local_adcirc_dir, self.base_adcirc_dir)

        if globus_client_id is not None:
            self.dt = ncar.NCARDataTransfer(globus_client_id)
        else:
            self.dt = None
        self.data_transfers = []

    def list_ADCIRC_inputs(self):
        """List ADCIRC input files configured"""
        input_files = tjm.list_files(self._jm["jm_id"], self.base_adcirc_dir / "inputs")
        return input_files

    def list_ADCIRC_execs(self):
        """List ADCIRC exec files configured"""
        input_files = tjm.list_files(self._jm["jm_id"], self.base_adcirc_dir / "execs")
        return input_files

    def compile_ADCIRC(
        self,
        version="v55.01",
        git_url="https://github.com/adcirc/adcirc-cg",
        is_tag=True,
    ):
        """Run ADCIRC Compile on Host to compile an ADCIRC executable"""
        scripts = tjm.list_scripts(self._jm["jm_id"])
        if "adcirc_compile" not in scripts:
            sp = Path(__file__).parents[1] / "scripts/adcirc_compile.sh"
            tjm.deploy_script(self._jm["jm_id"], script_name=str(sp))

        res = tjm.run_script(
            self._jm["jm_id"],
            "adcirc_compile",
            args=[version, git_url, 1 if is_tag else 0],
        )
        log = pd.read_csv(StringIO(res), delimiter="|")

        return log

    def list_ncar_data(
        self, ds_name, start_date, end_date, search=None, match=r".", print_res=True
    ):
        """Search NCAR data sets using Globus SDK"""
        if self.dt is None:
            raise AttributeError("NCAR data transfer not initialized")

        res = filter_res(
            db.dt.list_files(ds, start_date, end_date, pp=False),
            search=search,
            match=match,
            print_res=print_res,
        )

        return res

    def init_ncar_data_transfer(self, ds_name, fields, start_date, end_date):
        """Use Globus to Transfer NCAR data to TACC"""
        if self.dt is None:
            raise AttributeError("NCAR data transfer not initialized")

        dest_path = self.base_adcirc_dir / f"ncar_{ds_name}"
        tdata = self.dt.stage(
            ds_name, fields, GLOBUS_ENDPOINT, str(dest_path), start_date, end_date
        )
        tresults = self.dt.submit(tdata)
        self.data_transfers.append((tdata, tresults))

    def check_ncar_data_transfer(
        self,
        fields=["task_id", "time", "code", "details"],
        search=None,
        match=r".",
        print_res=True,
    ):
        """Check on statuses of NCAR data transfers"""
        if self.dt is None:
            raise AttributeError("NCAR data transfer not initialized")

        status_info = []
        for dt in self.data_transfers:
            si = self.dt.transfer_client.task_event_list(dt[1]["task_id"])
            for s in si:
                s["task_id"] = dt[1]["task_id"]
            status_info += si

        return filter_res(
            status_info, fields=fields, search=search, match=match, print_res=print_res
        )


class ADCIRCSim(object):
    """Class for running ADCIRC simulations on TACC resources

    Attributes
    ----------

    """

    def __init__(
        self,
        jm_id,
        name,
        input_dir,
        execs_dir,
    ):

        self._jm = tjm.get_jm("jm_id")
        self.job_config = None

    def deploy_job(
        self,
        write_proc=1,
        remora=False,
        rt=None,
        nodes=1,
        np=12,
        queue=None,
        allocation=None,
        desc=None,
    ):
        """Get root dir shared accross all TACC work systems"""
        job_config = {
            "name": name,
            "app": APPS["adcirc"]["name"],
            "desc": desc,
            "queue": queue,
            "allocation": allocation,
            "node_count": nodes,
            "processors_per_node": np,
            "max_run_time": rt,
            "inputs": {},
            "parameters": {
                "inputDirectory": self.input_dir,
                "execDirectory": self.execs_dir,
                "writeProcesses": write_proc,
                "remoray": remora,
            },
        }

        self.deployed_config = tjm.deploy_job(self._jm["jm_id"], job_config)

    def submit_job(self):
        """Get root dir shared accross all TACC work systems"""
        self.job_config = tjm.submit_job(self._jm["jm_id"], self.job_config["job_id"])

    def check_status(self):
        """Check status of ADCIRC simulation"""

    def list_inputs(self):
        tjm.list_files(self._jm["jm_id"], self.adcirc_base_dir)

    def get_allocations(self):
        """Get allocations on JM system"""
        allocs = tjm.get_allocations(self._jm["jm_id"])
        alloc = allocs[0]["name"]
        return allocs
