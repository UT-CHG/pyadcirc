from typing import List
from pathlib import Path
import numpy as np
from pyadcirc import io as pyio

def gen_uniform_beta_fort13(
    base_f13_path: str = "fort.13",
    targ_dir: str = None,
    name: str = "beta",
    num_samples: int = 10,
    domain: List[int] = [0.0, 2.0],
):
    """
    Generate fort.13 files w/beta vals from uniform distribution

    Parameters
    ----------
    base_f13_path : str, default='fort.13'
        Path to base fort.13 file to modify beta values for
    targ_dir : str, optional
        Path to output directory. Defaults to current working directory.
    name : str, default='beta'
        Name to give to output directory. Final name will be in the
        format {name}_{domain min}-{domain max}_u{num samples}
    num_samples : int, default=10
        Number of samples to take from a uniform distribution
    domain : List[int], default=[0.0, 2.0]
        Range for beta values.


    Returns
    ----------
    targ_path : str
        Path to directory containing all the seperate job directories
        with individual fort.13 files

    """

    targ_dir = Path.cwd() if targ_dir is None else targ_dir
    if not targ_dir.exists():
        raise ValueError(f"target directory {str(targ_dir)} does not exist")
    if not Path(base_f13_path).exists():
        raise ValueError(f"Unable to find base fort.13 file {base_f13_path}")

    targ_path = Path(
        f"{str(targ_dir)}/{name}_{domain[0]:.1f}-{domain[1]:.1f}_u{num_samples}"
    )
    targ_path.mkdir(exist_ok=True)

    beta_vals = np.random.uniform(domain[0], domain[1], size=num_samples)
    f13 = pyio.read_fort13(base_f13_path)

    files = []
    for idx, b in enumerate(beta_vals):
        f13["v0"][0] = b
        job_name = f"beta-{idx}_{b:.2f}"
        job_dir = targ_path / job_name
        job_dir.mkdir(exist_ok=True)
        fpath = str(job_dir / "fort.13")
        pyio.write_fort13(f13, fpath)
        files.append(fpath)

    return str(targ_path), files

def generator(base_dir:str,
    runs_dir:str,
    execs_dir:str,
    cores_per_job:int,
    write_proc_per_job:int = 0,
    remora:int = 0):
  """
  Generator for set of basic ADCIRC runs. Assumes base set of files (for
  example, fort.14 adn fort.15 contorl file) are in `base_dir` on TACC systems
  (accessible by compute nodes), and each run's files are in seperate
  directories within `runs_dir`, with the name of the directories being the
  name to give to each run. The same executables are used for each run,
  contained in `execs_dir` and each ADCIRC run is to be executed with a total
  of `cores_per_job` MPIE processes, with `write_proc_per_job` of those being
  dedicated to just writing ADCIRC data output.

  Parameters
  ----------
  base_dir : str
    Dir on TACC where base input files for each ADCIRC run are.
  runs_dir : str
    Dir on TACC containing sub-directories with each runs' job specific files.
  execs_dir : str
    Dir on TACC where ADCIRC executables are.
  cores_per_jobs : int
    Number of total MPI processes to use for each ADCIRC run.
  write_proc_per_job: int, default=0
    Number of teh total cores to use for dedicated writing of output files.
  remora: int, default=0
    If non-zero, runs each job with remora monitoring.

  Returns
  -------
  jobs : List[dict]
    List of pylauncher job configuration dictionaries, with the fields `cores,
    main, pre_process, post_process`.

  """

  # Get job directories in runs_directory
  job_dirs = []
  for it in Path(runs_dir).iterdir():
      if it.is_dir():
          job_dirs.append(it)

  job_configs = []
  run_proc = cores_per_job - write_proc_per_job
  for idx, job in enumerate(job_dirs):
    job_dir = job.resolve()
    job_name = job.name
    log_file = f"logs/job_{idx+1}_{job_name}.log"

    # Pre-process command
    pre_process = ''.join([f"./pre_process.sh {idx+1} {base_dir} {job_dir} ",
        f"{execs_dir} {run_proc} >> {log_file}"])

    # Main ADCIRC command
    main = f"{execs_dir}/padcirc -W {write_proc_per_job} >> {log_file}"

    # Post process command - For now does nothing
    post_process = (f"./post_process.sh {idx+1} >> {log_file}")

    job_configs.append({"cores": cores_per_job,
           "main": main,
           "pre_process": pre_process,
           "post_process": post_process,
           "remora": 1 if remora else 0})

  return job_configs
