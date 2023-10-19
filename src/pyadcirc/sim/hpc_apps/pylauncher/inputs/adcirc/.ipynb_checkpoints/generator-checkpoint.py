import argparse
import json
from pathlib import Path


def generator(base_dir:str,
    runs_dir:str,
    execs_dir:str,
    cores_per_job:int,
    write_proc_per_job:int = 0):
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
          jobs_dir.append(it)

  job_configs = []
  run_proc = cores_per_job - write_proc_per_job
  for idx, job in enumerate(job_dirs):
    job_dir = job.path
    job_name = job.name
    log_file = f"logs/job_{idx+1}_{job_name}.log"

    # Pre-process command
    pre_process = (f"./pre_process.sh {idx+1} {base_dir} {job_dir} ",
        f"{execs_dir} {run_proc} >> {log_file}")

    # Main ADCIRC command
    main = f"./padcirc -W {write_proc_per_job} >> {log_file}"

    # Post process command - For now does nothing
    post_process = (f"./post_process.sh {idx+1} >> {log_file}")

    job_configs.append({"cores": cores_per_job,
           "main": main,
           "pre_process": pre_process,
           "post_process": post_process})

  return job_configs


if __name__ == "__main__":

  # Parse command line options
  parser = argparse.ArgumentParser()
  parser.add_argument("iter", type=int)
  parser.add_argument("np", type=int)
  parser.add_argument("base_dir", type=str)
  parser.add_argument("runs_dir", type=str)
  parser.add_argument("execs_dir", type=str)
  parser.add_argument("--cores_per_job", type=int, default=4)
  parser.add_argument("--write_proc_per_job", type=int, default=0)
  args = parser.parse_args()

  # Only generate configs on first iteration (only one run-through)
  # TODO: On future iterations check for failed jobs and re-run
  # Or run image processing/post-processing on future runs after jobs complete.
  if args.iter == 1:

    if args.cores_per_job > args.np:
      args.cores_per_job = args.np

    print(f"Running generator.py with message: {args.message}")
    jobs = generator(args.base_dir, args.runs_dir, args.execs_dir,
        args.cores_per_job, args.write_proc_per_job)

    # Write jobs json file
    with open("jobs_list.json", "w") as fp:
      print(f"Writing {len(jobs)} jobs to json file.")
      json.dump(jobs, fp)
