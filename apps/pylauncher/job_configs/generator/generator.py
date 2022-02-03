import argparse
import json


def generator(message:str, num_jobs:int, cores_per_job:int):
  """
  Simple shell command generator with pre & post processing.

  Parameters
  ----------
  message : str
    String message to pass to main parallel script.
  num_jobs : int
    Number of jobs to configure.
  cores_per_jobs : int
    Number of cores to assing to each job.

  """

  # Parse json dictionary
  jobs = [{"cores": cores_per_job,
           "main": f"./main.sh {i} >> logs/job_{i}.log",
           "pre_process": f"./pre_process.sh {i} >> logs/job_{i}.log",
           "post_process": f"./post_process.sh {i} >> logs/job_{i}.log"}
           for i in range(num_jobs)]

  return jobs


if __name__ == "__main__":

  # Parse command line options
  parser = argparse.ArgumentParser()
  parser.add_argument("iter", type=int)
  parser.add_argument("np", type=int)
  parser.add_argument("--message", type=str, default="Hello World!")
  parser.add_argument("--num-jobs", type=int, default=5)
  parser.add_argument("--cores-per-job", type=int, default=1)
  args = parser.parse_args()

  # Only generate configs on first iteration (only one run-through)
  if args.iter == 1:
    print(f"Running generator.py with message: {args.message}")
    jobs = generator(args.message, args.num_jobs, args.cores_per_job)

    # Write jobs json file
    with open("jobs_list.json", "w") as fp:
      print(f"Writing jobs {jobs} to file.")
      json.dump(jobs, fp)
