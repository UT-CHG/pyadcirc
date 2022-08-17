import argparse
import time
import json
from pathlib import Path
from pyadcirc import generators as gens

if __name__ == "__main__":

  # Parse command line options
  parser = argparse.ArgumentParser()
  parser.add_argument("iter", type=int)
  parser.add_argument("np", type=int)
  parser.add_argument("base_dir", type=str)
  parser.add_argument("execs_dir", type=str)
  parser.add_argument("--runs_dir", type=str, default=None)
  parser.add_argument("--cores_per_job", type=int, default=4)
  parser.add_argument("--write_proc_per_job", type=int, default=0)
  parser.add_argument("--num_samples", type=int, default=10)
  parser.add_argument("--range-low", type=float, default=0.0)
  parser.add_argument("--range-high", type=float, default=2.0)
  parser.add_argument("--remora", type=bool, default=False, action=argparse.BooleanOptionalAction)
  args = parser.parse_args()

  # Only generate configs on first iteration (only one run-through)
  # TODO: On future iterations check for failed jobs and re-run
  # Or run image processing/post-processing on future runs after jobs complete.
  if args.iter == 1:

    if args.cores_per_job > args.np:
      args.cores_per_job = args.np

    if args.runs_dir is None:
        args.runs_dir, _ = gens.gen_uniform_beta_fort13(
            base_f13_path=f"{args.base_dir}/fort.13",
            targ_dir=Path.cwd(),
            name=f"{args.iter}_{time.strftime('%Y%m%d-%H%M%S')}",
            num_samples=args.num_samples,
            domain=[args.range_low, args.range_high])

    jobs = gens.generator(args.base_dir,
            args.runs_dir,
            args.execs_dir,
            args.cores_per_job,
            args.write_proc_per_job,
            remora=args.remora
            )

    # Write jobs json file
    with open("jobs_list.json", "w") as fp:
      print(f"Writing {len(jobs)} jobs to json file.")
      json.dump(jobs, fp)
