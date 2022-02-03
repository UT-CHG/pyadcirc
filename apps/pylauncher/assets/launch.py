import pylauncher4 as pyl4
import argparse as ap

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("infile", nargs="?", default="jobs_list.csv")
    args = parser.parse_args()
    # Launch pylauncher with IbrunLauncher, pre_post_processing set to True by default
    # Note name of input file expected is jobs_list.csv
    pyl4.IbrunLauncher(args.infile, cores="file", debug="job+host+task+exec", pre_post_process=True)
