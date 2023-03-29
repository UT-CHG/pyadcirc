#!/bin/bash
# 
# Shell Demo - Post Process Script
# Carlos del-Castillo-Negrete - cdelcastillo21@gmail.com
# June 2021
#
# Post-process script for shell demo. Gets passed in job number and creates a run directory
# for that job. 

DEBUG=false

# Read command line inputs
JOB_DIR=$1
FLAGS="${@:2}"

log () {
  echo "$(date) | POST_PROCESS | ${1} | ${2}" 
}

if [ "$DEBUG" = true ] ; then
  set -x
  log DEBUG "Setting debug."
fi

log INFO "STARTING"

log INFO "LOADING CONDA ENV"
source /home1/06307/clos21/miniconda3/etc/profile.d/conda.sh
conda activate adcirc
log INFO "PYTHON PATH $(which python)"

log INFO "Processing ADCIRC output"

python inputs/process_output.py $JOB_DIR $JOB_DIR/adcirc_output.nc ${FLAGS}

log INFO "Done processing ADCIRC OUTPUT"

log INFO "Cleaning Job Directory"
mv $JOB_DIR/adcirc .trash/
log INFO "Done Cleaning Job Directory"

log INFO "DONE"

if [ "$DEBUG" = true ] ; then
  log DEBUG "Unsetting debug."
  set +x
fi

exit 0

