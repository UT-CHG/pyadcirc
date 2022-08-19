#!/bin/bash
# 
# PYLAUNCHER ADCIRC - Generator 
# Carlos del-Castillo-Negrete - cdelcastillo21@gmail.com
# June 2021
#
# Main entrypoint called by pylauncher application to generate files for pylauncher.
# Offload generator work to python script generator.py

log () {
  # Note this file logs to stdout, since it is expected
  # to be run by the main application and capture output
  # in the main log file. Using this log function helps 
  # keep logs consistent.
  echo "$(date) | GENERATOR | ${1} | ${2}" 
}

DEBUG=false

if [ "$DEBUG" = true ] ; then
  set -x
fi

ITER=$1
NP=$2
GEN_ARGS="${@:3}"

log INFO "STARTING GENERATOR"
log DEBUG "Current Directory: $(pwd)"

log INFO "LOADING CONDA ENV"
source /home1/06307/clos21/miniconda3/etc/profile.d/conda.sh
conda activate adcirc
log INFO "PYTHON PATH $(which python)"

log INFO "Calling generator.py for iteration ${ITER} with $NP processes."
log INFO "Additional arguments passed - ${GEN_ARGS}"

python generator.py $ITER $NP ${GEN_ARGS}

log INFO "GENERATOR FINISHED"

if [ "$DEBUG" = true ] ; then
  set +x
fi

exit 0
