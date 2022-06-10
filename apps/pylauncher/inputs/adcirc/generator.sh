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

log INFO "STARTING GENERATOR"
log DEBUG "Current Directory: $(pwd)"

log INFO "Calling generator.py for iteration ${ITER} with $NP processes."
log INFO "Additional arguments passed - ${@:3}"

python3 generator.py $ITER $NP ${@:3}

log INFO "GENERATOR FINISHED"

if [ "$DEBUG" = true ] ; then
  set +x
fi

exit 0
