#!/bin/bash
# 
# Shell Demo - Main Shell Script
# Carlos del-Castillo-Negrete - cdelcastillo21@gmail.com
# June 2021
#
# Main shell script called by each job
# This script will be called with ibrun by pylauncher and be executed by the adequate number of 
# independent processes, so be careful with creating/updating common job run files. 

JOB_NUM=$1

DEBUG=false

log () {
  # Note only master process logs
  if [ $MPI_LOCALRANKD -eq 0 ]
  then
    echo "$(date) - MAIN - ${1} - ${2}" 
  fi
}

if [ "$DEBUG" = true ] ; then
  set -x
  log DEBUG "Setting debug."
fi

log INFO "STARTING"

RUN_DIR="runs/job_${JOB_NUM}"
log INFO "Navigating to run directory ${RUN_DIR}"
cd $RUN_DIR

# Do something
if [ $MPI_LOCALRANKD -ne 0 ]
then
  sleep 5
else
  log INFO "Hi I am main process $MPI_LOCALRANKID. Writing output file..." 
  echo $RANDOM > output_${JOB_NUM}.csv
fi

log INFO "DONE"

exit 0
