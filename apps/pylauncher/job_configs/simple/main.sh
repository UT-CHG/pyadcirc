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
  # Note this file logs to job specific run directory
  echo "$(date) : ${1} : ${2}" 
}

if [ "$DEBUG" = true ] ; then
  set -x
  log DEBUG "Setting debug."
fi

log INFO "Starting main script for Job ${JOB_NUM}"
log DEBUG "Current Directory: $(pwd)"

# Do something
if [ $MPI_LOCALRANKD -ne 0 ]
then
  log INFO "Hi I am worker process $MPI_LOCALRANKID. Sleeping... " 
  sleep 5
else
  log INFO "Hi I am main process $MPI_LOCALRANKID. Writing output file..." 
  echo $RANDOM > output_${JOB_NUM}.csv
fi

log INFO "Finished Job ${JOB_NUM} parallel job."

exit 0
