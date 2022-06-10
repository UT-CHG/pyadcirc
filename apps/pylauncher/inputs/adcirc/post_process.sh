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
JOB_NUM=$1

log () {
  echo "$(date) | POST_PROCESS | ${1} | ${2}" 
}

if [ "$DEBUG" = true ] ; then
  set -x
  log DEBUG "Setting debug."
fi

log INFO "STARTING"

# For now do nothing in post-process
# Future: Generate figures and relevatn plots and compile results into singular netcdf file.
# log INFO "Moving output file to outputs directory"
# RUN_DIR="runs/job_${JOB_NUM}"
# mv "${RUN_DIR}/output.csv" "outputs/job_${JOB_NUM}.csv"

log INFO "DONE"

if [ "$DEBUG" = true ] ; then
  log DEBUG "Unsetting debug."
  set +x
fi

exit 0

