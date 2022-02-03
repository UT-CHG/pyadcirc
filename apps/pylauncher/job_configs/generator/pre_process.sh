#!/bin/bash
# 
# Shell Demo - Pre Process Script
# Carlos del-Castillo-Negrete - cdelcastillo21@gmail.com
# June 2021
#
# Pre-process script for shell demo. Gets passed in job number and creates a run directory
# for that job. 

DEBUG=false

# Read command line inputs
JOB_NUM=$1

log () {
  echo "$(date) - PRE_PROCESS - ${1} - ${2}" 
}

if [ "$DEBUG" = true ] ; then
  log DEBUG "Setting debug."
  set -x
fi

log INFO "STARTING"

# Create direcotry for job in runs directory
RUN_DIR="runs/job_${JOB_NUM}"
log INFO "Making run directory ${RUN_DR}"
mkdir $RUN_DIR

log INFO "DONE"

if [ "$DEBUG" = true ] ; then
  log DEBUG "Unsetting debug."
  set +x
fi

exit 0
