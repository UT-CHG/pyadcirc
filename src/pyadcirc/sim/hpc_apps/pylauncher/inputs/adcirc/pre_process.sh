#!/bin/bash
#
# PYLAUNCHER ADCIRC - Pre Process Script
# Carlos del-Castillo-Negrete - cdelcastillo21@gmail.com
# June 2021
#
# Pre-process script for shell demo. Gets passed in job number and creates a run directory
# for that job.

DEBUG=false

# Read command line inputs
JOB_NUM=$1
BASE_INPUTS=$2
RUN_DIR=$3
EXECS_DIR=$4
RUN_PROC=$5

log () {
  echo "$(date) | PRE_PROCESS | ${1} | ${2}"
}

if [ "$DEBUG" = true ] ; then
  log DEBUG "Setting debug."
  set -x
fi

log INFO "STARTING"

# Load necessary modules - Netcdf for adcprep
module load netcdf

cd $RUN_DIR

# Make directory for adcirc data
mkdir adcirc
cd adcirc

# Move base inputs to current (job) directory
log INFO "Copying base inputs from ${BASE_INPUTS}"
ln -sf ${BASE_INPUTS}/* .

# Override base inputs with fort files in run directory (parent)
mv ../fort.* .

# Link symbolically the executables
log INFO "Copying adcirc executables from ${EXECS_DIR}"
ln -sf ${EXECS_DIR}/adcprep .
ln -sf ${EXECS_DIR}/padcirc .

# Run ADCPREP
log INFO "Running ADCPREP - partmesh for ${RUN_PROC} compute processes"
printf "${RUN_PROC}\n1\nfort.14\n" | adcprep

log INFO "Running ADCPREP - prep all for ${RUN_PROC} compute processes"
printf "${RUN_PROC}\n2\n" | adcprep

log INFO "DONE"

if [ "$DEBUG" = true ] ; then
  log DEBUG "Unsetting debug."
  set +x
fi

exit 0
