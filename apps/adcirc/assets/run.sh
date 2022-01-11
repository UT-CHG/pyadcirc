#!/usr/bin/env bash

# PADCIRC Application - Main entry-point
#   Carlos del-Castillo-Negrete - cdelcastillo21@gmail.com
#   June 2021
#
# Main entrypoint called in execution envrionment by adcirc application.
# Expects the following inputs (set as local environment variables):
# 	inputDirectory - Directory containing ADCIRC inputs
# 	execsDirectory - Directory containing ADCIRC executables
# 	writeProcesses - Number of write processe to use
# 
# Furthermore since assumed executing on TACC SLURM execution, we 
# use the following environment variables set by SLURM/TACC:
# 	SLURM_TACC_CORES - Total number of processes available job

DEBUG=true

log () {
  echo "$(date) : ${1} - ${2}"
}

if [ "$DEBUG" = true ] ; then
  set -x
  log DEBUG "Setting debug"
fi

module load netcdf

# Move inputs to current (job) directory
ln -sf ${inputDirectory}/* .

# Link symbolically the executables
ln -sf ${execDirectory}/adcprep .
ln -sf ${execDirectory}/padcirc .

# generate the two prep files
WRITE_PROC=${writeProcesses}

# Question, which of  these is correct for total node count?
CORES=${SLURM_TACC_CORES}
PCORES=$(( $CORES-$WRITE_PROC ))

log INFO "ADCIRC will run on a total of ${CORES} cores, ${PCORES} for computation, ${WRITE_PROC} for data otuput"

printf "${PCORES}\n1\nfort.14\n" | adcprep
printf "${PCORES}\n2\n" | adcprep

ls -lat

ibrun -np $CORES ./padcirc -W $WRITE_PROC >> output.eo.txt 2>&1

if [ ! $? ]; then
	log ERROR "ADCIRC exited with an error status. $?" 
	exit
fi

exit 0
