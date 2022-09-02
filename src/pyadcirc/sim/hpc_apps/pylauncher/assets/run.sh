#!/usr/bin/env bash

# Pylauncher Tapis Application - Main entry-point
#   Carlos del-Castillo-Negrete - cdelcastillo21@gmail.com
#   June 2021
#
# Main entrypoint called in execution envrionment by Tapis to start the job.

DEBUG=true

log () {
  echo "$(date) | PYLAUNCHER | ${1} | ${2}"
}

if [ "$DEBUG" = true ] ; then
  log DEBUG "Setting debug"
  set -x
fi

log INFO "STARTING"

log INFO "App ID: ${TACCJM_APPID}"
log INFO "Job Inputs : ${job_inputs}"
log INFO "Custom Modules : ${custom_modules}"
log INFO "Pylauncher Input Fname : ${pylauncher_input}"
log INFO "Generator Args : ${generator_args}"

# Load necessary modules - These are the modules required for all executed jobs.
log INFO "Loading modules python3 remora ${custom_modules}"
module load python3 1>&2
module load remora 1>&2
module load ${custom_modules} 1>&2


# Unzip job inptus directory (place in inputs)
log INFO "Unzipping job inputs directory"
unzip -j ${job_inputs} -d inputs 1>&2
rm ${job_inputs}

# If set-up script exists, run it first
if [ -e inputs/setup.sh ]
then
  chmod +x inputs/setup.sh
  log INFO "Running setup script"
  inputs/setup.sh
  if [ $? -ne 0 ]
  then
    log ERROR "Setup script failed!"
    exit 1
  else
    log INFO "Setup script complete."
  fi
fi

if [ -e inputs/generator.sh ]
then
  # Change permissions of generator script so it can be executed
  chmod +x inputs/generator.sh
fi

# Main Execution Loop:
#   - Call generator script.
#   - Calls pylauncher on generated input file. Expected name = jobs_list.csv
#   - Repeats until generator script returns no input file for pylauncher.
ITER=1
while :
do

  if [ -e inputs/generator.sh ]
  then
    # Call generator if it exists script
    log INFO "Calling Generator Script With arugments ${generator_args}"
    inputs/generator.sh ${ITER} ${SLURM_NPROCS} ${generator_args}
    if [ $? -ne 0 ]
    then
      log ERROR "Generator script failed on iteration ${ITER}!"
      # Fail gracefully here?
      exit 1
    else
      log INFO "Generator script completed for iteration ${ITER}."
    fi
  fi

  # If input file for pylauncher has been generated, then start pylauncher
  if [ -e ${pylauncher_input} ]
  then
    # Launch pylauncher on generated input file
    log INFO "Starting pylauncher for iteration ${ITER}"
    python3 ${TACCJM_APPID}/pylauncher4.py ${pylauncher_input}
    if [ $? -ne 0 ]
    then
      log ERROR "Pylauncher failed on iteration ${ITER}!"
      exit 1
    else
      log INFO "Pylauncher done for iteration ${ITER}"
    fi

    # Save pylauncher input file used.
    log INFO "Archiving ${ITER} pylauncher input file"
    mv ${pylauncher_input} ${pylauncher_input}_${ITER}
  else
    # No input for pylauncher, done.
    log INFO "No Input for Pylauncher found on iter ${ITER}, exiting"
    break
  fi

  ITER=$(( $ITER + 1 ))
done

log INFO "Done with execution of pylauncher applicaiton."

# If cleanup script exists, run it last
if [ -e inputs/cleanup.sh ]
then
  # Make set-up script executable and run
  chmod +x inputs/cleanup.sh
  log INFO "Running cleanup script"
  inputs/cleanup.sh
  if [ $? -ne 0 ]
  then
    log ERROR "Cleanup script failed!"
    exit 1
  else
    log INFO "Cleanup script complete. Exiting job"
  fi
fi

log INFO "DONE"

if [ "$DEBUG" = true ] ; then
  set +x
fi

exit 0
