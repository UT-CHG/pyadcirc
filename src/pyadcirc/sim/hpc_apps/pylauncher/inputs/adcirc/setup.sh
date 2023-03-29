#!/bin/bash
#
# ADCIRC PYLAUNCHER - Setup Script
# Carlos del-Castillo-Negrete - cdelcastillo21@gmail.com
# June 2021
#
# Basic set up required before running all job runs.
# Create run and output directories

log () {
  # Note this file logs to stdout, since it is expected
  # to be run by the main application and capture output
  # in the main log file. Using this log function helps
  # keep logs consistent.
  echo "$(date) | SETUP | ${1} | ${2}"
}

DEBUG=false

if [ "$DEBUG" = true ] ; then
  set -x
fi

log INFO "STARTING"
log DEBUG "cwd is $(pwd)"

# Create directories all runs will need
log INFO "Making runs, logs, and output directories"
mkdir jobs # active job runs
mkdir outputs # processed outputs

# Make pre/post shell scripts executable
log INFO "Making pre/post process scripts executable."
chmod +x inputs/post_process.sh inputs/pre_process.sh

# Trash directory - Clean-up periodically
mkdir .trash

log INFO "DONE"

if [ "$DEBUG" = true ] ; then
  set +x
fi

exit 0
