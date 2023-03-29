#!/bin/bash
# 
# SHELL DEMO  - Cleanup Script
# Carlos del-Castillo-Negrete - cdelcastillo21@gmail.com
# June 2021
#
# Basic cleanup and package outputs from runs

log () {
  echo "$(date) | CLEANUP | ${1} | ${2}" 
}

DEBUG=false

if [ "$DEBUG" = true ] ; then
  set -x
fi

log INFO "STARTING"

# TODO: Add cleanup routines for run

log INFO "DONE"

if [ "$DEBUG" = true ] ; then
  set +x
fi

exit 0
