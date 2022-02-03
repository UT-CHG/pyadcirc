#!/bin/bash
# 
# SHELL DEMO  - Cleanup Script
# Carlos del-Castillo-Negrete - cdelcastillo21@gmail.com
# June 2021
#
# Basic cleanup and package outputs from runs

log () {
  echo "$(date) - CLEANUP - ${1} - ${2}" 
}

DEBUG=false

if [ "$DEBUG" = true ] ; then
  set -x
fi

log INFO "STARTING"

log INFO "Packaing log files"
cp run.log logs/run.log
cp runs/*/*.log logs/
cd logs; zip -r ../logs.zip *; cd ..

log INFO "Packaging output files"
cd outputs; zip -r ../outputs.zip *; cd ..

log INFO "DONE"

if [ "$DEBUG" = true ] ; then
  set +x
fi

exit 0
