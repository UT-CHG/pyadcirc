#!/usr/bin/env bash

# Pylauncher Tapis Application - Test Script
#   Carlos del-Castillo-Negrete
#   June 2021
#
# Test script for Tapis

DEBUG=true

if [ "$DEBUG" = true ] ; then
  set -x
fi

# IDEAS HERE -> Run generator script, but only first couple of jobs? Maybe with a smaller job count? Explore more ideas to test common failures.
echo "Success"

if [ "$DEBUG" = true ] ; then
  set +x
fi

exit 0
