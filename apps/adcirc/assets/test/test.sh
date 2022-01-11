#!/bin/bash

DIR=$( cd "$( dirname "$0" )" && pwd )

# Make test run dir, copy app assets to there
mkdir $DIR/test_run
cp -r $DIR/inputs $DIR/test_run/inputs
cp $DIR/../wrapper.sh $DIR/test_run/wrapper.sh
cd $DIR/test_run


# set test variables
export inputDirectory="inputs"
export writeProcesses=2

# Agave env variables that need to be set to imitate execution
export AGAVE_JOB_NODE_COUNT=1
export AGAVE_JOB_PROCESSORS_PER_NODE=12
export AGAVE_JOB_NAME=adcirc_test_job

# call wrapper script as if the values had been injected by the API
sh -c ./wrapper.sh

# clean up after the run completes
# rm -rf $DIR/test_run
