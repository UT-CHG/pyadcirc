#!/bin/bash

esc() {
    # Escape single line characters and new lines
    v=${1//$'\''/\\\'}
    v=${v//$'\n'/\\n}
    echo $v
}

log () {
  local escaped=$(esc "$3")
  echo "$(date) | $1 | $2 | '$escaped'"
}

# Print helpFunction in case parameters are empty
if [ -z "$1" ] 
then
   log ERROR "Must specify at least a command";
   helpFunction
fi

# Begin script in case all parameters are correct
full_command=$(echo $1 ${@:2})
log START MAIN "Executing: $full_command" 
res=$($full_command  2>&1)
log DATA COMMAND_RES "$($full_command  2>&1)"
log STOP MAIN "Done"
