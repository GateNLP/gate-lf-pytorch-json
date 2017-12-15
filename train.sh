#!/bin/bash

echo '!!!!!!!!!!!!!!!!!!!!!! RUNNING train.sh'
echo 'args:' "$@"

# This script should get invoked with the following parameters:
# * absolute path to the meta file
# * absolute path (prefix) to the model file. 
# * any additional parameters set by the user to pass on to the training step
# In addition the script will make use of the following environment variables if set:
# GATE_LF_PYTHONCMD - if set will get used to invoke python instead of "python"

pythoncmd=python
if [[ "x${GATE_LF_PYTHONCMD}" != "x" ]]
then
  pythoncmd="${GATE_LF_PYTHONCMD}"
fi

"${pythoncmd}" 
