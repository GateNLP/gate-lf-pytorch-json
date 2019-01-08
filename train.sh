#!/bin/bash

# run training on file exported by GATE

metafile="$1"
shift
modelname="$1"
shift

# NOTE: all the missing arguments handling is done by the python script to make this more portable!

versionpython="UNKNOWN"
wherepython=`which python`
if [[ "x$wherepython" != "x" ]]
then
  versionpython=`python -V |& cut -f 2 -d " " | cut -f 1 -d'.'`
fi
if [[ "$versionpython" == "3" ]]
then
  pythoncmd=$wherepython
else
  wherepython=`which python3`
  if [[ "x$wherepython" == "x" ]]
  then
    echo 'ERROR: could not find a python 3 interpreter, exiting'
    exit 1
  fi
fi
PRG="$0"
CURDIR="`pwd`"
# need this for relative symlinks
while [ -h "$PRG" ] ; do
  ls=`ls -ld "$PRG"`
  link=`expr "$ls" : '.*-> \(.*\)$'`
  if expr "$link" : '/.*' > /dev/null; then
    PRG="$link"
  else
    PRG=`dirname "$PRG"`"/$link"
  fi
done
SCRIPTDIR=`dirname "$PRG"`
SCRIPTDIR=`cd "$SCRIPTDIR"; pwd -P`
GATELFDATA=`cd "$SCRIPTDIR"/../gate-lf-python-data; pwd -P`
export PYTHONPATH="$GATELFDATA"
echo DEBUG PYTHON ${wherepython}
echo DEBUG PYTHONPATH $PYTHONPATH
echo DEBUG SCRIPTDIR $SCRIPTDIR
echo DEBUG RUNNING ${wherepython} "${SCRIPTDIR}"/train.py "$metafile" "$modelname" "$@"
${wherepython} "${SCRIPTDIR}"/train.py "$metafile" "$modelname" "$@" 



