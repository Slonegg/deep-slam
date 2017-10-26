#!/bin/bash
# This script should be sourced before using this repo (for development).
# It creates the python virtualenv and using pip to populate it
# This only run to setup the development environment.
# Installation is handled by setup.py/disttools.

# create virtualenv
if [ -d env ]; then
    echo "virtual environment env exist"
else
    if [ "$(expr substr $(uname -s) 1 5)" == "MINGW" ]; then
        virtualenv env --prompt "(dslam)"
    else
        virtualenv -p python3 env --prompt "(dslam)"
    fi
fi

# activate env & install requirements
if [ -f ./env/bin/activate ]; then
    source ./env/bin/activate
elif [ -f ./env/Scripts/activate ]; then
    source ./env/Scripts/activate
else
    echo "virtualenv activation script not found!"
    return
fi
pip install -r requirements.txt

# make our importable
pip install -e .
