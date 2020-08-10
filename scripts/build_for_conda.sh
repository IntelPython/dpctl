#!/bin/bash

CONDA_PKG_DIR=${PWD}/dppl_conda_pkg
# Remove any old build directory
rm -rf ${CONDA_PKG_DIR}
# Recreate the build directory
mkdir ${CONDA_PKG_DIR}

CHANNELS="-c defaults"

# Check if conda-build is installed
conda-build --version

conda_build_ret=$?
if [[ conda_build_ret -ne 0 ]]; then
    echo "conda-build needs to be installed. Do you want to do it now? [y/N]"
    read ok
    shopt -s nocasematch
    case "$ok" in
    "y" )
        conda install conda-build ${CHANNELS}
        ;;
    *)
        echo "Aborting PyDPPL setup"
        exit 1
        ;;
    esac
fi

export ONEAPI_ROOT="/opt/intel/oneapi"

conda build --output-folder ${CONDA_PKG_DIR} ${CHANNELS} conda-recipe

# To install the package in your current conda environment, execute

# conda install pydppl -c ${CONDA_PKG_DIR} ${CHANNELS}
