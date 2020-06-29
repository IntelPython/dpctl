#!/bin/bash

CONDA_PKG_DIR=${PWD}/dppy_conda_pkg
# Remove any old build directory
rm -rf ${CONDA_PKG_DIR}
# Recreate the build directory
mkdir ${CONDA_PKG_DIR}

# Check if conda-build is installed
conda-build --version

conda_build_ret=$?
if [[ conda_build_ret -ne 0 ]]; then
    echo "conda-build needs to be installed. Do you want to do it now? [y/N]"
    read ok
    shopt -s nocasematch
    case "$ok" in
	"y" )
	    conda install conda-build -c conda-forge
	    ;;
	*)
	    echo "Aborting dp-glue setup"
	    exit 1
	    ;;
    esac
fi

export ONEAPI_ROOT="/opt/intel/inteloneapi"
export OpenCL_LIBDIR="/usr/lib/x86_64-linux-gnu"

CHANNELS="-c defaults"

conda build --output-folder ${CONDA_PKG_DIR} ${CHANNELS} conda.recipe
conda install dppy -c ${CONDA_PKG_DIR} ${CHANNELS}
echo "conda index"
conda index ${CONDA_PKG_DIR}
