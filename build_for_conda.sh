#!/bin/bash

CONDA_PKG_DIR=${PWD}/dppy_conda_pkg
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
	    echo "Aborting dp-glue setup"
	    exit 1
	    ;;
    esac
fi

export ONEAPI_ROOT="/opt/intel/inteloneapi"

conda build --output-folder ${CONDA_PKG_DIR} ${CHANNELS} conda.recipe

# Commented because main goal of this script is building the package.
# You can run the following commands manually if you want to install the package
# to your current conda environment.

# conda install dppy -c ${CONDA_PKG_DIR} ${CHANNELS}

# Indexing is commented because conda-build indexes the output folder.
# Aslo, the current script clears any old build directories.

# echo "conda index"
# conda index ${CONDA_PKG_DIR}
