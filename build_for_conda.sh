#!/bin/bash

CONDA_PKG_DIR=dppy_conda_pkg
# Remove any old build directory
rm -rf ./${CONDA_PKG_DIR}
# Recreate the build directory
mkdir ./${CONDA_PKG_DIR}

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

echo "conda-build"
conda-build ${DPPY_EXTRA_CHANNELS} --output-folder ./${CONDA_PKG_DIR} -c conda-forge conda.recipe/
echo "conda install"
conda install dppy -c `pwd`/${CONDA_PKG_DIR}  -c conda-forge
conda index `pwd`/${CONDA_PKG_DIR}
