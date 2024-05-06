#!/bin/bash

# This is necessary to help DPC++ find Intel libraries such as SVML, IRNG, etc in build prefix
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${BUILD_PREFIX}/lib"

# Intel LLVM must cooperate with compiler and sysroot from conda
echo "--gcc-toolchain=${BUILD_PREFIX} --sysroot=${BUILD_PREFIX}/${HOST}/sysroot -target ${HOST}" > icpx_for_conda.cfg
export ICPXCFG="$(pwd)/icpx_for_conda.cfg"
export ICXCFG="$(pwd)/icpx_for_conda.cfg"

if [ -e "_skbuild" ]; then
    ${PYTHON} setup.py clean --all
fi
export CMAKE_GENERATOR="Ninja"
SKBUILD_ARGS="-- -DCMAKE_C_COMPILER:PATH=icx -DCMAKE_CXX_COMPILER:PATH=icpx -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON"
echo "${PYTHON} setup.py install ${SKBUILD_ARGS}"

if [ -n "${WHEELS_OUTPUT_FOLDER}" ]; then
    # Install packages and assemble wheel package from built bits
    WHEELS_BUILD_ARGS="-p manylinux_2_28_x86_64 --build-number ${GIT_DESCRIBE_NUMBER}"
    ${PYTHON} setup.py install bdist_wheel ${WHEELS_BUILD_ARGS} ${SKBUILD_ARGS}
    cp dist/dpctl*.whl ${WHEELS_OUTPUT_FOLDER}
else
    # Perform regular install
    ${PYTHON} setup.py install ${SKBUILD_ARGS}
fi

# need to create this folder so ensure that .dpctl-post-link.sh can work correctly
mkdir -p $PREFIX/etc/OpenCL/vendors
echo "dpctl creates symbolic link to system installed /etc/OpenCL/vendors/intel.icd as a work-around." > $PREFIX/etc/OpenCL/vendors/.dpctl_readme

cp $RECIPE_DIR/dpctl-post-link.sh $PREFIX/bin/.dpctl-post-link.sh
cp $RECIPE_DIR/dpctl-pre-unlink.sh $PREFIX/bin/.dpctl-pre-unlink.sh
