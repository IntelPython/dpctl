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

export CC=icx
export CXX=icpx

export CMAKE_GENERATOR=Ninja
# Make CMake verbose
export VERBOSE=1

CMAKE_ARGS="${CMAKE_ARGS} -DDPCTL_LEVEL_ZERO_INCLUDE_DIR=${PREFIX}/include/level_zero"

# -wnx flags mean: --wheel --no-isolation --skip-dependency-check
${PYTHON} -m build -w -n -x
${PYTHON} -m wheel tags --remove --build "$GIT_DESCRIBE_NUMBER" \
    --platform-tag manylinux2014_x86_64 dist/dpctl*.whl
${PYTHON} -m pip install dist/dpctl*.whl \
    --no-build-isolation \
    --no-deps \
    --only-binary :all: \
    --no-index \
    --prefix "${PREFIX}" \
    -vv

# Recover symbolic links
# libDPCTLSyclInterface.so.0 -> libDPCTLSyclInterface.so.0.17
# libDPCTLSyclInterface.so -> libDPCTLSyclInterface.so.0
find $PREFIX | grep libDPCTLSyclInterface | sort -r | \
awk '{if ($0=="") ln=""; else if (ln=="") ln = $0; else system("rm " $0 ";\tln -s " ln " " $0); ln = $0 }'

# Copy wheel package
if [[ -v WHEELS_OUTPUT_FOLDER ]]; then
    cp dist/dpctl*.whl "${WHEELS_OUTPUT_FOLDER[@]}"
fi

# need to create this folder so ensure that .dpctl-post-link.sh can work correctly
mkdir -p $PREFIX/etc/OpenCL/vendors
echo "dpctl creates symbolic link to system installed /etc/OpenCL/vendors/intel.icd as a work-around." > $PREFIX/etc/OpenCL/vendors/.dpctl_readme

cp $RECIPE_DIR/dpctl-post-link.sh $PREFIX/bin/.dpctl-post-link.sh
cp $RECIPE_DIR/dpctl-pre-unlink.sh $PREFIX/bin/.dpctl-pre-unlink.sh
