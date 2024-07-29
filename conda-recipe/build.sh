#!/bin/bash

# This is necessary to help DPC++ find Intel libraries such as SVML, IRNG, etc in build prefix
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${BUILD_PREFIX}/lib"

# Intel LLVM must cooperate with compiler and sysroot from conda
echo "--gcc-toolchain=${BUILD_PREFIX} --sysroot=${BUILD_PREFIX}/${HOST}/sysroot -target ${HOST}" > icpx_for_conda.cfg
export ICPXCFG="$(pwd)/icpx_for_conda.cfg"
export ICXCFG="$(pwd)/icpx_for_conda.cfg"

read -r GLIBC_MAJOR GLIBC_MINOR <<<"$(conda list '^sysroot_linux-64$' \
    | tail -n 1 | awk '{print $2}' | grep -oP '\d+' | head -n 2 | tr '\n' ' ')"

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

pushd dist
${PYTHON} -m wheel unpack -d dpctl_wheel dpctl*.whl
export lib_name=libDPCTLSyclInterface
export so_full_path=$(find dpctl_wheel -regextype posix-extended -regex "^.*${lib_name}\.so")
export sox_full_path=$(find dpctl_wheel -regextype posix-extended -regex "^.*${lib_name}\.so\.[0-9]*$")
export soxxx_full_path=$(find dpctl_wheel -regextype posix-extended -regex "^.*${lib_name}\.so\.[0-9]*\.[0-9]*$")

rm -rf ${so_full_path} ${soxxx_full_path}

export so_name=$(basename ${so_full_path})
export sox_name=$(basename ${sox_full_path})
export soxxx_name=$(basename ${soxxx_full_path})
export wheel_path=$(dirname $(dirname ${so_full_path}))

# deal with hard copies
${PYTHON} -m wheel pack ${wheel_path}

rm -rf dpctl_wheel
popd

${PYTHON} -m wheel tags --remove --build "$GIT_DESCRIBE_NUMBER" \
    --platform-tag "manylinux_${GLIBC_MAJOR}_${GLIBC_MINOR}_x86_64" \
    dist/dpctl*.whl

${PYTHON} -m pip install dist/dpctl*.whl \
    --no-build-isolation \
    --no-deps \
    --only-binary :all: \
    --no-index \
    --prefix "${PREFIX}" \
    -vv

export libdir=$(find $PREFIX -name 'libDPCTLSyclInterface*' -exec dirname \{\} \;)

# Recover symbolic links
# libDPCTLSyclInterface.so.0 -> libDPCTLSyclInterface.so.0.17
# libDPCTLSyclInterface.so -> libDPCTLSyclInterface.so.0
mv ${libdir}/${sox_name} ${libdir}/${soxxx_name}
ln -s ${libdir}/${soxxx_name} ${libdir}/${sox_name}
ln -s ${libdir}/${sox_name} ${libdir}/${so_name}

# Copy wheel package
if [[ -d "${WHEELS_OUTPUT_FOLDER}" ]]; then
    cp dist/dpctl*.whl "${WHEELS_OUTPUT_FOLDER[@]}"
fi

# need to create this folder so ensure that .dpctl-post-link.sh can work correctly
mkdir -p $PREFIX/etc/OpenCL/vendors
echo "dpctl creates symbolic link to system installed /etc/OpenCL/vendors/intel.icd as a work-around." > $PREFIX/etc/OpenCL/vendors/.dpctl_readme

cp $RECIPE_DIR/dpctl-post-link.sh $PREFIX/bin/.dpctl-post-link.sh
cp $RECIPE_DIR/dpctl-pre-unlink.sh $PREFIX/bin/.dpctl-pre-unlink.sh
