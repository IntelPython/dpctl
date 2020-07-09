#!/bin/bash

# We need dpcpp to compile dppl_oneapi_interface
if [ ! -z "${ONEAPI_ROOT}" ]; then
    # Suppress error b/c it could fail on Ubuntu 18.04
    source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh || true
    export CC=clang
    export CXX=dpcpp
else
    echo "DPCPP is needed to build DPPL. Abort!"
    exit 1
fi

mkdir build
cd build

cmake                                    \
    -DCMAKE_BUILD_TYPE=Release           \
    -DCMAKE_INSTALL_PREFIX=${PREFIX}     \
    -DCMAKE_PREFIX_PATH=${PREFIX}        \
    ..

make -j 4 && make install

cd ../python_binding

# required by dpglue
export DP_GLUE_LIBDIR=${PREFIX}
export DP_GLUE_INCLDIR=${PREFIX}/include
export OpenCL_LIBDIR=${ONEAPI_ROOT}/compiler/latest/linux/lib
# required by oneapi_interface
export DPPL_ONEAPI_INTERFACE_LIBDIR=${INSTALL_PREFIX}/lib
export DPPL_ONEAPI_INTERFACE_INCLDIR=${INSTALL_PREFIX}/include

# FIXME: How to pass this using setup.py? This flags is needed when
# dpcpp compiles the generated cpp file.
export CFLAGS="-fPIC -O3 ${CFLAGS}"
export LDFLAGS="-L OpenCL_LIBDIR ${LDFLAGS}"
${PYTHON} setup.py clean --all
${PYTHON} setup.py build
${PYTHON} setup.py install
