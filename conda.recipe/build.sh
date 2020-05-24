#!/bin/bash

# We need dpcpp to compile dppy_oneapi_interface
if [ ! -z "${ONEAPI_ROOT}" ]; then
    source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh
    export CC=clang
    export CXX=dpcpp
else
    echo "DPCPP is needed to build DPPY. Abort!"
    exit 1
fi

rm -rf build
mkdir build
cd build

cmake                                    \
    -DCMAKE_BUILD_TYPE=Release           \
    -DCMAKE_INSTALL_PREFIX=${PREFIX}     \
    -DCMAKE_PREFIX_PATH=${PREFIX}        \
    ..

make -n && make V=1 -j 4 && make install

cd ../python_binding
export DP_GLUE_LIBDIR=${PREFIX}
export DP_GLUE_INCLDIR=${PREFIX}/include
export OPENCL_LIBDIR=${BUILD_PREFIX}/lib
export DPPY_ONEAPI_INTERFACE_LIBDIR=${INSTALL_PREFIX}/lib
export DPPY_ONEAPI_INTERFACE_INCLDIR=${INSTALL_PREFIX}/include

# FIXME: How to pass this using setup.py? This flags is needed when
# dpcpp compiles the generated cpp file.
export CFLAGS=-fPIC
${PYTHON} setup.py clean --all
${PYTHON} setup.py build
${PYTHON} setup.py install
