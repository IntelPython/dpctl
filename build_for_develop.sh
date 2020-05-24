#!/bin/bash
0;136;0c
rm -rf build
mkdir build
cd build

INSTALL_PREFIX=`pwd`/install
export ONEAPI_ROOT=/opt/intel/inteloneapi
DPCPP_ROOT=${ONEAPI_ROOT}/compiler/latest

cmake                                                       \
    -DCMAKE_BUILD_TYPE=Release                              \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}                \
    -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}                   \
    -DCMAKE_C_COMPILER:PATH=${DPCPP_ROOT}/linux/bin/clang   \
    -DCMAKE_CXX_COMPILER:PATH=${DPCPP_ROOT}/linux/bin/dpcpp \
    ..

make V=1 -n -j 4 && make install

cd ../python_binding
export DP_GLUE_LIBDIR=${INSTALL_PREFIX}/lib
export DP_GLUE_INCLDIR=${INSTALL_PREFIX}/include
export OPENCL_LIBDIR=/usr/lib/x86_64-linux-gnu/
export DPPY_ONEAPI_INTERFACE_LIBDIR=${INSTALL_PREFIX}/lib
export DPPY_ONEAPI_INTERFACE_INCLDIR=${INSTALL_PREFIX}/include

export CC=clang
export CXX=dpcpp
# FIXME: How to pass this using setup.py? The fPIC flag is needed when
# dpcpp compiles the Cython generated cpp file.
export CFLAGS=-fPIC
python setup.py clean --all
python setup.py build_ext --inplace
python setup.py develop
