#!/bin/bash
set +xe
rm -rf build_cmake
mkdir build_cmake
pushd build_cmake

INSTALL_PREFIX=`pwd`/../install
rm -rf ${INSTALL_PREFIX}
export ONEAPI_ROOT=/opt/intel/oneapi

PYTHON_INC=`python -c "import distutils.sysconfig;                  \
                        print(distutils.sysconfig.get_python_inc())"`
NUMPY_INC=`python -c "import numpy; print(numpy.get_include())"`
DPCPP_ROOT=${ONEAPI_ROOT}/compiler/latest/linux

cmake                                                       \
    -DCMAKE_BUILD_TYPE=Debug                                \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}                \
    -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}                   \
    -DDPCPP_ROOT=${DPCPP_ROOT}                              \
    -DCMAKE_C_COMPILER:PATH=${DPCPP_ROOT}/bin/clang         \
    -DCMAKE_CXX_COMPILER:PATH=${DPCPP_ROOT}/bin/dpcpp       \
    -DPYTHON_INCLUDE_DIR=${PYTHON_INC}                      \
    -DNUMPY_INCLUDE_DIR=${NUMPY_INC}                        \
    -DGTEST_INCLUDE_DIR=${CONDA_PREFIX}/include/            \
    -DGTEST_LIB_DIR=${CONDA_PREFIX}/lib                     \
    ../backends

make V=1 -n -j 4 && make install
popd
cp install/lib/*.so dpctl/

mkdir -p dpctl/include
cp -r backends/include/* dpctl/include

export DPPL_OPENCL_INTERFACE_LIBDIR=dpctl
export DPPL_OPENCL_INTERFACE_INCLDIR=dpctl/include
# /usr/lib/x86_64-linux-gnu/
export OpenCL_LIBDIR=${DPCPP_ROOT}/lib
export DPPL_SYCL_INTERFACE_LIBDIR=dpctl
export DPPL_SYCL_INTERFACE_INCLDIR=dpctl/include

export CC=clang
export CXX=dpcpp
# FIXME: How to pass this using setup.py? The fPIC flag is needed when
# dpcpp compiles the Cython generated cpp file.
export CFLAGS=-fPIC
python setup.py clean --all
python setup.py build develop
