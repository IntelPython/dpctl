#!/bin/bash
set +xe
rm -rf build
mkdir build
pushd build

INSTALL_PREFIX=`pwd`/../install
export ONEAPI_ROOT=/opt/intel/oneapi
DPCPP_ROOT=${ONEAPI_ROOT}/compiler/latest/linux
PYTHON_INC=`python -c "import distutils.sysconfig;                  \
                        print(distutils.sysconfig.get_python_inc())"`
NUMPY_INC=`python -c "import numpy; print(numpy.get_include())"`

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
    ..

make V=1 -n -j 4
make check
make install


popd
