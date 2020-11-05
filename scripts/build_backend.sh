#!/bin/bash

rm -rf build_cmake
mkdir build_cmake
pushd build_cmake

INSTALL_PREFIX=`pwd`/../install
rm -rf ${INSTALL_PREFIX}

PYTHON_INC=`${PYTHON} -c "import distutils.sysconfig;                  \
                        print(distutils.sysconfig.get_python_inc())"`
NUMPY_INC=`${PYTHON} -c "import numpy; print(numpy.get_include())"`
DPCPP_ROOT=${ONEAPI_ROOT}/compiler/latest/linux/

cmake                                                       \
    -DCMAKE_BUILD_TYPE=Release                              \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}                \
    -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}                   \
    -DDPCPP_ROOT=${DPCPP_ROOT}                              \
    -DPYTHON_INCLUDE_DIR=${PYTHON_INC}                      \
    -DNUMPY_INCLUDE_DIR=${NUMPY_INC}                        \
    ../backends

make -j 4 && make install

popd
cp install/lib/*.so dpctl/

mkdir -p dpctl/include
cp -r backends/include/* dpctl/include
