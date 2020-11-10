#!/bin/bash

rm -rf build_cmake
mkdir build_cmake
pushd build_cmake

INSTALL_PREFIX=`pwd`/../install
rm -rf ${INSTALL_PREFIX}

DPCPP_ROOT=${ONEAPI_ROOT}/compiler/latest/linux/

cmake                                                       \
    -DCMAKE_BUILD_TYPE=Release                              \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}                \
    -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}                   \
    -DDPCPP_ROOT=${DPCPP_ROOT}                              \
    -DCMAKE_C_COMPILER:PATH=${DPCPP_ROOT}/bin/clang         \
    -DCMAKE_CXX_COMPILER:PATH=${DPCPP_ROOT}/bin/dpcpp       \
    ../backends

make -j 4 && make install

popd
cp install/lib/*.so dpctl/

mkdir -p dpctl/include
cp -r backends/include/* dpctl/include
