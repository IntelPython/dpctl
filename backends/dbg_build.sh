#!/bin/bash
set +xe
rm -rf build
mkdir build
pushd build

INSTALL_PREFIX=`pwd`/../install
rm -rf ${INSTALL_PREFIX}
export ONEAPI_ROOT=/opt/intel/oneapi
DPCPP_ROOT=${ONEAPI_ROOT}/compiler/latest/linux

cmake                                                       \
    -DCMAKE_BUILD_TYPE=Debug                                \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}                \
    -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}                   \
    -DDPCPP_ROOT=${DPCPP_ROOT}                              \
    -DCMAKE_C_COMPILER:PATH=${DPCPP_ROOT}/bin/clang         \
    -DCMAKE_CXX_COMPILER:PATH=${DPCPP_ROOT}/bin/dpcpp       \
    -DBUILD_CAPI_TESTS=ON                                   \
    ..

make V=1 -n -j 4 && make check && make install

popd
