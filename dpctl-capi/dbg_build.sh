#!/bin/bash
set +xe
rm -rf build
mkdir build
pushd build

INSTALL_PREFIX=`pwd`/../install
rm -rf ${INSTALL_PREFIX}

if [[ -z "${DPCPP_HOME}" ]]; then
    ONEAPI_ROOT=/opt/intel/oneapi
    DPCPP_ROOT=${ONEAPI_ROOT}/compiler/latest/linux
else
    DPCPP_ROOT=${DPCPP_HOME}
fi

cmake                                                       \
    -DCMAKE_BUILD_TYPE=Debug                                \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}                \
    -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}                   \
    -DDPCPP_INSTALL_DIR=${DPCPP_ROOT}                       \
    -DCMAKE_C_COMPILER:PATH=${DPCPP_ROOT}/bin/clang         \
    -DCMAKE_CXX_COMPILER:PATH=${DPCPP_ROOT}/bin/clang++     \
    -DCMAKE_CMAKE_LINKER:PATH=${DPCPP_ROOT}/bin/lld         \
    -DCMAKE_CXX_FLAGS=" -fsycl "                            \
    -DDPCTL_ENABLE_LO_PROGRAM_CREATION=ON                   \
    -DDPCTL_BUILD_CAPI_TESTS=ON                             \
    -DDPCTL_GENERATE_COVERAGE=ON                            \
    ..

make V=1 -n -j 4 && make check && make install

# Turn on to generate coverage report html files
make lcov-genhtml

# For more verbose tests use:
# cd tests
# ctest -V --progress --output-on-failure -j 4
# cd ..

popd

