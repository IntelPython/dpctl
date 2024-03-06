#!/bin/bash
set +xe
rm -rf build
mkdir build
pushd build || exit 1

INSTALL_PREFIX=$(pwd)/../install
rm -rf ${INSTALL_PREFIX}

cmake                                                       \
    -DCMAKE_BUILD_TYPE=Debug                                \
    -DCMAKE_C_COMPILER=icx                                  \
    -DCMAKE_CXX_COMPILER=icpx                               \
    -DCMAKE_CXX_FLAGS=-fsycl                                \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}                \
    -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}                   \
    -DDPCTL_ENABLE_L0_PROGRAM_CREATION=ON                   \
    -DDPCTL_BUILD_CAPI_TESTS=ON                             \
    ..

make V=1 -n -j 4 && make check && make install

# Turn on to generate coverage report html files reconfigure with
# -DDPCTL_GENERATE_COVERAGE=ON and then
# make lcov-genhtml

# For more verbose tests use:
# cd tests
# ctest -V --progress --output-on-failure -j 4
# cd ..

popd || exit 1
