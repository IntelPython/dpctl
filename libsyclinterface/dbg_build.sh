#!/bin/bash
set +xe
rm -rf build
mkdir build
pushd build || exit 1

INSTALL_PREFIX=$(pwd)/../install
rm -rf ${INSTALL_PREFIX}

# With DPC++ 2024.0 adn newer set these to ensure that
# cmake can find llvm-cov and other utilities
LLVM_TOOLS_HOME=${CMPLR_ROOT}/bin/compiler
PATH=$PATH:${CMPLR_ROOT}/bin/compiler

cmake                                                       \
    -DCMAKE_BUILD_TYPE=Debug                                \
    -DCMAKE_C_COMPILER=icx                                  \
    -DCMAKE_CXX_COMPILER=icpx                               \
    -DCMAKE_CXX_FLAGS=-fsycl                                \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}                \
    -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}                   \
    -DDPCTL_ENABLE_L0_PROGRAM_CREATION=ON                   \
    -DDPCTL_BUILD_CAPI_TESTS=ON                             \
    -DDPCTL_GENERATE_COVERAGE=OFF                           \
    ..

# build
make V=1 -n -j 4
# run ctest
make check
# install
make install

# Turn on to generate coverage report html files reconfigure with
# -DDPCTL_GENERATE_COVERAGE=ON and then
# make llvm-cov-report

# For more verbose tests use:
# cd tests
# ctest -V --progress --output-on-failure -j 4
# cd ..

popd || exit 1
