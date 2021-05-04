#!/bin/bash
set +xe
rm -rf build
mkdir build
pushd build

INSTALL_PREFIX=`pwd`/../install
rm -rf ${INSTALL_PREFIX}

if [[ -z "${DPCPP_HOME}" ]]; then
    echo "Set the DPCPP_HOME environment variable to root directory."
fi

# Check if level-zero headers are installed. Currently, works only for Ubuntu.
# Check https://dgpu-docs.intel.com/technologies/level-zero.html for details
# about what development package should be checked for different distros.
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$NAME" == "Ubuntu" ]]; then
        dpkg -s level-zero-dev
        if [[ $? == 0 ]]; then
            export USE_LO_HEADERS="ON"
        else
            export USE_LO_HEADERS="OFF"
        fi
    fi
fi

cmake                                                       \
    -DCMAKE_BUILD_TYPE=Debug                                \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}                \
    -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}                   \
    -DDPCTL_CUSTOM_DPCPP_INSTALL_DIR=${DPCPP_HOME}          \
    -DCMAKE_LINKER:PATH=${DPCPP_HOME}/bin/lld               \
    -DDPCTL_ENABLE_LO_PROGRAM_CREATION=${USE_LO_HEADERS}    \
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
