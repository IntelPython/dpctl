#!/bin/bash
set +xe
rm -rf build_cmake
mkdir build_cmake
pushd build_cmake

INSTALL_PREFIX=`pwd`/../install
rm -rf ${INSTALL_PREFIX}
export ONEAPI_ROOT=/opt/intel/oneapi

DPCPP_ROOT=${ONEAPI_ROOT}/compiler/latest/linux

cmake                                                       \
    -DCMAKE_BUILD_TYPE=Debug                                \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}                \
    -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}                   \
    -DDPCPP_INSTALL_DIR=${DPCPP_ROOT}                       \
    -DCMAKE_C_COMPILER:PATH=${DPCPP_ROOT}/bin/clang         \
    -DCMAKE_CXX_COMPILER:PATH=${DPCPP_ROOT}/bin/dpcpp       \
    -DDPCTL_BUILD_CAPI_TESTS=ON                             \
    -DDPCTL_GENERATE_COVERAGE=ON                            \
    ../dpctl-capi

make V=1 -n -j 4 && make check && make install

if [ $? -ne 0 ]; then
    echo "Building of libDPCTLSyclInterface failed. Abort!"
    exit 1
fi

# To run code coverage for dpctl-c API
make llvm-cov
# For more verbose tests use:
# cd tests
# ctest -V --progress --output-on-failure -j 4
# cd ..

popd
cp install/lib/*.so dpctl/

mkdir -p dpctl/include
cp -r dpctl-capi/include/* dpctl/include

export DPCTL_SYCL_INTERFACE_LIBDIR=dpctl
export DPCTL_SYCL_INTERFACE_INCLDIR=dpctl/include

export CC=${DPCPP_ROOT}/bin/clang
export CXX=${DPCPP_ROOT}/bin/dpcpp
# FIXME: How to pass this using setup.py? The fPIC flag is needed when
# dpcpp compiles the Cython generated cpp file.
export CFLAGS=-fPIC
python setup.py clean --all
python setup.py build develop
python -m unittest -v dpctl.tests
