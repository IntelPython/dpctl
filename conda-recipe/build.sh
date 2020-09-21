#!/bin/bash

# We need dpcpp to compile dppl_sycl_interface
if [ ! -z "${ONEAPI_ROOT}" ]; then
    # Suppress error b/c it could fail on Ubuntu 18.04
    source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh || true
    export CC=clang
    export CXX=clang++
else
    echo "DPCPP is needed to build DPPL. Abort!"
    exit 1
fi

rm -rf build_cmake
mkdir build_cmake
cd build_cmake

PYTHON_INC=`${PYTHON} -c "import distutils.sysconfig;                  \
                        print(distutils.sysconfig.get_python_inc())"`
NUMPY_INC=`${PYTHON} -c "import numpy; print(numpy.get_include())"`
DPCPP_ROOT=${ONEAPI_ROOT}/compiler/latest/linux/

cmake                                                       \
    -DCMAKE_BUILD_TYPE=Release                              \
    -DCMAKE_INSTALL_PREFIX=${PREFIX}                        \
    -DCMAKE_PREFIX_PATH=${PREFIX}                           \
    -DDPCPP_ROOT=${DPCPP_ROOT}                              \
    -DPYTHON_INCLUDE_DIR=${PYTHON_INC}                      \
    -DNUMPY_INCLUDE_DIR=${NUMPY_INC}                        \
    ../backends

make -j 4 && make install

cd ..

# required by dpctl.opencl_core
export DPPL_OPENCL_INTERFACE_LIBDIR=${PREFIX}
export DPPL_OPENCL_INTERFACE_INCLDIR=${PREFIX}/include
export OpenCL_LIBDIR=${DPCPP_ROOT}/lib

# required by dpctl.sycl_core
export DPPL_SYCL_INTERFACE_LIBDIR=${PREFIX}/lib
export DPPL_SYCL_INTERFACE_INCLDIR=${PREFIX}/include


# FIXME: How to pass this using setup.py? This flags is needed when
# dpcpp compiles the generated cpp file.
export CFLAGS="-fPIC -O3 ${CFLAGS}"
export LDFLAGS="-L OpenCL_LIBDIR ${LDFLAGS}"
${PYTHON} setup.py clean --all
${PYTHON} setup.py build
${PYTHON} setup.py install
