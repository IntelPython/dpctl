#!/bin/bash -x

export PYBIND11_INCLUDES=$(python3 -m pybind11 --includes)
export DPCTL_INCLUDE_DIR=$(python -c "import dpctl; print(dpctl.get_include())")
export DPCTL_LIB_DIR=${DPCTL_INCLUDE_DIR}/..
export PY_EXT_SUFFIX=$(python3-config --extension-suffix)
export HOST_COMPILER_FLAGS="-g -std=c++2a -O3 -Wno-return-type -Wno-deprecated-declarations -fPIC ${PYBIND11_INCLUDES} -I${DPCTL_INCLUDE_DIR}"

#    -fsycl-host-compiler=g++ \
#    -fsycl-host-compiler-options="${HOST_COMPILER_FLAGS}" \

dpcpp -O3 -fsycl -Wno-deprecated-declarations \
    -fpic -fPIC -shared \
     ${PYBIND11_INCLUDES} -I${DPCTL_INCLUDE_DIR} \
    sycl_gemm.cpp -o _sycl_gemm${PY_EXT_SUFFIX}
