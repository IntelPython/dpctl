#!/bin/bash

${PYTHON} setup.py clean --all
${PYTHON} setup.py install --sycl-compiler-prefix=$CONDA_PREFIX

# Build wheel package
if [ "$CONDA_PY" == "36" ]; then
    WHEELS_BUILD_ARGS="-p manylinux1_x86_64"
else
    WHEELS_BUILD_ARGS="-p manylinux2014_x86_64"
fi
if [ -n "${WHEELS_OUTPUT_FOLDER}" ]; then
    # We need dpcpp to compile dpctl_sycl_interface
    if [ ! -z "${ONEAPI_ROOT}" ]; then
        # Suppress error b/c it could fail on Ubuntu 18.04
        source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh || true
    else
        echo "DPCPP is needed to build DPCTL. Abort!"
        exit 1
    fi
    $PYTHON setup.py bdist_wheel ${WHEELS_BUILD_ARGS}
    cp dist/dpctl*.whl ${WHEELS_OUTPUT_FOLDER}
fi
