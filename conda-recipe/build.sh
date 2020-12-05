#!/bin/bash

# We need dpcpp to compile dpctl_sycl_interface
if [ ! -z "${ONEAPI_ROOT}" ]; then
    # Suppress error b/c it could fail on Ubuntu 18.04
    source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh || true
else
    echo "DPCPP is needed to build DPCTL. Abort!"
    exit 1
fi

${PYTHON} setup.py clean --all
${PYTHON} setup.py install

# Build wheel package
if [ -n "${WHEELS_OUTPUT_FOLDER}" ]; then
    $PYTHON setup.py bdist_wheel -p manylinux1_x86_64
    cp dist/dpctl*.whl ${WHEELS_OUTPUT_FOLDER}
fi
