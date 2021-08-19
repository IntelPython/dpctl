#!/bin/bash

${PYTHON} setup.py clean --all
INSTALL_CMD="install --sycl-compiler-prefix=$CONDA_PREFIX"

if [ -n "${WHEELS_OUTPUT_FOLDER}" ]; then
    # Install packages and assemble wheel package from built bits
    if [ "$CONDA_PY" == "36" ]; then
        WHEELS_BUILD_ARGS="-p manylinux1_x86_64"
    else
        WHEELS_BUILD_ARGS="-p manylinux2014_x86_64"
    fi
    ${PYTHON} setup.py ${INSTALL_CMD} bdist_wheel ${WHEELS_BUILD_ARGS}
    cp dist/dpctl*.whl ${WHEELS_OUTPUT_FOLDER}
else
    # Perform regular install
    ${PYTHON} setup.py ${INSTALL_CMD}
fi
