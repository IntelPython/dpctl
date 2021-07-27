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
    $PYTHON setup.py bdist_wheel ${WHEELS_BUILD_ARGS}
    cp dist/dpctl*.whl ${WHEELS_OUTPUT_FOLDER}
fi
