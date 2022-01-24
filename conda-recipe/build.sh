#!/bin/bash

# Workaround to Klocwork overwriting LD_LIBRARY_PATH that was modified
# by DPC++ compiler conda packages. Will need to be added to DPC++ compiler
# activation scripts.
export LDFLAGS="$LDFLAGS -Wl,-rpath,$PREFIX/lib"

${PYTHON} setup.py clean --all
export CMAKE_GENERATOR="Unix Makefiles"
INSTALL_CMD="install -- -DCMAKE_C_COMPILER:PATH=icx -DCMAKE_CXX_COMPILER:PATH=icpx -DDPCTL_ENABLE_LO_PROGRAM_CREATION=ON -DDPCTL_DPCPP_HOME_DIR=${BUILD_PREFIX}"
echo "${PYTHON} setup.py ${INSTALL_CMD}"

# Workaround for:
# DPC++ launched by cmake does not see components of `dpcpp_cpp_rt`,
# because conda build isolates LD_LIBRARY_PATH to only $PREFIX subfolders.
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$BUILD_PREFIX/lib

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
