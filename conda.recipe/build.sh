#!/bin/bash

rm -rf build
mkdir build
cd build

cmake                                    \
    -DCMAKE_INSTALL_PREFIX=${PREFIX}     \
    -DCMAKE_PREFIX_PATH=${PREFIX}        \
    ..

make -n -j 4 && make install

cd ../python_binding
export DP_GLUE_LIBDIR=${PREFIX}
export DP_GLUE_INCLDIR=${PREFIX}
export OPENCL_LIBDIR=${BUILD_PREFIX}/lib

${PYTHON} setup.py clean --all
${PYTHON} setup.py build
${PYTHON} setup.py install
#${PYTHON} setup.py build_ext --inplace --single-version-externally-managed --record=record.txt 
