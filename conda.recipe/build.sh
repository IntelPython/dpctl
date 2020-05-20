#!/bin/bash

rm -rf build
mkdir build
cd build

cmake                                    \
    -DCMAKE_INSTALL_PREFIX=${PREFIX}     \
    -DCMAKE_PREFIX_PATH=${PREFIX}        \
    -DLIBUSM_INCLUDE_DIR=${PREFIX}/include/ \
    ..
    #-DCMAKE_BUILD_TYPE=Debug             \

make -n -j 4 && make install
#    -DLIBUSM_LIBDIR=${PREFIX}/lib/ \

cd ../python_binding
export DP_GLUE_LIBDIR=${PREFIX}
export DP_GLUE_INCLDIR=${PREFIX}/include
export OPENCL_LIBDIR=${BUILD_PREFIX}/lib
export LIBUSM_LIBDIR=${BUILD_PREFIX}/lib

${PYTHON} setup.py clean --all
${PYTHON} setup.py build
${PYTHON} setup.py install
