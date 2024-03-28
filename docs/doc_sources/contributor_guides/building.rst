.. _dpctl_building_from_source:

Building from the Source
========================

To build :py:mod:`dpctl` from the source, you need DPC++ compiler.
To run expamples and test suite you would need GPU drivers and/or CPU
OpenCL drivers. It is preferable to use the Intel(R) oneAPI DPC++ compiler
available as part of oneAPI Base-Kit. However, it is possible to use a custom
build of dpcpp to build :py:mod:`dpctl`, especially if you want to enable
CUDA support or try latest features.

Building using oneAPI dpcpp
---------------------------

Install oneAPI and graphics drivers to the system prior
to proceeding further.

Activate oneAPI
~~~~~~~~~~~~~~~

On Linux OS

.. code-block:: bash

  source ${ONEAPI_ROOT}/setvars.sh

On Windows OS

.. code-block:: bat

    call "%ONEAPI_ROOT%\setvars.bat"

Build and Install Using Conda-Build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use the conda-recipe included with the sources to build the dpctl
package. The advantage of this approach is that all dependencies are pulled in
from oneAPI's intelpython conda channel that is installed as a part of oneAPI.

.. code-block:: bash

    export ONEAPI_ROOT=/opt/intel/oneapi
    conda build conda-recipe -c ${ONEAPI_ROOT}/conda_channel

On Windows OS to cope with `long file names <https://github.com/IntelPython/dpctl/issues/15>`_,
use ``croot`` with a short folder path:

.. code-block:: bat

    set "ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI\"
    conda build --croot=C:/tmp conda-recipe -c "%ONEAPI_ROOT%\conda_channel"

After building the Conda package, install it by executing:

.. code-block:: bash

    conda install dpctl

.. note::

    You can face issues with conda-build version 3.20. Use conda-build
    3.18 instead.


Build and Install with scikit-build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To build using Python ``setuptools`` and ``scikit-build``, install the following Python packages:

    - ``cython``
    - ``numpy``
    - ``cmake``
    - ``scikit-build``
    - ``ninja``
    - ``gtest`` (optional to run C API tests)
    - ``gmock`` (optional to run C API tests)
    - ``pytest`` (optional to run Python API tests)

Once the prerequisites are installed, building using ``scikit-build`` involves the usual steps.

To build and install, run:

.. code-block:: bash

    python setup.py install -- -G Ninja -DCMAKE_C_COMPILER:PATH=icx -DCMAKE_CXX_COMPILER:PATH=icpx


To develop, run:

.. code-block:: bash

    python setup.py develop -G Ninja -DCMAKE_C_COMPILER:PATH=icx -DCMAKE_CXX_COMPILER:PATH=icpx

On Windows OS, use ``icx`` for both C and CXX compilers.

To develop on Linux OS, use the driver script:

.. code-block:: bash

    python scripts/build_locally.py


Building Using Custom dpcpp
---------------------------

You can build dpctl from the source using the `DPC++ toolchain <https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md>`_
instead of the DPC++ compiler that comes with oneAPI.

Do this, to enable support for CUDA devices.

Following steps in the `Build and install with scikit-build`_ use a command-line option to set
the relevant CMake variables, for example:

.. code-block:: bash

    python setup.py develop -- -G Ninja -DCMAKE_C_COMPILER:PATH=$(which clang) -DCMAKE_CXX_COMPILER:PATH=$(which clang++)


Or you can use the driver script:

.. code-block:: bash

    python scripts/build_locally.py --c-compiler=$(which clang) --cxx-compiler=$(which clang++)


You can retrieve available options and their descriptions using the option
:code:`--help`.


Building the libsyclinterface Library
=======================================

The libsyclinterface is a shared library used by the Python package.
To build the library, you need:

*  ``DPC++`` toolchain
* ``cmake``
* ``ninja`` or ``make``
* Optionally ``gtest 1.10`` if you want to build and run the test suite

For example, on Linux OS the following script can be used to build the C oneAPI
library.

.. code-block:: bash

    #!/bin/bash
    set +xe
    rm -rf build
    mkdir build
    pushd build || exit 1

    INSTALL_PREFIX=$(pwd)/../install
    rm -rf ${INSTALL_PREFIX}
    export ONEAPI_ROOT=/opt/intel/oneapi
    # Values are set as appropriate for oneAPI DPC++ 2024.0
    # or later.
    DPCPP_ROOT=${ONEAPI_ROOT}/compiler/latest/

    # Set these to ensure that cmake can find llvm-cov and
    # other utilities
    LLVM_TOOLS_HOME=${DPCPP_ROOT}/bin/compiler
    PATH=$PATH:${DPCPP_ROOT}/bin/compiler

    cmake                                                       \
        -DCMAKE_BUILD_TYPE=Debug                                \
        -DCMAKE_C_COMPILER=icx                                  \
        -DCMAKE_CXX_COMPILER=icpx                               \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}                \
        -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}                   \
        -DDPCTL_ENABLE_L0_PROGRAM_CREATION=ON                   \
        -DDPCTL_BUILD_CAPI_TESTS=ON                             \
        -DDPCTL_GENERATE_COVERAGE=ON                            \
        ..

    make V=1 -n -j 4 && make check && make install

    popd || exit 1
