.. _quick_start_guide:

#################
Quick Start Guide
#################

Installing from oneAPI
======================

Dpctl is available as part of the oneAPI Intel Distribution of Python (IDP).
Please follow `oneAPI installation guide`_ to install oneAPI. In this quick
start guide, it is assumed oneAPI is installed in the standard location and the
environment variable ``ONEAPI_ROOT`` points to the following installation root
directory:

    - Windows: ``C:\Program Files (x86)\Intel\oneAPI\``
    - Linux: ``/opt/intel/oneapi``

Once oneAPI is installed, dpctl is ready to be used by setting up IDP from
the oneAPI installation. IDP can be set up as follows:

On Linux

.. code-block:: bash

  source ${ONEAPI_ROOT}/intelpython/latest/env/vars.sh
  python -c "import dpctl; dpctl.lsplatform()"

On Windows

.. code-block:: bat

    call "%ONEAPI_ROOT%\intelpython\latest\env\vars.bat"
    python -c "import dpctl; dpctl.lsplatform()"


.. note::

    The ``dpctl.lsplatform()`` function is new in dpctl 0.7 and will be
    available in oneAPI 2021.3. If you are following the guide on an older
    oneAPI installation, use ``dpctl.dump()``. If no GPU platforms are shown,
    make sure your system has a supported GPU and has the necessary GPU drivers
    installed. You can install GPU drivers by following the
    `GPU driver installation guide`_.

Install Wheel package from Pypi
===============================

Dpctl can also be istalled from Pypi.

.. code-block:: bash

    python -m pip install --index-url https://pypi.anaconda.org/intel/simple dpctl

.. note::

    The dpctl wheel package is available on Pypi, but some of the dependencies
    (like Intel numpy) are available only in Anaconda Cloud. For this reason,
    please install the extra packages needed by dpctl from the Intel channel in
    Anaconda cloud. Additionally, you will need to set the ``LD_LIBRARY_PATH``
    or ``PATH`` correctly.

On Linux

.. code-block:: bash

    export LD_LIBRARY_PATH=<path_to_your_env>/lib

On Windows

.. code-block:: bat

    set PATH=<path_to_your_env>\bin;<path_to_your_env>\Library\bin;%PATH%

Building from source
====================

To build dpctl from source, we need dpcpp and GPU drivers (and optionally CPU
OpenCL drivers). It is preferable to use the dpcpp compiler packaged as part of
oneAPI. However, it is possible to use a custom build of dpcpp to build dpctl,
especially if you want to enable CUDA support.

Building using oneAPI dpcpp
---------------------------

As before, oneAPI and graphics drivers should be installed on the system prior
to proceeding further.

Activate oneAPI as follows
~~~~~~~~~~~~~~~~~~~~~~~~~~

On Linux

.. code-block:: bash

  source ${ONEAPI_ROOT}/setvars.sh

On Windows

.. code-block:: bat

    call "%ONEAPI_ROOT%\setvars.bat"

Build and install using conda-build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The conda-recipe included with the sources can be used to build the dpctl
package. The advantage of this approach is that all dependencies are pulled in
from oneAPI's intelpython conda channel that was installed as part of oneAPI.

.. code-block:: bash

    export ONEAPI_ROOT=/opt/intel/oneapi
    conda build conda-recipe -c ${ONEAPI_ROOT}/conda_channel

On Windows to cope with `long file names <https://github.com/IntelPython/dpctl/issues/15>`_,
use ``croot`` with short folder path:

.. code-block:: bat

    set "ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI\"
    conda build --croot=C:/tmp conda-recipe -c "%ONEAPI_ROOT%\conda_channel"

After building the conda package you may install it by executing:

.. code-block:: bash

    conda install dpctl

.. note::

    You could face issues with conda-build version 3.20. Use conda-build
    3.18 instead.


Build and install with scikit-build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To build using Python ``setuptools`` and ``scikit-build``, the following Python packages should be
installed:

    - ``cython``
    - ``numpy``
    - ``cmake``
    - ``scikit-build``
    - ``ninja``
    - ``gtest`` (optional to run C API tests)
    - ``gmock`` (optional to run C API tests)
    - ``pytest`` (optional to run Python API tests)

Once the prerequisites are installed, building using ``scikit-build`` involves the usual steps, to build and install:

.. code-block:: bash

    python setup.py install -- -G Ninja -DCMAKE_C_COMPILER:PATH=icx -DCMAKE_CXX_COMPILER:PATH=icpx

, and to develop:

.. code-block:: bash

    python setup.py develop -G Ninja -DCMAKE_C_COMPILER:PATH=icx -DCMAKE_CXX_COMPILER:PATH=icpx

On Windows, use ``icx`` for both C and CXX compilers.

Developing on Linux can also be done using driver script:

.. code-block:: bash

    python scripts/build_locally.py


Building using custom dpcpp
---------------------------

It is possible to build dpctl from source using .. _DPC++ toolchain: https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md
instead of the DPC++ compiler that comes with oneAPI. One reason for doing this
may be to enable support for CUDA devices.

Following steps in `Build and install with scikit-build`_ use command line option to set relevant cmake variables, for example:

.. code-block:: bash

    python setup.py develop -- -G Ninja -DCMAKE_C_COMPILER:PATH=$(which clang) -DCMAKE_CXX_COMPILER:PATH=$(which clang++)

Alterantively, the driver script can be used

.. code-block:: bash

    python scripts/build_locally.py --c-compiler=$(which clang) --cxx-compiler=$(which clang++)

Available options and their descriptions can be retrieved using option
:code:`--help`.

Using dpctl
===========

Dpctl requires a DPC++ runtime. When dpctl is installed via conda then it uses
the DPC++ runtime from ``dpcpp_cpp_rt`` package that is part of IDP. When using
``setuptools`` make sure a compatible version of DPC++ runtime is available on
the system. The easiest way to setup a DPC++ runtime will be by activating
oneAPI.

Running examples and tests
==========================

Running the examples
--------------------

After setting up dpctl you can try out the Python examples as follows:

.. code-block:: bash

    for script in `ls examples/python/`
    do
    echo "executing ${script}"
    python examples/python/${script}
    done

The dpctl repository also provides a set of examples of building Cython
extensions with DPC++ compiler, that interoperate with dpctl. These examples are
located under *examples/cython*. Each example in the folder can be built using
``CC=icx CXX=dpcpp python setup.py build_ext --inplace``. Please refer to
``run.py`` script in respective folders to execute the Cython extension
examples.

Running the Python tests
------------------------

The dpctl Python test suite can be executed as follows:

.. code-block:: bash

    pytest --pyargs dpctl


Building the DPCTLSyclInterface library
=======================================

The libDPCTLSyclInterface is a shared library used by the Python package.
To build the library you will need ``DPC++`` toolchain, ``cmake``,
``ninja`` or ``make``, and optionally ``gtest 1.10`` if you wish to run the
test suite.

For example, on Linux the following script can be used to build the C oneAPI
library.

.. code-block:: bash

    #!/bin/bash
    set +xe
    rm -rf build
    mkdir build
    pushd build

    INSTALL_PREFIX=`pwd`/../install
    rm -rf ${INSTALL_PREFIX}
    export ONEAPI_ROOT=/opt/intel/oneapi
    DPCPP_ROOT=${ONEAPI_ROOT}/compiler/latest/linux

    cmake                                                       \
        -DCMAKE_BUILD_TYPE=Release                              \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}                \
        -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}                   \
        -DDPCPP_INSTALL_DIR=${DPCPP_ROOT}                       \
        -DCMAKE_C_COMPILER:PATH=${DPCPP_ROOT}/bin/icx           \
        -DCMAKE_CXX_COMPILER:PATH=${DPCPP_ROOT}/bin/dpcpp       \
        -DDPCTL_BUILD_CAPI_TESTS=ON                             \
        ..

    make V=1 -n -j 4 && make check && make install





.. _oneAPI installation guide: https://software.intel.com/content/www/us/en/develop/articles/installation-guide-for-intel-oneapi-toolkits.html
.. _GPU driver installation guide : https://dgpu-docs.intel.com/installation-guides/index.html
