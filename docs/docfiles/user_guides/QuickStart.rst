.. _quick_start_guide:

#################
Quick Start Guide
#################

Installation from oneAPI
========================

Dpctl is available as part of the oneAPI Intel(R) Distribution for Python (IDP).
Refer to `Intel(R) oneAPI Toolkits Installation Guide <https://www.intel.com/content/www/us/en/developer/articles/guide/installation-guide-for-oneapi-toolkits.html>`_ 
to install it. 

In this topic, it is assumed that oneAPI is installed in the standard location and the
environment variable ``ONEAPI_ROOT`` points to the following installation root
directory:

    - Windows* OS: ``C:\Program Files (x86)\Intel\oneAPI\``
    - Linux* OS: ``/opt/intel/oneapi``

Once oneAPI is installed, dpctl is ready to be used by setting up IDP from
the oneAPI installation. IDP can be set up as follows:

On Linux* OS

.. code-block:: bash

  source ${ONEAPI_ROOT}/intelpython/latest/env/vars.sh
  python -c "import dpctl; dpctl.lsplatform()"

On Windows* OS

.. code-block:: bat

    call "%ONEAPI_ROOT%\intelpython\latest\env\vars.bat"
    python -c "import dpctl; dpctl.lsplatform()"


.. note::

    The ``dpctl.lsplatform()`` function is new in dpctl 0.7 and will be
    available in oneAPI 2021.3. If you are following the guide on an older
    oneAPI installation, use ``dpctl.dump()``. If no GPU platforms are shown,
    make sure your system has a supported GPU and the necessary GPU drivers
    installed. You can install GPU drivers by following the
    `GPU driver installation guide <https://dgpu-docs.intel.com/installation-guides/index.html>`_.

Install the Wheel Package from PyPi*
====================================

To install dpctl from PyPi*, run:

.. code-block:: bash

    python -m pip install --index-url https://pypi.anaconda.org/intel/simple dpctl

.. note::

    The dpctl wheel package is available on PyPi*, but some of the dependencies
    (like Intel(R) numpy) are available only on Anaconda* Cloud. For this reason,
    install the extra packages needed by dpctl from the Intel(R) channel on
    Anaconda cloud. You also need to set the ``LD_LIBRARY_PATH``
    or ``PATH`` correctly.

On Linux* OS

.. code-block:: bash

    export LD_LIBRARY_PATH=<path_to_your_env>/lib

On Windows* OS

.. code-block:: bat

    set PATH=<path_to_your_env>\bin;<path_to_your_env>\Library\bin;%PATH%

Building from the Source
========================

To build dpctl from the source, you need dpcpp and GPU drivers, and optionally CPU
OpenCL drivers. It is preferable to use the dpcpp compiler packaged as part of
oneAPI. However, it is possible to use a custom build of dpcpp to build dpctl,
especially if you want to enable CUDA* support.

Building using oneAPI dpcpp
---------------------------

Install oneAPI and graphics drivers to the system prior
to proceeding further.

Activate oneAPI 
~~~~~~~~~~~~~~~

On Linux* OS

.. code-block:: bash

  source ${ONEAPI_ROOT}/setvars.sh

On Windows* OS

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

On Windows* OS to cope with `long file names <https://github.com/IntelPython/dpctl/issues/15>`_,
use ``croot`` with a short folder path:

.. code-block:: bat

    set "ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI\"
    conda build --croot=C:/tmp conda-recipe -c "%ONEAPI_ROOT%\conda_channel"

After building the Conda* package, install it by executing:

.. code-block:: bash

    conda install dpctl

.. note::

    You can face issues with conda-build version 3.20. Use conda-build
    3.18 instead.


Build and Install with scikit-build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To build using Python* ``setuptools`` and ``scikit-build``, install the following Python* packages:

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

On Windows* OS, use ``icx`` for both C and CXX compilers.

To develop on Linux* OS, use the driver script:

.. code-block:: bash

    python scripts/build_locally.py


Building Using Custom dpcpp
---------------------------

You can build dpctl from the source using the `DPC++ toolchain <https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md>`_
instead of the DPC++ compiler that comes with oneAPI. 

Do this, to enable support for CUDA* devices.

Following steps in the *Build and Install with scikit-build* use a command-line option to set
the relevant CMake variables, for example:

.. code-block:: bash

    python setup.py develop -- -G Ninja -DCMAKE_C_COMPILER:PATH=$(which clang) -DCMAKE_CXX_COMPILER:PATH=$(which clang++)


Or you can use the driver script:

.. code-block:: bash

    python scripts/build_locally.py --c-compiler=$(which clang) --cxx-compiler=$(which clang++)


You can retrieve available options and their descriptions using the option
:code:`--help`.

Using dpctl
===========

Dpctl requires a DPC++ runtime. When dpctl is installed via Conda* it uses
the DPC++ runtime from the ``dpcpp_cpp_rt`` package that is a part of IDP. When using
``setuptools`` make sure a compatible version of DPC++ runtime is available on
the system. The easiest way to set up a DPC++ runtime is by activating
oneAPI.

Running Examples and Tests
==========================

Running the Examples
--------------------

After setting up dpctl, you can test the Python* examples as follows:

.. code-block:: bash

    for script in `ls examples/python/`
    do
    echo "executing ${script}"
    python examples/python/${script}
    done

The dpctl repository also provides a set of `examples <https://github.com/IntelPython/dpctl/tree/master/examples/cython>`_ 
of building the Cython extensions with the DPC++ compiler, that interoperates with dpctl.

To build each example, use
``CC=icx CXX=dpcpp python setup.py build_ext --inplace``. 
Refer to the ``run.py`` script in respective folders to execute the Cython extension
examples.

Running the Python Tests
------------------------

You can execute the dpctl Python* test suite as follow:

.. code-block:: bash

    pytest --pyargs dpctl


Building the libsyclinterface Library
=======================================

The libsyclinterface is a shared library used by the Python* package.
To build the library, you need: 

*  ``DPC++`` toolchain
* ``cmake``
* ``ninja`` or ``make``
* Optionally ``gtest 1.10`` if you want to run the test suite

For example, on Linux* OS the following script can be used to build the C oneAPI
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

