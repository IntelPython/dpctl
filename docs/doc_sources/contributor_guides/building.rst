.. _dpctl_building_from_source:

Building from the Source
========================

To build :py:mod:`dpctl` from the source, you need DPC++ compiler.
To run examples and test suite you would need GPU drivers and/or CPU
OpenCL drivers. It is preferable to use the Intel(R) oneAPI DPC++ compiler
available as part of oneAPI Base-Kit. However, it is possible to use a custom
build of DPC++ to build :py:mod:`dpctl`, especially if you want to enable
CUDA support or try latest features. On Linux OS, AdaptiveCpp can be used
instead of DPC++, see `Building Using AdaptiveCpp`_.

Building using oneAPI DPC++
---------------------------


Prerequisites
~~~~~~~~~~~~~

Install oneAPI and graphics drivers according to your targeted hardware:


- To target Intel GPUs, see the `Installation Page <https://dgpu-docs.intel.com/driver/installation.html>`_
  of the Intel(R) software for general purpose GPU capabilities document for
  driver information.
- To target NVIDIA* or AMD* GPUs, see the vendor website for drivers, as well
  as `CodePlay plugins <https://codeplay.com/solutions/oneapi/plugins/>`_ to
  enable hardware targeting.
- To target a CPU, the OpenCL* CPU driver is included as a part of the
  oneAPI DPC++ Compiler installation. The CPU
  driver is also packaged in conda, and is automatically made available using
  conda activation scripts on Linux*, and on Windows* (in user-mode).
  If conda is used with elevated privileges in Windows (similar to
  GitHub Actions CI), a PowerShell script must be run:

  .. code-block:: powershell

    &$Env:CONDA_PREFIX\Scripts\script_name.ps1

Use the script ``set-intel-ocl-icd-registry.ps1`` to set
appropriate registry key, and ``unset-intel-ocl-icd-registry.ps1``
to remove it.


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
package. The advantage of this approach is that all oneAPI library dependencies are
pulled in from oneAPI's local conda channel that is installed as a part of oneAPI.

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


Build and Install with scikit-build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To build using Python ``setuptools`` and ``scikit-build``, install the following Python packages:

- ``cython``
- ``numpy``
- ``cmake``
- ``scikit-build``
- ``ninja``
- ``versioneer``
- ``gtest`` (optional to run C API tests)
- ``gmock`` (optional to run C API tests)
- ``pytest`` (optional to run Python API tests)

Once the prerequisites are installed, building using ``scikit-build`` involves the usual steps.

To build and install, run:

.. tab-set::

    .. tab-item:: Linux
        :sync: lnx

        .. code-block:: bash

            python setup.py install -- -G Ninja -DCMAKE_C_COMPILER:PATH=icx -DCMAKE_CXX_COMPILER:PATH=icpx

    .. tab-item:: Windows
        :sync: win

        .. code-block:: bat

            python setup.py install -- -G Ninja -DCMAKE_C_COMPILER:PATH=icx -DCMAKE_CXX_COMPILER:PATH=icx


To develop, run:

.. tab-set::

    .. tab-item:: Linux
        :sync: lnx

        .. code-block:: bash

            python setup.py build_ext --inplace -G Ninja -DCMAKE_C_COMPILER:PATH=icx -DCMAKE_CXX_COMPILER:PATH=icpx
            python -m pip install -e .

    .. tab-item:: Windows
        :sync: win

        .. code-block:: bat

            python setup.py build_ext --inplace -G Ninja -DCMAKE_C_COMPILER:PATH=icx -DCMAKE_CXX_COMPILER:PATH=icx
            python -m pip install -e .


Developing can be streamlined using the driver script:

.. tab-set::

    .. tab-item:: Linux
        :sync: lnx

        .. code-block:: bash

            python scripts/build_locally.py --verbose

    .. tab-item:: Windows
        :sync: win

        .. code-block:: bat

            python scripts/build_locally.py --verbose


Building Using Custom DPC++
---------------------------

You can build dpctl from the source using the `DPC++ toolchain <https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md>`_
instead of the DPC++ compiler that comes with oneAPI.

Following steps in the `Build and install with scikit-build`_ use a command-line option to set
the relevant CMake variables, for example:

.. code-block:: bash

    python setup.py build_ext --inplace -G Ninja -DCMAKE_C_COMPILER:PATH=$(which clang) -DCMAKE_CXX_COMPILER:PATH=$(which clang++)
    python -m pip install -e .


Or you can use the driver script:

.. code-block:: bash

    python scripts/build_locally.py --c-compiler=$(which clang) --cxx-compiler=$(which clang++)


You can retrieve available options and their descriptions using the option
:code:`--help`.


Runtime loaders when using a nightly DPC++ build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:mod:`dpctl` creates kernel bundles from OpenCL* source and from SPIR-V by
calling into the OpenCL ICD loader and the Level Zero loader directly.

The oneAPI DPC++ Compiler installation and the conda packages provide these
loaders as dependencies. Nightly builds of the open source DPC++ toolchain,
however, bundle neither, so they must be installed separately:

- ``libOpenCL.so.1``, from the OpenCL ICD loader (for example, ``ocl-icd``
  conda package or ``ocl-icd-libopencl1`` system package)
- ``libze_loader.so.1``, from the Level Zero loader (for example, the
  ``level-zero`` conda package or the ``libze1`` system package)

Both can be installed from the toolchain's own dependencies. The DPC++ toolchain
pins them in `devops/dependencies.json <https://github.com/intel/llvm/blob/sycl/devops/dependencies.json>`_,
as the ``level_zero`` and ``oclcpu`` entries.

Should a loader be installed under a different file name, override the name that
:py:mod:`dpctl` uses with the ``LIBCL_SET_LOADER_FILENAME`` and
``LIBZE_SET_LOADER_FILENAME`` CMake variables.


Building Using AdaptiveCpp
--------------------------

:py:mod:`dpctl` can also be built with `AdaptiveCpp <https://adaptivecpp.github.io/>`_
as the SYCL implementation, on Linux OS. The build was tested with AdaptiveCpp
25.10.

Prerequisites for AdaptiveCpp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install AdaptiveCpp, the ``clang`` that it was built against, and a runtime for
the devices you intend to use. The conda packages, for example:

.. code-block:: bash

    conda install -c conda-forge "adaptivecpp=*=cpu_llvm19*" "clang=19.*" \
        "clangxx=19.*" intel-opencl-rt tbb

The ``clang`` version must match the LLVM version AdaptiveCpp was built with,
which the build string of the ``adaptivecpp`` package states, as in
``cpu_llvm19_h8255587_10``, hence the pins above. The ``intel-opencl-rt``
package provides an OpenCL CPU device; AdaptiveCpp always provides its own
OpenMP host device, which needs no runtime.

Run ``acpp-info -l`` to list the backends of the installed AdaptiveCpp, and set
the ``ACPP_VISIBILITY_MASK`` environment variable to select among them, for
example ``ACPP_VISIBILITY_MASK=omp`` for the OpenMP host device, or
``ACPP_VISIBILITY_MASK=ocl`` for the OpenCL devices.

Build and Install Using AdaptiveCpp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the driver script with the provider option:

.. code-block:: bash

    python scripts/build_locally.py --sycl-provider=AdaptiveCpp

The option defaults the compilers to ``clang`` and ``acpp``, which are taken
from the ``PATH`` unless ``--c-compiler`` and ``--cxx-compiler`` are given, and
it disables Level Zero kernel bundle creation, which only DPC++ implements.
Leaving it on would only check out the Level Zero headers for code that an
AdaptiveCpp build excludes; it does not affect whether a Level Zero device is
available, which depends on the backends AdaptiveCpp itself was built with.

Building with ``setuptools`` directly takes the equivalent CMake variables:

.. code-block:: bash

    python setup.py build_ext --inplace -G Ninja -- \
        -DCMAKE_C_COMPILER:PATH=clang -DCMAKE_CXX_COMPILER:PATH=acpp \
        -DDPCTL_SYCL_PROVIDER=AdaptiveCpp \
        -DDPCTL_ENABLE_L0_KERNEL_BUNDLE_CREATION=OFF
    python -m pip install -e .

Differences from a DPC++ build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An AdaptiveCpp build reports its SYCL implementation through
``dpctl._diagnostics.sycl_provider()``. The following is unavailable in such a
build:

- kernel bundle creation from OpenCL source and from SPIR-V, hence the
  :py:mod:`dpctl.compiler` functions and kernel submission through them, both
  through the OpenCL and the Level Zero backend
- the raw kernel argument extension
- peer access between devices, so ``enable_peer_access`` and
  ``disable_peer_access`` raise ``ValueError``
- the atomic memory order, fence order and scope capability descriptors of
  devices and of contexts
- the DPC++ version, the backend version and the default selector score

A few behaviors differ as well: a context can span platforms, and a platform is
reported for every backend AdaptiveCpp was built with, including backends with
no device.

Tests covering the above carry the ``unsupported_on_acpp`` pytest marker and are
skipped automatically.

The extension examples under ``examples/`` that build SYCL kernels take
``-DDPCTL_SYCL_PROVIDER=AdaptiveCpp`` as well, with the exception of
``pybind11/onemkl_gemv`` and ``pybind11/use_dpctl_sycl_kernel``, which are
DPC++ only.


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
    rm -rf "${INSTALL_PREFIX}"
    export ONEAPI_ROOT=/opt/intel/oneapi
    # Values are set as appropriate for oneAPI DPC++ 2026.0
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
        -DDPCTL_ENABLE_L0_KERNEL_BUNDLE_CREATION=ON             \
        -DDPCTL_BUILD_CAPI_TESTS=ON                             \
        -DDPCTL_GENERATE_COVERAGE=ON                            \
        ..

    make V=1 -n -j 4 && make check && make install

    popd || exit 1
