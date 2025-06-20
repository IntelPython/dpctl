.. _dpctl_installation:

####################
Installing ``dpctl``
####################

License
=======

:py:mod:`dpctl` is licensed under Apache* License 2.0 that can be found in the
`LICENSE <dpctl_license_>`_ file.
All usage and contributions to the project are subject to the terms and
conditions of this license.

.. _dpctl_license: https://github.com/IntelPython/dpctl/blob/master/LICENSE

See the user guide :ref:`document <user_guide_dpctl_license>` for additional information.

Installation using conda
========================

Binary builds of :py:mod:`dpctl` can be installed through the ``conda``/``mamba`` package managers,
from either the ``conda-forge`` channel, or from Intel's channel.

.. warning::
    Packages from the Intel channel are meant to be used together with dependencies from the **conda-forge** channel, and might not
    work correctly when used in an environment where packages from the ``anaconda`` default channel have been installed. It is
    advisable to use the `miniforge <https://github.com/conda-forge/miniforge>`__ installer for ``conda``/``mamba``, as it comes with
    ``conda-forge`` as the only default channel.

.. code-block:: bash
    :caption: Getting latest released version of ``dpctl`` using conda

    conda create --name dpctl_env --channel https://software.repos.intel.com/python/conda/ --channel conda-forge --override-channels dpctl

Development builds of ``dpctl`` can be installed from the ``dppy/label/dev`` channel:

.. code-block:: bash
    :caption: Getting latest development version

    conda create -n dpctl_nightly -c dppy/label/dev -c https://software.repos.intel.com/python/conda/ -c conda-forge --override-channels dpctl

.. note::
    If :py:mod:`dpctl` is not available for the Python version of interest,
    see `Building from source`_.


Installation using pip
======================

Binary wheels are published with Python Package Index (https://pypi.org/project/dpctl/).

.. code-block:: bash
    :caption: Getting latest released version of ``dpctl`` using ``pip``

    python -m pip install dpctl


Binary wheels of ``dpctl`` and its dependencies are also published on Intel(R) channel. To install from this non-default package index,
use

.. code-block:: bash

    python -m pip install --index-url https://software.repos.intel.com/python/pypi dpctl

.. note::
    As of April 2024, installation using ``pip`` on Linux* requires
    that host operating system had ``libstdc++.so`` library version 6.0.29
    or later.  Check the version you have by executing
    ``find /lib/x86_64-linux-gnu/ -name "libstdc++*"``

.. note::
    If :py:mod:`dpctl` is not available for the Python version of interest,
    see `Building from source`_.


Installation via Intel(R) Distribution for Python
=================================================

`Intel(R) Distribution for Python* <idp_page_>`_ is distributed as a conda-based installer
and includes :py:mod:`dpctl` along with its dependencies and sister projects :py:mod:`dpnp`
and :py:mod:`numba_dpex`.

.. _idp_page: https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html

Once the installed environment is activated, ``dpctl`` should be ready to use.

System requirements
===================

Since :py:mod:`dpctl` is compiled using the Intel(R) oneAPI DPC++ compiler,
the `compiler's system requirements for runtime <dpcpp_system_reqs_>`_ must be met.

In order for DPC++ runtime to recognize supported hardware appropriate drivers must be installed.
Directions to install drivers for Intel GPU devices are available at https://dgpu-docs.intel.com/

.. _dpcpp_system_reqs: https://www.intel.com/content/www/us/en/developer/articles/system-requirements/intel-oneapi-dpcpp-system-requirements.html

Once ``dpctl`` is installed, use ``python -m dpctl --full-list`` to list recognized devices.

For ``dpctl`` to target Intel GPU devices, appropriate drivers should be installed systemwide.
Please refer to `GPU installation guide <gpu_stack_installation_guide_>`_ for detailed
instructions on how to install required drivers on Linux.

.. _gpu_stack_installation_guide: https://dgpu-docs.intel.com/

.. note::
    Instructions for setting up GPU drivers in Windows Subsystem for Linux (WSL)
    will be added in a future release of this document.

Building from source
====================

There are several reasons to want to build ``dpctl`` from source:

1. To use it with Python version for which binary artifacts are not available
2. To be able to use DPC++ runtime libraries from local installation of DPC++ compiler and
   avoid installing them into Python environment
3. To build for custom SYCL targets, such as ``nvptx64-nvidia-cuda`` or ``"amdgcn-amd-amdhsa"``.

Building locally for use with oneAPI DPC++ installation
-------------------------------------------------------

Working with :py:mod:`dpctl` in this mode assumes that the DPC++ compiler is activated, and that
Python environment has all build and runtime dependencies of ``dpctl`` installed.

One way to create such environment is as follows:

.. code-block:: bash
    :caption: Creation of environment to build ``dpctl`` locally

    conda create -n dev_dpctl -c conda-forge python=3.12 pip
    conda activate dev_dpctl
    pip install --no-cache-dir numpy cython scikit-build cmake ninja pytest

Using such environment and with DPC++ compiler activated, build the project using

.. code-block:: bash

   python scripts/build_locally.py --verbose

.. note::
    Coming back to use this local build of ``dpctl`` remember to activate DPC++.

Building for custom SYCL targets
--------------------------------

Project :py:mod:`dpctl` is written using generic SYCL and supports building for
multiple SYCL targets, subject to limitations of `CodePlay <https://codeplay.com/>`_
plugins implementing  SYCL programming model for classes of devices.

Building ``dpctl`` for these targets requires that these CodePlay plugins be
installed into DPC++ installation layout of compatible version.
The following plugins from CodePlay are supported:

    - `oneAPI for NVIDIA(R) GPUs <codeplay_nv_plugin_>`_
    - `oneAPI for AMD GPUs <codeplay_amd_plugin_>`_

.. _codeplay_nv_plugin: https://developer.codeplay.com/products/oneapi/nvidia/
.. _codeplay_amd_plugin: https://developer.codeplay.com/products/oneapi/amd/

Builds for CUDA and AMD devices internally use SYCL alias targets that are passed to the compiler.
A full list of available SYCL alias targets is available in the
`DPC++ Compiler User Manual <https://intel.github.io/llvm/UsersManual.html>`_.

CUDA build
~~~~~~~~~~

``dpctl`` can be built for CUDA devices using the  ``--target-cuda`` argument.

To target a specific architecture (e.g., ``sm_80``):

.. code-block:: bash

    python scripts/build_locally.py --verbose --target-cuda=sm_80

To use the default architecture (``sm_50``), omit the value:

.. code-block:: bash

    python scripts/build_locally.py --verbose --target-cuda

Note that kernels are built for the default architecture (``sm_50``), allowing them to work on a
wider range of architectures, but limiting the usage of more recent CUDA features.

For reference, compute architecture strings like ``sm_80`` correspond to specific
CUDA Compute Capabilities (e.g., Compute Capability 8.0 corresponds to ``sm_80``).
A complete mapping between NVIDIA GPU models and their respective
Compute Capabilities can be found in the official
`CUDA GPU Compute Capability <https://developer.nvidia.com/cuda-gpus>`_ documentation.

AMD build
~~~~~~~~~

``dpctl`` can be built for AMD devices using the ``DPCTL_TARGET_HIP`` CMake option,
which requires specifying a compute architecture string:

.. code-block:: bash

    python scripts/build_locally.py --verbose --cmake-opts="-DDPCTL_TARGET_HIP=<arch>"

Note that the `oneAPI for AMD GPUs` plugin requires the architecture be specified and only
one architecture can be specified at a time.

To determine the architecture code (``<arch>``) for your AMD GPU, run:

.. code-block:: bash
    rocminfo | grep 'Name: *gfx.*'

This will print names like ``gfx90a``, ``gfx1030``, etc.
You can then use one of them as the argument to ``-DDPCTL_TARGET_HIP``.

For example:

.. code-block:: bash
    python scripts/build_locally.py --verbose --cmake-opts="-DDPCTL_TARGET_HIP=gfx1030"

Multi-target build
~~~~~~~~~~~~~~~~~~

The default ``dpctl`` build from the source enables support of Intel devices only.
Extending the build with a custom SYCL target additionally enables support of CUDA or AMD
device in ``dpctl``. Besides, the support can be also extended to enable both CUDA and AMD
devices at the same time:

.. code-block:: bash

    python scripts/build_locally.py --verbose --target-cuda --cmake-opts="-DDPCTL_TARGET_HIP=gfx1030"

Running Examples and Tests
==========================

Running the Examples
--------------------

After setting up dpctl, you can test the Python examples as follows:

.. code-block:: bash

    for script in `ls examples/python/`
    do
    echo "executing ${script}"
    python examples/python/${script}
    done

The :py:mod:`dpctl` repository also provides a set of `examples <examples_sources_>`_
of building Cython and pybind11 extensions with the DPC++ compiler that interoperate
with :py:mod:`dpctl`.

.. _examples_sources: https://github.com/IntelPython/dpctl/tree/master/examples/

Please refer to the ``README.md`` file in respective folders for instructions on how to build
each example Python project and how to execute its test suite.

Running the Python Tests
------------------------

You can execute Python test suite of :py:mod:`dpctl` with:

.. code-block:: bash

    pytest --pyargs dpctl
