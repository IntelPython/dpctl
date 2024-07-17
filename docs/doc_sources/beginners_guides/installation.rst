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

Binary builds of :py:mod:`dpctl` are available for the `conda package manager <conda_docs_>`_
ecosystem.

.. _conda_docs: https://docs.conda.io/projects/conda/en/stable/

Released versions of the package can be installed from the Intel channel, as
indicated by ``--channel`` option:

.. code-block:: bash
    :caption: Getting latest released version of ``dpctl`` using conda

    conda create --name dpctl_env --channel https://software.repos.intel.com/python/conda/ dpctl

Development builds of ``dpctl`` can be accessed from the ``dppy/label/dev`` channel:

.. code-block:: bash
    :caption: Getting latest development version

    conda create -n dpctl_nightly -c dppy/label/dev -c https://software.repos.intel.com/python/conda/ dpctl

.. note::
    If :py:mod:`dpctl` is not available for the Python version of interest,
    see `Building from source`_.


Installation using pip
======================

Binary wheels are published with Python Package Index (https://pypi.org/project/dpctl/).

.. code-block:: bash
    :caption: Getting latest released version of ``dpctl`` using ``pip``

    python -m pip install dpctl

Binary wheels of ``dpctl`` and its dependencies are also published on
http://anaconda.org/intel. To install from this non-default package index,
use

.. code-block:: bash

    python -m pip install --index-url https://pypi.anaconda.org/intel/simple dpctl

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

Build ``dpctl`` as follows:

.. code-block:: bash

    python scripts/build_locally.py --verbose --cmake-opts="-DDPCTL_TARGET_CUDA=ON"


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
