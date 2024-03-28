.. _dpctl_installation:

####################
Installing ``dpctl``
####################

Installation from oneAPI
========================

:py:mod:`dpctl` is available as part of the oneAPI Intel(R) Distribution for Python (IDP).
Refer to `Intel(R) oneAPI Toolkits Installation Guide <oneapi_installation_guide_>`_
to install it.

.. _oneapi_installation_guide: https://www.intel.com/content/www/us/en/developer/articles/guide/installation-guide-for-oneapi-toolkits.html
.. _gpu_stack_installation_guide: https://dgpu-docs.intel.com/installation-guides/index.html

In this topic, it is assumed that oneAPI is installed in the standard location and the
environment variable ``ONEAPI_ROOT`` points to the following installation root
directory:

    - Windows OS: ``C:\Program Files (x86)\Intel\oneAPI\``
    - Linux OS: ``/opt/intel/oneapi``

Once oneAPI is installed, :py:mod:`dpctl` is ready to be used by setting up IDP from
the oneAPI installation. IDP can be set up as follows:

On Linux OS

.. code-block:: bash

  source ${ONEAPI_ROOT}/intelpython/latest/env/vars.sh
  python -c "import dpctl; dpctl.lsplatform()"

On Windows OS

.. code-block:: bat

    call "%ONEAPI_ROOT%\intelpython\latest\env\vars.bat"
    python -c "import dpctl; dpctl.lsplatform()"


.. note::

    If no GPU platforms are shown, make sure your system has a supported
    GPU and the necessary GPU drivers installed.
    See `GPU driver installation guide <gpu_stack_installation_guide_>`_ to install GPU drivers.

Install the Wheel Package from PyPi
====================================

To install :py:mod:`dpctl` using ``pip``, run:

.. code-block:: bash

    python -m pip install --index-url https://pypi.anaconda.org/intel/simple dpctl

.. note::

    The :py:mod:`dpctl` wheel package is available on PyPi, but some of the dependencies
    (like Intel(R) numpy) are available only on Anaconda Cloud. For this reason,
    install the extra packages needed by :py:mod:`dpctl` from the Intel(R) channel on
    Anaconda cloud. You also need to set the ``LD_LIBRARY_PATH``
    or ``PATH`` correctly.

On Linux OS

.. code-block:: bash

    export LD_LIBRARY_PATH=<path_to_your_env>/lib

On Windows OS

.. code-block:: bat

    set PATH=<path_to_your_env>\bin;<path_to_your_env>\Library\bin;%PATH%


Using :mod:`dpctl`
==================

Dpctl requires a DPC++ runtime. When :py:mod:`dpctl` is installed via Conda it uses
the DPC++ runtime from the ``dpcpp_cpp_rt`` package that is a part of IDP.

When using local developer's build of :py:mod:`dpctl` ensure that a compatible version of
DPC++ runtime can be found by Python. The easiest way to set up a DPC++ runtime is by
activating oneAPI.

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

You can execute Python test suite of :py:mod:`dpctl` as follow:

.. code-block:: bash

    pytest --pyargs dpctl
