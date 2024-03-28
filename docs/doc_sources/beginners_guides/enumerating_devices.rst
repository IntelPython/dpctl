.. _beginners_guide_enumerating_devices:

Enumerating available devices
=============================



Listing platform from command-line
-----------------------------------

:py:mod:`dpctl` provides command-line interface to list available platforms:

.. code-block:: bash
    :caption: List platforms with detailed information on devices

    python -m dpctl --full-list

A sample output of executing such a command on a laptop:

.. code-block:: text
    :caption: Sample output of running ``python -m dpctl --full-list``

    Platform  0 ::
        Name        Intel(R) FPGA Emulation Platform for OpenCL(TM)
        Version     OpenCL 1.2 Intel(R) FPGA SDK for OpenCL(TM), Version 20.3
        Vendor      Intel(R) Corporation
        Backend     opencl
        Num Devices 1
        # 0
            Name                Intel(R) FPGA Emulation Device
            Version             2024.17.2.0.22_223154
            Filter string       opencl:accelerator:0
    Platform  1 ::
        Name        Intel(R) OpenCL
        Version     OpenCL 3.0 LINUX
        Vendor      Intel(R) Corporation
        Backend     opencl
        Num Devices 1
        # 0
            Name                11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz
            Version             2024.17.2.0.22_223154
            Filter string       opencl:cpu:0
    Platform  2 ::
        Name        Intel(R) OpenCL Graphics
        Version     OpenCL 3.0
        Vendor      Intel(R) Corporation
        Backend     opencl
        Num Devices 1
        # 0
            Name                Intel(R) Graphics [0x9a49]
            Version             23.52.28202.26
            Filter string       opencl:gpu:0
    Platform  3 ::
        Name        Intel(R) Level-Zero
        Version     1.3
        Vendor      Intel(R) Corporation
        Backend     ext_oneapi_level_zero
        Num Devices 1
        # 0
            Name                Intel(R) Graphics [0x9a49]
            Version             1.3.28202
            Filter string       level_zero:gpu:0

.. currentmodule:: dpctl

Command-line interface is useful for verifying that drivers are installed correctly.
It is implemented using :py:func:`lsplatform` function.

.. note::
    The output on your particular heterogeneous system may vary, depending on available hardware and drivers installed.

Listing devices programmatically
--------------------------------

Devices can also be discovered programmatically, either by using :py:func:`lsplatform` to :py:func:`print`` the listing or
by using :py:func:`get_devices` to obtain a list of :py:class:`SyclDevice` objects suitable for further processing.

.. code-block:: python
    :caption: Example: Obtaining list of available devices for processing

    import dpctl

    # get all available devices
    devices = dpctl.get_devices()

    # get memory of each in GB
    {d.name: d.global_mem_size // (1024 ** 3) for d in devices}


Interaction with DPC++ environment variables
--------------------------------------------

:py:mod:`dpctl` relies on DPC++ runtime for device discovery and is :ref:`subject <beginners_guide_env_variables>` to
environment variables that influence behavior of the runtime.
Setting ``ONEAPI_DEVICE_SELECTOR`` environment variable (see the `list of environment variables <dpcpp_env_vars_>`_
recognized by oneAPI DPC++ runtime for additional details) may restrict the set of devices visible to DPC++ runtime, and hence to :py:mod:`dpctl`

.. _dpcpp_env_vars: https://intel.github.io/llvm-docs/EnvironmentVariables.html

.. code-block:: bash
    :caption: Example: Setting ``ONEAPI_DEVICE_SELECTOR=*:cpu`` renders GPU devices unavailable even if they are present

    export ONEAPI_DEVICE_SELECTOR=*:cpu
    # would only show CPU device
    python -m dpctl -f

    unset ONEAPI_DEVICE_SELECTOR
    # all available devices are available now
    python -m dpctl -f
