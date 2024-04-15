.. _beginners_guide_managing_devices:

################
Managing devices
################

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

Devices can also be discovered programmatically, either by using :py:func:`lsplatform` to :py:func:`print` the listing or
by using :py:func:`get_devices` to obtain a list of :py:class:`SyclDevice` objects suitable for further processing.

.. code-block:: python
    :caption: Example: Obtaining list of available devices for processing

    import dpctl

    # get all available devices
    devices = dpctl.get_devices()

    # get memory of each in GB
    {d.name: d.global_mem_size // (1024 ** 3) for d in devices}


.. _beginners_guide_oneapi_device_selector:

Interaction with DPC++ environment variables
--------------------------------------------

:py:mod:`dpctl` relies on DPC++ runtime for device discovery and is :ref:`subject <user_guides_env_variables>` to
environment variables that influence behavior of the runtime.
Setting ``ONEAPI_DEVICE_SELECTOR`` environment variable may restrict the set of devices visible to DPC++ runtime,
and hence to :py:mod:`dpctl`.

The value of the variable must follow a specific syntax (please refer to
`list of environment variables <dpcpp_env_vars_>`_ recognized by oneAPI DPC++ runtime for additional detail). Some examples
of valid settings are:

.. list-table::
    :header-rows: 1

    * - Setting
      - Availability

    * - ``*:cpu``
      - Only CPU devices from all backends are available

    * - ``!*:cpu``
      - All devices except CPU devices are available

    * - ``*:gpu``
      - Only GPU devices from all backends are available

    * - ``cuda:*``
      - All devices only from CUDA backend are available

    * - ``level_zero:0,1``
      - Two specific devices from Level-Zero backend are available

    * - ``level_zero:gpu;cuda:gpu;opencl:cpu``
      - Level-Zero GPU devices, CUDA GPU devices, and OpenCL CPU devices are available

.. _dpcpp_env_vars: https://intel.github.io/llvm-docs/EnvironmentVariables.html

.. code-block:: bash
    :caption: Example: Setting ``ONEAPI_DEVICE_SELECTOR=*:cpu`` renders GPU devices unavailable even if they are present

    export ONEAPI_DEVICE_SELECTOR=*:cpu
    # would only show CPU device
    python -m dpctl -f

    unset ONEAPI_DEVICE_SELECTOR
    # all available devices are available now
    python -m dpctl -f

.. _beginners_guide_device_selection:

Device selection
================

DPC++ runtime provides a way to select a device with a highest score to for a set of selection scroring strategies.
Amongst these are a default selector, CPU selector, GPU selector, as well as filter-string selector.

Using fixed device selectors
----------------------------

:py:mod:`dpctl` exposes device selection using fixed selectors as free functions:

.. currentmodule:: dpctl

.. list-table::

    * - :py:func:`select_default_device`
      - :py:func:`select_gpu_device`
    * - :py:func:`select_cpu_device`
      - :py:func:`select_accelerator_device`

:Example:
    .. code-block:: python

        >>> import dpctl
        >>> dpctl.select_default_device()
        <dpctl.SyclDevice [backend_type.level_zero, device_type.gpu,  Intel(R) Graphics [0x9a49]] at 0x7fbce2f129f0>
        >>> dpctl.select_cpu_device()
        <dpctl.SyclDevice [backend_type.opencl, device_type.cpu,  11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz] at 0x7fbccbe90db0>

Also note, that default-constructor of :class:`dpctl.SyclDevice` also creates the default-selected device:

:Example:
    .. code-block:: python

        >>> import dpctl
        >>> dpctl.SyclDevice()
        <dpctl.SyclDevice [backend_type.level_zero, device_type.gpu,  Intel(R) Graphics [0x9a49]] at 0x7fbccb78d030>
        >>> dpctl.select_default_device()
        <dpctl.SyclDevice [backend_type.level_zero, device_type.gpu,  Intel(R) Graphics [0x9a49]] at 0x7fbce2f129f0>

Selecting device based on aspects
---------------------------------

In addition, :py:func:`select_device_with_aspects` permits selecting a device based on aspects it is required to have:

.. code-block:: python
    :caption: Example: Selecting devices based on their aspects

    import dpctl

    # select a device that support float64 data type
    dev1 = dpctl.select_device_with_aspects("fp64")

    # select a device that supports atomic operations on 64-bit types
    # in USM-shared allocations
    dev2 = dpctl.select_device_with_aspects(
        ["atomic64", "usm_atomic_shared_allocations"]
    )

An aspect string ``asp`` is valid if ``hasattr(dpctl.SyclDevice, "has_aspect_" + asp)`` evaluates to ``True``.

Selecting device using filter selector string
---------------------------------------------

:py:class:`SyclDevice` may also be created using :ref:`filter selector string <filter_selector_string>` specified
as argument to the class constructor:

.. code-block:: python
    :caption: Example: Creating device based on filter-selector string

    import dpctl

    # create any GPU device
    dev_gpu = dpctl.SyclDevice("gpu")

    # take second device GPU device in the list of GPU devices
    # 0-based number is used
    dev_gpu1 = dpctl.SyclDevice("gpu:1")

    # create GPU device, or CPU if GPU is not available
    dev_gpu_or_cpu = dpctl.SyclDevice("gpu,cpu")

.. _beginners_guide_oneapi_device_selector_usecase:

Selecting device using ``ONEAPI_DEVICE_SELECTOR``
-------------------------------------------------

The device returned by :py:func:`select_default_device`, as well the behavior
of default constructor of :py:class:`SyclDevice` class is influenced by settings
of ``ONEAPI_DEVICE_SELECTOR`` as explained earlier.

Some users may find it convenient to always use a default-selected device, but control
which device that may be by setting this environment variable.
For example, the following script:

.. code-block:: python
    :caption: Sample array computation script "run.py"

    from dpctl import tensor as dpt

    gamma = 0.34
    x = dpt.linspace(0, 2*dpt.pi, num=10**6)
    f = dpt.sin(gamma * x) * dpt.exp(-x)

    int_approx = dpt.sum(f)
    print(f"Approximate value of integral: {int_approx} running on {x.device}" )

This script may be executed on a CPU, or GPU as follows:

.. code-block:: bash

    # execute on CPU device
    ONEAPI_DEVICE_SELECTOR=*:cpu python run.py
    #   Output: Approximate value of integral: 48328.99708167 running on Device(opencl:cpu:0)

    # execute on GPU device
    ONEAPI_DEVICE_SELECTOR=*:gpu python run.py
    #   Output: Approximate value of integral: 48329. running on Device(level_zero:gpu:0)


.. _beginners_guide_device_info:

Obtaining information about device
==================================

.. currentmodule:: dpctl

An instance of :py:class:`SyclDevice` provides access to a collection of
descriptors characterizing underlying ``sycl::device``.

Properties ``has_aspect_*`` expose Boolean descriptors which can be either ``True`` or ``False``.
Other descriptions are exposed as properties of the instance.

.. code-block:: python
    :caption: Example: Obtaining information about a device

    import dpctl

    # create default-selected device
    dev = dpctl.SyclDevice()

    # number of compute units
    cu = dev.max_compute_units
    # maximal supported size of a work-group
    max_wg = dev.max_work_group_size
    # size of shared local memory in bytes
    loc_mem_sz = dev.local_mem_size

    # name of the device
    dname = dev.name
    # maximal clock frequency in MHz
    freq = dev.max_clock_frequency


.. currentmodule:: dpctl.utils

For Intel GPU devices, additional architectural information can be access with :py:func:`intel_device_info` function:

.. code-block:: python
    :caption: Example: Intel GPU-specific information

    In [1]: import dpctl, dpctl.utils

    In [2]: d_gpu = dpctl.SyclDevice()

    # Output for Iris Xe integerate GPU, with PCI ID 0x9a49
    # (corresponding decimal value: 39497)
    In [3]: dpctl.utils.intel_device_info(d_gpu)
    Out[3]:
    {'device_id': 39497,
    'gpu_eu_count': 96,
    'gpu_hw_threads_per_eu': 7,
    'gpu_eu_simd_width': 8,
    'gpu_slices': 1,
    'gpu_subslices_per_slice': 12,
    'gpu_eu_count_per_subslice': 8}

Please refer to "Intel(R) Xe GPU Architecture" section of the "`oneAPI GPU Optimization Guide <gpu_opt_guide_>`_"
for detailed explanation of these architectural descriptors.

.. _gpu_opt_guide: https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/

Creating sub-devices
====================

Some SYCL devices may support partitioning into logical sub-devices.
Devices created by way of partitioning are treated the same way as unpartitioned devices, and
are represented as instances of :class:`dpctl.SyclDevice` class.

To partition a device use :meth:`dpctl.SyclDevice.create_sub_devices`. If the device instance
can not be partitioned any further, an exception :exc:`dpctl.SyclSubDeviceCreationError` is raised.

:Example:

    .. code-block:: python

        >>> import dpctl
        >>> cpu = dpctl.select_cpu_device()
        >>> sub_devs = cpu.create_sub_devices(partition=[2, 2])
        >>> len(sub_device)
        2
        >>> [d.max_compute_units for d in sub_devs]
        [2, 2]

Sub-devices may be used by expert users to create multiple queues and experiment with load balancing,
study scaling, etc.
