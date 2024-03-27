.. _beginners_guide_device_info:

Obtaining information about device
==================================

An instance of :py:class:`SyclDevice` provides access to a collection of information
descriptors characterizing underlying ``sycl::device``.

Information of Boolean nature is exposed via ``has_aspect_*`` properties.
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
