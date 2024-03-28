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

Selecting device based on aspects
---------------------------------

In addition, a :py:func:`select_device_with_aspects` permits selecting a device based on aspects it is required to have:

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
