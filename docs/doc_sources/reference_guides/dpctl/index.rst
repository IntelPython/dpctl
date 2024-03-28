.. _dpctl_pyapi:

.. currentmodule:: dpctl

:py:mod:`dpctl`
===============

.. py:module:: dpctl

.. rubric:: Submodules

.. list-table::
    :widths: 10 50

    * - :py:mod:`dpctl.memory`
      - Unified Shared Memory operations
    * - :py:mod:`dpctl.program`
      - Support for working with SYCL kernels
    * - :py:mod:`dpctl.tensor`
      - Array library conforming to Python Array API specification
    * - :py:mod:`dpctl.utils`
      - A collection of utility functions

.. rubric:: Classes

.. autosummary::
    :toctree: generated
    :nosignatures:

    SyclDevice
    SyclContext
    SyclQueue
    SyclEvent
    SyclPlatform
    SyclTimer

.. rubric:: Device selection

.. _dpctl_device_selection_functions:

.. autosummary::
    :toctree: generated
    :nosignatures:

    select_default_device
    select_cpu_device
    select_gpu_device
    select_accelerator_device
    select_device_with_aspects

.. rubric:: Platform discovery

.. autosummary::
    :toctree: generated
    :nosignatures:

    get_devices
    lsplatform
    get_num_devices
    has_gpu_devices
    has_cpu_devices
    has_accelerator_devices

.. rubric:: Exceptions

.. autosummary::
    :toctree: generated
    :nosignatures:

    SyclDeviceCreationError
    SyclContextCreationError
    SyclQueueCreationError
    SyclSubDeviceCreationError

.. toctree::
    :hidden:

    filter_selector_string
