.. _dpctl_pyapi:

################
dpctl Python API
################

.. automodule:: dpctl

Sub-modules
-----------

    :mod:`dpctl.memory`
        USM allocators and deallocators and classes that implement Python's
        `buffer protocol`_.
    :mod:`dpctl.program`
        Experimental wrappers for SYCL 1.2 ``program`` and ``kernel`` classes.
        The module is going to be refactored in the future to support SYCL
        2020's ``kernel_bundle`` feature and the wrapper for the ``program``
        class is going to be removed.
    :mod:`dpctl.tensor`
        Implementation of different types of tensor classes that use USM memory.

Classes
-------

.. toctree::
    :maxdepth: 1

    dpctl_pyapi/SyclContext
    dpctl_pyapi/SyclDevice
    dpctl_pyapi/SyclEvent
    dpctl_pyapi/SyclPlatform
    dpctl_pyapi/SyclQueue

Enumerations
------------

.. autoclass:: dpctl.backend_type
    :members:

.. autoclass:: dpctl.device_type
    :members:

Exceptions
----------

.. autoexception:: dpctl.SyclKernelInvalidRangeError
.. autoexception:: dpctl.SyclKernelSubmitError
.. autoexception:: dpctl.SyclQueueCreationError

Device Selection Functions
--------------------------

.. autofunction:: get_devices
.. autofunction:: select_accelerator_device
.. autofunction:: select_cpu_device
.. autofunction:: select_default_device
.. autofunction:: select_gpu_device
.. autofunction:: select_host_device
.. autofunction:: get_num_devices
.. autofunction:: has_cpu_devices
.. autofunction:: has_gpu_devices
.. autofunction:: has_accelerator_devices
.. autofunction:: has_host_device

DPCTL Queue Management Functions
--------------------------------

.. autofunction:: device_context
.. autofunction:: get_current_backend
.. autofunction:: get_current_device_type
.. autofunction:: get_current_queue
.. autofunction:: get_num_activated_queues
.. autofunction:: is_in_device_context
.. autofunction:: set_global_queue

Other Helper Functions
----------------------
.. autofunction:: get_platforms
.. autofunction:: lsplatform

.. _Section 4.6: https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#_sycl_runtime_classes
.. _SYCL 2020 spec: https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html
.. _buffer protocol: https://docs.python.org/3/c-api/buffer.html
