.. _dpCtl_api:

################
dpctl Python API
################

.. automodule:: dpctl

Classes
-------

.. autoclass:: dpctl.SyclContext
    :members:
    :undoc-members:

.. autoclass:: dpctl.SyclDevice
    :members:
    :inherited-members:
    :undoc-members:

.. autoclass:: dpctl.SyclEvent
    :members:
    :undoc-members:

.. autoclass:: dpctl.SyclQueue
    :members:
    :undoc-members:

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

Functions
---------

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
.. autofunction:: get_platforms
.. autofunction:: lsplatform
.. autofunction:: device_context
.. autofunction:: get_current_backend
.. autofunction:: get_current_device_type
.. autofunction:: get_current_queue
.. autofunction:: get_num_activated_queues
.. autofunction:: is_in_device_context
.. autofunction:: set_global_queue