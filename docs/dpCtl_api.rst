.. _dpCtl_api:

################
dpCtl Python API
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
.. autoexception:: dpctl.UnsupportedBackendError
.. autoexception:: dpctl.UnsupportedDeviceError

Functions
---------

.. autofunction:: device_context
.. autofunction:: dump
.. autofunction:: get_current_backend
.. autofunction:: get_current_device_type
.. autofunction:: get_current_queue
.. autofunction:: get_include
.. autofunction:: get_num_activated_queues
.. autofunction:: get_num_platforms
.. autofunction:: get_num_queues
.. autofunction:: has_cpu_queues
.. autofunction:: has_gpu_queues
.. autofunction:: has_sycl_platforms
.. autofunction:: is_in_device_context
.. autofunction:: set_default_queue
