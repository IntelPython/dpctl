.. _dpCtl.memory_api:

#######################
dpCtl Memory Python API
#######################

.. automodule:: dpctl.memory

Classes
-------

.. autoclass:: dpctl.memory.MemoryUSMDevice
    :members:
    :undoc-members:

.. autoclass:: dpctl.memory.MemoryUSMHost
    :members:
    :undoc-members:

.. autoclass:: dpctl.memory.MemoryUSMShared
    :members:
    :undoc-members:

Comparison with Rapids Memory Manager (RMM)
-------------------------------------------

RMM implements DeviceBuffer which is Cython native class wrapping around something similar to ``std::vector<unsigned char, custom_cuda_allocator (calls resource manager)>`` which is called device_buffer.

DeviceBuffer stores a unique pointer to an instance of this class. DeviceBuffer implements ``__cuda_array_interface__``. Direct constructors always allocate
new memory and copy provided inputs into the newly allocated array.

Zero-copy construction is possible from a ``unique_ptr<device_ buffer>``, with
the ownership being moved to the Cython extension instance.

DeviceBuffer provides ``__reduce__`` method to support pickling (which works by copying content of the device buffer to host) and provides the following set of routines, among others:

    - ``copy_to_host(host_buf_obj)`` to copy content of the underlying device_buffer to a host buffer
    - ``copy_from_host(host_buf_obf)`` to copy content of the host buffer into memory of underlying device_buffer
    - ``copy_from_device(cuda_ary_obj)`` to copy device memory underlying cuda_ary_obj Python object implementing ``__cuda_array_interface__`` to the memory underlying DeviceBuffer instance.

RMM's methods are declared nogil.