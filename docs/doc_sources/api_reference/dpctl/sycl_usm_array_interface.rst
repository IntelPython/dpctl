.. _suai_attribute:

``__sycl_usm_array_interface__`` attribute
==========================================

Python objects representing USM allocations, such as :py:class:`dpctl.memory.MemoryUSMDevice`,
:py:class:`dpctl.memory.MemoryUSMShared`, :py:class:`dpctl.memory.MemoryUSMHost`,
or :py:class:`dpctl.tensor.usm_ndarray`, distinguish themselves from other Python objects
by providing ``__sycl_usm_array_interface__`` attribute describing the allocation in a
Python dictionary with the following fields:

``"shape"``
    a tuple of integers describing dimensions of an N-dimensional array

``"typestr"``
    a string encoding elemental data type of the array. A valid typestring is a subset of
    typestrings supported by NumPy's `array interface protocol <numpy_array_interface_>`_
    corresponding to numeric and boolean data types:

    =====  ================================================================
    ``b``  Boolean (integer type where all values are only ``True`` or
           ``False``)
    ``i``  Integer
    ``u``  Unsigned integer
    ``f``  Floating point
    ``c``  Complex floating point
    =====  ================================================================

``"data"``
    A 2-tuple whose first element is a Python integer encoding USM pointer value.
    The second entry in the tuple is a read-only flag (``True`` means the data area
    is read-only).

``"strides"``
    an optional tuple of integers describing number of array elements needed to jump
    to the next array element in the corresponding dimensions. The default value of ``None``
    implies a C-style contiguous (row-major compact) layout of the array.

``"offset"``
    an optional Python integer encoding offset in number of elements from the pointer
    provided in ``"data"`` field to the array element with zero indices. Default: `0`.

``"syclobj"``
    Python object from which SYCL context to which represented USM allocation is bound.

    ==============================================  =======================================
    Filter selector string                          Platform's default context for platform
                                                    the SYCL device selected by the
                                                    :ref:`filter selector string <filter_selector_string>`
                                                    is a part of.
    :py:class:`dpctl.SyclContext`                   An explicitly provided context
    Python capsule with name ``"SyclContextRef"``   A Python capsule carrying a
                                                    ``DPCTLSyclContextRef`` opaque pointer.
    :py:class:`dpctl.SyclQueue`                     An explicitly provided queue which
                                                    encapsulates context.
    Python capsule with name ``"SyclQueueRef"``     A Python capsule carrying a
                                                    ``DPCTLSyclQueueRef`` opaque pointer.
    Any Python object with method ``_get_capsule``  An object whose method call
                                                    ``_get_capsule()`` returns a Python
                                                    capsule of the two supported kinds.
    ==============================================  =======================================

``"version"``
    version of the interface. At present, the only supported value is `1`.


.. _numpy_array_interface: https://numpy.org/doc/stable/reference/arrays.interface.html
