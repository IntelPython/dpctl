#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True

""" Implements SyclQueue Cython extension type.
"""

from ._backend cimport (  # noqa: E211
    DPCTLContext_Create,
    DPCTLContext_Delete,
    DPCTLDefaultSelector_Create,
    DPCTLDevice_Copy,
    DPCTLDevice_CreateFromSelector,
    DPCTLDevice_Delete,
    DPCTLDeviceMgr_GetCachedContext,
    DPCTLDeviceSelector_Delete,
    DPCTLEvent_Delete,
    DPCTLEvent_Wait,
    DPCTLFilterSelector_Create,
    DPCTLQueue_AreEq,
    DPCTLQueue_Copy,
    DPCTLQueue_Create,
    DPCTLQueue_Delete,
    DPCTLQueue_GetBackend,
    DPCTLQueue_GetContext,
    DPCTLQueue_GetDevice,
    DPCTLQueue_HasEnableProfiling,
    DPCTLQueue_Hash,
    DPCTLQueue_IsInOrder,
    DPCTLQueue_MemAdvise,
    DPCTLQueue_Memcpy,
    DPCTLQueue_MemcpyWithEvents,
    DPCTLQueue_Prefetch,
    DPCTLQueue_SubmitBarrierForEvents,
    DPCTLQueue_SubmitNDRange,
    DPCTLQueue_SubmitRange,
    DPCTLQueue_Wait,
    DPCTLSyclContextRef,
    DPCTLSyclDeviceSelectorRef,
    DPCTLSyclEventRef,
    _arg_data_type,
    _backend_type,
    _queue_property_type,
)
from .memory._memory cimport _Memory

import ctypes

from .enum_types import backend_type

from cpython cimport pycapsule
from cpython.buffer cimport PyObject_CheckBuffer
from cpython.ref cimport Py_DECREF, Py_INCREF, PyObject
from libc.stdlib cimport free, malloc

import collections.abc
import logging


cdef extern from "_host_task_util.hpp":
    DPCTLSyclEventRef async_dec_ref(DPCTLSyclQueueRef, PyObject **, size_t, DPCTLSyclEventRef *, size_t, int *) nogil


__all__ = [
    "SyclQueue",
    "SyclKernelInvalidRangeError",
    "SyclKernelSubmitError",
    "SyclQueueCreationError",
]


_logger = logging.getLogger(__name__)


cdef class kernel_arg_type_attribute:
    cdef str parent_name
    cdef str attr_name
    cdef int attr_value

    def __cinit__(self, str parent, str name, int value):
        self.parent_name = parent
        self.attr_name = name
        self.attr_value = value

    def __repr__(self):
        return f"<{self.parent_name}.{self.attr_name}: {self.attr_value}>"

    def __str__(self):
        return f"<{self.parent_name}.{self.attr_name}: {self.attr_value}>"

    @property
    def name(self):
        return self.attr_name

    @property
    def value(self):
        return self.attr_value


cdef class _kernel_arg_type:
    """
    An enumeration of supported kernel argument types in
    :func:`dpctl.SyclQueue.submit`
    """
    cdef str _name

    def __cinit__(self):
        self._name = "kernel_arg_type"


    @property
    def __name__(self):
        return self._name

    def __repr__(self):
        return "<enum 'kernel_arg_type'>"

    def __str__(self):
        return "<enum 'kernel_arg_type'>"

    @property
    def dpctl_int8(self):
        cdef str p_name = "dpctl_int8"
        return kernel_arg_type_attribute(
            self._name,
            p_name,
            _arg_data_type._INT8_T
        )

    @property
    def dpctl_uint8(self):
        cdef str p_name = "dpctl_uint8"
        return kernel_arg_type_attribute(
            self._name,
            p_name,
            _arg_data_type._UINT8_T
        )

    @property
    def dpctl_int16(self):
        cdef str p_name = "dpctl_int16"
        return kernel_arg_type_attribute(
            self._name,
            p_name,
            _arg_data_type._INT16_T
        )

    @property
    def dpctl_uint16(self):
        cdef str p_name = "dpctl_uint16"
        return kernel_arg_type_attribute(
            self._name,
            p_name,
            _arg_data_type._UINT16_T
        )

    @property
    def dpctl_int32(self):
        cdef str p_name = "dpctl_int32"
        return kernel_arg_type_attribute(
            self._name,
            p_name,
            _arg_data_type._INT32_T
        )

    @property
    def dpctl_uint32(self):
        cdef str p_name = "dpctl_uint32"
        return kernel_arg_type_attribute(
            self._name,
            p_name,
            _arg_data_type._UINT32_T
        )

    @property
    def dpctl_int64(self):
        cdef str p_name = "dpctl_int64"
        return kernel_arg_type_attribute(
            self._name,
            p_name,
            _arg_data_type._INT64_T
        )

    @property
    def dpctl_uint64(self):
        cdef str p_name = "dpctl_uint64"
        return kernel_arg_type_attribute(
            self._name,
            p_name,
            _arg_data_type._UINT64_T
        )

    @property
    def dpctl_float32(self):
        cdef str p_name = "dpctl_float32"
        return kernel_arg_type_attribute(
            self._name,
            p_name,
            _arg_data_type._FLOAT
        )

    @property
    def dpctl_float64(self):
        cdef str p_name = "dpctl_float64"
        return kernel_arg_type_attribute(
            self._name,
            p_name,
            _arg_data_type._DOUBLE
        )

    @property
    def dpctl_void_ptr(self):
        cdef str p_name = "dpctl_void_ptr"
        return kernel_arg_type_attribute(
            self._name,
            p_name,
            _arg_data_type._VOID_PTR
        )

    @property
    def dpctl_local_accessor(self):
        cdef str p_name = "dpctl_local_accessor"
        return kernel_arg_type_attribute(
            self._name,
            p_name,
            _arg_data_type._LOCAL_ACCESSOR
        )


kernel_arg_type = _kernel_arg_type()


cdef class SyclKernelSubmitError(Exception):
    """
    A ``SyclKernelSubmitError`` exception is raised when
    the provided :class:`.program.SyclKernel` could not be
    submitted to the :class:`.SyclQueue`.

    """
    pass


cdef class SyclKernelInvalidRangeError(Exception):
    """
    A ``SyclKernelInvalidRangeError`` is raised when the provided
    range has less than one or more than three dimensions.
    """
    pass


cdef class SyclQueueCreationError(Exception):
    """
    A ``SyclQueueCreationError`` exception is raised when a
    :class:`.SyclQueue` could not be created.

    :class:`.SyclQueue` creation can fail if the filter
    string is invalid, or the backend or device type values are not supported.

    """
    pass


cdef int _parse_queue_properties(object prop) except *:
    cdef int res = 0
    cdef object props
    if isinstance(prop, int):
        return <int>prop
    if not isinstance(prop, (tuple, list)):
        props = (prop, )
    else:
        props = prop
    for p in props:
        if isinstance(p, int):
            res = res | <int> p
        elif isinstance(p, str):
            if (p == "in_order"):
                res = res | _queue_property_type._IN_ORDER
            elif (p == "enable_profiling"):
                res = res | _queue_property_type._ENABLE_PROFILING
            elif (p == "default"):
                res = res | _queue_property_type._DEFAULT_PROPERTY
            else:
                raise ValueError(
                    (
                        "queue property '{}' is not understood, "
                        "expecting 'in_order', 'enable_profiling', or 'default'"
                    ).format(prop)
                )
        else:
            raise ValueError(
                "queue property '{}' is not understood.".format(prop)
            )
    return res


cdef void _queue_capsule_deleter(object o) noexcept:
    cdef DPCTLSyclQueueRef QRef = NULL
    if pycapsule.PyCapsule_IsValid(o, "SyclQueueRef"):
        QRef = <DPCTLSyclQueueRef> pycapsule.PyCapsule_GetPointer(
            o, "SyclQueueRef"
        )
        DPCTLQueue_Delete(QRef)
    elif pycapsule.PyCapsule_IsValid(o, "used_SyclQueueRef"):
        QRef = <DPCTLSyclQueueRef> pycapsule.PyCapsule_GetPointer(
            o, "used_SyclQueueRef"
        )
        DPCTLQueue_Delete(QRef)


cdef bint _is_buffer(object o):
    return PyObject_CheckBuffer(o)


cdef DPCTLSyclEventRef _memcpy_impl(
     SyclQueue q,
     object dst,
     object src,
     size_t byte_count,
     DPCTLSyclEventRef *dep_events,
     size_t dep_events_count
) except *:
    cdef void *c_dst_ptr = NULL
    cdef void *c_src_ptr = NULL
    cdef DPCTLSyclEventRef ERef = NULL
    cdef const unsigned char[::1] src_host_buf = None
    cdef unsigned char[::1] dst_host_buf = None

    if isinstance(src, _Memory):
        c_src_ptr = <void*>(<_Memory>src).get_data_ptr()
    elif _is_buffer(src):
        src_host_buf = src
        c_src_ptr = <void *>&src_host_buf[0]
    else:
        raise TypeError(
             "Parameter `src` should have either type "
             "`dpctl.memory._Memory` or a type that "
             "supports Python buffer protocol"
       )

    if isinstance(dst, _Memory):
        c_dst_ptr = <void*>(<_Memory>dst).get_data_ptr()
    elif _is_buffer(dst):
        dst_host_buf = dst
        c_dst_ptr = <void *>&dst_host_buf[0]
    else:
        raise TypeError(
             "Parameter `dst` should have either type "
             "`dpctl.memory._Memory` or a type that "
             "supports Python buffer protocol"
       )

    if dep_events_count == 0 or dep_events is NULL:
        ERef = DPCTLQueue_Memcpy(q._queue_ref, c_dst_ptr, c_src_ptr, byte_count)
    else:
        ERef = DPCTLQueue_MemcpyWithEvents(
            q._queue_ref,
            c_dst_ptr,
            c_src_ptr,
            byte_count,
            dep_events,
            dep_events_count
        )
    return ERef


cdef class _SyclQueue:
    """ Barebone data owner class used by SyclQueue.
    """
    def __dealloc__(self):
        if (self._queue_ref):
            DPCTLQueue_Delete(self._queue_ref)
        # self._context is a Python object and will be GC-ed
        # self._device is a Python object


cdef class SyclQueue(_SyclQueue):
    """
    SyclQueue(*args, **kwargs)
    Python class representing ``sycl::queue``.

    There are multiple ways to create a :class:`dpctl.SyclQueue` object:

    - Invoking the constructor with no arguments creates a context using
      the default selector.

    :Example:
        .. code-block:: python

            import dpctl

            # Create a default SyclQueue
            q = dpctl.SyclQueue()
            print(q.sycl_device)

    - Invoking the constructor with specific filter selector string that
      creates a queue for the device corresponding to the filter string.

    :Example:
        .. code-block:: python

            import dpctl

            # Create in-order SyclQueue for either gpu, or cpu device
            q = dpctl.SyclQueue("gpu,cpu", property="in_order")
            print([q.sycl_device.is_gpu, q.sycl_device.is_cpu])

    - Invoking the constructor with a :class:`dpctl.SyclDevice` object
      creates a queue for that device, automatically finding/creating
      a :class:`dpctl.SyclContext` for the given device.

    :Example:
        .. code-block:: python

            import dpctl

            d = dpctl.SyclDevice("gpu")
            q = dpctl.SyclQueue(d)
            ctx = q.sycl_context
            print(q.sycl_device == d)
            print(any([ d == ctx_d for ctx_d in ctx.get_devices()]))

    - Invoking the constructor with a :class:`dpctl.SyclContext` and a
      :class:`dpctl.SyclDevice` creates a queue for given context and
      device.

    :Example:
        .. code-block:: python

            import dpctl

            # Create a CPU device using the opencl driver
            cpu_d = dpctl.SyclDevice("opencl:cpu")
            # Partition the CPU device into sub-devices with two cores each.
            sub_devices = cpu_d.create_sub_devices(partition=2)
            # Create a context common to all the sub-devices.
            ctx = dpctl.SyclContext(sub_devices)
            # create a queue for each sub-device using the common context
            queues = [dpctl.SyclQueue(ctx, sub_d) for sub_d in sub_devices]

    - Invoking the constructor with a named ``PyCapsule`` with the name
      **"SyclQueueRef"** that carries a pointer to a ``sycl::queue``
      object. The capsule will be renamed upon successful consumption
      to ensure one-time use. A new named capsule can be constructed by
      using :func:`dpctl.SyclQueue._get_capsule` method.

    Args:
        ctx (:class:`dpctl.SyclContext`, optional): Sycl context to create
            :class:`dpctl.SyclQueue` from. If not specified, a single-device
            context will be created from the specified device.
        dev (str, :class:`dpctl.SyclDevice`, capsule, optional): Sycl device
             to create :class:`dpctl.SyclQueue` from. If not specified, sycl
             device selected by ``sycl::default_selector`` is used.
             The argument must be explicitly specified if `ctxt` argument is
             provided.

             If `dev` is a named ``PyCapsule`` called **"SyclQueueRef"** and
             `ctxt` is not specified, :class:`dpctl.SyclQueue` instance is
             created from foreign `sycl::queue` object referenced by the
             capsule.
        property (str, tuple(str), list(str), optional): Defaults to None.
                The argument can be either "default", "in_order",
                "enable_profiling", or a tuple containing these.

    Raises:
        SyclQueueCreationError: If the :class:`dpctl.SyclQueue` object
                                creation failed.
        TypeError: In case of incorrect arguments given to constructors,
                   unexpected types of input arguments, or in the case the input
                   capsule contained a null pointer or could not be renamed.

    """
    def __cinit__(self, *args, **kwargs):
        cdef int len_args
        cdef int status = 0
        cdef const char *filter_c_str = NULL
        if len(args) > 2:
            raise TypeError(
                "SyclQueue constructor takes 0, 1, or 2 positinal arguments, "
                "but {} were given.".format(len(args))
            )
        props = _parse_queue_properties(
            kwargs.pop('property', _queue_property_type._DEFAULT_PROPERTY)
        )
        if (kwargs):
            raise TypeError(
                f"Unsupported keyword arguments {kwargs} to "
                "SyclQueue constructor encountered."
            )
        len_args = len(args)
        if len_args == 0:
            status = self._init_queue_default(props)
        elif len_args == 1:
            arg = args[0]
            if type(arg) is str:
                string = bytes(<str>arg, "utf-8")
                filter_c_str = string
                status = self._init_queue_from_filter_string(
                    filter_c_str, props)
            elif type(arg) is _SyclQueue:
                status = self._init_queue_from__SyclQueue(<_SyclQueue>arg)
            elif isinstance(arg, SyclDevice):
                status = self._init_queue_from_device(<SyclDevice>arg, props)
            elif pycapsule.PyCapsule_IsValid(arg, "SyclQueueRef"):
                status = self._init_queue_from_capsule(arg)
            else:
                raise TypeError(
                    "Positional argument {} is not a filter string or a "
                    "SyclDevice".format(arg)
                )
        else:
            ctx, dev = args
            if not isinstance(ctx, SyclContext):
                raise TypeError(
                    "SyclQueue constructor with two positional arguments "
                    "expected SyclContext as its first argument, but got {}."
                    .format(type(ctx))
                )
            if not isinstance(dev, SyclDevice):
                raise TypeError(
                    "SyclQueue constructor with two positional arguments "
                    "expected SyclDevice as its second argument, but got {}."
                    .format(type(dev))
                )
            status = self._init_queue_from_context_and_device(
                <SyclContext>ctx, <SyclDevice>dev, props
            )
        if status < 0:
            if status == -1:
                raise SyclQueueCreationError(
                    "Device filter selector string '{}' is not understood."
                    .format(arg)
                )
            elif status == -2 or status == -8:
                default_dev_error = (
                    "Default SYCL Device could not be created."
                )
                raise SyclQueueCreationError(
                    default_dev_error if (len_args == 0) else
                    "SYCL Device '{}' could not be created.".format(arg)
                )
            elif status == -3 or status == -7:
                raise SyclQueueCreationError(
                    "SYCL Context could not be created " +
                    ("by default constructor" if len_args == 0 else
                     "from '{}'.".format(arg)
                    )
                )
            elif status == -4 or status == -6:
                if len_args == 2:
                    arg = args
                raise SyclQueueCreationError(
                    "SYCL Queue failed to be created from '{}'.".format(arg)
                )
            elif status == -5:
                raise TypeError(
                    "Input capsule {} contains a null pointer or could not "
                    "be renamed".format(arg)
                )

    cdef int _init_queue_from__SyclQueue(self, _SyclQueue other):
        """ Copy data container _SyclQueue fields over.
        """
        cdef DPCTLSyclQueueRef QRef = DPCTLQueue_Copy(other._queue_ref)
        if (QRef is NULL):
            return -4
        self._queue_ref = QRef
        self._context = other._context
        self._device = other._device

    cdef int _init_queue_from_DPCTLSyclDeviceRef(
        self, DPCTLSyclDeviceRef DRef, int props
    ):
        """
        Initializes self by creating SyclQueue with specified error handler and
        specified properties from the given device instance. SyclContext is
        looked-up by DPCTL from a cache to avoid repeated construction of new
        context for performance reasons.

        Returns: 0 : normal execution
                -3 : Context creation/look-up failed
                -4 : queue could not be created from context,device, error
                     handler and properties
        """
        cdef DPCTLSyclContextRef CRef
        cdef DPCTLSyclQueueRef QRef

        CRef = DPCTLDeviceMgr_GetCachedContext(DRef)
        if (CRef is NULL):
            # look-up failed (was not a root device?)
            # create a new context
            CRef = DPCTLContext_Create(DRef, NULL, 0)
            if (CRef is NULL):
                DPCTLDevice_Delete(DRef)
                return -3
        QRef = DPCTLQueue_Create(
            CRef,
            DRef,
            NULL,
            props
        )
        if QRef is NULL:
            DPCTLContext_Delete(CRef)
            DPCTLDevice_Delete(DRef)
            return -4
        _dev = SyclDevice._create(DRef)
        _ctxt = SyclContext._create(CRef)
        self._device = _dev
        self._context = _ctxt
        self._queue_ref = QRef
        return 0  # normal return

    cdef int _init_queue_from_filter_string(self, const char *c_str, int props):
        """
        Initializes self from filter string, error handler and properties.
        Creates device from device selector, then calls helper function above.

        Returns:
             0 : normal execution
            -1 : filter selector could not be created (malformed?)
            -2 : Device could not be created from filter selector
            -3 : Context creation/look-up failed
            -4 : queue could not be created from context,device, error handler
                 and properties
        """
        cdef DPCTLSyclDeviceSelectorRef DSRef = NULL
        cdef DPCTLSyclDeviceRef DRef = NULL
        cdef int ret = 0

        DSRef = DPCTLFilterSelector_Create(c_str)
        if DSRef is NULL:
            ret = -1  # Filter selector failed to be created
        else:
            DRef = DPCTLDevice_CreateFromSelector(DSRef)
            DPCTLDeviceSelector_Delete(DSRef)
            if (DRef is NULL):
                ret = -2  # Device could not be created
            else:
                ret = self._init_queue_from_DPCTLSyclDeviceRef(DRef, props)
        return ret

    cdef int _init_queue_from_device(self, SyclDevice dev, int props):
        cdef DPCTLSyclDeviceRef DRef = NULL
        # The DRef will be stored in self._device and freed when self._device
        # is garbage collected.
        DRef = DPCTLDevice_Copy(dev.get_device_ref())
        if (DRef is NULL):
            return -2  # Device could not be created
        else:
            return self._init_queue_from_DPCTLSyclDeviceRef(DRef, props)

    cdef int _init_queue_default(self, int props):
        cdef DPCTLSyclDeviceSelectorRef DSRef = DPCTLDefaultSelector_Create()
        cdef int ret = 0
        # The DRef will be stored in self._device and freed when self._device
        # is garbage collected.
        DRef = DPCTLDevice_CreateFromSelector(DSRef)
        DPCTLDeviceSelector_Delete(DSRef)
        if (DRef is NULL):
            ret = -2  # Device could not be created
        else:
            ret = self._init_queue_from_DPCTLSyclDeviceRef(DRef, props)
        return ret

    cdef int _init_queue_from_context_and_device(
        self, SyclContext ctxt, SyclDevice dev, int props
    ):
        cdef DPCTLSyclContextRef CRef = NULL
        cdef DPCTLSyclDeviceRef DRef = NULL
        cdef DPCTLSyclQueueRef QRef = NULL
        CRef = ctxt.get_context_ref()
        DRef = dev.get_device_ref()
        QRef = DPCTLQueue_Create(
            CRef,
            DRef,
            NULL,
            props
        )
        if (QRef is NULL):
            return -4
        self._device = dev
        self._context = ctxt
        self._queue_ref = QRef
        return 0  # normal return

    cdef int _init_queue_from_capsule(self, object cap):
        """
        For named PyCapsule with name "SyclQueueRef", which carries pointer to
        ``sycl::queue`` object, interpreted as ``DPCTLSyclQueueRef``, creates
        corresponding :class:`.SyclQueue`.
        """
        cdef DPCTLSyclContextRef CRef = NULL
        cdef DPCTLSyclDeviceRef DRef = NULL
        cdef DPCTLSyclQueueRef QRef = NULL
        cdef DPCTLSyclQueueRef QRef_copy = NULL
        cdef int ret = 0
        if pycapsule.PyCapsule_IsValid(cap, "SyclQueueRef"):
            QRef = <DPCTLSyclQueueRef> pycapsule.PyCapsule_GetPointer(
                cap, "SyclQueueRef"
            )
            if (QRef is NULL):
                return -5
            ret = pycapsule.PyCapsule_SetName(cap, "used_SyclQueueRef")
            if (ret):
                return -5
            QRef_copy = DPCTLQueue_Copy(QRef)
            if (QRef_copy is NULL):
                return -6
            CRef = DPCTLQueue_GetContext(QRef_copy)
            if (CRef is NULL):
                DPCTLQueue_Delete(QRef_copy)
                return -7
            DRef = DPCTLQueue_GetDevice(QRef_copy)
            if (DRef is NULL):
                DPCTLContext_Delete(CRef)
                DPCTLQueue_Delete(QRef_copy)
                return -8
            self._context = SyclContext._create(CRef)
            self._device = SyclDevice._create(DRef)
            self._queue_ref = QRef_copy
            return 0
        else:
            # __cinit__ checks that capsule is valid, so one can be here only
            # if call to `_init_queue_from_capsule` was made outside of
            # __cinit__ and the capsule was not checked to be valid.
            return -128

    @staticmethod
    cdef SyclQueue _create(DPCTLSyclQueueRef qref):
        """
        This function calls ``DPCTLQueue_Delete(qref)``.
        The user of this function must pass a copy to keep the
        qref argument alive.
        """
        if qref is NULL:
            raise SyclQueueCreationError("Queue creation failed.")
        cdef _SyclQueue ret = _SyclQueue.__new__(_SyclQueue)
        ret._context = SyclContext._create(DPCTLQueue_GetContext(qref))
        ret._device = SyclDevice._create(DPCTLQueue_GetDevice(qref))
        ret._queue_ref = qref
        # ret is a temporary, and will call DPCTLQueue_Delete(qref)
        return SyclQueue(ret)

    @staticmethod
    cdef SyclQueue _create_from_context_and_device(
        SyclContext ctx, SyclDevice dev, int props=0
    ):
        """
        Static factory method to create :class:`dpctl.SyclQueue` instance
        from given :class:`dpctl.SyclContext`, :class:`dpctl.SyclDevice`
        and optional integer ``props`` encoding the queue properties.
        """
        cdef _SyclQueue ret = _SyclQueue.__new__(_SyclQueue)
        cdef DPCTLSyclContextRef cref = ctx.get_context_ref()
        cdef DPCTLSyclDeviceRef dref = dev.get_device_ref()
        cdef DPCTLSyclQueueRef qref = NULL

        qref = DPCTLQueue_Create(
            cref,
            dref,
            NULL,
            props
        )
        if qref is NULL:
            raise SyclQueueCreationError("Queue creation failed.")
        ret._queue_ref = qref
        ret._context = ctx
        ret._device = dev
        return SyclQueue(ret)

    cdef int _populate_args(
        self,
        list args,
        void **kargs,
        _arg_data_type *kargty
    ):
        cdef int ret = 0
        for idx, arg in enumerate(args):
            if isinstance(arg, ctypes.c_char):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._INT8_T
            elif isinstance(arg, ctypes.c_uint8):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._UINT8_T
            elif isinstance(arg, ctypes.c_short):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._INT16_T
            elif isinstance(arg, ctypes.c_ushort):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._UINT16_T
            elif isinstance(arg, ctypes.c_int):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._INT32_T
            elif isinstance(arg, ctypes.c_uint):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._UINT32_T
            elif isinstance(arg, ctypes.c_longlong):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._INT64_T
            elif isinstance(arg, ctypes.c_ulonglong):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._UINT64_T
            elif isinstance(arg, ctypes.c_float):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._FLOAT
            elif isinstance(arg, ctypes.c_double):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._DOUBLE
            elif isinstance(arg, _Memory):
                kargs[idx]= <void*>(<size_t>arg._pointer)
                kargty[idx] = _arg_data_type._VOID_PTR
            else:
                ret = -1
        return ret

    cdef int _populate_range(self, size_t Range[3], list S, size_t nS):

        cdef int ret = 0

        if nS == 1:
            Range[0] = <size_t>S[0]
            Range[1] = 1
            Range[2] = 1
        elif nS == 2:
            Range[0] = <size_t>S[0]
            Range[1] = <size_t>S[1]
            Range[2] = 1
        elif nS == 3:
            Range[0] = <size_t>S[0]
            Range[1] = <size_t>S[1]
            Range[2] = <size_t>S[2]
        else:
            ret = -1

        return ret

    cdef cpp_bool equals(self, SyclQueue q):
        """ Returns true if the :class:`.SyclQueue` argument ``q`` has the
            same ``._queue_ref`` attribute as this :class:`.SyclQueue`.
        """
        return DPCTLQueue_AreEq(self._queue_ref, q.get_queue_ref())

    def __eq__(self, other):
        """
        Returns True if two :class:`dpctl.SyclQueue` compared arguments have
        the same underlying ``DPCTLSyclQueueRef`` object.

        Returns:
            bool:
                ``True`` if the two :class:`dpctl.SyclQueue` objects
                point to the same ``DPCTLSyclQueueRef`` object, otherwise
                ``False``.
        """
        if isinstance(other, SyclQueue):
            return self.equals(<SyclQueue> other)
        else:
            return False

    @property
    def backend(self):
        """ Returns the ``backend_type`` enum value for this queue.

        Returns:
            backend_type:
                The backend for the queue.
        """
        cdef _backend_type BE = DPCTLQueue_GetBackend(self._queue_ref)
        if BE == _backend_type._OPENCL:
            return backend_type.opencl
        elif BE == _backend_type._LEVEL_ZERO:
            return backend_type.level_zero
        elif BE == _backend_type._CUDA:
            return backend_type.cuda
        else:
            raise ValueError("Unknown backend type.")

    @property
    def sycl_context(self):
        """
        Returns :class:`SyclContext` underlying this queue.

        Returns:
            :class:`SyclContext`
                SYCL context underlying this queue
        """
        return self._context

    @property
    def sycl_device(self):
        """
        Returns :class:`.SyclDevice` targeted by this queue.

        Returns:
            :class:`SyclDevice`
                SYCL device targeted by this queue
        """
        return self._device

    cpdef SyclContext get_sycl_context(self):
        return self._context

    cpdef SyclDevice get_sycl_device(self):
        return self._device

    cdef DPCTLSyclQueueRef get_queue_ref(self):
        return self._queue_ref

    def addressof_ref(self):
        """
        Returns the address of the C API ``DPCTLSyclQueueRef`` pointer as
        integral value of type ``size_t``.

        Returns:
            int:
                The address of the ``DPCTLSyclQueueRef`` object used to create
                this :class:`dpctl.SyclQueue` object cast to ``size_t`` type.
        """
        return <size_t>self._queue_ref


    cpdef SyclEvent _submit_keep_args_alive(
        self,
        object args,
        list dEvents
    ):
        """ SyclQueue._submit_keep_args_alive(args, events)

        Keeps objects in ``args`` alive until tasks associated with events
        complete.

        Args:
            args(object):
                Python object to keep alive.
                Typically a tuple with arguments to offloaded tasks
            events(Tuple[dpctl.SyclEvent]):
                Gating events.
                The list or tuple of events associated with tasks
                working on Python objects collected in ``args``.
        Returns:
            dpctl.SyclEvent
               The event associated with the submission of host task.

        Increments reference count of ``args`` and schedules asynchronous
        ``host_task`` to decrement the count once dependent events are
        complete.

        .. note::
            The ``host_task`` attempts to acquire Python GIL, and it is
            known to be unsafe during interpreter shutdown sequence. It is
            thus strongly advised to ensure that all submitted ``host_task``
            complete before the end of the Python script.
        """
        cdef size_t nDE = len(dEvents)
        cdef DPCTLSyclEventRef *depEvents = NULL
        cdef PyObject *args_raw = NULL
        cdef DPCTLSyclEventRef htERef = NULL
        cdef int status = -1

        # Create the array of dependent events if any
        if nDE > 0:
            depEvents = (
                <DPCTLSyclEventRef*>malloc(nDE*sizeof(DPCTLSyclEventRef))
            )
            if not depEvents:
                raise MemoryError()
            else:
                for idx, de in enumerate(dEvents):
                    if isinstance(de, SyclEvent):
                        depEvents[idx] = (<SyclEvent>de).get_event_ref()
                    else:
                        free(depEvents)
                        raise TypeError(
                            "A sequence of dpctl.SyclEvent is expected"
                        )

        # increment reference counts to list of arguments
        Py_INCREF(args)

        # schedule decrement
        args_raw = <PyObject *>args

        htERef = async_dec_ref(
            self.get_queue_ref(),
            &args_raw, 1,
            depEvents, nDE, &status
        )

        free(depEvents)
        if (status != 0):
            with nogil: DPCTLEvent_Wait(htERef)
            DPCTLEvent_Delete(htERef)
            raise RuntimeError("Could not submit keep_args_alive host_task")

        return SyclEvent._create(htERef)


    cpdef SyclEvent submit_async(
        self,
        SyclKernel kernel,
        list args,
        list gS,
        list lS=None,
        list dEvents=None
    ):
        """
        Asynchronously submit :class:`dpctl.program.SyclKernel` for execution.

        Args:
            kernel (dpctl.program.SyclKernel):
                SYCL kernel object
            args (List[object]):
                List of kernel arguments
            gS (List[int]):
                Global iteration range. Must be a list of length 1, 2, or 3.
            lS (List[int], optional):
                Local iteration range. Must be ``None`` or have the same
                length as ``gS`` and each element of ``gS`` must be divisible
                by respective element of ``lS``.
            dEvents (List[dpctl.SyclEvent], optional):
                List of events indicating ordering of this task relative
                to tasks associated with specified events.

        Returns:
            dpctl.SyclEvent:
                An event associated with submission of the kernel.

        .. note::
            One must ensure that the lifetime of all kernel arguments
            extends after the submitted task completes. It is not a concern for
            scalar arguments since they are passed by value, but for
            objects representing USM allocations which are passed to the kernel
            as unified address space pointers.

            One way of accomplishing this is to use
            :meth:`dpctl.SyclQueue._submit_keep_args_alive`.
        """
        cdef void **kargs = NULL
        cdef _arg_data_type *kargty = NULL
        cdef DPCTLSyclEventRef *depEvents = NULL
        cdef DPCTLSyclEventRef Eref = NULL
        cdef DPCTLSyclEventRef htEref = NULL
        cdef int ret = 0
        cdef size_t gRange[3]
        cdef size_t lRange[3]
        cdef size_t nGS = len(gS)
        cdef size_t nLS = len(lS) if lS is not None else 0
        cdef size_t nDE = len(dEvents) if dEvents is not None else 0
        cdef PyObject *args_raw = NULL
        cdef ssize_t i = 0

        # Allocate the arrays to be sent to DPCTLQueue_Submit
        kargs = <void**>malloc(len(args) * sizeof(void*))
        if not kargs:
            raise MemoryError()
        kargty = (
            <_arg_data_type*>malloc(len(args)*sizeof(_arg_data_type))
        )
        if not kargty:
            free(kargs)
            raise MemoryError()
        # Create the array of dependent events if any
        if dEvents is not None and nDE > 0:
            depEvents = (
                <DPCTLSyclEventRef*>malloc(nDE*sizeof(DPCTLSyclEventRef))
            )
            if not depEvents:
                free(kargs)
                free(kargty)
                raise MemoryError()
            else:
                for idx, de in enumerate(dEvents):
                    if isinstance(de, SyclEvent):
                        depEvents[idx] = (<SyclEvent>de).get_event_ref()
                    else:
                        free(kargs)
                        free(kargty)
                        free(depEvents)
                        raise TypeError(
                            "A sequence of dpctl.SyclEvent is expected"
                        )

        # populate the args and argstype arrays
        ret = self._populate_args(args, kargs, kargty)
        if ret == -1:
            free(kargs)
            free(kargty)
            free(depEvents)
            raise TypeError("Unsupported type for a kernel argument")

        if lS is None:
            ret = self._populate_range(gRange, gS, nGS)
            if ret == -1:
                free(kargs)
                free(kargty)
                free(depEvents)
                raise SyclKernelInvalidRangeError(
                    "Range with ", nGS, " not allowed. Range can only have "
                    "between one and three dimensions."
                )
            Eref = DPCTLQueue_SubmitRange(
                kernel.get_kernel_ref(),
                self.get_queue_ref(),
                kargs,
                kargty,
                len(args),
                gRange,
                nGS,
                depEvents,
                nDE
            )
        else:
            ret = self._populate_range(gRange, gS, nGS)
            if ret == -1:
                free(kargs)
                free(kargty)
                free(depEvents)
                raise SyclKernelInvalidRangeError(
                    "Range with ", nGS, " not allowed. Range can only have "
                    "between one and three dimensions."
                )
            ret = self._populate_range(lRange, lS, nLS)
            if ret == -1:
                free(kargs)
                free(kargty)
                free(depEvents)
                raise SyclKernelInvalidRangeError(
                    "Range with ", nLS, " not allowed. Range can only have "
                    "between one and three dimensions."
                )
            if nGS != nLS:
                free(kargs)
                free(kargty)
                free(depEvents)
                raise ValueError(
                    "Local and global ranges need to have same "
                    "number of dimensions."
                )
            Eref = DPCTLQueue_SubmitNDRange(
                kernel.get_kernel_ref(),
                self.get_queue_ref(),
                kargs,
                kargty,
                len(args),
                gRange,
                lRange,
                nGS,
                depEvents,
                nDE
            )
        free(kargs)
        free(kargty)
        free(depEvents)

        if Eref is NULL:
            raise SyclKernelSubmitError(
                "Kernel submission to Sycl queue failed."
            )

        return SyclEvent._create(Eref)

    cpdef SyclEvent submit(
        self,
        SyclKernel kernel,
        list args,
        list gS,
        list lS=None,
        list dEvents=None
    ):
        """
        Submit :class:`dpctl.program.SyclKernel` for execution.

        Args:
            kernel (dpctl.program.SyclKernel):
                SYCL kernel object
            args (List[object]):
                List of kernel arguments
            gS (List[int]):
                Global iteration range. Must be a list of length 1, 2, or 3.
            lS (List[int], optional):
                Local iteration range. Must be ``None`` or have the same
                length as ``gS`` and each element of ``gS`` must be divisible
                by respective element of ``lS``.
            dEvents (List[dpctl.SyclEvent], optional):
                List of events indicating ordering of this task relative
                to tasks associated with specified events.

        Returns:
            dpctl.SyclEvent:
                An event which is always complete. May be ignored.

        .. note::
            :meth:`dpctl.SyclQueue.submit` is a synchronizing method.
            Use :meth:`dpctl.SyclQueue.submit_async` to avoid synchronization.
        """
        cdef SyclEvent e = self.submit_async(kernel, args, gS, lS, dEvents)
        e.wait()
        return e

    cpdef void wait(self):
        with nogil: DPCTLQueue_Wait(self._queue_ref)

    cpdef memcpy(self, dest, src, size_t count):
        """Copy memory from `src` to `dst`"""
        cdef DPCTLSyclEventRef ERef = NULL

        ERef = _memcpy_impl(<SyclQueue>self, dest, src, count, NULL, 0)
        if (ERef is NULL):
            raise RuntimeError(
                "SyclQueue.memcpy operation encountered an error"
            )
        with nogil: DPCTLEvent_Wait(ERef)
        DPCTLEvent_Delete(ERef)

    cpdef SyclEvent memcpy_async(self, dest, src, size_t count, list dEvents=None):
        """Copy memory from ``src`` to ``dst``"""
        cdef DPCTLSyclEventRef ERef = NULL
        cdef DPCTLSyclEventRef *depEvents = NULL
        cdef size_t nDE = 0

        if dEvents is None:
            ERef = _memcpy_impl(<SyclQueue>self, dest, src, count, NULL, 0)
        else:
            nDE = len(dEvents)
            depEvents = (
                <DPCTLSyclEventRef*>malloc(nDE*sizeof(DPCTLSyclEventRef))
            )
            if depEvents is NULL:
                raise MemoryError()
            else:
                for idx, de in enumerate(dEvents):
                    if isinstance(de, SyclEvent):
                        depEvents[idx] = (<SyclEvent>de).get_event_ref()
                    else:
                        free(depEvents)
                        raise TypeError(
                            "A sequence of dpctl.SyclEvent is expected"
                        )
            ERef = _memcpy_impl(self, dest, src, count, depEvents, nDE)
            free(depEvents)

        if (ERef is NULL):
            raise RuntimeError(
                "SyclQueue.memcpy operation encountered an error"
            )

        return SyclEvent._create(ERef)

    cpdef prefetch(self, mem, size_t count=0):
        cdef void *ptr
        cdef DPCTLSyclEventRef ERef = NULL

        if isinstance(mem, _Memory):
            ptr = <void*>(<_Memory>mem).get_data_ptr()
        else:
            raise TypeError("Parameter `mem` should have type _Memory")

        if (count <=0 or count > mem.nbytes):
            count = mem.nbytes

        ERef = DPCTLQueue_Prefetch(self._queue_ref, ptr, count)
        if (ERef is NULL):
            raise RuntimeError(
                "SyclQueue.prefetch encountered an error"
            )
        with nogil: DPCTLEvent_Wait(ERef)
        DPCTLEvent_Delete(ERef)

    cpdef mem_advise(self, mem, size_t count, int advice):
        cdef void *ptr
        cdef DPCTLSyclEventRef ERef = NULL

        if isinstance(mem, _Memory):
            ptr = <void*>(<_Memory>mem).get_data_ptr()
        else:
            raise TypeError("Parameter `mem` should have type _Memory")

        if (count <=0 or count > mem.nbytes):
            count = mem.nbytes

        ERef = DPCTLQueue_MemAdvise(self._queue_ref, ptr, count, advice)
        if (ERef is NULL):
            raise RuntimeError(
                "SyclQueue.mem_advise operation encountered an error"
            )
        with nogil: DPCTLEvent_Wait(ERef)
        DPCTLEvent_Delete(ERef)

    @property
    def is_in_order(self):
        """``True`` if :class:`.SyclQueue`` is in-order,
        ``False`` if it is out-of-order.

        :Example:

            ..code-block:: python

                >>> import dpctl
                >>> q = dpctl.SyclQueue("cpu")
                >>> q.is_in_order
                False
                >>> q = dpctl.SyclQueue("cpu", property="in_order")
                >>> q.is_in_order
                True

        Returns:
            bool:
                Indicates whether this :class:`.SyclQueue` was created
                with ``property="in_order"``.

        .. note::
            Unless requested otherwise, :class:`.SyclQueue` is constructed
            to support out-of-order execution.
        """
        return DPCTLQueue_IsInOrder(self._queue_ref)

    @property
    def has_enable_profiling(self):
        """
        ``True`` if :class:`.SyclQueue` was constructed with
        ``"enabled_profiling"`` property, ``False`` otherwise.

        :Example:

            ..code-block:: python

                >>> import dpctl
                >>> q = dpctl.SyclQueue("cpu")
                >>> q.has_enable_profiling
                False
                >>> q = dpctl.SyclQueue("cpu", property="enable_profiling")
                >>> q.has_enable_profiling
                True

        Returns:
            bool:
                Whether profiling information for tasks submitted
                to this :class:`.SyclQueue` is being collected.

        .. note::
            Profiling information can be accessed using
            properties
            :attr:`dpctl.SyclEvent.profiling_info_submit`,
            :attr:`dpctl.SyclEvent.profiling_info_start`, and
            :attr:`dpctl.SyclEvent.profiling_info_end`. It is
            also necessary for proper working of
            :class:`dpctl.SyclTimer`.

            Collection of profiling information is not enabled
            by default.
        """
        return DPCTLQueue_HasEnableProfiling(self._queue_ref)

    @property
    def __name__(self):
        "The name of :class:`dpctl.SyclQueue` object"
        return "SyclQueue"

    def __repr__(self):
        cdef cpp_bool in_order = DPCTLQueue_IsInOrder(self._queue_ref)
        cdef cpp_bool en_prof = DPCTLQueue_HasEnableProfiling(self._queue_ref)
        if in_order or en_prof:
            prop = []
            if in_order:
                prop.append("in_order")
            if en_prof:
                prop.append("enable_profiling")
            return (
                "<dpctl."
                + self.__name__
                + " at {}, property={}>".format(hex(id(self)), prop)
            )
        else:
            return "<dpctl." + self.__name__ + " at {}>".format(hex(id(self)))

    def __hash__(self):
        """
        Returns a hash value by hashing the underlying ``sycl::queue`` object.

        Returns:
            int:
                Hash value of this :class:`.SyclQueue` instance
        """
        return DPCTLQueue_Hash(self._queue_ref)

    def _get_capsule(self):
        cdef DPCTLSyclQueueRef QRef = NULL
        QRef = DPCTLQueue_Copy(self._queue_ref)
        if (QRef is NULL):
            raise ValueError("SyclQueue copy failed.")
        return pycapsule.PyCapsule_New(
            <void *>QRef, "SyclQueueRef", &_queue_capsule_deleter
        )

    cpdef SyclEvent submit_barrier(self, dependent_events=None):
        """
        Submits a barrier to this queue.

        Args:
            dependent_events:
                List[dpctl.SyclEvent]:
                    List or tuple of events that must complete
                    before this task may begin execution.

        Returns:
            dpctl.SyclEvent:
                Event associated with the submitted task
        """
        cdef DPCTLSyclEventRef *depEvents = NULL
        cdef DPCTLSyclEventRef ERef = NULL
        cdef size_t nDE = 0
        # Create the array of dependent events if any
        if (dependent_events is None or
            (isinstance(dependent_events, collections.abc.Sequence) and
             all([type(de) is SyclEvent for de in dependent_events]))):
            nDE = 0 if dependent_events is None else len(dependent_events)
        else:
            raise TypeError(
                "dependent_events must either None, or a sequence of "
                ":class:`dpctl.SyclEvent` objects")
        if nDE > 0:
            depEvents = (
                <DPCTLSyclEventRef*>malloc(nDE*sizeof(DPCTLSyclEventRef))
            )
            if not depEvents:
                raise MemoryError()
            else:
                for idx, de in enumerate(dependent_events):
                    depEvents[idx] = (<SyclEvent>de).get_event_ref()

        ERef = DPCTLQueue_SubmitBarrierForEvents(
            self.get_queue_ref(), depEvents, nDE)
        if (depEvents is not NULL):
            free(depEvents)
        if ERef is NULL:
            raise SyclKernelSubmitError(
                "Barrier submission to Sycl queue failed."
            )

        return SyclEvent._create(ERef)

    @property
    def name(self):
        """Returns the device name for the device
        associated with this queue.

        Returns:
            str:
                The name of the device as a string.
        """
        return self.sycl_device.name

    @property
    def driver_version(self):
        """Returns the driver version for the device
        associated with this queue.

        Returns:
            str:
                The driver version of the device as a string.
        """
        return self.sycl_device.driver_version

    def print_device_info(self):
        """ Print information about the SYCL device
        associated with this queue.
        """
        self.sycl_device.print_device_info()


cdef api DPCTLSyclQueueRef SyclQueue_GetQueueRef(SyclQueue q):
    """
    C-API function to get opaque queue reference from
    :class:`dpctl.SyclQueue` instance.
    """
    return q.get_queue_ref()


cdef api SyclQueue SyclQueue_Make(DPCTLSyclQueueRef QRef):
    """
    C-API function to create :class:`dpctl.SyclQueue` instance
    from the given opaque queue reference.
    """
    cdef DPCTLSyclQueueRef copied_QRef = DPCTLQueue_Copy(QRef)
    return SyclQueue._create(copied_QRef)
