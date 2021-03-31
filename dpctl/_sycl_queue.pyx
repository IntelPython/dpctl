#                      Data Parallel Control (dpCtl)
#
# Copyright 2020-2021 Intel Corporation
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

""" Implements SyclQueue Cython extension type.
"""

from __future__ import print_function
from ._backend cimport (
    _arg_data_type,
    _backend_type,
    _queue_property_type,
    DPCTLContext_Delete,
    DPCTLDefaultSelector_Create,
    DPCTLDevice_CreateFromSelector,
    DPCTLDeviceMgr_GetCachedContext,
    DPCTLDeviceSelector_Delete,
    DPCTLDevice_Copy,
    DPCTLDevice_Delete,
    DPCTLFilterSelector_Create,
    DPCTLQueue_AreEq,
    DPCTLQueue_Copy,
    DPCTLQueue_Create,
    DPCTLQueue_Delete,
    DPCTLQueue_GetBackend,
    DPCTLQueue_GetContext,
    DPCTLQueue_GetDevice,
    DPCTLQueue_MemAdvise,
    DPCTLQueue_Memcpy,
    DPCTLQueue_Prefetch,
    DPCTLQueue_SubmitNDRange,
    DPCTLQueue_SubmitRange,
    DPCTLQueue_Wait,
    DPCTLSyclBackendType,
    DPCTLSyclContextRef,
    DPCTLSyclDeviceSelectorRef,
    DPCTLSyclEventRef,
    error_handler_callback,
)
from .memory._memory cimport _Memory
from . import backend_type
import ctypes
from libc.stdlib cimport malloc, free
import logging


__all__ = [
    "SyclQueue",
    "SyclKernelInvalidRangeError",
    "SyclKernelSubmitError",
    "SyclQueueCreationError",
]


_logger = logging.getLogger(__name__)


cdef class SyclKernelSubmitError(Exception):
    """
    A SyclKernelSubmitError exception is raised when the provided
    :class:`.SyclKernel` could not be submitted to the :class:`.SyclQueue`.

    """
    pass


cdef class SyclKernelInvalidRangeError(Exception):
    """
    A SyclKernelInvalidRangeError is raised when the provided range has less
    than one or more than three dimensions.
    """
    pass


cdef class SyclQueueCreationError(Exception):
    """
    A SyclQueueCreationError exception is raised when a :class:`.SyclQueue`
    could not be created. :class:`.SyclQueue` creation can fail if the filter
    string is invalid, or the backend or device type values are not supported.

    """
    pass


cdef class SyclAsynchronousError(Exception):
    """
    A SyclAsynchronousError exception is raised when SYCL operation submission
    or execution encounters an error.
    """


cdef void default_async_error_handler(int err) nogil except *:
    with gil:
        raise SyclAsynchronousError(err)


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
                raise ValueError("queue property '{}' is not understood.".format(prop))
        else:
            raise ValueError("queue property '{}' is not understood.".format(prop))
    return res


cdef class _SyclQueue:
    """ Internal helper metaclass to abstract `cl::sycl::queue` instance.
    """
    def __dealloc__(self):
        if (self._queue_ref):
            DPCTLQueue_Delete(self._queue_ref)
        # self._context is a Python object and will be GC-ed
        # self._device is a Python object


cdef class SyclQueue:
    """ Python wrapper class for cl::sycl::queue.
    """
    def __cinit__(self, *args, **kwargs):
        """
           SyclQueue(*, /, property=None)
               create SyclQueue from default selector
           SyclQueue(filter_string, *, /, propery=None)
               create SyclQueue from filter selector string
           SyclQueue(SyclDevice, *, / property=None)
               create SyclQueue from give SyclDevice automatically
               finding/creating SyclContext.
           SyclQueue(SyclContext, SyclDevice, *, /, property=None)
               create SyclQueue from give SyclContext, SyclDevice
        """
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
        len_args = len(args)
        if len_args == 0:
            status = self._init_queue_default(props)
        elif len_args == 1:
            arg = args[0]
            if type(arg) is unicode:
                string = bytes(<unicode>arg, "utf-8")
                filter_c_str = string
                status = self._init_queue_from_filter_string(
                    filter_c_str, props)
            elif type(arg) is _SyclQueue:
                status = self._init_queue_from__SyclQueue(<_SyclQueue>arg)
            elif isinstance(arg, unicode):
                string = bytes(<unicode>unicode(arg), "utf-8")
                filter_c_str = string
                status = self._init_queue_from_filter_string(
                    filter_c_str, props)
            elif isinstance(arg, SyclDevice):
                status = self._init_queue_from_device(<SyclDevice>arg, props)
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
            elif status == -2:
                raise SyclQueueCreationError(
                    "SYCL Device '{}' could not be created.".format(arg)
                )
            elif status == -3:
                raise SyclQueueCreationError(
                    "SYCL Context could not be created from '{}'.".format(arg)
                )
            elif status == -4:
                if len_args == 2:
                    arg = args
                raise SyclQueueCreationError(
                    "SYCL Queue failed to be created from '{}'.".format(arg)
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
            DPCTLDevice_Delete(DRef)
            return -3
        QRef = DPCTLQueue_Create(
            CRef,
            DRef,
            <error_handler_callback *>&default_async_error_handler,
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
        return 0 # normal return

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
            ret = -1 # Filter selector failed to be created
        else:
            DRef = DPCTLDevice_CreateFromSelector(DSRef)
            DPCTLDeviceSelector_Delete(DSRef)
            if (DRef is NULL):
                ret = -2 # Device could not be created
            else:
                ret = self._init_queue_from_DPCTLSyclDeviceRef(DRef, props)
        return ret

    cdef int _init_queue_from_device(self, SyclDevice dev, int props):
        cdef DPCTLSyclDeviceRef DRef = NULL
        # The DRef will be stored in self._device and freed when self._device
        # is garbage collected.
        DRef = DPCTLDevice_Copy(dev.get_device_ref())
        if (DRef is NULL):
            return -2 # Device could not be created
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
            ret = -2 # Device could not be created
        else:
            ret = self._init_queue_from_DPCTLSyclDeviceRef(DRef, props)
        return ret

    cdef int _init_queue_from_context_and_device(
        self, SyclContext ctxt, SyclDevice dev, int props
    ):
        """
        """
        cdef DPCTLSyclContextRef CRef = NULL
        cdef DPCTLSyclDeviceRef DRef = NULL
        cdef DPCTLSyclQueueRef QRef = NULL
        CRef = ctxt.get_context_ref()
        DRef = dev.get_device_ref()
        QRef = DPCTLQueue_Create(
            CRef,
            DRef,
            <error_handler_callback *>&default_async_error_handler,
            props
        )
        if (QRef is NULL):
            return -4
        self._device = dev
        self._context = ctxt
        self._queue_ref = QRef
        return 0 # normal return

    @staticmethod
    cdef SyclQueue _create(DPCTLSyclQueueRef qref):
        if qref is NULL:
            raise SyclQueueCreationError("Queue creation failed.")
        cdef _SyclQueue ret = _SyclQueue.__new__(_SyclQueue)
        ret._context = SyclContext._create(DPCTLQueue_GetContext(qref))
        ret._device = SyclDevice._create(DPCTLQueue_GetDevice(qref))
        ret._queue_ref = qref
        return SyclQueue(ret)

    @staticmethod
    cdef SyclQueue _create_from_context_and_device(
        SyclContext ctx, SyclDevice dev
    ):
        cdef _SyclQueue ret = _SyclQueue.__new__(_SyclQueue)
        cdef DPCTLSyclContextRef cref = ctx.get_context_ref()
        cdef DPCTLSyclDeviceRef dref = dev.get_device_ref()
        cdef DPCTLSyclQueueRef qref = DPCTLQueue_Create(cref, dref, NULL, 0)

        if qref is NULL:
            raise SyclQueueCreationError("Queue creation failed.")
        ret._queue_ref = qref
        ret._context = ctx
        ret._device = dev
        return SyclQueue(ret)

    cdef _raise_queue_submit_error(self, fname, errcode):
        e = SyclKernelSubmitError("Kernel submission to Sycl queue failed.")
        e.fname = fname
        e.code = errcode
        raise e

    cdef _raise_invalid_range_error(self, fname, ndims, errcode):
        e = SyclKernelInvalidRangeError(
            "Range with ", ndims, " not allowed. Range should have between "
            " one and three dimensions."
        )
        e.fname = fname
        e.code = errcode
        raise e

    cdef int _populate_args(
        self,
        list args,
        void **kargs,
        DPCTLKernelArgType *kargty
    ):
        cdef int ret = 0
        for idx, arg in enumerate(args):
            if isinstance(arg, ctypes.c_char):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._CHAR
            elif isinstance(arg, ctypes.c_int):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._INT
            elif isinstance(arg, ctypes.c_uint):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._UNSIGNED_INT
            elif isinstance(arg, ctypes.c_uint8):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._UNSIGNED_INT8
            elif isinstance(arg, ctypes.c_long):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._LONG
            elif isinstance(arg, ctypes.c_ulong):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._UNSIGNED_LONG
            elif isinstance(arg, ctypes.c_longlong):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._LONG_LONG
            elif isinstance(arg, ctypes.c_ulonglong):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._UNSIGNED_LONG_LONG
            elif isinstance(arg, ctypes.c_short):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._SHORT
            elif isinstance(arg, ctypes.c_size_t):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._SIZE_T
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

    cpdef cpp_bool equals(self, SyclQueue q):
        """ Returns true if the SyclQueue argument has the same _queue_ref
            as this SycleQueue.
        """
        return DPCTLQueue_AreEq(self._queue_ref, q.get_queue_ref())

    def get_sycl_backend (self):
        """ Returns the Sycl backend associated with the queue.
        """
        cdef DPCTLSyclBackendType BE = DPCTLQueue_GetBackend(self._queue_ref)
        if BE == _backend_type._OPENCL:
            return backend_type.opencl
        elif BE == _backend_type._LEVEL_ZERO:
            return backend_type.level_zero
        elif BE == _backend_type._HOST:
            return backend_type.host
        elif BE == _backend_type._CUDA:
            return backend_type.cuda
        else:
            raise ValueError("Unknown backend type.")

    @property
    def sycl_context(self):
        return self._context

    @property
    def sycl_device(self):
        return self._device

    cpdef SyclContext get_sycl_context(self):
        return self._context

    cpdef SyclDevice get_sycl_device(self):
        return self._device

    cdef DPCTLSyclQueueRef get_queue_ref(self):
        return self._queue_ref

    def addressof_ref(self):
        """
        Returns the address of the C API DPCTLSyclQueueRef pointer as a size_t.

        Returns:
            The address of the DPCTLSyclQueueRef object used to create this
            SyclQueue cast to a size_t.
        """
        return int(<size_t>self._queue_ref)

    cpdef SyclEvent submit(
        self,
        SyclKernel kernel,
        list args,
        list gS,
        list lS = None,
        list dEvents = None
    ):
        cdef void **kargs = NULL
        cdef DPCTLKernelArgType *kargty = NULL
        cdef DPCTLSyclEventRef *depEvents = NULL
        cdef DPCTLSyclEventRef Eref = NULL
        cdef int ret
        cdef size_t gRange[3]
        cdef size_t lRange[3]
        cdef size_t nGS = len(gS)
        cdef size_t nLS = len(lS) if lS is not None else 0
        cdef size_t nDE = len(dEvents) if dEvents is not None else 0

        # Allocate the arrays to be sent to DPCTLQueue_Submit
        kargs = <void**>malloc(len(args) * sizeof(void*))
        if not kargs:
            raise MemoryError()
        kargty = <DPCTLKernelArgType*>malloc(len(args)*sizeof(DPCTLKernelArgType))
        if not kargty:
            free(kargs)
            raise MemoryError()
        # Create the array of dependent events if any
        if dEvents is not None and nDE > 0:
            depEvents = <DPCTLSyclEventRef*>malloc(nDE*sizeof(DPCTLSyclEventRef))
            if not depEvents:
                free(kargs)
                free(kargty)
                raise MemoryError()
            else:
                for idx, de in enumerate(dEvents):
                    depEvents[idx] = (<SyclEvent>de).get_event_ref()

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
                self._raise_invalid_range_error("SyclQueue.submit", nGS, -1)
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
            ret = self._populate_range (gRange, gS, nGS)
            if ret == -1:
                free(kargs)
                free(kargty)
                free(depEvents)
                self._raise_invalid_range_error("SyclQueue.submit", nGS, -1)
            ret = self._populate_range (lRange, lS, nLS)
            if ret == -1:
                free(kargs)
                free(kargty)
                free(depEvents)
                self._raise_invalid_range_error("SyclQueue.submit", nLS, -1)
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
            self._raise_queue_submit_error("DPCTLQueue_Submit", -1)

        return SyclEvent._create(Eref, args)

    cpdef void wait(self):
        DPCTLQueue_Wait(self._queue_ref)

    cpdef memcpy(self, dest, src, size_t count):
        cdef void *c_dest
        cdef void *c_src

        if isinstance(dest, _Memory):
            c_dest = <void*>(<_Memory>dest).memory_ptr
        else:
            raise TypeError("Parameter `dest` should have type _Memory.")

        if isinstance(src, _Memory):
            c_src = <void*>(<_Memory>src).memory_ptr
        else:
            raise TypeError("Parameter `src` should have type _Memory.")

        DPCTLQueue_Memcpy(self._queue_ref, c_dest, c_src, count)

    cpdef prefetch(self, mem, size_t count=0):
       cdef void *ptr

       if isinstance(mem, _Memory):
           ptr = <void*>(<_Memory>mem).memory_ptr
       else:
           raise TypeError("Parameter `mem` should have type _Memory")

       if (count <=0 or count > self.nbytes):
           count = self.nbytes

       DPCTLQueue_Prefetch(self._queue_ref, ptr, count)

    cpdef mem_advise(self, mem, size_t count, int advice):
       cdef void *ptr

       if isinstance(mem, _Memory):
           ptr = <void*>(<_Memory>mem).memory_ptr
       else:
           raise TypeError("Parameter `mem` should have type _Memory")

       if (count <=0 or count > self.nbytes):
           count = self.nbytes

       DPCTLQueue_MemAdvise(self._queue_ref, ptr, count, advice)

    @property
    def __name__(self):
        return "SyclQueue"

    def __repr__(self):
        return "<dpctl." + self.__name__ + " at {}>".format(hex(id(self)))
