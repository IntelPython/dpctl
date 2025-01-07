#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2025 Intel Corporation
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

"""This file implements Python buffer protocol using Sycl USM shared and host
allocators. The USM device allocator is also exposed through this module for
use in other Python modules.
"""


import dpctl

from cpython cimport Py_buffer, pycapsule
from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_FromStringAndSize

from dpctl._backend cimport (  # noqa: E211
    DPCTLaligned_alloc_device,
    DPCTLaligned_alloc_host,
    DPCTLaligned_alloc_shared,
    DPCTLContext_AreEq,
    DPCTLContext_Delete,
    DPCTLDevice_Copy,
    DPCTLEvent_Delete,
    DPCTLEvent_Wait,
    DPCTLfree_with_queue,
    DPCTLmalloc_device,
    DPCTLmalloc_host,
    DPCTLmalloc_shared,
    DPCTLQueue_Copy,
    DPCTLQueue_Create,
    DPCTLQueue_Delete,
    DPCTLQueue_GetContext,
    DPCTLQueue_Memcpy,
    DPCTLQueue_MemcpyWithEvents,
    DPCTLQueue_Memset,
    DPCTLSyclContextRef,
    DPCTLSyclDeviceRef,
    DPCTLSyclEventRef,
    DPCTLSyclQueueRef,
    DPCTLSyclUSMRef,
    DPCTLUSM_GetPointerDevice,
    DPCTLUSM_GetPointerType,
    _usm_type,
)

from .._sycl_context cimport SyclContext
from .._sycl_device cimport SyclDevice
from .._sycl_queue cimport SyclQueue
from .._sycl_queue_manager cimport get_device_cached_queue

import collections
import numbers

import numpy as np

__all__ = [
    "MemoryUSMShared",
    "MemoryUSMHost",
    "MemoryUSMDevice",
    "USMAllocationError",
]

include "_sycl_usm_array_interface_utils.pxi"

cdef extern from "_opaque_smart_ptr.hpp":
    void * OpaqueSmartPtr_Make(void *, DPCTLSyclQueueRef) nogil
    void * OpaqueSmartPtr_Copy(void *) nogil
    void OpaqueSmartPtr_Delete(void *) nogil
    void * OpaqueSmartPtr_Get(void *) nogil

class USMAllocationError(Exception):
    """
    An exception raised when Universal Shared Memory (USM) allocation
    call returns a null pointer, signaling a failure to perform the allocation.
    Some common reasons for allocation failure are:

        * insufficient free memory to perform the allocation request
        * allocation size exceeds the maximum supported by targeted backend
    """
    pass


cdef void copy_via_host(void *dest_ptr, SyclQueue dest_queue,
                        void *src_ptr, SyclQueue src_queue, size_t nbytes):
    """
    Copies `nbytes` bytes from `src_ptr` USM memory to
    `dest_ptr` USM memory using host as the intermediary.

    This is useful when `src_ptr` and `dest_ptr` are bound to incompatible
    SYCL contexts.
    """
    # could also have used bytearray(nbytes)
    cdef unsigned char[::1] host_buf = np.empty((nbytes,), dtype="|u1")
    cdef DPCTLSyclEventRef E1Ref = NULL
    cdef DPCTLSyclEventRef *depEvs = [NULL,]
    cdef DPCTLSyclEventRef E2Ref = NULL

    E1Ref = DPCTLQueue_Memcpy(
        src_queue.get_queue_ref(),
        <void *>&host_buf[0],
        src_ptr,
        nbytes
    )
    depEvs[0] = E1Ref
    E2Ref = DPCTLQueue_MemcpyWithEvents(
        dest_queue.get_queue_ref(),
        dest_ptr,
        <void *>&host_buf[0],
        nbytes,
        depEvs,
        1
    )
    DPCTLEvent_Delete(E1Ref)
    with nogil: DPCTLEvent_Wait(E2Ref)
    DPCTLEvent_Delete(E2Ref)


def _to_memory(unsigned char[::1] b, str usm_kind):
    """
    Constructs Memory of the same size as the argument
    and copies data into it"""
    cdef _Memory res

    if (usm_kind == "shared"):
        res = MemoryUSMShared(len(b))
    elif (usm_kind == "device"):
        res = MemoryUSMDevice(len(b))
    elif (usm_kind == "host"):
        res = MemoryUSMHost(len(b))
    else:
        raise ValueError(
            "Unrecognized usm_kind={} stored in the "
            "pickle".format(usm_kind)
        )
    res.copy_from_host(b)

    return res


cdef class _Memory:
    """ Internal class implementing methods common to
        MemoryUSMShared, MemoryUSMDevice, MemoryUSMHost
    """
    cdef _cinit_empty(self):
        self._memory_ptr = NULL
        self._opaque_ptr = NULL
        self.nbytes = 0
        self.queue = None
        self.refobj = None

    cdef _cinit_alloc(self, Py_ssize_t alignment, Py_ssize_t nbytes,
                      bytes ptr_type, SyclQueue queue):
        cdef DPCTLSyclUSMRef p = NULL
        cdef DPCTLSyclQueueRef QRef = NULL

        self._cinit_empty()

        if (nbytes > 0):
            if queue is None:
                queue = get_device_cached_queue(dpctl.SyclDevice())

            QRef = queue.get_queue_ref()
            if (ptr_type == b"shared"):
                if alignment > 0:
                    with nogil: p = DPCTLaligned_alloc_shared(
                        alignment, nbytes, QRef
                    )
                else:
                    with nogil: p = DPCTLmalloc_shared(nbytes, QRef)
            elif (ptr_type == b"host"):
                if alignment > 0:
                    with nogil: p = DPCTLaligned_alloc_host(
                        alignment, nbytes, QRef
                    )
                else:
                    with nogil: p = DPCTLmalloc_host(nbytes, QRef)
            elif (ptr_type == b"device"):
                if (alignment > 0):
                    with nogil: p = DPCTLaligned_alloc_device(
                        alignment, nbytes, QRef
                    )
                else:
                    with nogil: p = DPCTLmalloc_device(nbytes, QRef)
            else:
                raise RuntimeError(
                    "Pointer type '{}' is not recognized".format(
                        ptr_type.decode("UTF-8")
                    )
                )

            if (p):
                self._memory_ptr = p
                self._opaque_ptr = OpaqueSmartPtr_Make(p, QRef)
                self.nbytes = nbytes
                self.queue = queue
            else:
                raise USMAllocationError(
                    "USM allocation failed"
                )
        else:
            raise ValueError(
                "Number of bytes of request allocation must be positive."
            )

    cdef _cinit_other(self, object other):
        cdef _Memory other_mem
        if isinstance(other, _Memory):
            other_mem = <_Memory> other
            self.nbytes = other_mem.nbytes
            self.queue = other_mem.queue
            if other_mem._opaque_ptr is NULL:
                self._memory_ptr = other_mem._memory_ptr
                self._opaque_ptr = NULL
                self.refobj = other.reference_obj
            else:
                self._memory_ptr = other_mem._memory_ptr
                self._opaque_ptr = OpaqueSmartPtr_Copy(other_mem._opaque_ptr)
                self.refobj = None
        elif hasattr(other, '__sycl_usm_array_interface__'):
            other_iface = other.__sycl_usm_array_interface__
            if isinstance(other_iface, dict):
                other_buf = _USMBufferData.from_sycl_usm_ary_iface(other_iface)
                self._opaque_ptr = NULL
                self._memory_ptr = <DPCTLSyclUSMRef>other_buf.p
                self.nbytes = other_buf.nbytes
                self.queue = other_buf.queue
                self.refobj = other
            else:
                raise ValueError(
                    "Argument {} does not correctly expose"
                    "`__sycl_usm_array_interface__`.".format(other)
                )
        else:
            raise ValueError(
                "Argument {} does not expose "
                "`__sycl_usm_array_interface__`.".format(other)
            )

    def __dealloc__(self):
        if not (self._opaque_ptr is NULL):
            OpaqueSmartPtr_Delete(self._opaque_ptr)
        self._cinit_empty()

    cdef DPCTLSyclUSMRef get_data_ptr(self):
        return self._memory_ptr

    cdef void* get_opaque_ptr(self):
        return self._opaque_ptr

    cdef _getbuffer(self, Py_buffer *buffer, int flags):
        # memory_ptr is Ref which is pointer to SYCL type. For USM it is void*.
        cdef SyclContext ctx = self._context
        cdef _usm_type UsmTy = DPCTLUSM_GetPointerType(
            self._memory_ptr, ctx.get_context_ref()
        )
        if UsmTy == _usm_type._USM_DEVICE:
            raise ValueError("USM Device memory is not host accessible")
        buffer.buf = <void *>self._memory_ptr
        buffer.format = 'B'                     # byte
        buffer.internal = NULL                  # see References
        buffer.itemsize = 1
        buffer.len = self.nbytes
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = &self.nbytes
        buffer.strides = &buffer.itemsize
        buffer.suboffsets = NULL                # for pointer arrays only

    property nbytes:
        """Extent of this USM buffer in bytes."""
        def __get__(self):
            return self.nbytes

    property size:
        """Extent of this USM buffer in bytes."""
        def __get__(self):
            return self.nbytes

    property _pointer:
        """
        USM pointer at the start of this buffer
        represented as Python integer.
        """
        def __get__(self):
            return <size_t>(self._memory_ptr)

    property _context:
        """:class:`dpctl.SyclContext` the USM pointer is bound to. """
        def __get__(self):
            return self.queue.get_sycl_context()

    property _queue:
        """
        :class:`dpctl.SyclQueue` with :class:`dpctl.SyclContext` the
        USM allocation is bound to and :class:`dpctl.SyclDevice` it was
        allocated on.
        """
        def __get__(self):
            return self.queue

    property reference_obj:
        """
        Reference to the Python object owning this USM buffer.
        """
        def __get__(self):
            return self.refobj

    property sycl_context:
        """:class:`dpctl.SyclContext` the USM pointer is bound to."""
        def __get__(self):
            return self.queue.get_sycl_context()

    property sycl_device:
        """:class:`dpctl.SyclDevice` the USM pointer is bound to."""
        def __get__(self):
            return self.queue.get_sycl_device()

    property sycl_queue:
        """
        :class:`dpctl.SyclQueue` with :class:`dpctl.SyclContext` the
        USM allocation is bound to and :class:`dpctl.SyclDevice` it was
        allocated on.
        """
        def __get__(self):
            return self.queue

    def __repr__(self):
        return (
            "<SYCL(TM) USM-{} allocation of {} bytes at {}>"
            .format(
                self.get_usm_type(),
                self.nbytes,
                hex(<object>(<size_t>self._memory_ptr))
            )
        )

    def __len__(self):
        return self.nbytes

    def __sizeof__(self):
        return self.nbytes

    def __bytes__(self):
        return self.tobytes()

    def __reduce__(self):
        return _to_memory, (self.copy_to_host(), self.get_usm_type())

    property __sycl_usm_array_interface__:
        """
        Dictionary encoding information about USM allocation.

        Contains the following fields:

            * ``"data"`` (Tuple[int, bool])
                unified address space pointer presented as Python integer
                and a Boolean value of 'writable' flag. If ``False`` the
                allocation is read-only. The return flag is always set to
                writable.
            * ``"shape"`` (Tuple[int])
                Extent of array in bytes. Shape is always 1-tuple for
                this object.
            * ``"strides"`` (Options[Tuple[int]])
                Strides describing array layout, or ``None`` if allocation is
                C-contiguous. Always ``None``.
            * ``"typestr"`` (str)
                Typestring encoding values of allocation. This field is always
                set to ``"|u1"`` representing unsigned bytes.
            * ``"version"`` (int)
                Always ``1``.
            * ``"syclobj"`` (:class:`dpctl.SyclQueue`)
                Queue associated with this class instance.

        """
        def __get__(self):
            cdef dict iface = {
                "data": (<size_t>(<void *>self._memory_ptr),
                         True),  # bool(self.writable)),
                "shape": (self.nbytes,),
                "strides": None,
                "typestr": "|u1",
                "version": 1,
                "syclobj": self.queue
            }
            return iface

    def get_usm_type(self, syclobj=None):
        """
        get_usm_type(syclobj=None)

        Returns the type of USM allocation using :class:`dpctl.SyclContext`
        carried by ``syclobj`` keyword argument. Value of ``None`` is understood
        to query against ``self.sycl_context`` - the context used to create the
        allocation.
        """
        cdef const char* kind
        cdef SyclContext ctx
        cdef SyclQueue q
        if syclobj is None:
            ctx = self._context
            return _Memory.get_pointer_type(
                self._memory_ptr, ctx
            ).decode("UTF-8")
        elif isinstance(syclobj, SyclContext):
            ctx = <SyclContext>(syclobj)
            return _Memory.get_pointer_type(
                self._memory_ptr, ctx
            ).decode("UTF-8")
        elif isinstance(syclobj, SyclQueue):
            q = <SyclQueue>(syclobj)
            ctx = q.get_sycl_context()
            return _Memory.get_pointer_type(
                self._memory_ptr, ctx
            ).decode("UTF-8")
        raise TypeError(
            "syclobj keyword can be either None, or an instance of "
            "SyclContext or SyclQueue"
        )

    def get_usm_type_enum(self, syclobj=None):
        """
        get_usm_type(syclobj=None)

        Returns the type of USM allocation using :class:`dpctl.SyclContext`
        carried by ``syclobj`` keyword argument. Value of ``None`` is understood
        to query against ``self.sycl_context`` - the context used to create the
        allocation.
        """
        cdef const char* kind
        cdef SyclContext ctx
        cdef SyclQueue q
        if syclobj is None:
            ctx = self._context
            return _Memory.get_pointer_type_enum(
                self._memory_ptr, ctx
            )
        elif isinstance(syclobj, SyclContext):
            ctx = <SyclContext>(syclobj)
            return _Memory.get_pointer_type_enum(
                self._memory_ptr, ctx
            )
        elif isinstance(syclobj, SyclQueue):
            q = <SyclQueue>(syclobj)
            ctx = q.get_sycl_context()
            return _Memory.get_pointer_type_enum(
                self._memory_ptr, ctx
            )
        raise TypeError(
            "syclobj keyword can be either None, or an instance of "
            "SyclContext or SyclQueue"
        )

    cpdef copy_to_host(self, obj=None):
        """
        Copy content of instance's memory into memory of ``obj``, or allocate
        NumPy array of ``obj`` is ``None``.
        """
        # Cython does the right thing here
        cdef unsigned char[::1] host_buf = obj
        cdef DPCTLSyclEventRef ERef = NULL

        if (host_buf is None):
            # Python object did not have buffer interface
            # allocate new memory
            obj = np.empty((self.nbytes,), dtype="|u1")
            host_buf = obj
        elif (<Py_ssize_t>len(host_buf) < self.nbytes):
            raise ValueError(
                "Destination object is too small to accommodate {} bytes"
                .format(self.nbytes)
            )
        # call kernel to copy from
        ERef = DPCTLQueue_Memcpy(
            self.queue.get_queue_ref(),
            <void *>&host_buf[0],      # destination
            <void *>self._memory_ptr,  # source
            <size_t>self.nbytes
        )
        with nogil: DPCTLEvent_Wait(ERef)
        DPCTLEvent_Delete(ERef)

        return obj

    cpdef copy_from_host(self, object obj):
        """
        Copy content of Python buffer provided by ``obj`` to instance memory.
        """
        cdef const unsigned char[::1] host_buf = obj
        cdef Py_ssize_t buf_len = len(host_buf)
        cdef DPCTLSyclEventRef ERef = NULL

        if (buf_len > self.nbytes):
            raise ValueError(
                "Source object is too large to be accommodated in {} bytes "
                "buffer".format(self.nbytes)
            )
        # call kernel to copy from
        ERef = DPCTLQueue_Memcpy(
            self.queue.get_queue_ref(),
            <void *>self._memory_ptr,  # destination
            <void *>&host_buf[0],      # source
            <size_t>buf_len
        )
        with nogil: DPCTLEvent_Wait(ERef)
        DPCTLEvent_Delete(ERef)

    cpdef copy_from_device(self, object sycl_usm_ary):
        """
        Copy SYCL memory underlying the argument object into
        the memory of the instance
        """
        cdef _USMBufferData src_buf
        cdef DPCTLSyclEventRef ERef = NULL
        cdef bint same_contexts = False
        cdef SyclQueue this_queue = None
        cdef SyclQueue src_queue = None

        if not hasattr(sycl_usm_ary, '__sycl_usm_array_interface__'):
            raise ValueError(
                "Object does not implement "
                "`__sycl_usm_array_interface__` protocol"
            )
        sycl_usm_ary_iface = sycl_usm_ary.__sycl_usm_array_interface__
        if isinstance(sycl_usm_ary_iface, dict):
            src_buf = _USMBufferData.from_sycl_usm_ary_iface(sycl_usm_ary_iface)

            if (src_buf.nbytes > self.nbytes):
                raise ValueError(
                    "Source object is too large to "
                    "be accommondated in {} bytes buffer".format(self.nbytes)
                )

            src_queue = src_buf.queue
            this_queue = self.queue
            same_contexts = DPCTLContext_AreEq(
                src_queue.get_sycl_context().get_context_ref(),
                this_queue.get_sycl_context().get_context_ref()
                )
            if (same_contexts):
                ERef = DPCTLQueue_Memcpy(
                    this_queue.get_queue_ref(),
                    <void *>self._memory_ptr,
                    <void *>src_buf.p,
                    <size_t>src_buf.nbytes
                )
                with nogil: DPCTLEvent_Wait(ERef)
                DPCTLEvent_Delete(ERef)
            else:
                copy_via_host(
                    <void *>self._memory_ptr, this_queue, # dest
                    <void *>src_buf.p, src_queue,         # src
                    <size_t>src_buf.nbytes
                )
        else:
            raise TypeError

    cpdef memset(self, unsigned short val = 0):
        """
        Populates this USM allocation with given value.
        """
        cdef DPCTLSyclEventRef ERef = NULL

        ERef = DPCTLQueue_Memset(
            self.queue.get_queue_ref(),
            <void *>self._memory_ptr,  # destination
            <int> val,
            self.nbytes)

        if ERef is not NULL:
            DPCTLEvent_Wait(ERef)
            DPCTLEvent_Delete(ERef)
            return
        else:
            raise RuntimeError(
                "Call to memset resulted in an error"
            )

    cpdef bytes tobytes(self):
        """
        Constructs bytes object populated with copy of USM memory.
        """
        cdef Py_ssize_t nb = self.nbytes
        cdef bytes b = PyBytes_FromStringAndSize(NULL, nb)
        # convert bytes to memory view
        cdef unsigned char* ptr = <unsigned char*>PyBytes_AS_STRING(b)
        # string is null terminated
        cdef unsigned char[::1] mv = (<unsigned char[:(nb + 1):1]>ptr)[:nb]
        self.copy_to_host(mv)  # output is discarded
        return b

    @staticmethod
    cdef SyclDevice get_pointer_device(DPCTLSyclUSMRef p, SyclContext ctx):
        """
        Returns sycl device used to allocate given pointer ``p`` in
        given sycl context ``ctx``
        """
        cdef DPCTLSyclDeviceRef dref = DPCTLUSM_GetPointerDevice(
            p, ctx.get_context_ref()
        )
        cdef DPCTLSyclDeviceRef dref_copy = DPCTLDevice_Copy(dref)
        if (dref_copy is NULL):
            raise RuntimeError("Could not create a copy of sycl device")
        return SyclDevice._create(dref_copy)  # deletes the argument

    @staticmethod
    cdef bytes get_pointer_type(DPCTLSyclUSMRef p, SyclContext ctx):
        """
        get_pointer_type(p, ctx)

        Gives the SYCL(TM) USM pointer type, using ``sycl::get_pointer_type``,
        returning one of 4 possible strings: ``'shared'``, ``'host'``,
        ``'device'``, or ``'unknown'``.

        Args:
            p: DPCTLSyclUSMRef
                A pointer to test the type of.
            ctx: :class:`dpctl.SyclContext`
                Python object providing :class:`dpctl.SyclContext` against
                which to query for the pointer type.
        Returns:
            ``b'unknown'`` if the pointer does not represent USM allocation
            made using given context. Otherwise, returns ``b'shared'``,
            ``b'device'``, or ``b'host'`` type of the allocation.
        """
        cdef _usm_type usm_ty = DPCTLUSM_GetPointerType(
            p, ctx.get_context_ref()
        )
        if usm_ty == _usm_type._USM_DEVICE:
            return b'device'
        elif usm_ty == _usm_type._USM_SHARED:
            return b'shared'
        elif usm_ty == _usm_type._USM_HOST:
            return b'host'
        else:
            return b'unknown'

    @staticmethod
    cdef _usm_type get_pointer_type_enum(DPCTLSyclUSMRef p, SyclContext ctx):
        """
        get_pointer_type(p, ctx)

        Gives the SYCL(TM) USM pointer type, using ``sycl::get_pointer_type``,
        returning an enum value.

        Args:
            p: DPCTLSyclUSMRef
                A pointer to test the type of.
            ctx: :class:`dpctl.SyclContext`
                Python object providing :class:`dpctl.SyclContext` against
                which to query for the pointer type.
        Returns:
            An enum value corresponding to the type of allocation.
        """
        cdef _usm_type usm_ty = DPCTLUSM_GetPointerType(
            p, ctx.get_context_ref()
        )
        return usm_ty

    @staticmethod
    cdef object create_from_usm_pointer_size_qref(
        DPCTLSyclUSMRef USMRef, Py_ssize_t nbytes,
        DPCTLSyclQueueRef QRef, object memory_owner=None
    ):
        r"""
        Create appropriate ``MemoryUSM*`` object from pre-allocated
        USM memory bound to SYCL context in the reference SYCL queue.

        Memory will be freed by ``MemoryUSM*`` object for default
        value of ``memory_owner`` keyword. The non-default value should
        be an object whose dealloc slot frees the memory.

        The object may not be a no-op dummy Python object to
        delay freeing the memory until later times.
        """
        cdef _usm_type usm_ty
        cdef DPCTLSyclContextRef CRef = NULL
        cdef DPCTLSyclQueueRef QRef_copy = NULL
        cdef _Memory _mem
        cdef object mem_ty
        if nbytes <= 0:
            raise ValueError("Number of bytes must must be positive")
        if (QRef is NULL):
            raise TypeError("Argument DPCTLSyclQueueRef is NULL")
        CRef = DPCTLQueue_GetContext(QRef)
        if (CRef is NULL):
            raise ValueError("Could not retrieve context from QRef")
        usm_ty = DPCTLUSM_GetPointerType(USMRef, CRef)
        DPCTLContext_Delete(CRef)
        if usm_ty == _usm_type._USM_SHARED:
            mem_ty = MemoryUSMShared
        elif usm_ty == _usm_type._USM_DEVICE:
            mem_ty = MemoryUSMDevice
        elif usm_ty == _usm_type._USM_HOST:
            mem_ty = MemoryUSMHost
        else:
            raise ValueError(
                "Argument pointer is not bound to "
                "context in the given queue"
            )
        res = _Memory.__new__(_Memory)
        _mem = <_Memory> res
        _mem._cinit_empty()
        _mem.nbytes = nbytes
        QRef_copy = DPCTLQueue_Copy(QRef)
        if QRef_copy is NULL:
            raise ValueError("Referenced queue could not be copied.")
        try:
            # _create steals ownership of QRef_copy
            _mem.queue = SyclQueue._create(QRef_copy)
        except dpctl.SyclQueueCreationError as sqce:
            raise ValueError(
                "SyclQueue object could not be created from "
                "copy of referenced queue"
            ) from sqce
        if memory_owner is None:
            _mem._memory_ptr = USMRef
            # assume ownership of USM allocation via smart pointer
            _mem._opaque_ptr = OpaqueSmartPtr_Make(<void *>USMRef, QRef)
            _mem.refobj = None
        else:
            _mem._memory_ptr = USMRef
            _mem._opaque_ptr = NULL
            _mem.refobj = memory_owner
        _out = mem_ty(<object>_mem)
        return _out


cdef class MemoryUSMShared(_Memory):
    """
    MemoryUSMShared(nbytes, alignment=0, queue=None)

    An object representing allocation of SYCL USM-shared memory.

    Args:
        nbytes (int)
            number of bytes to allocated.
            Expected to be positive.
        alignment (Optional[int]):
            allocation alignment request. Non-positive
            ``alignment`` values are not ignored and
            the unaligned allocator ``sycl::malloc_device``
            is used to make allocation instead.
            Default: `0`.
        queue (Optional[:class:`dpctl.SyclQueue`]):
            SYCL queue associated with return allocation
            instance. Allocation is performed on the device
            associated with the queue, and is bound to
            SYCL context from the queue.
            If ``queue`` is ``None`` a cached
            default-constructed :class:`dpctl.SyclQueue` is
            used to allocate memory.
    """
    def __cinit__(self, other, *, Py_ssize_t alignment=0,
                  SyclQueue queue=None, int copy=False):
        if (isinstance(other, numbers.Integral)):
            self._cinit_alloc(alignment, <Py_ssize_t>other, b"shared", queue)
        else:
            self._cinit_other(other)
            if (self.get_usm_type() != "shared"):
                if copy:
                    self._cinit_alloc(0, <Py_ssize_t>self.nbytes,
                                      b"shared", queue)
                    self.copy_from_device(other)
                else:
                    raise ValueError(
                        "USM pointer in the argument {} is not a "
                        "USM shared pointer. "
                        "Zero-copy operation is not possible with "
                        "copy=False. "
                        "Either use copy=True, or use a constructor "
                        "appropriate for "
                        "type '{}'".format(other, self.get_usm_type())
                    )

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        self._getbuffer(buffer, flags)


cdef class MemoryUSMHost(_Memory):
    """
    MemoryUSMHost(nbytes, alignment=0, queue=None)

    An object representing allocation of SYCL USM-host memory.

    Args:
        nbytes (int)
            number of bytes to allocated.
            Expected to be positive.
        alignment (Optional[int]):
            allocation alignment request. Non-positive
            ``alignment`` values are not ignored and
            the unaligned allocator ``sycl::malloc_device``
            is used to make allocation instead.
            Default: `0`.
        queue (Optional[:class:`dpctl.SyclQueue`]):
            SYCL queue associated with return allocation
            instance. Allocation is made in host memory accessible
            to all device in the SYCL context from the queue.
            Allocation is bound to SYCL context from the queue.
            If ``queue`` is ``None`` a cached
            default-constructed :class:`dpctl.SyclQueue` is
            used to allocate memory.
    """
    def __cinit__(self, other, *, Py_ssize_t alignment=0,
                  SyclQueue queue=None, int copy=False):
        if (isinstance(other, numbers.Integral)):
            self._cinit_alloc(alignment, <Py_ssize_t>other, b"host", queue)
        else:
            self._cinit_other(other)
            if (self.get_usm_type() != "host"):
                if copy:
                    self._cinit_alloc(
                        0, <Py_ssize_t>self.nbytes, b"host", queue
                    )
                    self.copy_from_device(other)
                else:
                    raise ValueError(
                        "USM pointer in the argument {} is "
                        "not a USM host pointer. "
                        "Zero-copy operation is not possible with copy=False. "
                        "Either use copy=True, or use a constructor "
                        "appropriate for type '{}'".format(
                            other, self.get_usm_type()
                        )
                    )

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        self._getbuffer(buffer, flags)


cdef class MemoryUSMDevice(_Memory):
    """
    MemoryUSMDevice(nbytes, alignment=0, queue=None)

    Class representing allocation of SYCL USM-device memory.

    Args:
        nbytes (int)
            number of bytes to allocated.
            Expected to be positive.
        alignment (Optional[int]):
            allocation alignment request. Non-positive
            ``alignment`` values are not ignored and
            the unaligned allocator ``sycl::malloc_device``
            is used to make allocation instead.
            Default: `0`.
        queue (Optional[:class:`dpctl.SyclQueue`]):
            SYCL queue associated with return allocation
            instance. Allocation is performed on the device
            associated with the queue, and is bound to
            SYCL context from the queue.
            If ``queue`` is ``None`` a cached
            default-constructed :class:`dpctl.SyclQueue` is
            used to allocate memory.
    """
    def __cinit__(self, other, *, Py_ssize_t alignment=0,
                  SyclQueue queue=None, int copy=False):
        if (isinstance(other, numbers.Integral)):
            self._cinit_alloc(alignment, <Py_ssize_t>other, b"device", queue)
        else:
            self._cinit_other(other)
            if (self.get_usm_type() != "device"):
                if copy:
                    self._cinit_alloc(
                        0, <Py_ssize_t>self.nbytes, b"device", queue
                    )
                    self.copy_from_device(other)
                else:
                    raise ValueError(
                        "USM pointer in the argument {} is not "
                        "a USM device pointer. "
                        "Zero-copy operation is not possible with copy=False. "
                        "Either use copy=True, or use a constructor "
                        "appropriate for type '{}'".format(
                            other, self.get_usm_type()
                        )
                    )


def as_usm_memory(obj):
    """
    as_usm_memory(obj)

    Converts Python object with ``__sycl_usm_array_interface__`` property
    to one of :class:`.MemoryUSMShared`, :class:`.MemoryUSMDevice`, or
    :class:`.MemoryUSMHost` instances depending on the type of USM allocation
    they represent.

    Raises:
        ValueError
            When object does not expose the ``__sycl_usm_array_interface__``,
            or it is malformed
        TypeError
            When unexpected types of entries in the interface are encountered
        SyclQueueCreationError
            When a :class:`dpctl.SyclQueue` could not be created from the
            information given by the interface
    """
    cdef _Memory res = _Memory.__new__(_Memory)
    cdef str kind
    res._cinit_empty()
    res._cinit_other(obj)
    kind = res.get_usm_type()
    if kind == "shared":
        return MemoryUSMShared(res)
    elif kind == "device":
        return MemoryUSMDevice(res)
    elif kind == "host":
        return MemoryUSMHost(res)
    else:
        raise ValueError(
            "Could not determine the type "
            "USM allocation represented by argument {}".
            format(obj)
        )

cdef api void * Memory_GetOpaquePointer(_Memory obj):
    "Opaque pointer value"
    return obj.get_opaque_ptr()

cdef api DPCTLSyclUSMRef Memory_GetUsmPointer(_Memory obj):
    "Pointer of USM allocation"
    return obj.get_data_ptr()

cdef api DPCTLSyclContextRef Memory_GetContextRef(_Memory obj):
    "Context reference to which USM allocation is bound"
    return obj.queue._context.get_context_ref()

cdef api DPCTLSyclQueueRef Memory_GetQueueRef(_Memory obj):
    """Queue associated with this allocation, used
    for copying, population, etc."""
    return obj.queue.get_queue_ref()

cdef api size_t Memory_GetNumBytes(_Memory obj):
    "Size of the allocation in bytes."
    return <size_t>obj.nbytes

cdef api object Memory_Make(
    DPCTLSyclUSMRef ptr,
    size_t nbytes,
    DPCTLSyclQueueRef QRef,
    object owner
):
    "Create _Memory Python object from preallocated memory."
    return _Memory.create_from_usm_pointer_size_qref(
        ptr, nbytes, QRef, memory_owner=owner
    )
