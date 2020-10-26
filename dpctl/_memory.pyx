##===--------------- _memory.pyx - dpctl module --------*- Cython -*-------===##
##
##                      Data Parallel Control (dpCtl)
##
## Copyright 2020 Intel Corporation
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##
##===----------------------------------------------------------------------===##
##
## \file
## This file implements Python buffer protocol using Sycl USM shared and host
## allocators. The USM device allocator is also exposed through this module for
## use in other Python modules.
##
##===----------------------------------------------------------------------===##

# distutils: language = c++
# cython: language_level=3

import dpctl
from dpctl._backend cimport *
from ._sycl_core cimport SyclContext, SyclQueue, SyclDevice
from ._sycl_core cimport get_current_queue

from cpython cimport Py_buffer
from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_FromStringAndSize

import numpy as np

cdef _throw_sycl_usm_ary_iface():
    raise ValueError("__sycl_usm_array_interface__ is malformed")


cdef void copy_via_host(void *dest_ptr, SyclQueue dest_queue,
                   void *src_ptr, SyclQueue src_queue, size_t nbytes):
   """
   Copies `nbytes` bytes from `src_ptr` USM memory to 
   `dest_ptr` USM memory using host as the intemediary.

   This is useful when `src_ptr` and `dest_ptr` are bound to incompatible
   SYCL contexts.
   """
   # could also have used bytearray(nbytes)
   cdef unsigned char[::1] host_buf = np.empty((nbytes,), dtype="|u1")
   
   DPPLQueue_Memcpy(
       src_queue.get_queue_ref(),
       <void *>&host_buf[0],
       src_ptr,
       nbytes
   )

   DPPLQueue_Memcpy(
       dest_queue.get_queue_ref(),
       dest_ptr,
       <void *>&host_buf[0],
       nbytes
   )


cdef class _BufferData:
    """
    Internal data struct populated from parsing 
    `__sycl_usm_array_interface__` dictionary
    """
    cdef DPPLSyclUSMRef p
    cdef int writeable
    cdef object dt
    cdef Py_ssize_t itemsize
    cdef Py_ssize_t nbytes
    cdef SyclQueue queue

    @staticmethod
    cdef _BufferData from_sycl_usm_ary_iface(dict ary_iface):
        cdef object ary_data_tuple = ary_iface.get('data', None)
        cdef object ary_typestr = ary_iface.get('typestr', None)
        cdef object ary_shape = ary_iface.get('shape', None)
        cdef object ary_strides = ary_iface.get('strides', None)
        cdef object ary_syclobj = ary_iface.get('syclobj', None)
        cdef Py_ssize_t ary_offset = ary_iface.get('offset', 0)
        cdef int ary_version = ary_iface.get('version', 0)
        cdef object dt
        cdef _BufferData buf
        cdef Py_ssize_t arr_data_ptr
        cdef SyclDevice dev

        if ary_version != 1:
            _throw_sycl_usm_ary_iface()
        if not ary_data_tuple or len(ary_data_tuple) != 2:
            _throw_sycl_usm_ary_iface()
        if not ary_shape or len(ary_shape) != 1 or ary_shape[0] < 1:
            raise ValueError
        try:
            dt = np.dtype(ary_typestr)
        except TypeError:
            _throw_sycl_usm_ary_iface()
        if ary_strides and len(ary_strides) != dt.itemsize:
            raise ValueError("Must be contiguous")

        if not ary_syclobj or not isinstance(ary_syclobj,
                                             (dpctl.SyclQueue, dpctl.SyclContext)):
            _throw_sycl_usm_ary_iface()

        buf = _BufferData.__new__(_BufferData)
        arr_data_ptr = <Py_ssize_t>ary_data_tuple[0]
        buf.p = <DPPLSyclUSMRef>(<void*>arr_data_ptr)
        buf.writeable = 1 if ary_data_tuple[1] else 0
        buf.itemsize = <Py_ssize_t>(dt.itemsize)
        buf.nbytes = (<Py_ssize_t>ary_shape[0]) * buf.itemsize

        if isinstance(ary_syclobj, dpctl.SyclQueue):
            buf.queue = <SyclQueue>ary_syclobj
        else:
            # FIXME: need a way to construct a queue from
            # context and device, which can be obtaine from the
            # pointer and the context.
            # 
            # cdef SyclQueue new_queue = SyclQueue._create_from_dev_context(dev, <SyclContext> ary_syclobj)
            # buf.queue = new_queue
            dev = Memory.get_pointer_device(buf.p, <SyclContext> ary_syclobj)
            buf.queue = get_current_queue()

        return buf


def _to_memory(unsigned char [::1] b, str usm_kind):
    """
    Constructs Memory of the same size as the argument 
    and copies data into it"""
    cdef Memory res

    if (usm_kind == "shared"):
        res = MemoryUSMShared(len(b))
    elif (usm_kind == "device"):
        res = MemoryUSMDevice(len(b))
    elif (usm_kind == "host"):
        res = MemoryUSMHost(len(b))
    else:
        raise ValueError(
            "Unrecognized usm_kind={} stored in the "
            "pickle".format(usm_kind))
    res.copy_from_host(b)
    
    return res


cdef class Memory:
    cdef _cinit_empty(self):
        self.memory_ptr = NULL
        self.nbytes = 0
        self.queue = None
        self.refobj = None        

    cdef _cinit_alloc(self, Py_ssize_t alignment, Py_ssize_t nbytes,
                      bytes ptr_type, SyclQueue queue):
        cdef DPPLSyclUSMRef p

        self._cinit_empty()

        if (nbytes > 0):
            if queue is None:
                queue = get_current_queue()

            if (ptr_type == b"shared"):
                if alignment > 0:
                    p = DPPLaligned_alloc_shared(alignment, nbytes,
                                                 queue.get_queue_ref())
                else:
                    p = DPPLmalloc_shared(nbytes, queue.get_queue_ref())
            elif (ptr_type == b"host"):
                if alignment > 0:
                    p = DPPLaligned_alloc_host(alignment, nbytes,
                                               queue.get_queue_ref())
                else:
                    p = DPPLmalloc_host(nbytes, queue.get_queue_ref())
            elif (ptr_type == b"device"):
                if (alignment > 0):
                    p = DPPLaligned_alloc_device(alignment, nbytes,
                                                  queue.get_queue_ref())
                else:
                    p = DPPLmalloc_device(nbytes, queue.get_queue_ref())
            else:
                raise RuntimeError("Pointer type is unknown: {}" \
                    .format(ptr_type.decode("UTF-8")))

            if (p):
                self.memory_ptr = p
                self.nbytes = nbytes
                self.queue = queue
            else:
                raise RuntimeError("Null memory pointer returned")
        else:
            raise ValueError("Non-positive number of bytes found.")

    cdef _cinit_other(self, object other):
        cdef Memory other_mem
        if isinstance(other, Memory):
            other_mem = <Memory> other
            self.memory_ptr = other_mem.memory_ptr
            self.nbytes = other_mem.nbytes
            self.queue = other_mem.queue
            if other_mem.refobj is None:
                self.refobj = other
            else:
                self.refobj = other_mem.refobj
        elif hasattr(other, '__sycl_usm_array_interface__'):
            other_iface = other.__sycl_usm_array_interface__
            if isinstance(other_iface, dict):
                other_buf = _BufferData.from_sycl_usm_ary_iface(other_iface)
                self.memory_ptr = other_buf.p
                self.nbytes = other_buf.nbytes
                self.queue = other_buf.queue
                # self.writeable = other_buf.writeable
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
        if (self.refobj is None and self.memory_ptr):
            DPPLfree_with_queue(self.memory_ptr,
                                self.queue.get_queue_ref())
        self._cinit_empty()

    cdef _getbuffer(self, Py_buffer *buffer, int flags):
        # memory_ptr is Ref which is pointer to SYCL type. For USM it is void*.
        buffer.buf = <char *>self.memory_ptr
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
        def __get__(self):
            return self.nbytes

    property size:
        def __get__(self):
            return self.nbytes

    property _pointer:
        def __get__(self):
            return <size_t>(self.memory_ptr)

    property _context:
        def __get__(self):
            return self.queue.get_sycl_context()

    property _queue:
        def __get__(self):
            return self.queue

    property reference_obj:
        def __get__(self):
            return self.refobj

    def __repr__(self):
        return "<Intel(R) USM allocated memory block of {} bytes at {}>" \
            .format(self.nbytes, hex(<object>(<Py_ssize_t>self.memory_ptr)))

    def __len__(self):
        return self.nbytes

    def __sizeof__(self):
        return self.nbytes

    def __bytes__(self):
        return self.tobytes()

    def __reduce__(self):
        return _to_memory, (self.copy_to_host(), self.get_usm_type())
    
    property __sycl_usm_array_interface__:
        def __get__ (self):
            cdef dict iface = {
                "data": (<Py_ssize_t>(<void *>self.memory_ptr),
                         True), # bool(self.writeable)),
                "shape": (self.nbytes,),
                "strides": None,
                "typestr": "|u1",
                "version": 1,
                "syclobj": self.queue
            }
            return iface

    def get_usm_type(self, syclobj=None):
        cdef const char* kind
        cdef SyclContext ctx
        cdef SyclQueue q
        if syclobj is None:
            ctx = self._context
            kind = DPPLUSM_GetPointerType(self.memory_ptr,
                                          ctx.get_context_ref())
        elif isinstance(syclobj, SyclContext):
            ctx = <SyclContext>(syclobj)
            kind = DPPLUSM_GetPointerType(self.memory_ptr,
                                          ctx.get_context_ref())
        elif isinstance(syclobj, SyclQueue):
            q = <SyclQueue>(syclobj)
            ctx = q.get_sycl_context()
            kind = DPPLUSM_GetPointerType(self.memory_ptr,
                                          ctx.get_context_ref())
        else:
            raise ValueError("syclobj keyword can be either None, "
                             "or an instance of SyclContext or SyclQueue")
        return kind.decode('UTF-8')

    cpdef copy_to_host (self, obj=None):
        """Copy content of instance's memory into memory of 
        `obj`, or allocate NumPy array of obj is None"""
        # Cython does the right thing here
        cdef unsigned char[::1] host_buf = obj

        if (host_buf is None):
            # Python object did not have buffer interface
            # allocate new memory
            obj = np.empty((self.nbytes,), dtype="|u1")
            host_buf = obj
        elif (<Py_ssize_t>len(host_buf) < self.nbytes):
            raise ValueError("Destination object is too small to "
                             "accommodate {} bytes".format(self.nbytes))
        # call kernel to copy from
        DPPLQueue_Memcpy(
            self.queue.get_queue_ref(),
            <void *>&host_buf[0],     # destination
            <void *>self.memory_ptr,  # source
            <size_t>self.nbytes
        )

        return obj

    cpdef copy_from_host (self, object obj):
        """Copy contant of Python buffer provided by `obj` to instance memory."""
        cdef const unsigned char[::1] host_buf = obj
        cdef Py_ssize_t buf_len = len(host_buf)

        if (buf_len > self.nbytes):
            raise ValueError("Source object is too large to be "
                             "accommodated in {} bytes buffer".format(self.nbytes))
        # call kernel to copy from
        DPPLQueue_Memcpy(
            self.queue.get_queue_ref(),
            <void *>self.memory_ptr,  # destination
            <void *>&host_buf[0],     # source
            <size_t>buf_len
        )

    cpdef copy_from_device (self, object sycl_usm_ary):
        """Copy SYCL memory underlying the argument object into 
        the memory of the instance"""
        cdef _BufferData src_buf
        cdef const char* kind
    
        if not hasattr(sycl_usm_ary, '__sycl_usm_array_interface__'):
            raise ValueError("Object does not implement "
                             "`__sycl_usm_array_interface__` protocol")
        sycl_usm_ary_iface = sycl_usm_ary.__sycl_usm_array_interface__
        if isinstance(sycl_usm_ary_iface, dict):
            src_buf = _BufferData.from_sycl_usm_ary_iface(sycl_usm_ary_iface)

            if (src_buf.nbytes > self.nbytes):
                raise ValueError("Source object is too large to "
                                 "be accommondated in {} bytes buffer".format(self.nbytes))
            kind = DPPLUSM_GetPointerType(
                src_buf.p, self.queue.get_sycl_context().get_context_ref())
            if (kind == b'unknown'):
                copy_via_host(
                    <void *>self.memory_ptr, self.queue,  # dest
                    <void *>src_buf.p, src_buf.queue,     # src
                    <size_t>src_buf.nbytes
                )
            else:
                DPPLQueue_Memcpy(
                    self.queue.get_queue_ref(),
                    <void *>self.memory_ptr,
                    <void *>src_buf.p,
                    <size_t>src_buf.nbytes
                )
        else:
            raise TypeError
    
    cpdef bytes tobytes (self):
        """"""
        cdef Py_ssize_t nb = self.nbytes
        cdef bytes b = PyBytes_FromStringAndSize(NULL, nb)
        # convert bytes to memory view
        cdef unsigned char* ptr = <unsigned char*>PyBytes_AS_STRING(b)
        # string is null terminated
        cdef unsigned char[::1] mv = (<unsigned char[:(nb + 1):1]>ptr)[:nb]
        self.copy_to_host(mv) # output is discarded
        return b

    @staticmethod
    cdef SyclDevice get_pointer_device(DPPLSyclUSMRef p, SyclContext ctx):
        cdef DPPLSyclDeviceRef dref = DPPLUSM_GetPointerDevice(p, ctx.get_context_ref())

        return SyclDevice._create(dref)


cdef class MemoryUSMShared(Memory):
    """
    MemoryUSMShared(nbytes, alignment=0, queue=None) allocates nbytes of USM shared memory.

    Non-positive alignments are not used (malloc_shared is used instead).
    The queue=None the current `dpctl.get_current_queue()` is used to allocate memory.

    MemoryUSMShared(usm_obj) constructor create instance from `usm_obj` expected to 
    implement `__sycl_usm_array_interface__` protocol and exposing a contiguous block of 
    USM memory.
    """
    def __cinit__(self, other, *, Py_ssize_t alignment=0, SyclQueue queue=None):
        if (isinstance(other, int)):
            self._cinit_alloc(alignment, <Py_ssize_t>other, b"shared", queue)
        else:
            self._cinit_other(other)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        self._getbuffer(buffer, flags)


cdef class MemoryUSMHost(Memory):

    def __cinit__(self, other, *, Py_ssize_t alignment=0, SyclQueue queue=None):
        if (isinstance(other, int)):
            self._cinit_alloc(alignment, <Py_ssize_t>other, b"host", queue)
        else:
            self._cinit_other(other)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        self._getbuffer(buffer, flags)


cdef class MemoryUSMDevice(Memory):

    def __cinit__(self, other, *, Py_ssize_t alignment=0, SyclQueue queue=None):
        if (isinstance(other, int)):
            self._cinit_alloc(alignment, <Py_ssize_t>other, b"device", queue)
        else:
            self._cinit_other(other)
