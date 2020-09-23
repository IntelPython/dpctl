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
from dpctl.backend cimport *
from ._sycl_core cimport SyclContext, SyclQueue

from cpython cimport Py_buffer


cdef class Memory:
    cdef DPPLSyclUSMRef memory_ptr
    cdef Py_ssize_t nbytes
    cdef SyclQueue queue

    cdef _cinit(self, Py_ssize_t nbytes, ptr_type, SyclQueue queue):
        cdef DPPLSyclUSMRef p

        self.memory_ptr = NULL
        self.nbytes = 0
        self.queue = None

        if (nbytes > 0):
            if queue is None:
                queue = dpctl.get_current_queue()

            if (ptr_type == "shared"):
                p = DPPLmalloc_shared(nbytes, queue.get_queue_ref())
            elif (ptr_type == "host"):
                p = DPPLmalloc_host(nbytes, queue.get_queue_ref())
            elif (ptr_type == "device"):
                p = DPPLmalloc_device(nbytes, queue.get_queue_ref())
            else:
                raise RuntimeError("Pointer type is unknown: {}" \
                    .format(ptr_type))

            if (p):
                self.memory_ptr = p
                self.nbytes = nbytes
                self.queue = queue
            else:
                raise RuntimeError("Null memory pointer returned")
        else:
            raise ValueError("Non-positive number of bytes found.")

    def __dealloc__(self):
        if (self.memory_ptr):
            DPPLfree_with_queue(self.memory_ptr,
                                self.queue.get_queue_ref())
        self.memory_ptr = NULL
        self.nbytes = 0
        self.queue = None

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

    property _context:
        def __get__(self):
            return self.queue.get_sycl_context()

    property _queue:
        def __get__(self):
            return self.queue

    def __repr__(self):
        return "<Intel(R) USM allocated memory block of {} bytes at {}>" \
            .format(self.nbytes, hex(<object>(<Py_ssize_t>self.memory_ptr)))

    def _usm_type(self, syclobj=None):
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
                             "or an instance of SyclConext or SyclQueue")
        return kind.decode('UTF-8')


cdef class MemoryUSMShared(Memory):

    def __cinit__(self, Py_ssize_t nbytes, SyclQueue queue=None):
        self._cinit(nbytes, "shared", queue)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        self._getbuffer(buffer, flags)


cdef class MemoryUSMHost(Memory):

    def __cinit__(self, Py_ssize_t nbytes, SyclQueue queue=None):
        self._cinit(nbytes, "host", queue)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        self._getbuffer(buffer, flags)


cdef class MemoryUSMDevice(Memory):

    def __cinit__(self, Py_ssize_t nbytes, SyclQueue queue=None):
        self._cinit(nbytes, "device", queue)
