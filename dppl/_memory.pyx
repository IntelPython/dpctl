import dppl
from dppl.backend cimport *
from ._sycl_core cimport SyclContext, SyclQueue

from cython.operator cimport dereference as deref

from cpython cimport Py_buffer

cdef extern from "CL/sycl.hpp" namespace "cl::sycl::usm":
    cdef enum alloc:
       host 'cl::sycl::usm::alloc::host'
       device 'cl::sycl::usm::alloc::device'
       shared 'cl::sycl::usm::alloc::shared'
       unknown 'cl::sycl::usm::alloc::unknown'

cdef extern from "CL/sycl.hpp" namespace "cl::sycl":
    cdef cppclass context nogil

    cdef alloc get_pointer_type(void *, context&) nogil


cdef class Memory:
    cdef DPPLMemoryUSMSharedRef memory_ptr
    cdef Py_ssize_t nbytes
    cdef SyclQueue queue

    def __cinit__(self, Py_ssize_t nbytes):
        cdef SyclQueue q
        cdef DPPLMemoryUSMSharedRef p

        self.memory_ptr = NULL
        self.queue = None
        self.nbytes = 0

        if (nbytes > 0):
            q = dppl.get_current_queue()
            p = DPPLmalloc_shared(nbytes, q.get_queue_ref())
            if (p):
                self.memory_ptr = p
                self.nbytes = nbytes
                self.queue = q
            else:
                raise RuntimeError("Null memory pointer returned")
        else:
            raise ValueError("Non-positive number of bytes found.")

    def __dealloc__(self):
        if (self.memory_ptr):
            DPPLfree(self.memory_ptr, self.queue.get_queue_ref())
        self.memory_ptr = NULL
        self.nbytes = 0
        self.queue = None

    def __getbuffer__(self, Py_buffer *buffer, int flags):
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

    property pointer:
        def __get__(self):
            return <object>(<Py_ssize_t>self.memory_ptr)

    property nbytes:
        def __get__(self):
            return self.nbytes

    property _queue:
        def __get__(self):
            return self.queue

    def __repr__(self):
        return "<Intel(R) USM allocated memory block of {} bytes at {}>".format(self.nbytes, hex(<object>(<Py_ssize_t>self.memory_ptr)))

    def _usm_type(self):
        cdef SyclContext ctxt
        cdef alloc ptr_type

        ctxt = self.queue.get_sycl_context()
        ptr_type = get_pointer_type(self.memory_ptr, deref(<context*>ctxt.get_context_ref()))
        if (ptr_type == alloc.shared):
            return "shared"
        elif (ptr_type == alloc.host):
            return "host"
        elif (ptr_type == alloc.device):
            return "device"
        else:
            return "unknown"
