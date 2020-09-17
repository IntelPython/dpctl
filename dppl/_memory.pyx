import dppl
cimport dppl._sycl_core

from cython.operator cimport dereference as deref

from cpython cimport Py_buffer

cdef extern from "CL/sycl.hpp" namespace "cl::sycl::usm":
    cdef enum alloc:
       host 'cl::sycl::usm::alloc::host'
       device 'cl::sycl::usm::alloc::device'
       shared 'cl::sycl::usm::alloc::shared'
       unknown 'cl::sycl::usm::alloc::unknown'

cdef extern from "CL/sycl.hpp" namespace "cl::sycl":
    cdef cppclass context nogil:
       pass

    cdef cppclass queue nogil:
       context get_context() nogil
       pass

    cdef void* malloc_shared(Py_ssize_t, queue&) nogil
    cdef void free(void *, queue&) nogil
    cdef alloc get_pointer_type(void *, context&) nogil


cdef class SyclQueue:
    cdef dppl._sycl_core.SyclQueue queue_cap
    cdef queue q

    def __cinit__(self):
        cdef void* q_ptr
        self.queue_cap = dppl.get_current_queue()
        q_ptr = self.queue_cap.get_queue_ref()
        if (q_ptr):
            self.q = deref(<queue *>q_ptr)
        else:
            raise ValueError("NULL pointer returned by the Capsule")

    def get_pointer_type(self, Py_ssize_t p):
        cdef context ctx = self.q.get_context()
        cdef void * p_ptr = <void *> p

        ptr_type = get_pointer_type(p_ptr, ctx)
        if (ptr_type == alloc.shared):
            return "shared"
        elif (ptr_type == alloc.host):
            return "host"
        elif (ptr_type == alloc.device):
            return "device"
        else:
            return "unknown"

    property get_capsule:
        def __get__(self):
            return self.queue_cap

    cdef queue get_queue(self):
        return self.q


cdef class Memory:
    cdef void* _ptr
    cdef Py_ssize_t nbytes
    cdef dppl._sycl_core.SyclQueue queue_cap

    def __cinit__(self, Py_ssize_t nbytes):
        cdef dppl._sycl_core.SyclQueue q_cap
        cdef void* queue_ptr
        cdef void* p

        self._ptr = NULL
        self.queue_cap = None
        self.nbytes = 0

        if (nbytes > 0):
            q_cap = dppl.get_current_queue()
            queue_ptr = q_cap.get_queue_ref()
            p = malloc_shared(nbytes, deref(<queue *>queue_ptr))
            if (p):
                self._ptr = p
                self.nbytes = nbytes
                self.queue_cap = q_cap
            else:
                raise RuntimeError("Null memory pointer returned")
        else:
            raise ValueError("Non-positive number of bytes found.")

    def __dealloc__(self):
        cdef void* queue_ptr

        if (self._ptr):
            queue_ptr = self.queue_cap.get_queue_ref()
            free(self._ptr, deref(<queue *>queue_ptr))
        self._ptr = NULL
        self.nbytes = 0
        self.queue_cap = None

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        buffer.buf = <char *>self._ptr
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
            return <object>(<Py_ssize_t>self._ptr)

    property nbytes:
        def __get__(self):
            return self.nbytes

    property _queue:
        def __get__(self):
            return self.queue_cap

    def __repr__(self):
        return "<Intel(R) USM allocated memory block of {} bytes at {}>".format(self.nbytes, hex(<object>(<Py_ssize_t>self._ptr)))

    def _usm_type(self, qcaps=None):
        cdef void *q_ptr
        cdef alloc ptr_type
        cdef dppl._sycl_core.SyclQueue _cap

        _cap = qcaps if (qcaps) else self.queue_cap
        q_ptr = _cap.get_queue_ref()
        ptr_type = get_pointer_type(self._ptr, deref(<queue*>q_ptr).get_context())
        if (ptr_type == alloc.shared):
            return "shared"
        elif (ptr_type == alloc.host):
            return "host"
        elif (ptr_type == alloc.device):
            return "device"
        else:
            return "unknown"

#    cdef void* _ptr
#    cdef Py_ssize_t nbytes
#    cdef object queue_cap

    @staticmethod
    cdef Memory create(void *p, Py_ssize_t nbytes, object queue_cap):
        cdef Memory ret = Memory.__new__()
        ret._ptr = p
        ret.nbytes = nbytes
        ret.q_cap = queue_cap
        return ret
