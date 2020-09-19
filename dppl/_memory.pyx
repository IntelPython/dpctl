import dppl
from dppl.backend cimport *
from ._sycl_core cimport SyclContext, SyclQueue

from cpython cimport Py_buffer


cdef class Memory:
    cdef DPPLSyclUSMRef memory_ptr
    cdef Py_ssize_t nbytes
    cdef SyclContext context

    cdef _cinit(self, Py_ssize_t nbytes, ptr_type, SyclQueue queue):
        cdef DPPLSyclUSMRef p

        self.memory_ptr = NULL
        self.nbytes = 0
        self.context = None

        if (nbytes > 0):
            if queue is None:
                queue = dppl.get_current_queue()

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
                self.context = queue.get_sycl_context()
            else:
                raise RuntimeError("Null memory pointer returned")
        else:
            raise ValueError("Non-positive number of bytes found.")

    def __dealloc__(self):
        if (self.memory_ptr):
            DPPLfree_with_context(self.memory_ptr,
                                  self.context.get_context_ref())
        self.memory_ptr = NULL
        self.nbytes = 0
        self.context = None

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
            return self.context

    def __repr__(self):
        return "<Intel(R) USM allocated memory block of {} bytes at {}>" \
            .format(self.nbytes, hex(<object>(<Py_ssize_t>self.memory_ptr)))

    def _usm_type(self):
        cdef const char* kind
        kind = DPPLUSM_GetPointerType(self.memory_ptr,
                                      self.context.get_context_ref())
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
