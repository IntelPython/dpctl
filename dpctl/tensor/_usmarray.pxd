# distutils: language = c++
# cython: language_level=3

cimport dpctl


cdef public api int USM_ARRAY_C_CONTIGUOUS
cdef public api int USM_ARRAY_F_CONTIGUOUS
cdef public api int USM_ARRAY_WRITEABLE

cdef public api int UAR_BOOL
cdef public api int UAR_BYTE
cdef public api int UAR_UBYTE
cdef public api int UAR_SHORT
cdef public api int UAR_USHORT
cdef public api int UAR_INT
cdef public api int UAR_UINT
cdef public api int UAR_LONG
cdef public api int UAR_ULONG
cdef public api int UAR_LONGLONG
cdef public api int UAR_ULONGLONG
cdef public api int UAR_FLOAT
cdef public api int UAR_DOUBLE
cdef public api int UAR_CFLOAT
cdef public api int UAR_CDOUBLE
cdef public api int UAR_TYPE_SENTINEL
cdef public api int UAR_HALF


cdef api class usm_ndarray [object PyUSMArrayObject, type PyUSMArrayType]:
    # data fields
    cdef char* data_
    cdef int nd_
    cdef Py_ssize_t *shape_
    cdef Py_ssize_t *strides_
    cdef int typenum_
    cdef int flags_
    cdef object base_
    cdef object array_namespace_
    # make usm_ndarray weak-referenceable
    cdef object __weakref__

    cdef void _reset(usm_ndarray self)
    cdef void _cleanup(usm_ndarray self)
    cdef usm_ndarray _clone(usm_ndarray self)
    cdef Py_ssize_t get_offset(usm_ndarray self) except *

    cdef char* get_data(self)
    cdef int get_ndim(self)
    cdef Py_ssize_t * get_shape(self)
    cdef Py_ssize_t * get_strides(self)
    cdef int get_typenum(self)
    cdef int get_itemsize(self)
    cdef int get_flags(self)
    cdef object get_base(self)
    cdef dpctl.DPCTLSyclQueueRef get_queue_ref(self) except *
    cdef dpctl.SyclQueue get_sycl_queue(self)

    cdef __cythonbufferdefaults__ = {"mode": "strided"}
