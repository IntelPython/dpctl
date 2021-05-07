# distutils: language = c++
# cython: language_level=3

cimport dpctl


cdef public int USM_ARRAY_C_CONTIGUOUS
cdef public int USM_ARRAY_F_CONTIGUOUS
cdef public int USM_ARRAY_WRITEABLE


cdef public class usm_ndarray [object PyUSMArrayObject, type PyUSMArrayType]:
    # data fields
    cdef char* data_
    cdef readonly int nd_
    cdef Py_ssize_t *shape_
    cdef Py_ssize_t *strides_
    cdef readonly int typenum_
    cdef readonly int flags_
    cdef readonly object base_
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
