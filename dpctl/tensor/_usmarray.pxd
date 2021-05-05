# distutils: language = c++
# cython: language_level=3

cdef public int USM_ARRAY_C_CONTIGUOUS
cdef public int USM_ARRAY_F_CONTIGUOUS
cdef public int USM_ARRAY_WRITEABLE


cdef public class usm_ndarray [object PyUSMArrayObject, type PyUSMArrayType]:
    cdef char* data
    cdef int nd
    cdef Py_ssize_t *shape
    cdef Py_ssize_t *strides
    cdef int typenum
    cdef int flags
    cdef object base

    cdef void _reset(usm_ndarray self)
    cdef void _cleanup(usm_ndarray self)
    cdef usm_ndarray _clone(usm_ndarray self)
    cdef Py_ssize_t get_offset(usm_ndarray self) except *

    cdef __cythonbufferdefaults__ = {"mode": "strided"}
