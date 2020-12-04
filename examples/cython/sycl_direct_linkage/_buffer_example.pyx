cimport numpy as cnp
import numpy as np
from cython.operator cimport dereference as deref

cdef extern from "CL/sycl.hpp" namespace "cl::sycl":
    cdef cppclass queue nogil:
       pass

cdef extern from "sycl_function.hpp":
    int c_columnwise_total(queue& q, size_t n, size_t m, double *m, double *ct) nogil

def columnwise_total(double[:, ::1] v):
    cdef cnp.ndarray res_array = np.empty((v.shape[1],), dtype='d')
    cdef double[::1] res_memslice = res_array
    cdef int ret_status
    cdef queue* q

    q = new queue()

    with nogil:
        ret_status = c_columnwise_total(deref(q), v.shape[0], v.shape[1], &v[0,0], &res_memslice[0])

    del q
        
    return res_array
