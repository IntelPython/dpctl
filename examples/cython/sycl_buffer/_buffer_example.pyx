cimport numpy as cnp
import numpy as np

cimport dpctl as c_dpctl
import dpctl

cdef extern from "use_sycl_buffer.h":
    int c_columnwise_total(c_dpctl.DPCTLSyclQueueRef q, size_t n, size_t m, double *m, double *ct) nogil
    int c_columnwise_total_no_mkl(c_dpctl.DPCTLSyclQueueRef q, size_t n, size_t m, double *m, double *ct) nogil

def columnwise_total(double[:, ::1] v, method='mkl'):
    cdef cnp.ndarray res_array = np.empty((v.shape[1],), dtype='d')
    cdef double[::1] res_memslice = res_array
    cdef int ret_status
    cdef c_dpctl.SyclQueue q
    cdef c_dpctl.DPCTLSyclQueueRef q_ref

    q = c_dpctl.get_current_queue()
    q_ref = q.get_queue_ref()

    if method == 'mkl':
        with nogil:
            ret_status = c_columnwise_total(q_ref, v.shape[0], v.shape[1], &v[0,0], &res_memslice[0])
    else:
        with nogil:
            ret_status = c_columnwise_total_no_mkl(q_ref, v.shape[0], v.shape[1], &v[0,0], &res_memslice[0])        

    return res_array
