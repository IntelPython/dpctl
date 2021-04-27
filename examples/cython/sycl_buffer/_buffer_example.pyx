#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cimport numpy as cnp
import numpy as np

cimport dpctl as c_dpctl
import dpctl

cdef extern from "use_sycl_buffer.h":
    int c_columnwise_total(c_dpctl.DPCTLSyclQueueRef q, size_t n, size_t m, double *m, double *ct) nogil
    int c_columnwise_total_no_mkl(c_dpctl.DPCTLSyclQueueRef q, size_t n, size_t m, double *m, double *ct) nogil

def columnwise_total(double[:, ::1] v, method='mkl', queue=None):
    cdef cnp.ndarray res_array = np.empty((v.shape[1],), dtype='d')
    cdef double[::1] res_memslice = res_array
    cdef int ret_status
    cdef c_dpctl.SyclQueue q
    cdef c_dpctl.DPCTLSyclQueueRef q_ref

    if (queue is None):
        q = c_dpctl.SyclQueue()
    elif isinstance(queue, dpctl.SyclQueue):
        q = <c_dpctl.SyclQueue> queue
    else:
        q = c_dpctl.SyclQueue(queue)
    q_ref = q.get_queue_ref()

    if method == 'mkl':
        with nogil:
            ret_status = c_columnwise_total(q_ref, v.shape[0], v.shape[1], &v[0,0], &res_memslice[0])
    else:
        with nogil:
            ret_status = c_columnwise_total_no_mkl(q_ref, v.shape[0], v.shape[1], &v[0,0], &res_memslice[0])

    return res_array
