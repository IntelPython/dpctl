#                      Data Parallel Control (dpCtl)
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
