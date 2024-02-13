#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2024 Intel Corporation
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

# distutils: language = c++
# cython: language_level=3

cimport cython

cimport dpctl as c_dpctl
from dpctl.sycl cimport queue as dpcpp_queue
from dpctl.sycl cimport unwrap_queue

import numpy as np

import dpctl


cdef extern from "use_sycl_buffer.hpp":
    void native_columnwise_total "columnwise_total"[T](
        dpcpp_queue,    # execution queue
        size_t,         # number of rows of the input matrix
        size_t,         # number of columns of the input matrix
        const T *,      # data pointer of the input matrix
        T *             # pointer for the resulting vector
    ) nogil except+


def columnwise_total(cython.floating[:, ::1] mat, queue=None):
    """ columntiwse_total(mat, queue=None)

    Returns column-wise total of the input matrix.

    Args:
        mat: ndarray
            C-contiguous non-empty matrix of single- or double-precision
            floating point type.
        queue: dpctl.SyclQueue or None
            Execution queue targeting a SYCL device for offload. Default
            value of `None` means use default-constructed `dpctl.SyclQueue`
            that targets default-selected device.

    Note:
        It is advantageous to create `dpctl.SyclQueue` and reuse it as queue
        construction may be expensive.
    """
    cdef cython.floating[:] res_memslice
    cdef c_dpctl.SyclQueue q
    cdef dpcpp_queue* exec_queue_ptr = NULL
    cdef size_t n_cols
    cdef size_t n_rows

    n_rows = mat.shape[0]
    n_cols = mat.shape[1]

    if cython.floating is float:
        res_memslice = np.empty(n_cols, dtype=np.single)
    elif cython.floating is double:
        res_memslice = np.empty(n_cols, dtype=np.double)
    else:
        raise TypeError(
	        "Use single or double precision floating point types are supported"
	    )

    if (queue is None):
        # use default-constructed queue
        q = c_dpctl.SyclQueue()
    elif isinstance(queue, dpctl.SyclQueue):
        q = <c_dpctl.SyclQueue> queue
    else:
        q = c_dpctl.SyclQueue(queue)
    exec_queue_ptr = unwrap_queue(q.get_queue_ref())

    with nogil:
        native_columnwise_total(
            exec_queue_ptr[0], n_rows, n_cols, &mat[0,0], &res_memslice[0]
        )

    return np.asarray(res_memslice)
