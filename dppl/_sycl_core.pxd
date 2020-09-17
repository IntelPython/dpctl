##===------------- sycl_core.pxd - DPPL interface ------*- Cython -*-------===##
##
##               Python Data Parallel Processing Library (PyDPPL)
##
## Copyright 2020 Intel Corporation
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##
##===----------------------------------------------------------------------===##
##
## \file
## This file defines the Cython interface for the Sycl API of PyDPPL.
##
##===----------------------------------------------------------------------===##

cdef extern from "dppl_sycl_types.h":
    cdef struct DPPLOpaqueSyclQueue

    ctypedef DPPLOpaqueSyclQueue* DPPLSyclQueueRef


cdef class SyclQueue:
    ''' Wrapper class for a Sycl queue.
    '''

    cdef DPPLSyclQueueRef queue_ptr

    @staticmethod
    cdef SyclQueue _create (DPPLSyclQueueRef qref)
    cpdef get_sycl_context (self)
    cpdef get_sycl_device (self)
    cdef DPPLSyclQueueRef get_queue_ref (self)

