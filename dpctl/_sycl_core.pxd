##===------------- sycl_core.pxd - dpctl interface ------*- Cython -*------===##
##
##                      Data Parallel Control (dpctl)
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
## This file declares the extension types and functions for the Cython API
## implemented in sycl_core.pyx.
##
##===----------------------------------------------------------------------===##

# distutils: language = c++
# cython: language_level=3

from .backend cimport *


cdef class SyclContext:
    ''' Wrapper class for a Sycl Context
    '''
    cdef DPPLSyclContextRef ctxt_ptr

    @staticmethod
    cdef SyclContext _create (DPPLSyclContextRef ctxt)
    cdef DPPLSyclContextRef get_context_ref (self)


cdef class SyclDevice:
    ''' Wrapper class for a Sycl Device
    '''
    cdef DPPLSyclDeviceRef device_ptr
    cdef const char *vendor_name
    cdef const char *device_name
    cdef const char *driver_version

    @staticmethod
    cdef SyclDevice _create (DPPLSyclDeviceRef dref)
    cdef DPPLSyclDeviceRef get_device_ptr (self)


cdef class SyclQueue:
    ''' Wrapper class for a Sycl queue.
    '''
    cdef DPPLSyclQueueRef queue_ptr

    @staticmethod
    cdef SyclQueue _create (DPPLSyclQueueRef qref)
    cpdef SyclContext get_sycl_context (self)
    cpdef SyclDevice get_sycl_device (self)
    cdef DPPLSyclQueueRef get_queue_ref (self)
