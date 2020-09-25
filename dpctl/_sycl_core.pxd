##===------------- sycl_core.pxd - dpctl module --------*- Cython -*-------===##
##
##                      Data Parallel Control (dpCtl)
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
    cdef DPPLSyclContextRef _ctxt_ptr

    @staticmethod
    cdef SyclContext _create (DPPLSyclContextRef ctxt)
    cdef DPPLSyclContextRef get_context_ref (self)


cdef class SyclDevice:
    ''' Wrapper class for a Sycl Device
    '''
    cdef DPPLSyclDeviceRef _device_ptr
    cdef const char *_vendor_name
    cdef const char *_device_name
    cdef const char *_driver_version

    @staticmethod
    cdef SyclDevice _create (DPPLSyclDeviceRef dref)
    cdef DPPLSyclDeviceRef get_device_ptr (self)


cdef class SyclEvent:
    ''' Wrapper class for a Sycl Event
    '''
    cdef  DPPLSyclEventRef _event_ptr

    @staticmethod
    cdef  SyclEvent _create (DPPLSyclEventRef e)
    cpdef void wait (self)


cdef class SyclKernel:
    ''' Wraps a sycl::kernel object created from an OpenCL interoperability
        kernel.
    '''
    cdef DPPLSyclKernelRef _kernel_ptr
    cdef const char *_function_name
    cdef DPPLSyclKernelRef get_kernel_ptr (self)

    @staticmethod
    cdef SyclKernel _create (DPPLSyclKernelRef kref)


cdef class SyclProgram:
    ''' Wraps a sycl::program object created from an OpenCL interoperability
        program.

        SyclProgram exposes the C API from dppl_sycl_program_interface.h. A
        SyclProgram can be created from either a source string or a SPIR-V
        binary file.
    '''
    cdef DPPLSyclProgramRef _program_ptr

    @staticmethod
    cdef  SyclProgram _create (DPPLSyclProgramRef pref)
    cdef  DPPLSyclProgramRef get_program_ptr (self)
    cpdef SyclKernel get_sycl_kernel(self, str kernel_name)


cdef class SyclQueue:
    ''' Wrapper class for a Sycl queue.
    '''
    cdef DPPLSyclQueueRef _queue_ptr
    cdef SyclContext _context
    cdef SyclDevice _device
    cdef list _events

    @staticmethod
    cdef  SyclQueue _create (DPPLSyclQueueRef qref)
    cpdef SyclContext get_sycl_context (self)
    cpdef SyclDevice get_sycl_device (self)
    cdef  DPPLSyclQueueRef get_queue_ref (self)
    cpdef SyclEvent submit (self, SyclKernel kernel, list args,                \
                            const size_t [:]range, size_t ndims)
    cpdef void wait (self)
