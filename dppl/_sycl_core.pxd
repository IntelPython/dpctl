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


cdef class SyclKernel:
    ''' Wraps a sycl::kernel object created from an OpenCL interoperability
        kernel.
    '''
    cdef DPPLSyclKernelRef kernel_ptr
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
    cdef DPPLSyclProgramRef program_ptr

    @staticmethod
    cdef SyclProgram _create (DPPLSyclProgramRef pref)
    cdef DPPLSyclProgramRef get_program_ptr (self)
    cpdef SyclKernel get_sycl_kernel(self, kernel_name)


cdef class SyclQueue:
    ''' Wrapper class for a Sycl queue.
    '''
    cdef DPPLSyclQueueRef queue_ptr

    @staticmethod
    cdef SyclQueue _create (DPPLSyclQueueRef qref)
    cpdef SyclContext get_sycl_context (self)
    cpdef SyclDevice get_sycl_device (self)
    cdef DPPLSyclQueueRef get_queue_ref (self)
