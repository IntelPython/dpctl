##===-------------- _program.pxd - dpctl.experimental module -*- Cython -*-===##
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
## This file has the Cython function declarations for the functions defined
## in dpctl.experimental._program.pyx
##
##===----------------------------------------------------------------------===##

# distutils: language = c++
# cython: language_level=3

from .._backend cimport DPPLSyclKernelRef, DPPLSyclProgramRef
from .._sycl_core cimport SyclQueue, SyclDevice, SyclContext


cdef class SyclKernel:
    ''' Wraps a sycl::kernel object created from an OpenCL interoperability
        kernel.
    '''
    cdef DPPLSyclKernelRef _kernel_ref
    cdef const char *_function_name
    cdef DPPLSyclKernelRef get_kernel_ref (self)

    @staticmethod
    cdef SyclKernel _create (DPPLSyclKernelRef kref)


cdef class SyclProgram:
    ''' Wraps a sycl::program object created from an OpenCL interoperability
        program.

        SyclProgram exposes the C API from dppl_sycl_program_interface.h. A
        SyclProgram can be created from either a source string or a SPIR-V
        binary file.
    '''
    cdef DPPLSyclProgramRef _program_ref

    @staticmethod
    cdef  SyclProgram _create (DPPLSyclProgramRef pref)
    cdef  DPPLSyclProgramRef get_program_ref (self)
    cpdef SyclKernel get_sycl_kernel(self, str kernel_name)


cpdef create_program_from_source (SyclQueue q, unicode source, unicode copts=*)
cpdef create_program_from_spirv (SyclQueue q, const unsigned char[:] IL)
