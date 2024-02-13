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


"""The Cython declarations for the functions defined in _program.pyx.
"""


from .._backend cimport DPCTLSyclKernelBundleRef, DPCTLSyclKernelRef
from .._sycl_context cimport SyclContext
from .._sycl_device cimport SyclDevice
from .._sycl_queue cimport SyclQueue


cdef api class SyclKernel [object PySyclKernelObject, type PySyclKernelType]:
    ''' Wraps a sycl::kernel object created from an OpenCL interoperability
        kernel.
    '''
    cdef DPCTLSyclKernelRef _kernel_ref
    cdef str _function_name
    cdef DPCTLSyclKernelRef get_kernel_ref (self)

    @staticmethod
    cdef SyclKernel _create (DPCTLSyclKernelRef kref, str name)


cdef api class SyclProgram [object PySyclProgramObject, type PySyclProgramType]:
    ''' Wraps a sycl::kernel_bundle<sycl::bundle_state::executable> object created by
        using SYCL interoperability layer for OpenCL and Level-Zero backends.

        SyclProgram exposes the C API from dpctl_sycl_kernel_bundle_interface.h. A
        SyclProgram can be created from either a source string or a SPIR-V
        binary file.
    '''
    cdef DPCTLSyclKernelBundleRef _program_ref

    @staticmethod
    cdef  SyclProgram _create (DPCTLSyclKernelBundleRef pref)
    cdef  DPCTLSyclKernelBundleRef get_program_ref (self)
    cpdef SyclKernel get_sycl_kernel(self, str kernel_name)


cpdef create_program_from_source (SyclQueue q, unicode source, unicode copts=*)
cpdef create_program_from_spirv (SyclQueue q, const unsigned char[:] IL,
                                 unicode copts=*)
