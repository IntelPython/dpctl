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

# distutils: language = c++
# cython: language_level=3

"""Implements a Python interface for SYCL's program and kernel runtime classes.

The module also provides functions to create a SYCL program from either
a OpenCL source string or a SPIR-V binary file.

"""


from __future__ import print_function

from dpctl._backend cimport *

__all__ = [
    "create_program_from_source",
    "create_program_from_spirv",
    "SyclKernel",
    "SyclProgram",
    "SyclProgramCompilationError",
]

cdef class SyclProgramCompilationError(Exception):
    """This exception is raised when a SYCL program could not be built from
       either a SPIR-V binary file or a string source.
    """
    pass

cdef class SyclKernel:
    """
    """
    @staticmethod
    cdef SyclKernel _create(DPCTLSyclKernelRef kref):
        cdef SyclKernel ret = SyclKernel.__new__(SyclKernel)
        ret._kernel_ref = kref
        ret._function_name = DPCTLKernel_GetFunctionName(kref)
        return ret

    def __dealloc__(self):
        DPCTLKernel_Delete(self._kernel_ref)
        DPCTLCString_Delete(self._function_name)

    def get_function_name (self):
        """ Returns the name of the Kernel function.
        """
        return self._function_name.decode()

    def get_num_args(self):
        """ Returns the number of arguments for this kernel function.
        """
        return DPCTLKernel_GetNumArgs(self._kernel_ref)

    cdef DPCTLSyclKernelRef get_kernel_ref (self):
        """ Returns the DPCTLSyclKernelRef pointer for this SyclKernel.
        """
        return self._kernel_ref

    def addressof_ref(self):
        """ Returns the address of the C API DPCTLSyclKernelRef pointer
        as a size_t.

        Returns:
            The address of the DPCTLSyclKernelRef object used to create this
            SyclKernel cast to a size_t.
        """
        return int(<size_t>self._kernel_ref)

cdef class SyclProgram:
    """ Wraps a sycl::program object created from an OpenCL interoperability
        program.

        SyclProgram exposes the C API from dpctl_sycl_program_interface.h. A
        SyclProgram can be created from either a source string or a SPIR-V
        binary file.
    """

    @staticmethod
    cdef SyclProgram _create (DPCTLSyclProgramRef pref):
        cdef SyclProgram ret = SyclProgram.__new__(SyclProgram)
        ret._program_ref = pref
        return ret

    def __dealloc__(self):
        DPCTLProgram_Delete(self._program_ref)

    cdef DPCTLSyclProgramRef get_program_ref (self):
        return self._program_ref

    cpdef SyclKernel get_sycl_kernel(self, str kernel_name):
        name = kernel_name.encode('utf8')
        return SyclKernel._create(DPCTLProgram_GetKernel(self._program_ref,
                                                         name))

    def has_sycl_kernel(self, str kernel_name):
        name = kernel_name.encode('utf8')
        return DPCTLProgram_HasKernel(self._program_ref, name)

    def addressof_ref(self):
        """Returns the address of the C API DPCTLSyclProgramRef pointer
        as a long.

        Returns:
            The address of the DPCTLSyclProgramRef object used to create this
            SyclProgram cast to a long.
        """
        return int(<size_t>self._program_ref)

cpdef create_program_from_source(SyclQueue q, unicode src, unicode copts=""):
    """
        Creates a Sycl interoperability program from an OpenCL source string.

        We use the DPCTLProgram_CreateFromOCLSource() C API function to create
        a Sycl progrma from an OpenCL source program that can contain multiple
        kernels. Note currently only supported for OpenCL.

        Parameters:
            q (SyclQueue)   : The :class:`SyclQueue` for which the
                              :class:`SyclProgram` is going to be built.
            src (unicode): Source string for an OpenCL program.
            copts (unicode) : Optional compilation flags that will be used
                              when compiling the program.

        Returns:
            program (SyclProgram): A :class:`SyclProgram` object wrapping the  sycl::program returned by the C API.

        Raises:
            SyclProgramCompilationError: If a SYCL program could not be created.
    """

    cdef DPCTLSyclProgramRef Pref
    cdef bytes bSrc = src.encode('utf8')
    cdef bytes bCOpts = copts.encode('utf8')
    cdef const char *Src = <const char*>bSrc
    cdef const char *COpts = <const char*>bCOpts
    cdef DPCTLSyclContextRef CRef = q.get_sycl_context().get_context_ref()
    Pref = DPCTLProgram_CreateFromOCLSource(CRef, Src, COpts)

    if Pref is NULL:
        raise SyclProgramCompilationError()

    return SyclProgram._create(Pref)

cimport cython.array


cpdef create_program_from_spirv(SyclQueue q, const unsigned char[:] IL,
                                unicode copts=""):
    """
        Creates a Sycl interoperability program from an SPIR-V binary.

        We use the DPCTLProgram_CreateFromOCLSpirv() C API function to create
        a Sycl progrma from an compiled SPIR-V binary file.

        Parameters:
            q (SyclQueue): The :class:`SyclQueue` for which the
                           :class:`SyclProgram` is going to be built.
            IL (const char[:]) : SPIR-V binary IL file for an OpenCL program.
            copts (unicode) : Optional compilation flags that will be used
                              when compiling the program.

        Returns:
            program (SyclProgram): A :class:`SyclProgram` object wrapping the  sycl::program returned by the C API.

        Raises:
            SyclProgramCompilationError: If a SYCL program could not be created.
    """

    cdef DPCTLSyclProgramRef Pref
    cdef const unsigned char *dIL = &IL[0]
    cdef DPCTLSyclContextRef CRef = q.get_sycl_context().get_context_ref()
    cdef size_t length = IL.shape[0]
    cdef bytes bCOpts = copts.encode('utf8')
    cdef const char *COpts = <const char*>bCOpts
    Pref = DPCTLProgram_CreateFromSpirv(CRef, <const void*>dIL, length, COpts)
    if Pref is NULL:
        raise SyclProgramCompilationError()

    return SyclProgram._create(Pref)
