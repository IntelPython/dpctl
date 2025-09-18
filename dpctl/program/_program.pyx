#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2025 Intel Corporation
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
# cython: linetrace=True

"""Implements a Python interface for SYCL's program and kernel runtime classes.

The module also provides functions to create a SYCL program from either
a OpenCL source string or a SPIR-V binary file.

"""

from libc.stdint cimport uint32_t

from dpctl._backend cimport (  # noqa: E211, E402;
    DPCTLBuildOptionList_Append,
    DPCTLBuildOptionList_Create,
    DPCTLBuildOptionList_Delete,
    DPCTLBuildOptionListRef,
    DPCTLKernel_Copy,
    DPCTLKernel_Delete,
    DPCTLKernel_GetCompileNumSubGroups,
    DPCTLKernel_GetCompileSubGroupSize,
    DPCTLKernel_GetMaxNumSubGroups,
    DPCTLKernel_GetMaxSubGroupSize,
    DPCTLKernel_GetNumArgs,
    DPCTLKernel_GetPreferredWorkGroupSizeMultiple,
    DPCTLKernel_GetPrivateMemSize,
    DPCTLKernel_GetWorkGroupSize,
    DPCTLKernelBuildLog_Create,
    DPCTLKernelBuildLog_Delete,
    DPCTLKernelBuildLog_Get,
    DPCTLKernelBuildLogRef,
    DPCTLKernelBundle_Copy,
    DPCTLKernelBundle_CreateFromOCLSource,
    DPCTLKernelBundle_CreateFromSpirv,
    DPCTLKernelBundle_CreateFromSYCLSource,
    DPCTLKernelBundle_Delete,
    DPCTLKernelBundle_GetKernel,
    DPCTLKernelBundle_GetSyclKernel,
    DPCTLKernelBundle_HasKernel,
    DPCTLKernelBundle_HasSyclKernel,
    DPCTLKernelNameList_Append,
    DPCTLKernelNameList_Create,
    DPCTLKernelNameList_Delete,
    DPCTLKernelNameListRef,
    DPCTLSyclContextRef,
    DPCTLSyclDeviceRef,
    DPCTLSyclKernelBundleRef,
    DPCTLSyclKernelRef,
    DPCTLVirtualHeaderList_Append,
    DPCTLVirtualHeaderList_Create,
    DPCTLVirtualHeaderList_Delete,
    DPCTLVirtualHeaderListRef,
)

__all__ = [
    "create_program_from_source",
    "create_program_from_spirv",
    "SyclKernel",
    "SyclProgram",
    "SyclProgramCompilationError",
]

cdef class SyclProgramCompilationError(Exception):
    """This exception is raised when a ``sycl::kernel_bundle`` could not be
       built from either a SPIR-V binary file or a string source.
    """
    pass


cdef class SyclKernel:
    """
    """
    @staticmethod
    cdef SyclKernel _create(DPCTLSyclKernelRef kref, str name):
        cdef SyclKernel ret = SyclKernel.__new__(SyclKernel)
        ret._kernel_ref = kref
        ret._function_name = name
        return ret

    def __dealloc__(self):
        DPCTLKernel_Delete(self._kernel_ref)

    def get_function_name(self):
        """ Returns the name of the ``sycl::kernel`` function.
        """
        return self._function_name

    def get_num_args(self):
        """ Returns the number of arguments for this kernel function.
        """
        return DPCTLKernel_GetNumArgs(self._kernel_ref)

    cdef DPCTLSyclKernelRef get_kernel_ref(self):
        """ Returns the ``DPCTLSyclKernelRef`` pointer for this SyclKernel.
        """
        return self._kernel_ref

    def addressof_ref(self):
        """ Returns the address of the C API ``DPCTLSyclKernelRef`` pointer
        as a ``size_t``.

        Returns:
            The address of the ``DPCTLSyclKernelRef`` pointer used to create
            this :class:`dpctl.SyclKernel` object cast to a ``size_t``.
        """
        return int(<size_t>self._kernel_ref)

    @property
    def num_args(self):
        """ Property equivalent to method call `SyclKernel.get_num_args()`
        """
        return self.get_num_args()

    @property
    def work_group_size(self):
        """ Returns the maximum number of work-items in a work-group that can
        be used to execute the kernel on device it was built for.
        """
        cdef size_t v = DPCTLKernel_GetWorkGroupSize(self._kernel_ref)
        return v

    @property
    def preferred_work_group_size_multiple(self):
        """ Returns a value, of which work-group size is preferred to be
        a multiple, for executing the kernel on the device it was built for.
        """
        cdef size_t v = DPCTLKernel_GetPreferredWorkGroupSizeMultiple(
            self._kernel_ref
        )
        return v

    @property
    def private_mem_size(self):
        """ Returns the minimum amount of private memory, in bytes, used by each
        work-item in the kernel.
        """
        cdef size_t v = DPCTLKernel_GetPrivateMemSize(self._kernel_ref)
        return v

    @property
    def max_num_sub_groups(self):
        """ Returns the maximum number of sub-groups for this kernel.
        """
        cdef uint32_t n = DPCTLKernel_GetMaxNumSubGroups(self._kernel_ref)
        return n

    @property
    def max_sub_group_size(self):
        """ Returns the maximum sub-groups size for this kernel.
        """
        cdef uint32_t sz = DPCTLKernel_GetMaxSubGroupSize(self._kernel_ref)
        return sz

    @property
    def compile_num_sub_groups(self):
        """ Returns the number of sub-groups specified by this kernel,
        or 0 (if not specified).
        """
        cdef size_t n = DPCTLKernel_GetCompileNumSubGroups(self._kernel_ref)
        return n

    @property
    def compile_sub_group_size(self):
        """ Returns the required sub-group size specified by this kernel,
        or 0 (if not specified).
        """
        cdef size_t n = DPCTLKernel_GetCompileSubGroupSize(self._kernel_ref)
        return n


cdef api DPCTLSyclKernelRef SyclKernel_GetKernelRef(SyclKernel ker):
    """ C-API function to access opaque kernel reference from
    Python object of type :class:`dpctl.program.SyclKernel`.
    """
    return ker.get_kernel_ref()


cdef api SyclKernel SyclKernel_Make(DPCTLSyclKernelRef KRef, const char *name):
    """
    C-API function to create :class:`dpctl.program.SyclKernel`
    instance from opaque sycl kernel reference.
    """
    cdef DPCTLSyclKernelRef copied_KRef = DPCTLKernel_Copy(KRef)
    if (name is NULL):
        return SyclKernel._create(copied_KRef, "default_name")
    else:
        return SyclKernel._create(copied_KRef, name.decode("utf-8"))


cdef class SyclProgram:
    """ Wraps a ``sycl::kernel_bundle<sycl::bundle_state::executable>`` object
    created using SYCL interoperability layer with underlying backends. Only the
    OpenCL and Level-Zero backends are currently supported.

    SyclProgram exposes the C API from ``dpctl_sycl_kernel_bundle_interface.h``.
    A SyclProgram can be created from either a source string or a SPIR-V
    binary file.
    """

    @staticmethod
    cdef SyclProgram _create(DPCTLSyclKernelBundleRef KBRef,
                             bint is_sycl_source):
        cdef SyclProgram ret = SyclProgram.__new__(SyclProgram)
        ret._program_ref = KBRef
        ret._is_sycl_source = is_sycl_source
        return ret

    def __dealloc__(self):
        DPCTLKernelBundle_Delete(self._program_ref)

    cdef DPCTLSyclKernelBundleRef get_program_ref(self):
        return self._program_ref

    cpdef SyclKernel get_sycl_kernel(self, str kernel_name):
        name = kernel_name.encode("utf8")
        if self._is_sycl_source:
            return SyclKernel._create(
                    DPCTLKernelBundle_GetSyclKernel(self._program_ref, name),
                    kernel_name)
        return SyclKernel._create(
            DPCTLKernelBundle_GetKernel(self._program_ref, name),
            kernel_name
        )

    def has_sycl_kernel(self, str kernel_name):
        name = kernel_name.encode("utf8")
        if self._is_sycl_source:
            return DPCTLKernelBundle_HasSyclKernel(self._program_ref, name)
        return DPCTLKernelBundle_HasKernel(self._program_ref, name)

    def addressof_ref(self):
        """Returns the address of the C API DPCTLSyclKernelBundleRef pointer
        as a long.

        Returns:
            The address of the ``DPCTLSyclKernelBundleRef`` pointer used to
            create this :class:`dpctl.SyclProgram` object cast to a ``size_t``.
        """
        return int(<size_t>self._program_ref)


cpdef create_program_from_source(SyclQueue q, str src, str copts=""):
    """
        Creates a Sycl interoperability program from an OpenCL source string.

        We use the ``DPCTLKernelBundle_CreateFromOCLSource()`` C API function
        to create a ``sycl::kernel_bundle<sycl::bundle_state::executable>``
        from an OpenCL source program that can contain multiple kernels.
        Note: This function is currently only supported for the OpenCL backend.

        Parameters:
            q (:class:`dpctl.SyclQueue`)
                The :class:`dpctl.SyclQueue` for which the
                :class:`.SyclProgram` is going to be built.
            src (str)
                Source string for an OpenCL program.
            copts (str, optional)
                Optional compilation flags that will be used
                when compiling the program. Default: ``""``.

        Returns:
            program (:class:`.SyclProgram`)
                A :class:`.SyclProgram` object wrapping the
                ``sycl::kernel_bundle<sycl::bundle_state::executable>``
                returned by the C API.

        Raises:
            SyclProgramCompilationError
                If a SYCL kernel bundle could not be created.
    """

    cdef DPCTLSyclKernelBundleRef KBref
    cdef bytes bSrc = src.encode("utf8")
    cdef bytes bCOpts = copts.encode("utf8")
    cdef const char *Src = <const char*>bSrc
    cdef const char *COpts = <const char*>bCOpts
    cdef DPCTLSyclContextRef CRef = q.get_sycl_context().get_context_ref()
    cdef DPCTLSyclDeviceRef DRef = q.get_sycl_device().get_device_ref()
    KBref = DPCTLKernelBundle_CreateFromOCLSource(CRef, DRef, Src, COpts)

    if KBref is NULL:
        raise SyclProgramCompilationError()

    return SyclProgram._create(KBref, False)


cpdef create_program_from_spirv(SyclQueue q, const unsigned char[:] IL,
                                str copts=""):
    """
        Creates a Sycl interoperability program from an SPIR-V binary.

        We use the :c:func:`DPCTLKernelBundle_CreateFromOCLSpirv` C API function
        to create a ``sycl::kernel_bundle<sycl::bundle_state::executable>``
        object from an compiled SPIR-V binary file.

        Parameters:
            q (:class:`dpctl.SyclQueue`)
                The :class:`dpctl.SyclQueue` for which the
                :class:`.SyclProgram` is going to be built.
            IL (bytes)
                SPIR-V binary IL file for an OpenCL program.
            copts (str, optional)
                Optional compilation flags that will be used
                when compiling the program. Default: ``""``.

        Returns:
            program (:class:`.SyclProgram`)
                A :class:`.SyclProgram` object wrapping the
                ``sycl::kernel_bundle<sycl::bundle_state::executable>``
                returned by the C API.

        Raises:
            SyclProgramCompilationError
                If a SYCL kernel bundle could not be created.
    """

    cdef DPCTLSyclKernelBundleRef KBref
    cdef const unsigned char *dIL = &IL[0]
    cdef DPCTLSyclContextRef CRef = q.get_sycl_context().get_context_ref()
    cdef DPCTLSyclDeviceRef DRef = q.get_sycl_device().get_device_ref()
    cdef size_t length = IL.shape[0]
    cdef bytes bCOpts = copts.encode("utf8")
    cdef const char *COpts = <const char*>bCOpts
    KBref = DPCTLKernelBundle_CreateFromSpirv(
        CRef, DRef, <const void*>dIL, length, COpts
    )
    if KBref is NULL:
        raise SyclProgramCompilationError()

    return SyclProgram._create(KBref, False)


cpdef create_program_from_sycl_source(SyclQueue q, unicode source,
                                      list headers=None,
                                      list registered_names=None,
                                      list copts=None):
    """
        Creates an executable SYCL kernel_bundle from SYCL source code.

        This uses the DPC++ ``kernel_compiler`` extension to create a
        ``sycl::kernel_bundle<sycl::bundle_state::executable>`` object from
        SYCL source code.

        Parameters:
            q (:class:`dpctl.SyclQueue`)
                The :class:`dpctl.SyclQueue` for which the
                :class:`.SyclProgram` is going to be built.
            source (unicode)
                SYCL source code string.
            headers (list)
                Optional list of virtual headers, where each entry in the list
                needs to be a tuple of header name and header content. See the
                documentation of the ``include_files`` property in the DPC++
                ``kernel_compiler`` extension for more information.
                Default: []
            registered_names (list, optional)
                Optional list of kernel names to register. See the
                documentation of the ``registered_names`` property in the DPC++
                ``kernel_compiler`` extension for more information.
                Default: []
            copts (list)
                Optional list of compilation flags that will be used
                when compiling the program. Default: ``""``.

        Returns:
            program (:class:`.SyclProgram`)
                A :class:`.SyclProgram` object wrapping the
                ``sycl::kernel_bundle<sycl::bundle_state::executable>``
                returned by the C API.

        Raises:
            SyclProgramCompilationError
                If a SYCL kernel bundle could not be created. The exception
                message contains the build log for more details.
    """
    cdef DPCTLSyclKernelBundleRef KBref
    cdef DPCTLSyclContextRef CRef = q.get_sycl_context().get_context_ref()
    cdef DPCTLSyclDeviceRef DRef = q.get_sycl_device().get_device_ref()
    cdef bytes bSrc = source.encode("utf8")
    cdef const char *Src = <const char*>bSrc
    cdef DPCTLBuildOptionListRef BuildOpts = DPCTLBuildOptionList_Create()
    cdef bytes bOpt
    cdef const char* sOpt
    cdef bytes bName
    cdef const char* sName
    cdef bytes bContent
    cdef const char* sContent
    cdef const char* buildLogContent
    for opt in copts:
        if not isinstance(opt, unicode):
            DPCTLBuildOptionList_Delete(BuildOpts)
            raise SyclProgramCompilationError()
        bOpt = opt.encode("utf8")
        sOpt = <const char*>bOpt
        DPCTLBuildOptionList_Append(BuildOpts, sOpt)

    cdef DPCTLKernelNameListRef KernelNames = DPCTLKernelNameList_Create()
    for name in registered_names:
        if not isinstance(name, unicode):
            DPCTLBuildOptionList_Delete(BuildOpts)
            DPCTLKernelNameList_Delete(KernelNames)
            raise SyclProgramCompilationError()
        bName = name.encode("utf8")
        sName = <const char*>bName
        DPCTLKernelNameList_Append(KernelNames, sName)

    cdef DPCTLVirtualHeaderListRef VirtualHeaders
    VirtualHeaders = DPCTLVirtualHeaderList_Create()

    for name, content in headers:
        if not isinstance(name, unicode) or not isinstance(content, unicode):
            DPCTLBuildOptionList_Delete(BuildOpts)
            DPCTLKernelNameList_Delete(KernelNames)
            DPCTLVirtualHeaderList_Delete(VirtualHeaders)
            raise SyclProgramCompilationError()
        bName = name.encode("utf8")
        sName = <const char*>bName
        bContent = content.encode("utf8")
        sContent = <const char*>bContent
        DPCTLVirtualHeaderList_Append(VirtualHeaders, sName, sContent)

    cdef DPCTLKernelBuildLogRef BuildLog
    BuildLog = DPCTLKernelBuildLog_Create()

    KBref = DPCTLKernelBundle_CreateFromSYCLSource(CRef, DRef, Src,
                                                   VirtualHeaders, KernelNames,
                                                   BuildOpts, BuildLog)

    if KBref is NULL:
        buildLogContent = DPCTLKernelBuildLog_Get(BuildLog)
        buildLogStr = str(buildLogContent, "utf-8")
        DPCTLBuildOptionList_Delete(BuildOpts)
        DPCTLKernelNameList_Delete(KernelNames)
        DPCTLVirtualHeaderList_Delete(VirtualHeaders)
        DPCTLKernelBuildLog_Delete(BuildLog)
        raise SyclProgramCompilationError(buildLogStr)

    DPCTLBuildOptionList_Delete(BuildOpts)
    DPCTLKernelNameList_Delete(KernelNames)
    DPCTLVirtualHeaderList_Delete(VirtualHeaders)
    DPCTLKernelBuildLog_Delete(BuildLog)

    return SyclProgram._create(KBref, True)


cdef api DPCTLSyclKernelBundleRef SyclProgram_GetKernelBundleRef(
    SyclProgram pro
):
    """ C-API function to access opaque kernel bundle reference from
    Python object of type :class:`dpctl.program.SyclKernel`.
    """
    return pro.get_program_ref()


cdef api SyclProgram SyclProgram_Make(DPCTLSyclKernelBundleRef KBRef):
    """
    C-API function to create :class:`dpctl.program.SyclProgram`
    instance from opaque ``sycl::kernel_bundle<sycl::bundle_state::executable>``
    reference.
    """
    cdef DPCTLSyclKernelBundleRef copied_KBRef = DPCTLKernelBundle_Copy(KBRef)
    return SyclProgram._create(copied_KBRef, False)
