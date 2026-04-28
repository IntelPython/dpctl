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

"""Implements a Python interface for SYCL's kernel bundle and kernel runtime
classes.

The module also provides functions to create a SYCL kernel bundle from either
an OpenCL source string or a SPIR-V binary file.

"""

from cpython.buffer cimport (
    Py_buffer,
    PyBUF_ANY_CONTIGUOUS,
    PyBUF_SIMPLE,
    PyBuffer_Release,
    PyObject_CheckBuffer,
    PyObject_GetBuffer,
)
from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdint cimport uint32_t
from libc.string cimport memcmp

import warnings

from dpctl._backend cimport (  # noqa: E211, E402;
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
    DPCTLKernelBundle_Copy,
    DPCTLKernelBundle_CreateFromOCLSource,
    DPCTLKernelBundle_CreateFromSpirv,
    DPCTLKernelBundle_Delete,
    DPCTLKernelBundle_GetKernel,
    DPCTLKernelBundle_HasKernel,
    DPCTLSyclContextRef,
    DPCTLSyclDeviceRef,
    DPCTLSyclKernelBundleRef,
    DPCTLSyclKernelRef,
    _spec_const,
)

import numbers

import numpy as np

__all__ = [
    "create_kernel_bundle_from_source",
    "create_kernel_bundle_from_spirv",
    "SyclKernel",
    "SyclKernelBundle",
    "SyclKernelBundleCompilationError",
    "SpecializationConstant",
]

cdef class SyclKernelBundleCompilationError(Exception):
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


cdef class SyclKernelBundle:
    """ Wraps a ``sycl::kernel_bundle<sycl::bundle_state::executable>`` object
    created using SYCL interoperability layer with underlying backends. Only the
    OpenCL and Level-Zero backends are currently supported.

    SyclKernelBundle exposes the C API from
    ``dpctl_sycl_kernel_bundle_interface.h``. A SyclKernelBundle can be
    created from either a source string or a SPIR-V binary file.
    """

    @staticmethod
    cdef SyclKernelBundle _create(DPCTLSyclKernelBundleRef KBRef):
        cdef SyclKernelBundle ret = SyclKernelBundle.__new__(SyclKernelBundle)
        ret._kernel_bundle_ref = KBRef
        return ret

    def __dealloc__(self):
        DPCTLKernelBundle_Delete(self._kernel_bundle_ref)

    cdef DPCTLSyclKernelBundleRef get_kernel_bundle_ref(self):
        return self._kernel_bundle_ref

    cpdef SyclKernel get_sycl_kernel(self, str kernel_name):
        name = kernel_name.encode("utf8")
        return SyclKernel._create(
            DPCTLKernelBundle_GetKernel(self._kernel_bundle_ref, name),
            kernel_name
        )

    def has_sycl_kernel(self, str kernel_name):
        name = kernel_name.encode("utf8")
        return DPCTLKernelBundle_HasKernel(self._kernel_bundle_ref, name)

    def addressof_ref(self):
        """Returns the address of the C API DPCTLSyclKernelBundleRef pointer
        as a long.

        Returns:
            The address of the ``DPCTLSyclKernelBundleRef`` pointer used to
            create this :class:`dpctl.SyclKernelBundle` object cast to a
            ``size_t``.
        """
        return int(<size_t>self._kernel_bundle_ref)


cdef api DPCTLSyclKernelBundleRef SyclKernelBundle_GetKernelBundleRef(
    SyclKernelBundle kb
):
    """ C-API function to access opaque kernel bundle reference from
    Python object of type :class:`dpctl.program.SyclKernelBundle`.
    """
    return kb.get_kernel_bundle_ref()


cdef api SyclKernelBundle SyclKernelBundle_Make(DPCTLSyclKernelBundleRef KBRef):
    """
    C-API function to create :class:`dpctl.program.SyclKernelBundle`
    instance from opaque ``sycl::kernel_bundle<sycl::bundle_state::executable>``
    reference.
    """
    cdef DPCTLSyclKernelBundleRef copied_KBRef = DPCTLKernelBundle_Copy(KBRef)
    return SyclKernelBundle._create(copied_KBRef)


cdef class SpecializationConstant:
    """
    SpecializationConstant(spec_id, *args)

    Python class representing SYCL specialization constants that can be used
    when creating a :class:`dpctl.program.SyclKernelBundle` from SPIR-V.

    There are multiple ways to create a :class:`.SpecializationConstant`:

    - ``SpecializationConstant(spec_id, obj)``
      If the constructor is invoked with a single variadic argument, the
      argument is expected to either expose the Python buffer protocol or be
      coercible to a NumPy array. If the argument is coercible to a NumPy array
      or is one, it must have a supported data type (bool, integral, floating
      point, or void). The specialization constant will be constructed from the
      data in the buffer

    - ``SpecializationConstant(spec_id, dtype, obj)``
    If the constructor is invoked with two variadic arguments, and the first
    argument is a string, it is interpreted as a NumPy ``dtype`` string and the
    second argument will be coerced to a NumPy array with that data type.
    The data type specified by the first argument must be a supported data
    type (bool, integral, floating point, or void).

    - ``SpecializationConstant(spec_id, nbytes, raw_ptr)``
      If the constructor is invoked with two variadic arguments where both are
      integers, the first argument is interpreted as the number of bytes and
      the second argument is interpreted as a pointer to the data.

    Note that when constructing from a buffer, the
    :class:`.SpecializationConstant`, shares memory with the original object.
    Modifications to the original object's data after construction will be
    reflected when the :class:`.SpecializationConstant` is used to create a
    :class:`.SyclKernelBundle`. This is not the case when constructing from a
    raw pointer, as the data is copied.

    Args:
        spec_id (int):
            The SPIR-V specialization ID.
        args:
            Variadic argument, see class documentation.

    Raises:
        TypeError: In case of incorrect arguments given to constructor,
                   failure to coerce to a buffer, or unsupported data type when
                   coercing to a buffer.
        ValueError: If the provided object fails to construct a buffer.
    """

    cdef _spec_const _spec_const
    cdef Py_buffer _buffer

    def __cinit__(self, spec_id, *args):
        cdef int ret_code = 0
        cdef object target_obj = None

        if not isinstance(spec_id, numbers.Integral):
            raise TypeError(
                "Specialization constant ID must be of type `int`, got "
                f"{type(spec_id)}"
            )

        if len(args) == 0 or len(args) > 2:
            raise TypeError(
                f"Constructor takes 2 or 3 arguments, got {len(args)}."
            )

        self._spec_const.id = <uint32_t>spec_id

        if len(args) == 2:
            if (
                isinstance(args[0], numbers.Integral) and
                isinstance(args[1], numbers.Integral)
            ):
                target_obj = PyBytes_FromStringAndSize(
                    <const char *><size_t>args[1], <Py_ssize_t>args[0]
                )
            elif isinstance(args[0], str):
                target_obj = np.ascontiguousarray(args[1], dtype=args[0])

        elif len(args) == 1:
            target_obj = args[0]
            if not PyObject_CheckBuffer(target_obj):
                # attempt to coerce to a numpy array
                target_obj = np.ascontiguousarray(target_obj)
        else:
            raise TypeError(
                "Invalid arguments."
            )

        if isinstance(target_obj, np.ndarray):
            if target_obj.dtype.kind not in ("b", "i", "u", "f", "c", "V"):
                raise TypeError(
                    "Coercion of input to buffer resulted in an unsupported "
                    f"data type '{target_obj.dtype}'. When coercing objects, "
                    "`SpecializationConstant` expects the data to coerce to a "
                    "supported type: bool, integral, real or complex floating "
                    "point, or void. To pass arbitrary data, use a "
                    "`memoryview` or `bytes` object, or pass the pointer and "
                    "size directly."
                )

        ret_code = PyObject_GetBuffer(
            target_obj, &(self._buffer), PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS
        )
        if ret_code != 0:
            raise ValueError(
                "Failed to get buffer view for the provided object."
            )
        self._spec_const.value = <void*>self._buffer.buf
        self._spec_const.size = <size_t>self._buffer.len

    def __dealloc__(self):
        PyBuffer_Release(&(self._buffer))

    def __repr__(self):
        return f"SpecializationConstant({self._spec_const.id})"

    def __eq__(self, other):
        if not isinstance(other, SpecializationConstant):
            return False
        cdef SpecializationConstant _other = <SpecializationConstant>other
        if (
            self._spec_const.id != _other._spec_const.id or
            self._spec_const.size != _other._spec_const.size or
            self._spec_const.value != _other._spec_const.value
        ):
            return False
        return memcmp(
            self._spec_const.value,
            _other._spec_const.value,
            self._spec_const.size
        ) == 0

    @property
    def id(self):
        """Returns the specialization ID for this specialization constant."""
        return self._spec_const.id

    @property
    def size(self):
        """
        Returns the size in bytes of the data for this specialization constant.
        """
        return self._spec_const.size

    cdef size_t addressof(self):
        """
        Returns the address of the _spec_const for this
        :class:`.SpecializationConstant` cast to ``size_t``.
        """
        return <size_t>&(self._spec_const)


cpdef create_kernel_bundle_from_source(SyclQueue q, str src, str copts=""):
    """
        Creates a Sycl interoperability kernel bundle from an OpenCL source
        string.

        We use the ``DPCTLKernelBundle_CreateFromOCLSource()`` C API function
        to create a ``sycl::kernel_bundle<sycl::bundle_state::executable>``
        from an OpenCL source program that can contain multiple kernels.
        Note: This function is currently only supported for the OpenCL backend.

        Parameters:
            q (:class:`dpctl.SyclQueue`)
                The :class:`dpctl.SyclQueue` for which the
                :class:`.SyclKernelBundle` is going to be built.
            src (str)
                Source string for an OpenCL program.
            copts (str, optional)
                Optional compilation flags that will be used
                when compiling the kernel bundle. Default: ``""``.

        Returns:
            kernel_bundle (:class:`.SyclKernelBundle`)
                A :class:`.SyclKernelBundle` object wrapping the
                ``sycl::kernel_bundle<sycl::bundle_state::executable>``
                returned by the C API.

        Raises:
            SyclKernelBundleCompilationError
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
        raise SyclKernelBundleCompilationError()

    return SyclKernelBundle._create(KBref)


cpdef create_kernel_bundle_from_spirv(
    SyclQueue q, const unsigned char[:] IL, str copts=""
):
    """
        Creates a Sycl interoperability kernel bundle from an SPIR-V binary.

        We use the :c:func:`DPCTLKernelBundle_CreateFromOCLSpirv` C API function
        to create a ``sycl::kernel_bundle<sycl::bundle_state::executable>``
        object from an compiled SPIR-V binary file.

        Parameters:
            q (:class:`dpctl.SyclQueue`)
                The :class:`dpctl.SyclQueue` for which the
                :class:`.SyclKernelBundle` is going to be built.
            IL (bytes)
                SPIR-V binary IL file for an OpenCL program.
            copts (str, optional)
                Optional compilation flags that will be used
                when compiling the kernel bundle. Default: ``""``.

        Returns:
            kernel_bundle (:class:`.SyclKernelBundle`)
                A :class:`.SyclKernelBundle` object wrapping the
                ``sycl::kernel_bundle<sycl::bundle_state::executable>``
                returned by the C API.

        Raises:
            SyclKernelBundleCompilationError
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
        raise SyclKernelBundleCompilationError()

    return SyclKernelBundle._create(KBref)


cpdef create_program_from_source(SyclQueue q, str src, str copts=""):
    """This function is a deprecated alias for
    :func:`dpctl.program.create_kernel_bundle_from_source`.
    New code should use :func:`dpctl.program.create_kernel_bundle_from_source`.
    """
    warnings.warn(
        "create_program_from_source is deprecated and will be removed in a "
        "future release. Use create_kernel_bundle_from_source instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_kernel_bundle_from_source(q, src, copts)


cpdef create_program_from_spirv(
    SyclQueue q, const unsigned char[:] IL, str copts=""
):
    """This function is a deprecated alias for
    :func:`dpctl.program.create_kernel_bundle_from_spirv`.
    New code should use :func:`dpctl.program.create_kernel_bundle_from_spirv`.
    """
    warnings.warn(
        "create_program_from_spirv is deprecated and will be removed in a "
        "future release. Use create_kernel_bundle_from_spirv instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_kernel_bundle_from_spirv(q, IL, copts)
