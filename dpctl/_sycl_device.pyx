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

""" Implements SyclDevice Cython extension type.
"""

from ._backend cimport (  # noqa: E211
    DPCTLCString_Delete,
    DPCTLDefaultSelector_Create,
    DPCTLDevice_AreEq,
    DPCTLDevice_CanAccessPeer,
    DPCTLDevice_CanCompileOpenCL,
    DPCTLDevice_CanCompileSPIRV,
    DPCTLDevice_CanCompileSYCL,
    DPCTLDevice_Copy,
    DPCTLDevice_CreateFromSelector,
    DPCTLDevice_CreateSubDevicesByAffinity,
    DPCTLDevice_CreateSubDevicesByCounts,
    DPCTLDevice_CreateSubDevicesEqually,
    DPCTLDevice_Delete,
    DPCTLDevice_DisablePeerAccess,
    DPCTLDevice_EnablePeerAccess,
    DPCTLDevice_GetBackend,
    DPCTLDevice_GetComponentDevices,
    DPCTLDevice_GetCompositeDevice,
    DPCTLDevice_GetDeviceType,
    DPCTLDevice_GetDriverVersion,
    DPCTLDevice_GetGlobalMemCacheLineSize,
    DPCTLDevice_GetGlobalMemCacheSize,
    DPCTLDevice_GetGlobalMemCacheType,
    DPCTLDevice_GetGlobalMemSize,
    DPCTLDevice_GetImage2dMaxHeight,
    DPCTLDevice_GetImage2dMaxWidth,
    DPCTLDevice_GetImage3dMaxDepth,
    DPCTLDevice_GetImage3dMaxHeight,
    DPCTLDevice_GetImage3dMaxWidth,
    DPCTLDevice_GetLocalMemSize,
    DPCTLDevice_GetMaxClockFrequency,
    DPCTLDevice_GetMaxComputeUnits,
    DPCTLDevice_GetMaxMemAllocSize,
    DPCTLDevice_GetMaxNumSubGroups,
    DPCTLDevice_GetMaxReadImageArgs,
    DPCTLDevice_GetMaxWorkGroupSize,
    DPCTLDevice_GetMaxWorkItemDims,
    DPCTLDevice_GetMaxWorkItemSizes1d,
    DPCTLDevice_GetMaxWorkItemSizes2d,
    DPCTLDevice_GetMaxWorkItemSizes3d,
    DPCTLDevice_GetMaxWriteImageArgs,
    DPCTLDevice_GetName,
    DPCTLDevice_GetNativeVectorWidthChar,
    DPCTLDevice_GetNativeVectorWidthDouble,
    DPCTLDevice_GetNativeVectorWidthFloat,
    DPCTLDevice_GetNativeVectorWidthHalf,
    DPCTLDevice_GetNativeVectorWidthInt,
    DPCTLDevice_GetNativeVectorWidthLong,
    DPCTLDevice_GetNativeVectorWidthShort,
    DPCTLDevice_GetParentDevice,
    DPCTLDevice_GetPartitionMaxSubDevices,
    DPCTLDevice_GetPlatform,
    DPCTLDevice_GetPreferredVectorWidthChar,
    DPCTLDevice_GetPreferredVectorWidthDouble,
    DPCTLDevice_GetPreferredVectorWidthFloat,
    DPCTLDevice_GetPreferredVectorWidthHalf,
    DPCTLDevice_GetPreferredVectorWidthInt,
    DPCTLDevice_GetPreferredVectorWidthLong,
    DPCTLDevice_GetPreferredVectorWidthShort,
    DPCTLDevice_GetProfilingTimerResolution,
    DPCTLDevice_GetSubGroupIndependentForwardProgress,
    DPCTLDevice_GetSubGroupSizes,
    DPCTLDevice_GetVendor,
    DPCTLDevice_HasAspect,
    DPCTLDevice_Hash,
    DPCTLDevice_IsAccelerator,
    DPCTLDevice_IsCPU,
    DPCTLDevice_IsGPU,
    DPCTLDeviceMgr_GetDeviceInfoStr,
    DPCTLDeviceMgr_GetPositionInDevices,
    DPCTLDeviceMgr_GetRelativeId,
    DPCTLDeviceSelector_Delete,
    DPCTLDeviceSelector_Score,
    DPCTLDeviceVector_Delete,
    DPCTLDeviceVector_GetAt,
    DPCTLDeviceVector_Size,
    DPCTLDeviceVectorRef,
    DPCTLFilterSelector_Create,
    DPCTLSize_t_Array_Delete,
    DPCTLSyclDeviceRef,
    DPCTLSyclDeviceSelectorRef,
    DPCTLSyclPlatformRef,
    _aspect_type,
    _backend_type,
    _device_type,
    _global_mem_cache_type,
    _partition_affinity_domain_type,
    _peer_access,
)

from .enum_types import backend_type, device_type, global_mem_cache_type

from libc.stdint cimport int64_t, uint32_t, uint64_t
from libc.stdlib cimport free, malloc

from ._sycl_platform cimport SyclPlatform

import collections
import functools
import warnings

__all__ = [
    "SyclDevice", "SyclDeviceCreationError", "SyclSubDeviceCreationError",
]


cdef class SyclDeviceCreationError(Exception):
    """
    A ``SyclDeviceCreationError`` exception is raised when
    :class:`.SyclDevice` instance could not created.
    """
    pass


cdef class SyclSubDeviceCreationError(Exception):
    """
    A ``SyclSubDeviceCreationError`` exception is raised
    by :meth:`.SyclDevice.create_sub_devices` when
    :class:`.SyclDevice` instance could not be partitioned
    into sub-devices.
    """
    pass


cdef class _SyclDevice:
    """
    A helper data-owner class to abstract ``sycl::device``
    instance.
    """

    def __dealloc__(self):
        DPCTLDevice_Delete(self._device_ref)
        DPCTLCString_Delete(self._name)
        DPCTLCString_Delete(self._vendor)
        DPCTLCString_Delete(self._driver_version)
        DPCTLSize_t_Array_Delete(self._max_work_item_sizes)


cdef list _get_devices(DPCTLDeviceVectorRef DVRef):
    """
    Deletes DVRef. Pass a copy in case an original reference is needed.
    """
    cdef list devices = []
    cdef size_t nelems = 0
    if DVRef:
        nelems = DPCTLDeviceVector_Size(DVRef)
        for i in range(0, nelems):
            DRef = DPCTLDeviceVector_GetAt(DVRef, i)
            D = SyclDevice._create(DRef)
            devices.append(D)
        DPCTLDeviceVector_Delete(DVRef)

    return devices


cdef str _backend_type_to_filter_string_part(_backend_type BTy):
    if BTy == _backend_type._CUDA:
        return "cuda"
    elif BTy == _backend_type._HIP:
        return "hip"
    elif BTy == _backend_type._LEVEL_ZERO:
        return "level_zero"
    elif BTy == _backend_type._OPENCL:
        return "opencl"
    else:
        return "unknown"


cdef str _device_type_to_filter_string_part(_device_type DTy):
    if DTy == _device_type._ACCELERATOR:
        return "accelerator"
    elif DTy == _device_type._AUTOMATIC:
        return "automatic"
    elif DTy == _device_type._CPU:
        return "cpu"
    elif DTy == _device_type._GPU:
        return "gpu"
    else:
        return "unknown"


cdef void _init_helper(_SyclDevice device, DPCTLSyclDeviceRef DRef) except *:
    "Populate attributes of device from opaque device reference DRef"
    device._device_ref = DRef
    device._name = DPCTLDevice_GetName(DRef)
    if device._name is NULL:
        raise RuntimeError("Descriptor 'name' not available")
    device._driver_version = DPCTLDevice_GetDriverVersion(DRef)
    if device._driver_version is NULL:
        raise RuntimeError("Descriptor 'driver_version' not available")
    device._vendor = DPCTLDevice_GetVendor(DRef)
    if device._vendor is NULL:
        raise RuntimeError("Descriptor 'vendor' not available")
    device._max_work_item_sizes = DPCTLDevice_GetMaxWorkItemSizes3d(DRef)
    if device._max_work_item_sizes is NULL:
        raise RuntimeError("Descriptor 'max_work_item_sizes3d' not available")


cdef inline bint _check_peer_access(SyclDevice dev, SyclDevice peer) except *:
    """
    Check peer access ahead of time to avoid errors from unified runtime or
    compiler implementation.
    """
    cdef list _peer_access_backends = [
        _backend_type._CUDA,
        _backend_type._HIP,
        _backend_type._LEVEL_ZERO
    ]
    cdef _backend_type BTy1 = DPCTLDevice_GetBackend(dev._device_ref)
    cdef _backend_type BTy2 = DPCTLDevice_GetBackend(peer.get_device_ref())
    if (
        BTy1 == BTy2 and
        BTy1 in _peer_access_backends and
        BTy2 in _peer_access_backends and
        dev != peer
    ):
        return True
    return False


cdef inline void _raise_invalid_peer_access(
    SyclDevice dev,
    SyclDevice peer,
) except *:
    """
    Check peer access ahead of time and raise errors for invalid cases.
    """
    cdef list _peer_access_backends = [
        _backend_type._CUDA,
        _backend_type._HIP,
        _backend_type._LEVEL_ZERO
    ]
    cdef _backend_type BTy1 = DPCTLDevice_GetBackend(dev._device_ref)
    cdef _backend_type BTy2 = DPCTLDevice_GetBackend(peer.get_device_ref())
    if (BTy1 != BTy2):
        raise ValueError(
            f"Device with backend {_backend_type_to_filter_string_part(BTy1)} "
            "cannot peer access device with backend "
            f"{_backend_type_to_filter_string_part(BTy2)}"
        )
    if (BTy1 not in _peer_access_backends):
        raise ValueError(
            "Peer access not supported for backend "
            f"{_backend_type_to_filter_string_part(BTy1)}"
        )
    if (BTy2 not in _peer_access_backends):
        raise ValueError(
            "Peer access not supported for backend "
            f"{_backend_type_to_filter_string_part(BTy2)}"
        )
    if (dev == peer):
        raise ValueError(
            "Peer access cannot be enabled between a device and itself"
        )
    return


@functools.lru_cache(maxsize=None)
def _cached_filter_string(d : SyclDevice):
    """
    Internal utility to compute filter_string of input SyclDevice
    and cached with `functools.cache`.

    Args:
        d (:class:`dpctl.SyclDevice`):
            A device for which to compute the filter string.
    Returns:
        out(str):
            Filter string that can be used to create input device,
            if the device is a root (unpartitioned) device.

    Raises:
        ValueError: if the input device is a sub-device.
    """
    cdef _backend_type BTy
    cdef _device_type DTy
    cdef int64_t relId = -1
    cdef SyclDevice cd = <SyclDevice> d
    relId = DPCTLDeviceMgr_GetRelativeId(cd._device_ref)
    if (relId == -1):
        raise ValueError("This SyclDevice is not a root device")
    BTy = DPCTLDevice_GetBackend(cd._device_ref)
    br_str = _backend_type_to_filter_string_part(BTy)
    DTy = DPCTLDevice_GetDeviceType(cd._device_ref)
    dt_str = _device_type_to_filter_string_part(DTy)
    return ":".join((br_str, dt_str, str(relId)))


cdef class SyclDevice(_SyclDevice):
    """ SyclDevice(arg=None)
    A Python wrapper for the ``sycl::device`` C++ class.

    There are two ways of creating a SyclDevice instance:

    - by directly passing in a filter string to the class
      constructor. The filter string needs to conform to the
      :oneapi_filter_selection:`DPC++ filter selector SYCL extension <>`.

    :Example:

        .. code-block:: python

            import dpctl

            # Create a SyclDevice with an explicit filter string,
            # in this case the first level_zero gpu device.
            level_zero_gpu = dpctl.SyclDevice("level_zero:gpu:0")
            level_zero_gpu.print_device_info()

    - by calling one of the device selector helper functions:
      :py:func:`dpctl.select_accelerator_device()`,
      :py:func:`dpctl.select_cpu_device()`,
      :py:func:`dpctl.select_default_device()`,
      :py:func:`dpctl.select_gpu_device()`

    :Example:

        .. code-block:: python

            import dpctl

            # Create a SyclDevice of type GPU based on whatever is returned
            # by the SYCL `gpu_selector` device selector class.
            gpu = dpctl.select_gpu_device()
            gpu.print_device_info()

    Args:
        arg (str, optional):
            The argument can be a selector string, another
            :class:`dpctl.SyclDevice`, or ``None``.
            Defaults to ``None``.

    Raises:
        MemoryError:
            If the constructor could not allocate necessary
            temporary memory.
        SyclDeviceCreationError:
            If the :class:`dpctl.SyclDevice` object creation failed.
        TypeError:
            If the argument is not a :class:`dpctl.SyclDevice` or string.
    """
    @staticmethod
    cdef SyclDevice _create(DPCTLSyclDeviceRef dref):
        """
        This function calls DPCTLDevice_Delete(dref).

        The user of this function must pass a copy to keep the
        dref argument alive.
        """
        cdef _SyclDevice ret = _SyclDevice.__new__(_SyclDevice)
        # Initialize the attributes of the SyclDevice object
        _init_helper(<_SyclDevice> ret, dref)
        # ret is a temporary, and _SyclDevice.__dealloc__ will delete dref
        return SyclDevice(ret)

    cdef int _init_from__SyclDevice(self, _SyclDevice other):
        self._device_ref = DPCTLDevice_Copy(other._device_ref)
        if (self._device_ref is NULL):
            return -1
        self._name = DPCTLDevice_GetName(self._device_ref)
        self._driver_version = DPCTLDevice_GetDriverVersion(self._device_ref)
        self._max_work_item_sizes = (
            DPCTLDevice_GetMaxWorkItemSizes3d(self._device_ref)
        )
        self._vendor = DPCTLDevice_GetVendor(self._device_ref)
        return 0

    cdef int _init_from_selector(self, DPCTLSyclDeviceSelectorRef DSRef):
        # Initialize the attributes of the SyclDevice object
        cdef DPCTLSyclDeviceRef DRef = DPCTLDevice_CreateFromSelector(DSRef)
        # Free up the device selector
        DPCTLDeviceSelector_Delete(DSRef)
        if DRef is NULL:
            return -1
        else:
            _init_helper(self, DRef)
            return 0

    def __cinit__(self, arg=None):
        cdef DPCTLSyclDeviceSelectorRef DSRef = NULL
        cdef const char *filter_c_str = NULL
        cdef int ret = 0

        if type(arg) is str:
            string = bytes(<str>arg, "utf-8")
            filter_c_str = string
            DSRef = DPCTLFilterSelector_Create(filter_c_str)
            ret = self._init_from_selector(DSRef)
            if ret == -1:
                raise SyclDeviceCreationError(
                    "Could not create a SyclDevice with the selector string "
                    "'{selector_string}'".format(selector_string=arg)
                )
        elif isinstance(arg, _SyclDevice):
            ret = self._init_from__SyclDevice(arg)
            if ret == -1:
                raise SyclDeviceCreationError(
                    "Could not create a SyclDevice from _SyclDevice instance"
                )
        elif arg is None:
            DSRef = DPCTLDefaultSelector_Create()
            ret = self._init_from_selector(DSRef)
            if ret == -1:
                raise SyclDeviceCreationError(
                    "Could not create a SyclDevice from default selector"
                )
        else:
            raise TypeError(
                "Invalid argument. Argument should be a str object specifying "
                "a SYCL filter selector string or another SyclDevice."
            )

    def print_device_info(self):
        """
        Print information about the SYCL device.
        """
        cdef const char * info_str = DPCTLDeviceMgr_GetDeviceInfoStr(
            self._device_ref
        )
        py_info = <bytes> info_str
        DPCTLCString_Delete(info_str)
        print(py_info.decode("utf-8"))

    cdef DPCTLSyclDeviceRef get_device_ref(self):
        """
        Returns the :c:struct:`DPCTLSyclDeviceRef` pointer for this class.
        """
        return self._device_ref

    def addressof_ref(self):
        """
        Returns the address of the :c:struct:`DPCTLSyclDeviceRef` pointer as a
        ``size_t``.

        :Example:

            .. code-block:: python

                >>> import dpctl
                >>> dev = dpctl.select_cpu_device()
                >>> hex(dev.addressof_ref())
                '0x55b18ec649d0'

        Returns:
            int: The address of the :c:struct:`DPCTLSyclDeviceRef` object used
            to create this :class:`dpctl.SyclDevice` cast to a ``size_t``.
        """
        return <size_t>self._device_ref

    @property
    def backend(self):
        """Returns the ``backend_type`` enum value for this device

        :Example:

            .. code-block:: python

                >>> import dpctl
                >>> dev = dpctl.select_cpu_device()
                >>> dev.backend
                <backend_type.opencl: 4>

        Returns:
            backend_type:
                The backend for the device.
        """
        cdef _backend_type BTy = (
            DPCTLDevice_GetBackend(self._device_ref)
        )
        if BTy == _backend_type._CUDA:
            return backend_type.cuda
        elif BTy == _backend_type._HIP:
            return backend_type.hip
        elif BTy == _backend_type._LEVEL_ZERO:
            return backend_type.level_zero
        elif BTy == _backend_type._OPENCL:
            return backend_type.opencl
        else:
            raise ValueError("Unknown backend type.")

    @property
    def device_type(self):
        """ Returns the type of the device as a ``device_type`` enum.

        :Example:

            .. code-block:: python

                >>> import dpctl
                >>> dev = dpctl.select_cpu_device()
                >>> dev.device_type
                <device_type.cpu: 4>

        Returns:
            device_type:
                The type of device encoded as a ``device_type`` enum.

        Raises:
            ValueError:
                If the device type is not recognized.
        """
        cdef _device_type DTy = (
            DPCTLDevice_GetDeviceType(self._device_ref)
        )
        if DTy == _device_type._ACCELERATOR:
            return device_type.accelerator
        elif DTy == _device_type._AUTOMATIC:
            return device_type.automatic
        elif DTy == _device_type._CPU:
            return device_type.cpu
        elif DTy == _device_type._GPU:
            return device_type.gpu
        else:
            raise ValueError("Unknown device type.")

    @property
    def has_aspect_cpu(self):
        """ Returns ``True`` if this device is a CPU device,
        ``False`` otherwise.

        :Example:

            .. code-block:: python

                >>> import dpctl
                >>> dev = dpctl.select_cpu_device()
                >>> dev.has_aspect_cpu
                True

        Returns:
            bool:
                Indicates whether the device is a cpu.
        """
        cdef _aspect_type AT = _aspect_type._cpu
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_gpu(self):
        """ Returns ``True`` if this device is a GPU device,
        ``False`` otherwise.

        :Example:

            .. code-block:: python

                >>> import dpctl
                >>> dev = dpctl.select_cpu_device()
                >>> dev.has_aspect_gpu
                False

        Returns:
            bool:
                Indicates whether the device is a gpu.
        """
        cdef _aspect_type AT = _aspect_type._gpu
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_accelerator(self):
        """ Returns ``True`` if this device is an accelerator device,
        ``False`` otherwise.

        SYCL considers an accelerator to be a device that usually uses a
        peripheral interconnect for communication.

        :Example:

            .. code-block:: python

                >>> import dpctl
                >>> dev = dpctl.select_cpu_device()
                >>> dev.has_aspect_accelerator
                False

        Returns:
            bool:
                Indicates whether the device is an accelerator.
        """
        cdef _aspect_type AT = _aspect_type._accelerator
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_custom(self):
        """ Returns ``True`` if this device is a custom device,
        ``False`` otherwise.

        A custom device can be a dedicated accelerator that can use the
        SYCL API, but programmable kernels cannot be dispatched to the device,
        only fixed functionality is available. Refer SYCL spec for more details.

        :Example:

            .. code-block:: python

                >>> import dpctl
                >>> dev = dpctl.select_cpu_device()
                >>> dev.has_aspect_custom
                False

        Returns:
            bool:
                Indicates if the device is a custom SYCL device.
        """
        cdef _aspect_type AT = _aspect_type._custom
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_fp16(self):
        """ Returns ``True`` if the device supports half-precision floating
        point operations, ``False`` otherwise.

        :Example:

            .. code-block:: python

                >>> import dpctl
                >>> dev = dpctl.select_cpu_device()
                >>> dev.has_aspect_fp16
                True

        Returns:
            bool:
                Indicates that the device supports half precision floating
                point operations.
        """
        cdef _aspect_type AT = _aspect_type._fp16
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_fp64(self):
        """ Returns ``True`` if the device supports 64-bit precision floating
        point operations, ``False`` otherwise.

        :Example:

            .. code-block:: python

                >>> import dpctl
                >>> dev = dpctl.select_cpu_device()
                >>> dev.has_aspect_fp64
                True

        Returns:
            bool:
                Indicates that the device supports 64-bit precision floating
                point operations.
        """
        cdef _aspect_type AT = _aspect_type._fp64
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_atomic64(self):
        """ Returns ``True`` if the device supports a basic set of atomic
        operations, ``False`` otherwise.

        Indicates that the device supports the following atomic operations on
        64-bit values:

            - ``sycl::atomic_ref::load``
            - ``sycl::atomic_ref::store``
            - ``sycl::atomic_ref::fetch_add``
            - ``sycl::atomic_ref::fetch_sub``
            - ``sycl::atomic_ref::exchange``
            - ``sycl::atomic_ref::compare_exchange_strong``
            - ``sycl::atomic_ref::compare_exchange_weak``

        :Example:

            .. code-block:: python

                >>> import dpctl
                >>> dev = dpctl.select_cpu_device()
                >>> dev.has_aspect_atomic64
                True

        Returns:
            bool:
                Indicates that the device supports a basic set of atomic
                operations on 64-bit values.
        """
        cdef _aspect_type AT = _aspect_type._atomic64
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_image(self):
        """ Returns ``True`` if the device supports images, ``False`` otherwise
        (refer Sec 4.15.3 of SYCL 2020 spec).

        Returns:
            bool:
                Indicates that the device supports images
        """
        cdef _aspect_type AT = _aspect_type._image
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_online_compiler(self):
        """ Returns ``True`` if this device supports online compilation of
        device code, ``False`` otherwise.

        Returns:
            bool:
                Indicates that the device supports online compilation of
                device code.
        """
        cdef _aspect_type AT = _aspect_type._online_compiler
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_online_linker(self):
        """ Returns ``True`` if this device supports online linking of
        device code, ``False`` otherwise.

        Returns:
            bool:
                Indicates that the device supports online linking of device
                code.
        """
        cdef _aspect_type AT = _aspect_type._online_linker
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_queue_profiling(self):
        """ Returns ``True`` if this device supports queue profiling,
        ``False`` otherwise.

        Returns:
            bool:
                Indicates that the device supports queue profiling.
        """
        cdef _aspect_type AT = _aspect_type._queue_profiling
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_usm_device_allocations(self):
        """ Returns ``True`` if this device supports explicit USM allocations,
        ``False`` otherwise (refer Section 4.8 of SYCL 2020 specs).

        Returns:
            bool:
                Indicates that the device supports explicit USM allocations.
        """
        cdef _aspect_type AT = _aspect_type._usm_device_allocations
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_usm_host_allocations(self):
        """ Returns ``True`` if this device can access USM-host memory,
        ``False`` otherwise (refer Section 4.8 of SYCL 2020 specs).

        Returns:
            bool:
                Indicates that the device can access USM memory
                allocated using ``sycl::malloc_host``.
        """
        cdef _aspect_type AT = _aspect_type._usm_host_allocations
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_usm_shared_allocations(self):
        """ Returns ``True`` if this device supports USM-shared memory
        allocated on the same device, ``False`` otherwise.

        Returns:
            bool:
                Indicates that the device supports USM memory
                allocated using ``sycl::malloc_shared``.
        """
        cdef _aspect_type AT = _aspect_type._usm_shared_allocations
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_usm_system_allocations(self):
        """ Returns ``True`` if system allocator may be used instead of
        SYCL USM allocation mechanism for USM-shared allocations on this
        device, ``False`` otherwise.

        Returns:
            bool:
                Indicates that system allocator may be used instead of
                ``sycl::malloc_shared``.
        """
        cdef _aspect_type AT = _aspect_type._usm_system_allocations
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_usm_atomic_host_allocations(self):
        """ Returns ``True`` if this device supports USM-host allocations
        and the host and this device may concurrently access and atomically
        modify host allocations, ``False`` otherwise.

        Returns:
            bool:
                Indicates if the device supports USM atomic host allocations.
        """
        cdef _aspect_type AT = _aspect_type._usm_atomic_host_allocations
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_usm_atomic_shared_allocations(self):
        """ Returns ``True`` if this device supports USM-shared allocations
        and the host and other devices in the same context as this device may
        concurrently access and atomically modify shared allocations,
        ``False`` otherwise.

        Returns:
            bool:
                Indicates if this device supports concurrent atomic modification
                of USM-shared allocation by host and device.
        """
        cdef _aspect_type AT = _aspect_type._usm_atomic_shared_allocations
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_host_debuggable(self):
        """ Returns ``True`` if kernels running on this device can be debugged
        using standard debuggers that are normally available on the host
        system, ``False`` otherwise.

        Returns:
            bool:
                Indicates if host debugger may be used to debug device code.
        """
        cdef _aspect_type AT = _aspect_type._host_debuggable
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_emulated(self):
        """ Returns ``True`` if this device is somehow emulated, ``False``
        otherwise. A device with this aspect is not intended for performance,
        and instead will generally have another purpose such as emulation
        or profiling.

        Returns:
            bool:
                Indicates if device is somehow emulated.
        """
        cdef _aspect_type AT = _aspect_type._emulated
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_is_component(self):
        """ Returns ``True`` if this device is a component device, ``False``
        otherwise. A device with this aspect will have a composite device
        from which it is descended.

        Returns:
            bool:
                Indicates if device is a component device.
        """
        cdef _aspect_type AT = _aspect_type._is_component
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_is_composite(self):
        """ Returns ``True`` if this device is a composite device, ``False``
        otherwise. A device with this aspect contains component devices.

        Returns:
            bool:
                Indicates if device is a composite device.
        """
        cdef _aspect_type AT = _aspect_type._is_composite
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def image_2d_max_width(self):
        """ Returns the maximum width of a 2D image or 1D image in pixels.
            The minimum value is 8192 if the SYCL device has
            ``sycl::aspect::image``.

            Returns:
                int:
                    Maximum width of a 2D image or 1D image in pixels.
        """
        return DPCTLDevice_GetImage2dMaxWidth(self._device_ref)

    @property
    def image_2d_max_height(self):
        """ Returns the maximum height of a 2D image or 1D image in pixels.
            The minimum value is 8192 if the SYCL device has
            ``sycl::aspect::image``.

            Returns:
                int:
                    Maximum height of a 2D image or 1D image in pixels.
        """
        return DPCTLDevice_GetImage2dMaxHeight(self._device_ref)

    @property
    def image_3d_max_width(self):
        """ Returns the maximum width of a 3D image in pixels.
            The minimum value is 2048 if the SYCL device has
            ``sycl::aspect::image``.

            Returns:
                int:
                    Maximum width of a 3D image in pixels.
        """
        return DPCTLDevice_GetImage3dMaxWidth(self._device_ref)

    @property
    def image_3d_max_height(self):
        """ Returns the maximum height of a 3D image in pixels.
            The minimum value is 2048 if the SYCL device has
            ``sycl::aspect::image``.

            Returns:
                int:
                    Maximum height of a 3D image in pixels.
        """
        return DPCTLDevice_GetImage3dMaxHeight(self._device_ref)

    @property
    def image_3d_max_depth(self):
        """ Returns the maximum depth of a 3D image in pixels.
            The minimum value is 2048 if the SYCL device has
            ``sycl::aspect::image``.

            Returns:
                int:
                    Maximum depth of a 3D image in pixels.
        """
        return DPCTLDevice_GetImage3dMaxDepth(self._device_ref)

    @property
    def default_selector_score(self):
        """ Integral score assigned to this device by DPC++ runtime's default
        selector's scoring function. Score of -1 denotes that this device
        was rejected and may not be properly programmed by the DPC++ runtime.

        Returns:
            int:
                Score assign to this device by ``sycl::default_selector_v``
                function.
        """
        cdef DPCTLSyclDeviceSelectorRef DSRef = DPCTLDefaultSelector_Create()
        cdef int score = -1
        if (DSRef):
            score = DPCTLDeviceSelector_Score(DSRef, self._device_ref)
            DPCTLDeviceSelector_Delete(DSRef)
        return score

    @property
    def max_read_image_args(self):
        """ Returns the maximum number of simultaneous image objects that
        can be read from by a kernel. The minimum value is 128 if the
        SYCL device has ``sycl::aspect::image``.

        Returns:
            int:
                Maximum number of image objects that can be read from by
                a kernel.
        """
        return DPCTLDevice_GetMaxReadImageArgs(self._device_ref)

    @property
    def max_write_image_args(self):
        """ Returns the maximum number of simultaneous image objects that
        can be written to by a kernel. The minimum value is 8 if the SYCL
        device has ``sycl::aspect::image``.

        Return:
            int:
                Maximum number of simultaneous image objects that
                can be written to by a kernel.
        """
        return DPCTLDevice_GetMaxWriteImageArgs(self._device_ref)

    @property
    def is_accelerator(self):
        """ Returns ``True`` if this instance is a SYCL
        accelerator device.

        Returns:
            bool:
                ``True`` if the :class:`.SyclDevice` is a SYCL accelerator
                device, else ``False``.
        """
        return DPCTLDevice_IsAccelerator(self._device_ref)

    @property
    def is_cpu(self):
        """ Returns ``True`` if this instance is a SYCL CPU device.

        Returns:
            bool:
                ``True`` if the :class:`.SyclDevice` is a SYCL CPU device,
                else ``False``.
        """
        return DPCTLDevice_IsCPU(self._device_ref)

    @property
    def is_gpu(self):
        """ Returns ``True`` if this instance is a SYCL GPU device.

        Returns:
            bool:
                ``True`` if the :class:`.SyclDevice` is a SYCL GPU device,
                else ``False``.
        """
        return DPCTLDevice_IsGPU(self._device_ref)

    @property
    def max_work_item_dims(self):
        """ Returns the maximum dimensions that specify the global and local
        work-item IDs used by the data parallel execution model.

        Returns:
            int:
                The maximum number of work items supported by the device.
        """
        cdef uint32_t max_work_item_dims = 0
        max_work_item_dims = DPCTLDevice_GetMaxWorkItemDims(self._device_ref)
        return max_work_item_dims

    @property
    def max_work_item_sizes1d(self):
        """ Returns the maximum number of work-items that are permitted in each
        dimension of the work-group of the ``sycl::nd_range<1>``. The minimum
        value is ``(1, )`` for devices that evaluate to ``False`` for
        :py:attr:`~has_aspect_custom`.

        Returns:
            Tuple[int]:
                A one-tuple with the maximum allowed value for a 1D range
                used to enqueue a kernel on the device.
        """
        cdef size_t *max_work_item_sizes1d = NULL
        cdef size_t s0
        max_work_item_sizes1d = DPCTLDevice_GetMaxWorkItemSizes1d(
            self._device_ref
        )
        if max_work_item_sizes1d is NULL:
            raise RuntimeError("error obtaining 'max_work_item_sizes1d'")
        s0 = max_work_item_sizes1d[0]
        DPCTLSize_t_Array_Delete(max_work_item_sizes1d)
        return (s0, )

    @property
    def max_work_item_sizes2d(self):
        """ Returns the maximum number of work-items that are permitted in each
        dimension of the work-group of the ``sycl::nd_range<2>``. The minimum
        value is ``(1, 1,)`` for devices that evaluate to ``False`` for
        :py:attr:`~has_aspect_custom`.

        Returns:
            Tuple[int]:
                A two-tuple with the maximum allowed value for each
                dimension of a 2D range used to enqueue a kernel on the device.
        """
        cdef size_t *max_work_item_sizes2d = NULL
        cdef size_t s0
        cdef size_t s1
        max_work_item_sizes2d = DPCTLDevice_GetMaxWorkItemSizes2d(
            self._device_ref
        )
        if max_work_item_sizes2d is NULL:
            raise RuntimeError("error obtaining 'max_work_item_sizes2d'")
        s0 = max_work_item_sizes2d[0]
        s1 = max_work_item_sizes2d[1]
        DPCTLSize_t_Array_Delete(max_work_item_sizes2d)
        return (s0, s1,)

    @property
    def max_work_item_sizes3d(self):
        """ Returns the maximum number of work-items that are permitted in each
        dimension of the work-group of the ``sycl::nd_range<3>``. The minimum
        value is ``(1, 1, 1,)`` for devices that evaluate to ``False`` for
        :py:attr:`~has_aspect_custom`.

        Returns:
            Tuple[int]:
                A three-tuple with the maximum allowed value for
                each dimension of a 3D range used to enqueue a kernel on
                the device.
        """
        return (
            self._max_work_item_sizes[0],
            self._max_work_item_sizes[1],
            self._max_work_item_sizes[2],
        )

    @property
    def max_work_item_sizes(self):
        """ Returns the maximum number of work-items that are permitted in each
        dimension of the work-group of the nd_range. The minimum value is
        `(1; 1; 1)` for devices that evaluate to ``False`` for
        :py:attr:`~has_aspect_custom`.

        Returns:
            Tuple[int]:
                A three-tuple whose length depends on the number of
                work-group dimensions supported by the device.

        .. deprecated:: 0.14
           The property is deprecated use :py:attr:`~max_work_item_sizes3d`
           instead.
        """
        warnings.warn(
            "dpctl.SyclDevice.max_work_item_sizes is deprecated, "
            "use dpctl.SyclDevice.max_work_item_sizes3d instead",
            DeprecationWarning,
        )
        return (
            self._max_work_item_sizes[0],
            self._max_work_item_sizes[1],
            self._max_work_item_sizes[2],
        )

    @property
    def max_compute_units(self):
        """ Returns the number of parallel compute units available to the
        device. The minimum value is 1.

        Returns:
            int:
                The number of compute units in the device.
        """
        cdef uint32_t max_compute_units = 0
        max_compute_units = DPCTLDevice_GetMaxComputeUnits(self._device_ref)
        return max_compute_units

    @property
    def max_work_group_size(self):
        """ Returns the maximum number of work-items that are permitted in a
        work-group executing a kernel on a single compute unit. The minimum
        value is 1.

        Returns:
            int:
                The maximum supported work-group size.
        """
        cdef uint32_t max_work_group_size = 0
        max_work_group_size = DPCTLDevice_GetMaxWorkGroupSize(self._device_ref)
        return max_work_group_size

    @property
    def max_num_sub_groups(self):
        """ Returns the maximum number of sub-groups
        in a work-group for any kernel executed on the
        device. The minimum value is 1.

        Returns:
            int:
                The maximum number of sub-groups support per work-group by
                the device.
        """
        cdef uint32_t max_num_sub_groups = (
            DPCTLDevice_GetMaxNumSubGroups(self._device_ref)
        )
        return max_num_sub_groups

    @property
    def sub_group_independent_forward_progress(self):
        """ Returns ``True`` if the device supports independent forward progress
        of sub-groups with respect to other sub-groups in the same work-group.

        Returns:
            bool:
                Indicates if the device supports independent forward progress
                of sub-groups.
        """
        return DPCTLDevice_GetSubGroupIndependentForwardProgress(
            self._device_ref
        )

    @property
    def sub_group_sizes(self):
        """ Returns list of supported sub-group sizes for this device.

        :Example:

            .. code-block:: python

                >>> import dpctl
                >>> dev = dpctl.select_cpu_device()
                >>> dev.sub_group_sizes
                [4, 8, 16, 32, 64]

        Returns:
            List[int]:
                List of supported sub-group sizes.
        """
        cdef size_t *sg_sizes = NULL
        cdef size_t sg_sizes_len = 0
        cdef size_t i

        sg_sizes = DPCTLDevice_GetSubGroupSizes(
            self._device_ref, &sg_sizes_len)
        if (sg_sizes is not NULL and sg_sizes_len > 0):
            res = list()
            for i in range(sg_sizes_len):
                res.append(sg_sizes[i])
            DPCTLSize_t_Array_Delete(sg_sizes)
            return res
        else:
            return []

    @property
    def sycl_platform(self):
        """ Returns the platform associated with this device.

        Returns:
            :class:`dpctl.SyclPlatform`:
                The platform associated with this device.
        """
        cdef DPCTLSyclPlatformRef PRef = (
            DPCTLDevice_GetPlatform(self._device_ref)
        )
        if (PRef == NULL):
            raise RuntimeError("Could not get platform for device.")
        else:
            return SyclPlatform._create(PRef)

    @property
    def preferred_vector_width_char(self):
        """ Returns the preferred native vector width size for built-in scalar
        types that can be put into vectors.

        Returns:
            int:
                Preferred native vector width size for C type ``char``.

        :Example:

            .. code-block:: python

                import dpctl

                dev = dpctl.select_cpu_device()
                pvw_c = dev.preferred_vector_width_char
        """
        return DPCTLDevice_GetPreferredVectorWidthChar(self._device_ref)

    @property
    def preferred_vector_width_short(self):
        """ Returns the preferred native vector width size for built-in scalar
        types that can be put into vectors.

        Returns:
            int:
                Preferred native vector width size for C type ``short``.

        :Example:

            .. code-block:: python

                import dpctl

                dev = dpctl.select_cpu_device()
                pvw_s = dev.preferred_vector_width_short
        """
        return DPCTLDevice_GetPreferredVectorWidthShort(self._device_ref)

    @property
    def preferred_vector_width_int(self):
        """ Returns the preferred native vector width size for built-in scalar
        types that can be put into vectors.

        Returns:
            int:
                Preferred native vector width size for C type ``int``.

        :Example:

            .. code-block:: python

                import dpctl

                dev = dpctl.select_cpu_device()
                pvw_i = dev.preferred_vector_width_int
        """
        return DPCTLDevice_GetPreferredVectorWidthInt(self._device_ref)

    @property
    def preferred_vector_width_long(self):
        """ Returns the preferred native vector width size for built-in scalar
        types that can be put into vectors.

        Returns:
            int:
                Preferred native vector width size for C type ``long``.

        :Example:

            .. code-block:: python

                import dpctl

                dev = dpctl.select_cpu_device()
                pvw_l = dev.preferred_vector_width_long
        """
        return DPCTLDevice_GetPreferredVectorWidthLong(self._device_ref)

    @property
    def preferred_vector_width_float(self):
        """ Returns the preferred native vector width size for built-in scalar
        types that can be put into vectors.

        Returns:
            int:
                Preferred native vector width size for C type ``float``.

        :Example:

            .. code-block:: python

                import dpctl

                dev = dpctl.select_cpu_device()
                pvw_f = dev.preferred_vector_width_float
        """
        return DPCTLDevice_GetPreferredVectorWidthFloat(self._device_ref)

    @property
    def preferred_vector_width_double(self):
        """ Returns the preferred native vector width size for built-in scalar
        types that can be put into vectors.

        Returns:
            int:
                Preferred native vector width size for C type ``double``.

        If device does not support double-precision floating point operations,
        the native width is zero.

        :Example:

            .. code-block:: python

                import dpctl

                dev = dpctl.select_cpu_device()
                pvw_d = dev.preferred_vector_width_double
        """
        return DPCTLDevice_GetPreferredVectorWidthDouble(self._device_ref)

    @property
    def preferred_vector_width_half(self):
        """ Returns the preferred native vector width size for built-in scalar
        types that can be put into vectors.

        Returns:
            int:
                Preferred native vector width size for C type ``sycl::half``.

        If device does not support half-precision floating point operations,
        the native width is zero.
        """
        return DPCTLDevice_GetPreferredVectorWidthHalf(self._device_ref)

    @property
    def native_vector_width_char(self):
        """ Returns the native ISA vector width size for built-in scalar
        types that can be put into vectors.

        Returns:
            int:
                Native ISA vector width size for C type ``char``.

        :Example:

            .. code-block:: python

                import dpctl

                dev = dpctl.select_cpu_device()
                nvw_c = dev.native_vector_width_char
        """
        return DPCTLDevice_GetNativeVectorWidthChar(self._device_ref)

    @property
    def native_vector_width_short(self):
        """ Returns the native ISA vector width size for built-in scalar
        types that can be put into vectors.

        Returns:
            int:
                Native ISA vector width size for C type ``short``.

        :Example:

            .. code-block:: python

                import dpctl

                dev = dpctl.select_cpu_device()
                nvw_s = dev.native_vector_width_short
        """
        return DPCTLDevice_GetNativeVectorWidthShort(self._device_ref)

    @property
    def native_vector_width_int(self):
        """ Returns the native ISA vector width size for built-in scalar
        types that can be put into vectors.

        Returns:
            int:
                Native ISA vector width size for C type ``int``.

        :Example:

            .. code-block:: python

                import dpctl

                dev = dpctl.select_cpu_device()
                nvw_i = dev.native_vector_width_int
        """
        return DPCTLDevice_GetNativeVectorWidthInt(self._device_ref)

    @property
    def native_vector_width_long(self):
        """ Returns the native ISA vector width size for built-in scalar
        types that can be put into vectors.

        Returns:
            int:
                Native ISA vector width size for C type ``long``.

        :Example:

            .. code-block:: python

                import dpctl

                dev = dpctl.select_cpu_device()
                nvw_l = dev.native_vector_width_long
        """
        return DPCTLDevice_GetNativeVectorWidthLong(self._device_ref)

    @property
    def native_vector_width_float(self):
        """ Returns the native ISA vector width size for built-in scalar
        types that can be put into vectors.

        Returns:
            int:
                Native ISA vector width size for C type ``float``.

        :Example:

            .. code-block:: python

                import dpctl

                dev = dpctl.select_cpu_device()
                nvw_f = dev.native_vector_width_float
        """
        return DPCTLDevice_GetNativeVectorWidthFloat(self._device_ref)

    @property
    def native_vector_width_double(self):
        """ Returns the native ISA vector width size for built-in scalar
        types that can be put into vectors.

        Returns:
            int:
                Native ISA vector width size for C type ``double``.

        :Example:

            .. code-block:: python

                import dpctl

                dev = dpctl.select_cpu_device()
                nvw_d = dev.native_vector_width_double
        """
        return DPCTLDevice_GetNativeVectorWidthDouble(self._device_ref)

    @property
    def native_vector_width_half(self):
        """ Returns the native ISA vector width size for built-in scalar
        types that can be put into vectors.

        Returns:
            int:
                Native ISA vector width size for C type ``sycl::half``.
        """
        return DPCTLDevice_GetNativeVectorWidthHalf(self._device_ref)

    @property
    def global_mem_size(self):
        """ Returns the size of global memory on this device in bytes.

        Returns:
            int:
                Size of global memory in bytes.
        """
        cdef size_t global_mem_size = 0
        global_mem_size = DPCTLDevice_GetGlobalMemSize(self._device_ref)
        return global_mem_size

    @property
    def local_mem_size(self):
        """ Returns the size of local memory on this device in bytes.

        Returns:
            int:
                Size of global memory in bytes.
        """
        cdef size_t local_mem_size = 0
        local_mem_size = DPCTLDevice_GetLocalMemSize(self._device_ref)
        return local_mem_size

    @property
    def vendor(self):
        """ Returns the device vendor name as a string.

        Returns:
            str:
                The vendor name for the device as a string.
        """
        return self._vendor.decode()

    @property
    def driver_version(self):
        """ Returns a backend-defined driver version as a string.

        Returns:
            str:
                The driver version of the device as a string.
        """
        return self._driver_version.decode()

    @property
    def name(self):
        """ Returns the name of the device as a string

        Returns:
            str:
                The name of the device as a string.
        """
        return self._name.decode()

    @property
    def __name__(self):
        """ Returns the name of the class  :class:`dpctl.SyclDevice`

        Returns:
            str:
                Name of the class as a string.
        """
        return "SyclDevice"

    def __repr__(self):
        return (
            "<dpctl."
            + self.__name__
            + " ["
            + str(self.backend)
            + ", "
            + str(self.device_type)
            + ", "
            + " "
            + self.name
            + "] at {}>".format(hex(id(self)))
        )

    def __hash__(self):
        """Returns a hash value by hashing the underlying ``sycl::device``
        object.

        Returns:
            int:
                Hash value.
        """
        return DPCTLDevice_Hash(self._device_ref)

    cdef list create_sub_devices_equally(self, size_t count):
        """ Returns a list of sub-devices partitioned from this SYCL device
        based on the ``count`` parameter.

        The returned list contains as many sub-devices as can be created
        such that each sub-device contains count compute units. If the
        device’s total number of compute units is not evenly divided by
        count, then the remaining compute units are not included in any of
        the sub-devices.

        Args:
            count (int):
                Number of sub-devices to partition into.

        Returns:
            List[:class:`dpctl.SyclDevice`]:
                Created sub-devices.

        Raises:
            dpctl.SyclSubDeviceCreationError:
                if sub-devices can not be created.
        """
        cdef DPCTLDeviceVectorRef DVRef = NULL
        if count > 0:
            DVRef = DPCTLDevice_CreateSubDevicesEqually(self._device_ref, count)
        if DVRef is NULL:
            raise SyclSubDeviceCreationError(
                "Sub-devices were not created." if (count > 0) else
                "Sub-devices were not created, "
                "requested compute units count was zero."
            )
        return _get_devices(DVRef)

    cdef list create_sub_devices_by_counts(self, object counts):
        """ Returns a list of sub-devices partitioned from this SYCL device
        based on the ``counts`` parameter.

        For each non-zero value ``M`` in the counts vector, a sub-device
        with ``M`` compute units is created.

        Returns:
            List[:class:`dpctl.SyclDevice`]:
                Created sub-devices.

        Raises:
            dpctl.SyclSubDeviceCreationError:
                if sub-devices can not be created.
        """
        cdef int ncounts = len(counts)
        cdef size_t *counts_buff = NULL
        cdef size_t min_count = 1
        cdef DPCTLDeviceVectorRef DVRef = NULL
        cdef int i

        if ncounts == 0:
            raise ValueError(
                "Non-empty object representing list of counts is expected."
            )
        counts_buff = <size_t *> malloc((<size_t> ncounts) * sizeof(size_t))
        if counts_buff is NULL:
            raise MemoryError(
                "Allocation of counts array of size {} failed.".format(ncounts)
            )
        for i in range(ncounts):
            counts_buff[i] = counts[i]
            if counts_buff[i] == 0:
                min_count = 0
        if min_count:
            DVRef = DPCTLDevice_CreateSubDevicesByCounts(
                self._device_ref, counts_buff, ncounts
            )
        free(counts_buff)
        if DVRef is NULL:
            raise SyclSubDeviceCreationError(
                "Sub-devices were not created." if (min_count > 0) else
                "Sub-devices were not created, "
                "sub-device execution units counts must be positive."
            )
        return _get_devices(DVRef)

    cdef list create_sub_devices_by_affinity(
        self, _partition_affinity_domain_type domain
    ):
        """ Returns a list of sub-devices partitioned from this SYCL device by
        affinity domain based on the ``domain`` parameter.

        Returns:
            List[:class:`dpctl.SyclDevice`]:
                Created sub-devices.

        Raises:
            dpctl.SyclSubDeviceCreationError:
                if sub-devices can not be created.
        """
        cdef DPCTLDeviceVectorRef DVRef = NULL
        DVRef = DPCTLDevice_CreateSubDevicesByAffinity(self._device_ref, domain)
        if DVRef is NULL:
            raise SyclSubDeviceCreationError("Sub-devices were not created.")
        return _get_devices(DVRef)

    def create_sub_devices(self, **kwargs):
        """create_sub_devices(partition=parition_spec)
        Creates a list of sub-devices by partitioning a root device based on the
        provided partition specifier.

        A partition specifier must be provided using a ``partition``
        keyword argument. Possible values for the specifier are: an integer, a
        string specifying the affinity domain, or a collection of integers.

        :Example:
            .. code-block:: python

                import dpctl

                cpu_d = dpctl.SyclDevice("cpu")
                cpu_count = cpu_d.max_compute_units
                sub_devs = cpu_d.create_sub_devices(partition=cpu_count // 2)
                for d in sub_devs:
                    d.print_device_info()

                # Create sub-devices partitioning by affinity.
                try:
                    sd = cpu_d.create_sub_devices(partition="numa")
                    print(
                        "{0} sub-devices were created with respective "
                        "#EUs being {1}".format(
                            len(sd), [d.max_compute_units for d in sd]
                        )
                    )
                except Exception:
                    print("Device partitioning by affinity was not successful.")

        Args:
            partition (Union[int, str, List[int]]):
                Specification to partition the device as follows:

                - Specifying an int (``count``)
                    The returned list contains as
                    many sub-devices as can be created such that each
                    sub-device contains ``count`` compute units. If the
                    device’s total number of compute units is not evenly
                    divided by ``count``, then the remaining compute units
                    are not included in any of the sub-devices.

                - Specifying an affinity domain as a string
                    The supported values are: ``"numa"``, ``"L4_cache"``,
                    ``"L3_cache"``, ``"L2_cache"``, ``"L1_cache"``,
                    ``"next_partitionable"``.

                - Specifying a collection of integral values
                    For each non-zero value ``M`` in the collection, a
                    sub-device with ``M`` compute units is created.

        Returns:
            List[:class:`dpctl.SyclDevice`]:
                Created sub-devices.

        Raises:
            ValueError:
                If the ``partition`` keyword argument is not specified or
                the affinity domain string is not legal or is not one of the
                three supported options.
            dpctl.SyclSubDeviceCreationError:
                If sub-devices can not be created.
        """
        if "partition" not in kwargs:
            raise TypeError(
                "create_sub_devices(partition=parition_spec) is expected."
            )
        partition = kwargs.pop("partition")
        if kwargs:
            raise TypeError(
                "create_sub_devices(partition=parition_spec) is expected."
            )
        if isinstance(partition, int) and partition >= 0:
            return self.create_sub_devices_equally(partition)
        elif isinstance(partition, str):
            if partition == "not_applicable":
                domain_type = _partition_affinity_domain_type._not_applicable
            elif partition == "numa":
                domain_type = _partition_affinity_domain_type._numa
            elif partition == "L4_cache":
                domain_type = _partition_affinity_domain_type._L4_cache
            elif partition == "L3_cache":
                domain_type = _partition_affinity_domain_type._L3_cache
            elif partition == "L2_cache":
                domain_type = _partition_affinity_domain_type._L2_cache
            elif partition == "L1_cache":
                domain_type = _partition_affinity_domain_type._L1_cache
            elif partition == "next_partitionable":
                domain_type = (
                    _partition_affinity_domain_type._next_partitionable
                )
            else:
                raise ValueError(
                    "Partition affinity domain {} is not understood.".format(
                        partition
                    )
                )
            return self.create_sub_devices_by_affinity(domain_type)
        elif isinstance(partition, collections.abc.Sized) and isinstance(
            partition, collections.abc.Iterable
        ):
            return self.create_sub_devices_by_counts(partition)
        else:
            try:
                partition = int(partition)
            except Exception as e:
                raise TypeError(
                    "Unsupported type of sub-device argument"
                ) from e
            return self.create_sub_devices_equally(partition)

    @property
    def parent_device(self):
        """ Parent device for a sub-device, or None for a root device.

        Returns:
            dpctl.SyclDevice:
                A parent :class:`dpctl.SyclDevice` instance if the
                device is a sub-device, ``None`` otherwise.
        """
        cdef DPCTLSyclDeviceRef pDRef = NULL
        pDRef = DPCTLDevice_GetParentDevice(self._device_ref)
        if (pDRef is NULL):
            return None
        return SyclDevice._create(pDRef)

    @property
    def composite_device(self):
        """ The composite device for a component device, or ``None`` for a
        non-component device.

        Returns:
            dpctl.SyclDevice:
                The composite :class:`dpctl.SyclDevice` instance for a
                component device, or ``None`` for a non-component device.
        """
        cdef DPCTLSyclDeviceRef CDRef = NULL
        CDRef = DPCTLDevice_GetCompositeDevice(self._device_ref)
        if (CDRef is NULL):
            return None
        return SyclDevice._create(CDRef)

    def component_devices(self):
        """ Returns a list of component devices contained in this SYCL device.

        The returned list will be empty if this SYCL device is not a composite
        device, i.e., if `is_composite` is ``False``.

        Returns:
            List[:class:`dpctl.SyclDevice`]:
                List of component devices.

        Raises:
            ValueError:
                If the ``DPCTLDevice_GetComponentDevices`` call returned
                ``NULL`` instead of a ``DPCTLDeviceVectorRef`` object.
        """
        cdef DPCTLDeviceVectorRef cDVRef = NULL
        cDVRef = DPCTLDevice_GetComponentDevices(self._device_ref)
        if cDVRef is NULL:
            raise ValueError("Internal error: NULL device vector encountered")
        return _get_devices(cDVRef)

    def can_access_peer(self, peer, value="access_supported"):
        """ Returns ``True`` if this device (``self``) can enable peer access
        to USM device memory on ``peer``, ``False`` otherwise.

        If peer access is supported, it may be enabled by calling
        :meth:`.enable_peer_access`.

        For details, see
        :oneapi_peer_access:`DPC++ peer access SYCL extension <>`.

        Args:
            peer (:class:`dpctl.SyclDevice`):
                The :class:`dpctl.SyclDevice` instance to check for peer access
                by this device.
            value (str, optional):
                Specifies the kind of peer access being queried.

                The supported values are

                - ``"access_supported"``
                    Returns ``True`` if it is possible for this device to
                    enable peer access to USM device memory on ``peer``.

                - ``"atomics_supported"``
                    Returns ``True`` if it is possible for this device to
                    concurrently access and atomically modify USM device
                    memory on ``peer`` when enabled. Atomics must have
                    ``memory_scope::system`` when modifying memory on a peer
                    device.

                If ``False`` is returned, these operations result in
                undefined behavior.

                Default: ``"access_supported"``

        Returns:
            bool:
                ``True`` if the kind of peer access specified by ``value`` is
                supported between this device and ``peer``, otherwise ``False``.

        Raises:
            TypeError:
                If ``peer`` is not :class:`dpctl.SyclDevice`.
        """
        cdef SyclDevice p_dev

        if not isinstance(value, str):
            raise TypeError(
                f"Expected `value` to be of type str, got {type(value)}"
            )
        if value == "access_supported":
            access_type = _peer_access._access_supported
        elif value == "atomics_supported":
            access_type = _peer_access._atomics_supported
        else:
            raise ValueError(
                "`value` must be 'access_supported' or 'atomics_supported', "
                f"got {value}"
            )
        if not isinstance(peer, SyclDevice):
            raise TypeError(
                "peer device must be a `dpctl.SyclDevice`, got "
                f"{type(peer)}"
            )
        p_dev = <SyclDevice>peer
        if _check_peer_access(self, p_dev):
            return DPCTLDevice_CanAccessPeer(
                self._device_ref,
                p_dev.get_device_ref(),
                access_type
            )
        return False

    def enable_peer_access(self, peer):
        """ Enables this device (``self``) to access USM device allocations
        located on ``peer``.

        Peer access may be disabled by calling :meth:`.disable_peer_access`.

        For details, see
        :oneapi_peer_access:`DPC++ peer access SYCL extension <>`.

        Args:
            peer (:class:`dpctl.SyclDevice`):
                The :class:`dpctl.SyclDevice` instance to enable peer access
                to.

        Raises:
            TypeError:
                If ``peer`` is not :class:`dpctl.SyclDevice`.
            ValueError:
                If the backend associated with this device or ``peer`` does not
                support peer access.
        """
        cdef SyclDevice p_dev

        if not isinstance(peer, SyclDevice):
            raise TypeError(
                "peer device must be a `dpctl.SyclDevice`, got "
                f"{type(peer)}"
            )
        p_dev = <SyclDevice>peer
        _raise_invalid_peer_access(self, p_dev)
        DPCTLDevice_EnablePeerAccess(
            self._device_ref,
            p_dev.get_device_ref()
        )
        return

    def disable_peer_access(self, peer):
        """ Disables peer access to ``peer`` from this device (``self``).

        Peer access may be enabled by calling :meth:`.enable_peer_access`.

        For details, see
        :oneapi_peer_access:`DPC++ peer access SYCL extension <>`.

        Args:
            peer (:class:`dpctl.SyclDevice`):
                The :class:`dpctl.SyclDevice` instance to
                disable peer access to.

        Raises:
            TypeError:
                If ``peer`` is not :class:`dpctl.SyclDevice`.
            ValueError:
                If the backend associated with this device or ``peer`` does not
                support peer access.
        """
        cdef SyclDevice p_dev

        if not isinstance(peer, SyclDevice):
            raise TypeError(
                "peer device must be a `dpctl.SyclDevice`, got "
                f"{type(peer)}"
            )
        p_dev = <SyclDevice>peer
        _raise_invalid_peer_access(self, p_dev)
        DPCTLDevice_DisablePeerAccess(
            self._device_ref,
            p_dev.get_device_ref()
        )
        return

    @property
    def profiling_timer_resolution(self):
        """ Profiling timer resolution.

        Returns:
            int:
                The resolution of device timer in nanoseconds.
        """
        cdef size_t timer_res = 0
        timer_res = DPCTLDevice_GetProfilingTimerResolution(self._device_ref)
        if (timer_res == 0):
            raise RuntimeError("Failed to get device timer resolution.")
        return timer_res

    @property
    def max_clock_frequency(self):
        """ Maximal clock frequency in MHz.

        Returns:
            int: Frequency in MHz
        """
        cdef uint32_t clock_fr = DPCTLDevice_GetMaxClockFrequency(
            self._device_ref
        )
        return clock_fr

    @property
    def max_mem_alloc_size(self):
        """ Maximum size of memory object than can be allocated.

        Returns:
            int:
                Maximum size of memory object in bytes
        """
        cdef uint64_t max_alloc_sz = DPCTLDevice_GetMaxMemAllocSize(
            self._device_ref
        )
        return max_alloc_sz

    @property
    def global_mem_cache_type(self):
        """ Global device cache memory type.

        :Example:

            .. code-block:: python

                >>> import dpctl
                >>> dev = dpctl.select_cpu_device()
                >>> dev.global_mem_cache_type
                <global_mem_cache_type.read_write: 4>

        Returns:
            global_mem_cache_type:
                type of cache memory

        Raises:
            RuntimeError:
                If an unrecognized memory type is reported by runtime.
        """
        cdef _global_mem_cache_type gmcTy = (
           DPCTLDevice_GetGlobalMemCacheType(self._device_ref)
        )
        if gmcTy == _global_mem_cache_type._MEM_CACHE_TYPE_READ_WRITE:
            return global_mem_cache_type.read_write
        elif gmcTy == _global_mem_cache_type._MEM_CACHE_TYPE_READ_ONLY:
            return global_mem_cache_type.read_only
        elif gmcTy == _global_mem_cache_type._MEM_CACHE_TYPE_NONE:
            return global_mem_cache_type.none
        elif gmcTy == _global_mem_cache_type._MEM_CACHE_TYPE_INDETERMINATE:
            raise RuntimeError("Unrecognized global memory cache type reported")

    @property
    def global_mem_cache_size(self):
        """ Global device memory cache size.

        Returns:
            int:
                Cache size in bytes
        """
        cdef uint64_t cache_sz = DPCTLDevice_GetGlobalMemCacheSize(
            self._device_ref
        )
        return cache_sz

    @property
    def global_mem_cache_line_size(self):
        """ Global device memory cache line size.

        Returns:
            int:
                Cache size in bytes
        """
        cdef uint64_t cache_line_sz = DPCTLDevice_GetGlobalMemCacheLineSize(
            self._device_ref
        )
        return cache_line_sz

    @property
    def partition_max_sub_devices(self):
        """ The maximum number of sub-devices this :class:`dpctl.SyclDevice`
        instance can be partitioned into. The value returned cannot exceed the
        value returned by :attr:`dpctl.SyclDevice.max_compute_units`.

        Returns:
            int:
                The maximum number of sub-devices that can be created when this
                device is partitioned. Zero value indicates that device can not
                be partitioned.
        """
        cdef uint32_t max_part = DPCTLDevice_GetPartitionMaxSubDevices(
            self._device_ref
        )
        return max_part

    cdef cpp_bool equals(self, SyclDevice other):
        """ Returns ``True`` if the :class:`dpctl.SyclDevice` argument has the
        same _device_ref as this SyclDevice.

        Args:
            other (:class:`dpctl.SyclDevice`):
                A :class:`dpctl.SyclDevice` instance to
                compare against.

        Returns:
            bool:
                ``True`` if the devices point to the same underlying
                ``sycl::device``, otherwise ``False``.
        """
        return DPCTLDevice_AreEq(self._device_ref, other.get_device_ref())

    def __eq__(self, other):
        "Returns ``True`` if two devices are the same"
        if isinstance(other, SyclDevice):
            return self.equals(<SyclDevice> other)
        else:
            return False

    @property
    def filter_string(self):
        """ For a root device, returns a fully specified filter selector
        string ``"backend:device_type:relative_id"`` selecting the device.

        Returns:
            str:
                A Python string representing a filter selector string.

        Raises:
            ValueError:
                If the device is a sub-device.

        :Example:
            .. code-block:: python

                import dpctl

                # Create a SyclDevice with an explicit filter string,
                # in this case the first level_zero gpu device.
                level_zero_gpu = dpctl.SyclDevice("level_zero:gpu:0")
                # filter_string property should be "level_zero:gpu:0"
                dev = dpctl.SyclDevice(level_zero_gpu.filter_string)
                assert level_zero_gpu == dev
        """
        cdef DPCTLSyclDeviceRef pDRef = NULL
        pDRef = DPCTLDevice_GetParentDevice(self._device_ref)
        if (pDRef is NULL):
            return _cached_filter_string(self)
        else:
            # this a sub-device, free it, and raise an exception
            DPCTLDevice_Delete(pDRef)
            raise ValueError("This SyclDevice is not a root device")

    cdef int get_backend_and_device_type_ordinal(self):
        """ If this device is a root ``sycl::device``, returns the ordinal
        position of this device in the vector
        ``sycl::device::get_devices(device_type_of_this_device)``
        filtered to contain only devices with the same backend as this
        device.

        Returns -1 if the device is a sub-device, or the device could not
        be found in the vector.
        """
        cdef int64_t relId = DPCTLDeviceMgr_GetRelativeId(self._device_ref)
        return relId

    cdef int get_device_type_ordinal(self):
        """ If this device is a root ``sycl::device``, returns the ordinal
        position of this device in the vector
        ``sycl::device::get_devices(device_type_of_this_device)``

        Returns -1 if the device is a sub-device, or the device could not
        be found in the vector.
        """
        cdef _device_type DTy
        cdef int64_t relId = -1

        DTy = DPCTLDevice_GetDeviceType(self._device_ref)
        relId = DPCTLDeviceMgr_GetPositionInDevices(self._device_ref, DTy)
        return relId

    cdef int get_backend_ordinal(self):
        """ If this device is a root ``sycl::device``, returns the ordinal
        position of this device in the vector ``sycl::device::get_devices()``
        filtered to contain only devices with the same backend as this
        device.

        Returns -1 if the device is a sub-device, or the device could not
        be found in the vector.
        """
        cdef _backend_type BTy
        cdef int64_t relId = -1

        BTy = DPCTLDevice_GetBackend(self._device_ref)
        relId = DPCTLDeviceMgr_GetPositionInDevices(self._device_ref, BTy)
        return relId

    cdef int get_overall_ordinal(self):
        """ If this device is a root ``sycl::device``, returns the ordinal
        position of this device in the vector ``sycl::device::get_devices()``.

        Returns -1 if the device is a sub-device, or the device could not
        be found in the vector.
        """
        cdef int64_t relId = -1

        relId = DPCTLDeviceMgr_GetPositionInDevices(
            self._device_ref,
            (_backend_type._ALL_BACKENDS | _device_type._ALL_DEVICES)
        )
        return relId

    def get_filter_string(self, include_backend=True, include_device_type=True):
        """ get_filter_string(include_backend=True, include_device_type=True)

        For a parent device, returns a filter selector string
        that includes backend or device type based on the value
        of the given keyword arguments.

        Args:
            include_backend (bool, optional):
                A flag indicating if the backend should be included in
                the filter string. Default: ``True``.
            include_device_type (bool, optional):
                A flag indicating if the device type should be included
                in the filter string. Default: ``True``.

        Returns:
            str:
                A Python string representing a filter selector string.

        Raises:
            ValueError:
                If the device is a sub-device.

                If no match for the device was found in the vector
                returned by ``sycl::device::get_devices()``

        :Example:
            .. code-block:: python

                import dpctl

                # Create a GPU SyclDevice
                gpu_dev = dpctl.SyclDevice("gpu:0")
                # filter string should be "gpu:0"
                fs = gpu_dev.get_filter_string(use_backend=False)
                dev = dpctl.SyclDevice(fs)
                assert gpu _dev == dev
        """
        cdef int relId = -1
        cdef DPCTLSyclDeviceRef pDRef = NULL
        cdef _device_type DTy
        cdef _backend_type BTy

        if include_backend:
            if include_device_type:
                relId = self.get_backend_and_device_type_ordinal()
            else:
                relId = self.get_backend_ordinal()
        else:
            if include_device_type:
                relId = self.get_device_type_ordinal()
            else:
                relId = self.get_overall_ordinal()

        if relId < 0:
            pDRef = DPCTLDevice_GetParentDevice(self._device_ref)
            if (pDRef is NULL):
                raise ValueError
            else:
                # this a sub-device, free it, and raise an exception
                DPCTLDevice_Delete(pDRef)
                raise ValueError("This SyclDevice is not a root device")
        else:
            if include_backend:
                BTy = DPCTLDevice_GetBackend(self._device_ref)
                be_str = _backend_type_to_filter_string_part(BTy)
                if include_device_type:
                    DTy = DPCTLDevice_GetDeviceType(self._device_ref)
                    dt_str = _device_type_to_filter_string_part(DTy)
                    return ":".join((be_str, dt_str, str(relId)))
                else:
                    return ":".join((be_str, str(relId)))
            else:
                if include_device_type:
                    DTy = DPCTLDevice_GetDeviceType(self._device_ref)
                    dt_str = _device_type_to_filter_string_part(DTy)
                    return ":".join((dt_str, str(relId)))
                else:
                    return str(relId)

    def get_unpartitioned_parent_device(self):
        """ get_unpartitioned_parent_device()

        Returns the unpartitioned parent device of this device.

        If this device is already an unpartitioned, root device,
        the same device is returned.

        Returns:
            dpctl.SyclDevice:
                A parent, unpartitioned :class:`dpctl.SyclDevice` instance, or
                ``self`` if already a root device.
        """
        cdef DPCTLSyclDeviceRef pDRef = NULL
        cdef DPCTLSyclDeviceRef tDRef = NULL
        pDRef = DPCTLDevice_GetParentDevice(self._device_ref)
        if pDRef is NULL:
            return self
        else:
            tDRef = DPCTLDevice_GetParentDevice(pDRef)
            while tDRef is not NULL:
                DPCTLDevice_Delete(pDRef)
                pDRef = tDRef
                tDRef = DPCTLDevice_GetParentDevice(pDRef)
            return SyclDevice._create(pDRef)

    def get_device_id(self):
        """ get_device_id()
        For an unpartitioned device, returns the canonical index of this device
        in the list of devices visible to dpctl.

        Returns:
            int:
                The index of the device.

        Raises:
            ValueError:
                If the device could not be found.

        :Example:
            .. code-block:: python

                import dpctl
                gpu_dev = dpctl.SyclDevice("gpu")
                i = gpu_dev.get_device_id
                devs = dpctl.get_devices()
                assert devs[i] == gpu_dev
        """
        cdef int dev_id = -1
        cdef SyclDevice dev

        dev = self.get_unpartitioned_parent_device()
        dev_id = dev.get_overall_ordinal()
        if dev_id < 0:
            raise ValueError("device could not be found")
        return dev_id

    cpdef bint can_compile(self, str language):
        """
        Check whether it is possible to create an executable kernel_bundle
        for this device from the given source language.

        Parameters:
            language
                Input language. Possible values are "spirv" for SPIR-V binary
                files, "opencl" for OpenCL C device code and "sycl" for SYCL
                device code.

        Returns:
            bool:
                True if compilation is supported, False otherwise.

        Raises:
            ValueError:
                If an unknown source language is used.
        """
        if language == "spirv" or language == "spv":
            return DPCTLDevice_CanCompileSPIRV(self._device_ref)
        if language == "opencl" or language == "ocl":
            return DPCTLDevice_CanCompileOpenCL(self._device_ref)
        if language == "sycl":
            return DPCTLDevice_CanCompileSYCL(self._device_ref)

        raise ValueError(f"Unknown source language {language}")


cdef api DPCTLSyclDeviceRef SyclDevice_GetDeviceRef(SyclDevice dev):
    """
    C-API function to get opaque device reference from
    :class:`dpctl.SyclDevice` instance.
    """
    return dev.get_device_ref()


cdef api SyclDevice SyclDevice_Make(DPCTLSyclDeviceRef DRef):
    """
    C-API function to create :class:`dpctl.SyclDevice` instance
    from the given opaque device reference.
    """
    cdef DPCTLSyclDeviceRef copied_DRef = DPCTLDevice_Copy(DRef)
    return SyclDevice._create(copied_DRef)
