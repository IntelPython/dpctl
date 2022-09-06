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
# cython: linetrace=True

""" Implements SyclDevice Cython extension type.
"""

from ._backend cimport (  # noqa: E211
    DPCTLCString_Delete,
    DPCTLDefaultSelector_Create,
    DPCTLDevice_AreEq,
    DPCTLDevice_Copy,
    DPCTLDevice_CreateFromSelector,
    DPCTLDevice_CreateSubDevicesByAffinity,
    DPCTLDevice_CreateSubDevicesByCounts,
    DPCTLDevice_CreateSubDevicesEqually,
    DPCTLDevice_Delete,
    DPCTLDevice_GetBackend,
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
    DPCTLDevice_GetMaxComputeUnits,
    DPCTLDevice_GetMaxNumSubGroups,
    DPCTLDevice_GetMaxReadImageArgs,
    DPCTLDevice_GetMaxWorkGroupSize,
    DPCTLDevice_GetMaxWorkItemDims,
    DPCTLDevice_GetMaxWorkItemSizes1d,
    DPCTLDevice_GetMaxWorkItemSizes2d,
    DPCTLDevice_GetMaxWorkItemSizes3d,
    DPCTLDevice_GetMaxWriteImageArgs,
    DPCTLDevice_GetName,
    DPCTLDevice_GetParentDevice,
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
    DPCTLDevice_GetVendor,
    DPCTLDevice_HasAspect,
    DPCTLDevice_Hash,
    DPCTLDevice_IsAccelerator,
    DPCTLDevice_IsCPU,
    DPCTLDevice_IsGPU,
    DPCTLDevice_IsHost,
    DPCTLDeviceMgr_GetDeviceInfoStr,
    DPCTLDeviceMgr_GetDevices,
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
)

from .enum_types import backend_type, device_type, global_mem_cache_type

from libc.stdint cimport int64_t, uint32_t, uint64_t
from libc.stdlib cimport free, malloc

from ._sycl_platform cimport SyclPlatform

import collections
import warnings

__all__ = [
    "SyclDevice", "SyclDeviceCreationError", "SyclSubDeviceCreationError",
]


cdef class SyclDeviceCreationError(Exception):
    """
    A SyclDeviceCreationError exception is raised when
    SyclDevice instance could not created.

    """
    pass


cdef class SyclSubDeviceCreationError(Exception):
    """
    A SyclSubDeviceCreationError exception is raised when
    sub-devices were not created.

    """
    pass


cdef class _SyclDevice:
    """
    A helper data-owner class to abstract a cl::sycl::device instance.
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
    elif BTy == _backend_type._HOST:
        return "host"
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
    elif DTy == _device_type._HOST_DEVICE:
        return "host"
    else:
        return "unknown"


cdef void _init_helper(_SyclDevice device, DPCTLSyclDeviceRef DRef):
    "Populate attributes of device from opaque device reference DRef"
    device._device_ref = DRef
    device._name = DPCTLDevice_GetName(DRef)
    device._driver_version = DPCTLDevice_GetDriverVersion(DRef)
    device._vendor = DPCTLDevice_GetVendor(DRef)
    device._max_work_item_sizes = DPCTLDevice_GetMaxWorkItemSizes3d(DRef)


cdef class SyclDevice(_SyclDevice):
    """ SyclDevice(arg=None)
    A Python wrapper for the :sycl_device:`sycl::device <>` C++ class.

    There are two ways of creating a SyclDevice instance:

        - by directly passing in a filter string to the class
          constructor. The filter string needs to conform to the
          :oneapi_filter_selection:`DPC++ filter selector SYCL extension <>`.

        :Example:
            .. code-block:: python

                import dpctl

                # Create a SyclDevice with an explicit filter string,
                # in this case the first level_zero gpu device.
                level_zero_gpu = dpctl.SyclDevice("level_zero:gpu:0"):
                level_zero_gpu.print_device_info()

        - by calling one of the device selector helper functions:

          :func:`dpctl.select_accelerator_device()`,
          :func:`dpctl.select_cpu_device()`,
          :func:`dpctl.select_default_device()`,
          :func:`dpctl.select_gpu_device()`,
          :func:`dpctl.select_host_device()`.


        :Example:
            .. code-block:: python

                import dpctl

                # Create a SyclDevice of type GPU based on whatever is returned
                # by the SYCL `gpu_selector` device selector class.
                gpu = dpctl.select_gpu_device():
                gpu.print_device_info()

    Args:
        arg (optional): The argument can be a selector string or None.
                        Defaults to ``None``.

    Raises:
        MemoryError: If the constructor could not allocate necessary
                     temporary memory.
        SyclDeviceCreationError: If the :class:`dpctl.SyclDevice` object
                                 creation failed.
        TypeError: If the list of :class:`dpctl.SyclDevice` objects was empty,
                   or the input capsule contained a null pointer or could not
                   be renamed.


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
            raise ValueError(
                "Invalid argument. Argument should be a str object specifying "
                "a SYCL filter selector string."
            )

    def print_device_info(self):
        """ Print information about the SYCL device.
        """
        cdef const char * info_str = DPCTLDeviceMgr_GetDeviceInfoStr(
            self._device_ref
        )
        py_info = <bytes> info_str
        DPCTLCString_Delete(info_str)
        print(py_info.decode("utf-8"))

    cdef DPCTLSyclDeviceRef get_device_ref(self):
        """ Returns the DPCTLSyclDeviceRef pointer for this class.
        """
        return self._device_ref

    def addressof_ref(self):
        """
        Returns the address of the ``DPCTLSyclDeviceRef`` pointer as a
        ``size_t``.

        Returns:
            int: The address of the ``DPCTLSyclDeviceRef`` object used to create
            this :class:`dpctl.SyclDevice` cast to a ``size_t``.
        """
        return <size_t>self._device_ref

    @property
    def backend(self):
        """Returns the backend_type enum value for this device

        Returns:
            backend_type: The backend for the device.
        """
        cdef _backend_type BTy = (
            DPCTLDevice_GetBackend(self._device_ref)
        )
        if BTy == _backend_type._CUDA:
            return backend_type.cuda
        elif BTy == _backend_type._HOST:
            return backend_type.host
        elif BTy == _backend_type._LEVEL_ZERO:
            return backend_type.level_zero
        elif BTy == _backend_type._OPENCL:
            return backend_type.opencl
        else:
            raise ValueError("Unknown backend type.")

    @property
    def device_type(self):
        """ Returns the type of the device as a `device_type` enum.

        Returns:
            device_type: The type of device encoded as a device_type enum.
        Raises:
            A ValueError is raised if the device type is not recognized.
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
        elif DTy == _device_type._HOST_DEVICE:
            return device_type.host
        else:
            raise ValueError("Unknown device type.")

    @property
    def has_aspect_host(self):
        "Returns True if this device is a host device, False otherwise"
        cdef _aspect_type AT = _aspect_type._host
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_cpu(self):
        "Returns True if this device is a CPU device, False otherwise"
        cdef _aspect_type AT = _aspect_type._cpu
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_gpu(self):
        "Returns True if this device is a GPU device, False otherwise"
        cdef _aspect_type AT = _aspect_type._gpu
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_accelerator(self):
        "Returns True if this device is an accelerator device, False otherwise"
        cdef _aspect_type AT = _aspect_type._accelerator
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_custom(self):
        "Returns True if this device is a custom device, False otherwise"
        cdef _aspect_type AT = _aspect_type._custom
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_fp16(self):
        """ Returns True if kernels submitted to this device
        may use 16-bit floating point types, False otherwise
        """
        cdef _aspect_type AT = _aspect_type._fp16
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_fp64(self):
        """  Returns True if kernels submitted to this device
        may use 64-bit floating point types, False otherwise
        """
        cdef _aspect_type AT = _aspect_type._fp64
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_atomic64(self):
        """ Returns True if kernels submitted to this device
        may perform 64-bit atomic operations, False otherwise
        """
        cdef _aspect_type AT = _aspect_type._atomic64
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_image(self):
        """ Returns True if this device supports images, False otherwise
        """
        cdef _aspect_type AT = _aspect_type._image
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_online_compiler(self):
        """ Returns True if this device supports online compilation of
        device code, False otherwise
        """
        cdef _aspect_type AT = _aspect_type._online_compiler
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_online_linker(self):
        """ Returns True if this device supports online linking of
        device code, False otherwise
        """
        cdef _aspect_type AT = _aspect_type._online_linker
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_queue_profiling(self):
        """ Returns True if this device supports queue profiling,
        False otherwise
        """
        cdef _aspect_type AT = _aspect_type._queue_profiling
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_usm_device_allocations(self):
        """ Returns True if this device supports explicit USM allocations,
        False otherwise
        """
        cdef _aspect_type AT = _aspect_type._usm_device_allocations
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_usm_host_allocations(self):
        """ Returns True if this device can access USM-host memory,
        False otherwise
        """
        cdef _aspect_type AT = _aspect_type._usm_host_allocations
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_usm_shared_allocations(self):
        """ Returns True if this device supports USM-shared memory
        allocated on the same device, False otherwise
        """
        cdef _aspect_type AT = _aspect_type._usm_shared_allocations
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_usm_restricted_shared_allocations(self):
        """ Deprecated property, do not use.
        """
        cdef _aspect_type AT = _aspect_type._usm_restricted_shared_allocations
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_usm_system_allocations(self):
        """ Returns True if system allocator may be used instead of SYCL USM
        allocation mechanism for USM-shared allocations on this device,
        False otherwise
        """
        cdef _aspect_type AT = _aspect_type._usm_system_allocations
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_usm_atomic_host_allocations(self):
        """ Returns True if this device supports USM-host allocations and
        the host and this device may concurrently access and atomically
        modify host allocations, False otherwise
        """
        cdef _aspect_type AT = _aspect_type._usm_atomic_host_allocations
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_usm_atomic_shared_allocations(self):
        """ Returns True if this device supports USM-shared allocations and
        the host and other devices in the same context as this device may
        concurrently access and atomically modify shared allocations,
        False otherwise
        """
        cdef _aspect_type AT = _aspect_type._usm_atomic_shared_allocations
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_host_debuggable(self):
        """ Returns True if kernels running on this device can be debugged
        using standard debuggers that are normally available on the host
        system, False otherwise
        """
        cdef _aspect_type AT = _aspect_type._host_debuggable
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def image_2d_max_width(self):
        """ Returns the maximum width of a 2D image or 1D image in pixels.
            The minimum value is 8192 if the SYCL device has aspect::image.
        """
        return DPCTLDevice_GetImage2dMaxWidth(self._device_ref)

    @property
    def image_2d_max_height(self):
        """ Returns the maximum height of a 2D image or 1D image in pixels.
            The minimum value is 8192 if the SYCL device has aspect::image.
        """
        return DPCTLDevice_GetImage2dMaxHeight(self._device_ref)

    @property
    def image_3d_max_width(self):
        """ Returns the maximum width of a 3D image in pixels.
            The minimum value is 2048 if the SYCL device has aspect::image.
        """
        return DPCTLDevice_GetImage3dMaxWidth(self._device_ref)

    @property
    def image_3d_max_height(self):
        """ Returns the maximum height of a 3D image in pixels.
            The minimum value is 2048 if the SYCL device has aspect::image.
        """
        return DPCTLDevice_GetImage3dMaxHeight(self._device_ref)

    @property
    def image_3d_max_depth(self):
        """ Returns the maximum depth of a 3D image in pixels.
            The minimum value is 2048 if the SYCL device has aspect::image.
        """
        return DPCTLDevice_GetImage3dMaxDepth(self._device_ref)

    @property
    def default_selector_score(self):
        """ Integer score assigned to this device by DPC++ runtime's default
        scoring function. Score of -1 denotes that this device was rejected
        and may not be properly programmed by the DPC++ runtime.
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
        SYCL device has aspect::image.
        """
        return DPCTLDevice_GetMaxReadImageArgs(self._device_ref)

    @property
    def max_write_image_args(self):
        """ Returns the maximum number of simultaneous image objects that
        can be written to by a kernel. The minimum value is 8 if the SYCL
        device has aspect::image.
        """
        return DPCTLDevice_GetMaxWriteImageArgs(self._device_ref)

    @property
    def is_accelerator(self):
        """ Returns True if the SyclDevice instance is a SYCL accelerator
        device.

        Returns:
            bool: True if the SyclDevice is a SYCL accelerator device,
            else False.
        """
        return DPCTLDevice_IsAccelerator(self._device_ref)

    @property
    def is_cpu(self):
        """ Returns True if the SyclDevice instance is a SYCL CPU device.

        Returns:
            bool: True if the SyclDevice is a SYCL CPU device, else False.
        """
        return DPCTLDevice_IsCPU(self._device_ref)

    @property
    def is_gpu(self):
        """ Returns True if the SyclDevice instance is a SYCL GPU device.

        Returns:
            bool: True if the SyclDevice is a SYCL GPU device, else False.
        """
        return DPCTLDevice_IsGPU(self._device_ref)

    @property
    def is_host(self):
        """ Returns True if the SyclDevice instance is a SYCL host device.

        Returns:
            bool: True if the SyclDevice is a SYCL host device, else False.
        """
        return DPCTLDevice_IsHost(self._device_ref)

    @property
    def max_work_item_dims(self):
        """ Returns the maximum dimensions that specify the global and local
        work-item IDs used by the data parallel execution model.
        """
        cdef uint32_t max_work_item_dims = 0
        max_work_item_dims = DPCTLDevice_GetMaxWorkItemDims(self._device_ref)
        return max_work_item_dims

    @property
    def max_work_item_sizes1d(self):
        """ Returns the maximum number of work-items that are permitted in each
        dimension of the work-group of the nd_range<1>. The minimum value is
        `(1 )` for devices that are not of device type
         ``info::device_type::custom``.
        """
        cdef size_t *max_work_item_sizes1d = NULL
        max_work_item_sizes1d = DPCTLDevice_GetMaxWorkItemSizes1d(
            self._device_ref
        )
        res = (max_work_item_sizes1d[0], )
        DPCTLSize_t_Array_Delete(max_work_item_sizes1d)
        return res

    @property
    def max_work_item_sizes2d(self):
        """ Returns the maximum number of work-items that are permitted in each
        dimension of the work-group of the nd_range<2>. The minimum value is
        `(1; 1)` for devices that are not of device type
         ``info::device_type::custom``.
        """
        cdef size_t *max_work_item_sizes2d = NULL
        max_work_item_sizes2d = DPCTLDevice_GetMaxWorkItemSizes2d(
            self._device_ref
        )
        res = (max_work_item_sizes2d[0], max_work_item_sizes2d[1],)
        DPCTLSize_t_Array_Delete(max_work_item_sizes2d)
        return res

    @property
    def max_work_item_sizes3d(self):
        """ Returns the maximum number of work-items that are permitted in each
        dimension of the work-group of the nd_range<3>. The minimum value is
        `(1; 1; 1)` for devices that are not of device type
         ``info::device_type::custom``.
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
        `(1; 1; 1)` for devices that are not of device type
        ``info::device_type::custom``.
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
        """
        cdef uint32_t max_compute_units = 0
        max_compute_units = DPCTLDevice_GetMaxComputeUnits(self._device_ref)
        return max_compute_units

    @property
    def max_work_group_size(self):
        """ Returns the maximum number of work-items
        that are permitted in a work-group executing a
        kernel on a single compute unit. The minimum
        value is 1.
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
            int: The maximum number of sub-groups support per work-group by
            the device.
        """
        cdef uint32_t max_num_sub_groups = 0
        if (not self.is_host):
            max_num_sub_groups = (
                DPCTLDevice_GetMaxNumSubGroups(self._device_ref)
            )
        return max_num_sub_groups

    @property
    def sub_group_independent_forward_progress(self):
        """ Returns true if the device supports independent forward progress of
        sub-groups with respect to other sub-groups in the same work-group.
        """
        return DPCTLDevice_GetSubGroupIndependentForwardProgress(
            self._device_ref
        )

    @property
    def sycl_platform(self):
        """ Returns the platform associated with this device.

        Returns:
            :class:`dpctl.SyclPlatform`: The platform associated with this
            device.
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
        """
        return DPCTLDevice_GetPreferredVectorWidthChar(self._device_ref)

    @property
    def preferred_vector_width_short(self):
        """ Returns the preferred native vector width size for built-in scalar
        types that can be put into vectors.
        """
        return DPCTLDevice_GetPreferredVectorWidthShort(self._device_ref)

    @property
    def preferred_vector_width_int(self):
        """ Returns the preferred native vector width size for built-in scalar
        types that can be put into vectors.
        """
        return DPCTLDevice_GetPreferredVectorWidthInt(self._device_ref)

    @property
    def preferred_vector_width_long(self):
        """ Returns the preferred native vector width size for built-in scalar
        types that can be put into vectors.
        """
        return DPCTLDevice_GetPreferredVectorWidthLong(self._device_ref)

    @property
    def preferred_vector_width_float(self):
        """ Returns the preferred native vector width size for built-in scalar
        types that can be put into vectors.
        """
        return DPCTLDevice_GetPreferredVectorWidthFloat(self._device_ref)

    @property
    def preferred_vector_width_double(self):
        """ Returns the preferred native vector width size for built-in scalar
        types that can be put into vectors.
        """
        return DPCTLDevice_GetPreferredVectorWidthDouble(self._device_ref)

    @property
    def preferred_vector_width_half(self):
        """ Returns the preferred native vector width size for built-in scalar
        types that can be put into vectors.
        """
        return DPCTLDevice_GetPreferredVectorWidthHalf(self._device_ref)

    @property
    def global_mem_size(self):
        """ Returns the size of global memory on this device in bytes.
        """
        cdef size_t global_mem_size = 0
        global_mem_size = DPCTLDevice_GetGlobalMemSize(self._device_ref)
        return global_mem_size

    @property
    def local_mem_size(self):
        """ Returns the size of local memory on this device in bytes.
        """
        cdef size_t local_mem_size = 0
        local_mem_size = DPCTLDevice_GetLocalMemSize(self._device_ref)
        return local_mem_size

    @property
    def vendor(self):
        """ Returns the device vendor name as a string.

        Returns:
            str: The vendor name for the device as a string.
        """
        return self._vendor.decode()

    @property
    def driver_version(self):
        """ Returns a backend-defined driver version as a string.

        Returns:
            str: The driver version of the device as a string.
        """
        return self._driver_version.decode()

    @property
    def name(self):
        """ Returns the name of the device as a string

        Returns:
            str: The name of the device as a string.
        """
        return self._name.decode()

    @property
    def __name__(self):
        "Name of the class `dpctl.SyclDevice`"
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
        """
        Returns a hash value by hashing the underlying ``sycl::device`` object.

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
        """
        cdef int ncounts = len(counts)
        cdef size_t *counts_buff = NULL
        cdef size_t min_count = 1
        cdef DPCTLDeviceVectorRef DVRef = NULL
        cdef int i

        if ncounts == 0:
            raise TypeError(
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
        """
        cdef DPCTLDeviceVectorRef DVRef = NULL
        DVRef = DPCTLDevice_CreateSubDevicesByAffinity(self._device_ref, domain)
        if DVRef is NULL:
            raise SyclSubDeviceCreationError("Sub-devices were not created.")
        return _get_devices(DVRef)

    def create_sub_devices(self, **kwargs):
        """create_sub_devices(partition=parition_spec)
        Creates a list of sub-devices by partitioning a root device based on the
        provided partion specifier.

        A partition specifier must be provided using a "partition"
        keyword argument. Possible values for the specifier are: an int, a
        string specifying the affinity domain, or a collection of ints.

        :Example:
            .. code-block:: python

                import dpctl

                cpu_d = dpctl.SyclDevice("cpu")
                cpu_count = cpu_d.max_compute_units
                sub_devs = cpu_d.create_sub_devices(partition=cpu_count // 2)
                for d in sub_devs:
                    d.print_device_info()

                #Create sub-devices partitioning by affinity.
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
            partition: Specification to partition the device as follows:

                - Specifying an int (`count`). The returned list contains as
                  many sub-devices as can be created such that each sub-device
                  contains `count` compute units. If the device’s total number
                  of compute units is not evenly divided by `count`, then the
                  remaining compute units are not included in any of the
                  sub-devices.

                - Specifying an affinity domain as a string. The supported
                  values are: `numa`, `L4_cache`, `L3_cache`, `L2_cache`,
                  `L1_cache`, `next_partitionable`.

                - Specifying a collection of int values. For each non-zero value
                  `M` in the collection, a sub-device with `M` compute units is
                  created.

        Returns:
            list: List of :class:`dpctl.SyclDevice` objects

        Raises:
            TypeError: If the "partition" keyword arg is not specified or the
                       affinity domain string is not legal or is not one of the
                       three supported options.

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
                raise TypeError(
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
                return self.create_sub_devices_equally(partition)
            except Exception as e:
                raise TypeError(
                    "Unsupported type of sub-device argument"
                ) from e

    @property
    def parent_device(self):
        """ Parent device for a sub-device, or None for a root device.

        Returns:
            dpctl.SyclDevice: A parent :class:`dpctl.SyclDevice` instance if the
            device is a sub-device, ``None`` otherwise.
        """
        cdef DPCTLSyclDeviceRef pDRef = NULL
        pDRef = DPCTLDevice_GetParentDevice(self._device_ref)
        if (pDRef is NULL):
            return None
        return SyclDevice._create(pDRef)

    @property
    def profiling_timer_resolution(self):
        """ Profiling timer resolution.

        Returns:
            int: The resolution of device timer in nanoseconds.
        """
        cdef size_t timer_res = 0
        timer_res = DPCTLDevice_GetProfilingTimerResolution(self._device_ref)
        if (timer_res == 0):
            raise RuntimeError("Failed to get device timer resolution.")
        return timer_res

    @property
    def global_mem_cache_type(self):
        """ Global device cache memory type.

        Returns:
            global_mem_cache_type: type of cache memory
        Raises:
            A RuntimeError is raised if an unrecognized memory type
            is reported by runtime.
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
            int: Cache size in bytes
        """
        cdef uint64_t cache_sz = DPCTLDevice_GetGlobalMemCacheSize(
            self._device_ref
	)
        return cache_sz

    @property
    def global_mem_cache_line_size(self):
        """ Global device memory cache line size.

        Returns:
            int: Cache size in bytes
        """
        cdef uint64_t cache_line_sz = DPCTLDevice_GetGlobalMemCacheLineSize(
            self._device_ref
	)
        return cache_line_sz

    cdef cpp_bool equals(self, SyclDevice other):
        """ Returns ``True`` if the :class:`dpctl.SyclDevice` argument has the
        same _device_ref as this SyclDevice.

        Args:
            other (dpctl.SyclDevice): A :class:`dpctl.SyclDevice` instance to
            compare against.

        Returns:
            bool: ``True`` if the devices point to the same underlying
            ``sycl::device``, otherwise ``False``.
        """
        return DPCTLDevice_AreEq(self._device_ref, other.get_device_ref())

    def __eq__(self, other):
        "Returns True if two devices are the same"
        if isinstance(other, SyclDevice):
            return self.equals(<SyclDevice> other)
        else:
            return False

    @property
    def filter_string(self):
        """ For a parent device, returns a fully specified filter selector
        string``backend:device_type:relative_id`` selecting the device.

        Returns:
            str: A Python string representing a filter selector string.

        Raises:
            TypeError: If the device is a sub-devices.

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
        cdef _backend_type BTy
        cdef _device_type DTy
        cdef int64_t relId = -1
        pDRef = DPCTLDevice_GetParentDevice(self._device_ref)
        if (pDRef is NULL):
            BTy = DPCTLDevice_GetBackend(self._device_ref)
            DTy = DPCTLDevice_GetDeviceType(self._device_ref)
            relId = DPCTLDeviceMgr_GetRelativeId(self._device_ref)
            if (relId == -1):
                raise TypeError("This SyclDevice is not a root device")
            br_str = _backend_type_to_filter_string_part(BTy)
            dt_str = _device_type_to_filter_string_part(DTy)
            return ":".join((br_str, dt_str, str(relId)))
        else:
            # this a sub-device, free it, and raise an exception
            DPCTLDevice_Delete(pDRef)
            raise TypeError("This SyclDevice is not a root device")

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
        position of this device in the vector ``sycl::device::get_devices()``
        filtered to contain only devices with the same backend as this
        device.

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
            include_backend (optional): Defaults to ``True``.
                A boolean flag indicating if the backend should be included in
                the filter string.
            include_device_type (optional): Defaults to ``True``.
                A boolean flag indicating if the device type should be included
                in the filter string.

        Returns:
            str: A Python string representing a filter selector string.

        Raises:
            TypeError: If the device is a sub-device.
            ValueError: If no match for the device was found in the vector
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
                raise TypeError("This SyclDevice is not a root device")
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
