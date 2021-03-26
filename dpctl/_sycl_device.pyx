#                      Data Parallel Control (dpCtl)
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

""" Implements SyclDevice Cython extension type.
"""

from ._backend cimport (
    _aspect_type,
    _backend_type,
    _device_type,
    DPCTLCString_Delete,
    DPCTLDefaultSelector_Create,
    DPCTLDevice_Copy,
    DPCTLDevice_CreateFromSelector,
    DPCTLDevice_Delete,
    DPCTLDevice_GetBackend,
    DPCTLDevice_GetDeviceType,
    DPCTLDevice_GetDriverInfo,
    DPCTLDevice_GetMaxComputeUnits,
    DPCTLDevice_GetMaxNumSubGroups,
    DPCTLDevice_GetMaxWorkGroupSize,
    DPCTLDevice_GetMaxWorkItemDims,
    DPCTLDevice_GetMaxWorkItemSizes,
    DPCTLDevice_GetVendorName,
    DPCTLDevice_GetName,
    DPCTLDevice_IsAccelerator,
    DPCTLDevice_IsCPU,
    DPCTLDevice_IsGPU,
    DPCTLDevice_IsHost,
    DPCTLDeviceMgr_PrintDeviceInfo,
    DPCTLFilterSelector_Create,
    DPCTLDeviceSelector_Delete,
    DPCTLDeviceSelector_Score,
    DPCTLSize_t_Array_Delete,
    DPCTLSyclBackendType,
    DPCTLSyclDeviceRef,
    DPCTLSyclDeviceSelectorRef,
    DPCTLDevice_HasAspect,
    DPCTLSyclDeviceType,
    DPCTLDevice_GetImage2dMaxWidth,
    DPCTLDevice_GetImage2dMaxHeight,
    DPCTLDevice_GetImage3dMaxWidth,
    DPCTLDevice_GetImage3dMaxHeight,
    DPCTLDevice_GetImage3dMaxDepth,
    DPCTLDevice_GetSubGroupIndependentForwardProgress,
    DPCTLDevice_GetPreferredVectorWidthChar,
    DPCTLDevice_GetPreferredVectorWidthShort,
    DPCTLDevice_GetPreferredVectorWidthInt,
    DPCTLDevice_GetPreferredVectorWidthLong,
    DPCTLDevice_GetPreferredVectorWidthFloat,
    DPCTLDevice_GetPreferredVectorWidthDouble,
    DPCTLDevice_GetPreferredVectorWidthHalf,
)
from . import backend_type, device_type
from libc.stdint cimport uint32_t
import warnings

__all__ = [
    "SyclDevice",
]


cdef class _SyclDevice:
    """ A helper metaclass to abstract a cl::sycl::device instance.
    """

    def __dealloc__(self):
        DPCTLDevice_Delete(self._device_ref)
        DPCTLCString_Delete(self._device_name)
        DPCTLCString_Delete(self._vendor_name)
        DPCTLCString_Delete(self._driver_version)
        DPCTLSize_t_Array_Delete(self._max_work_item_sizes)


cdef class SyclDevice(_SyclDevice):
    """ Python equivalent for cl::sycl::device class.

    There are two ways of creating a SyclDevice instance:

        - by directly passing in a filter string to the class constructor. The
        filter string needs to conform to the the `DPC++ filter selector SYCL
        extension <https://bit.ly/37kqANT>`_.

        :Example:
            .. code-block:: python

                import dpctl

                # Create a SyclDevice with an explicit filter string, in
                # this case the first level_zero gpu device.
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

    """
    @staticmethod
    cdef void _init_helper(SyclDevice device, DPCTLSyclDeviceRef DRef):
        device._device_ref = DRef
        device._device_name = DPCTLDevice_GetName(DRef)
        device._driver_version = DPCTLDevice_GetDriverInfo(DRef)
        device._vendor_name = DPCTLDevice_GetVendorName(DRef)
        device._max_work_item_sizes = DPCTLDevice_GetMaxWorkItemSizes(DRef)

    @staticmethod
    cdef SyclDevice _create(DPCTLSyclDeviceRef dref):
        cdef SyclDevice ret = <SyclDevice>_SyclDevice.__new__(_SyclDevice)
        # Initialize the attributes of the SyclDevice object
        SyclDevice._init_helper(ret, dref)
        return SyclDevice(ret)

    cdef int _init_from__SyclDevice(self, _SyclDevice other):
        self._device_ref = DPCTLDevice_Copy(other._device_ref)
        if (self._device_ref is NULL):
            return -1
        self._device_name = DPCTLDevice_GetName(self._device_ref)
        self._driver_version = DPCTLDevice_GetDriverInfo(self._device_ref)
        self._max_work_item_sizes =  (
            DPCTLDevice_GetMaxWorkItemSizes(self._device_ref)
        )
        self._vendor_name = DPCTLDevice_GetVendorName(self._device_ref)

    cdef int _init_from_selector(self, DPCTLSyclDeviceSelectorRef DSRef):
        # Initialize the attributes of the SyclDevice object
        cdef DPCTLSyclDeviceRef DRef = DPCTLDevice_CreateFromSelector(DSRef)
        if DRef is NULL:
            return -1
        else:
            SyclDevice._init_helper(self, DRef)
            return 0

    def __cinit__(self, arg=None):
        cdef DPCTLSyclDeviceSelectorRef DSRef = NULL
        cdef const char *filter_c_str = NULL
        cdef int ret = 0

        if type(arg) is unicode:
            string = bytes(<unicode>arg, "utf-8")
            filter_c_str = string
            DSRef = DPCTLFilterSelector_Create(filter_c_str)
            ret = self._init_from_selector(DSRef)
            if ret == -1:
                raise ValueError(
                    "Could not create a SyclDevice with the selector string"
                )
            # Free up the device selector
            DPCTLDeviceSelector_Delete(DSRef)
        elif isinstance(arg, unicode):
            string = bytes(<unicode>unicode(arg), "utf-8")
            filter_c_str = string
            DSRef = DPCTLFilterSelector_Create(filter_c_str)
            ret = self._init_from_selector(DSRef)
            if ret == -1:
                raise ValueError(
                    "Could not create a SyclDevice with the selector string"
                )
            # Free up the device selector
            DPCTLDeviceSelector_Delete(DSRef)
        elif isinstance(arg, _SyclDevice):
            ret = self._init_from__SyclDevice(arg)
            if ret == -1:
                raise ValueError(
                    "Could not create a SyclDevice from _SyclDevice instance"
                )
        elif arg is None:
            DSRef = DPCTLDefaultSelector_Create()
            ret = self._init_from_selector(DSRef)
            if ret == -1:
                raise ValueError(
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
        DPCTLDeviceMgr_PrintDeviceInfo(self._device_ref)

    cdef DPCTLSyclDeviceRef get_device_ref (self):
        """ Returns the DPCTLSyclDeviceRef pointer for this class.
        """
        return self._device_ref

    def addressof_ref(self):
        """
        Returns the address of the DPCTLSyclDeviceRef pointer as a size_t.

        Returns:
            int: The address of the DPCTLSyclDeviceRef object used to create
            this SyclDevice cast to a size_t.
        """
        return int(<size_t>self._device_ref)

    @property
    def backend(self):
        """Returns the backend_type enum value for this device

        Returns:
            backend_type: The backend for the device.
        """
        cdef DPCTLSyclBackendType BTy = (
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
        cdef DPCTLSyclDeviceType DTy = (
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
            return device_type.host_device
        else:
            raise ValueError("Unknown device type.")

    @property
    def has_aspect_host(self):
        cdef _aspect_type AT = _aspect_type._host
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_cpu(self):
        cdef _aspect_type AT = _aspect_type._cpu
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_gpu(self):
        cdef _aspect_type AT = _aspect_type._gpu
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_accelerator(self):
        cdef _aspect_type AT = _aspect_type._accelerator
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_custom(self):
        cdef _aspect_type AT = _aspect_type._custom
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_fp16(self):
        cdef _aspect_type AT = _aspect_type._fp16
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_fp64(self):
        cdef _aspect_type AT = _aspect_type._fp64
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_int64_base_atomics(self):
        cdef _aspect_type AT = _aspect_type._int64_base_atomics
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_int64_extended_atomics(self):
        cdef _aspect_type AT = _aspect_type._int64_extended_atomics
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_image(self):
        cdef _aspect_type AT = _aspect_type._image
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_online_compiler(self):
        cdef _aspect_type AT = _aspect_type._online_compiler
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_online_linker(self):
        cdef _aspect_type AT = _aspect_type._online_linker
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_queue_profiling(self):
        cdef _aspect_type AT = _aspect_type._queue_profiling
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_usm_device_allocations(self):
        cdef _aspect_type AT = _aspect_type._usm_device_allocations
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_usm_host_allocations(self):
        cdef _aspect_type AT = _aspect_type._usm_host_allocations
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_usm_shared_allocations(self):
        cdef _aspect_type AT = _aspect_type._usm_shared_allocations
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_usm_restricted_shared_allocations(self):
        cdef _aspect_type AT = _aspect_type._usm_restricted_shared_allocations
        return DPCTLDevice_HasAspect(self._device_ref, AT)

    @property
    def has_aspect_usm_system_allocator(self):
        cdef _aspect_type AT = _aspect_type._usm_system_allocator
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
        cdef DPCTLSyclDeviceSelectorRef DSRef = DPCTLDefaultSelector_Create()
        cdef int score = -1
        if (DSRef):
            score = DPCTLDeviceSelector_Score(DSRef, self._device_ref)
            DPCTLDeviceSelector_Delete(DSRef)
        return score

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
        """ Returns the maximum dimensions that specify
            the global and local work-item IDs used by the
            data parallel execution model. The cb
            value is 3 if this SYCL device is not of device
            type ``info::device_type::custom``.
        """
        cdef uint32_t max_work_item_dims = 0
        max_work_item_dims = DPCTLDevice_GetMaxWorkItemDims(self._device_ref)
        return max_work_item_dims

    @property
    def max_work_item_sizes(self):
        """ Returns the maximum number of work-items
            that are permitted in each dimension of the
            work-group of the nd_range. The minimum
            value is (1; 1; 1) for devices that are not of
            device type ``info::device_type::custom``.
        """
        return (
            self._max_work_item_sizes[0],
            self._max_work_item_sizes[1],
            self._max_work_item_sizes[2],
        )

    @property
    def max_compute_units(self):
        """ Returns the number of parallel compute units
            available to the device. The minimum value is 1.
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
        return DPCTLDevice_GetSubGroupIndependentForwardProgress(self._device_ref)

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
    def vendor_name(self):
        """ Returns the device vendor name as a string.
        """
        return self._vendor_name.decode()

    @property
    def driver_version(self):
        """ Returns a backend-defined driver version as a string.
        """
        return self._driver_version.decode()

    @property
    def device_name(self):
        """ Returns the name of the device as a string
        """
        return self._device_name.decode()

    @property
    def __name__(self):
        return "SyclDevice"

    def __repr__(self):
        return ("<dpctl." + self.__name__ + " [" +
                str(self.backend) + ", " + str(self.device_type) +", " +
                " " + self.device_name + "] at {}>".format(hex(id(self))) )
