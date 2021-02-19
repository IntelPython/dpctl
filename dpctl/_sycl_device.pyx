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
    DPCTLAcceleratorSelector_Create,
    DPCTLCPUSelector_Create,
    DPCTLDefaultSelector_Create,
    DPCTLGPUSelector_Create,
    DPCTLHostSelector_Create,
    DPCTLCString_Delete,
    DPCTLDevice_Copy,
    DPCTLDevice_CreateFromSelector,
    DPCTLDevice_Delete,
    DPCTLDevice_DumpInfo,
    DPCTLDevice_GetVendorName,
    DPCTLDevice_GetName,
    DPCTLDevice_GetDriverInfo,
    DPCTLDevice_GetMaxComputeUnits,
    DPCTLDevice_GetMaxWorkItemDims,
    DPCTLDevice_GetMaxWorkItemSizes,
    DPCTLDevice_GetMaxWorkGroupSize,
    DPCTLDevice_GetMaxNumSubGroups,
    DPCTLDevice_HasInt64BaseAtomics,
    DPCTLDevice_HasInt64ExtendedAtomics,
    DPCTLDevice_IsAccelerator,
    DPCTLDevice_IsCPU,
    DPCTLDevice_IsGPU,
    DPCTLDevice_IsHost,
    DPCTLFilterSelector_Create,
    DPCTLDeviceSelector_Delete,
    DPCTLSize_t_Array_Delete,
    DPCTLSyclDeviceRef,
    DPCTLSyclDeviceSelectorRef,
)
from . import device_type

__all__ = [
    "SyclDevice",
    "select_accelerator_device",
    "select_cpu_device",
    "select_default_device",
    "select_gpu_device",
    "select_host_device",
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

    def dump_device_info(self):
        """ Print information about the SYCL device.
        """
        DPCTLDevice_DumpInfo(self._device_ref)

    cpdef get_device_name(self):
        """ Returns the name of the device as a string
        """
        return self._device_name.decode()

    cpdef get_device_type(self):
        """ Returns the type of the device as a `device_type` enum
        """
        if DPCTLDevice_IsGPU(self._device_ref):
            return device_type.gpu
        elif DPCTLDevice_IsCPU(self._device_ref):
            return device_type.cpu
        else:
            raise ValueError("Unknown device type.")

    cpdef get_vendor_name(self):
        """ Returns the device vendor name as a string
        """
        return self._vendor_name.decode()

    cpdef get_driver_version(self):
        """ Returns the OpenCL software driver version as a string
            in the form: major number.minor number, if this SYCL
            device is an OpenCL device. Returns a string class
            with the value "1.2" if this SYCL device is a host device.
        """
        return self._driver_version.decode()

    cpdef has_int64_base_atomics(self):
        """ Returns true if device has int64_base_atomics else returns false.
        """
        return self._int64_base_atomics

    cpdef has_int64_extended_atomics(self):
        """ Returns true if device has int64_extended_atomics else returns false.
        """
        return self._int64_extended_atomics

    cpdef get_max_compute_units(self):
        """ Returns the number of parallel compute units
            available to the device. The minimum value is 1.
        """
        return self._max_compute_units

    cpdef get_max_work_item_dims(self):
        """ Returns the maximum dimensions that specify
            the global and local work-item IDs used by the
            data parallel execution model. The minimum
            value is 3 if this SYCL device is not of device
            type ``info::device_type::custom``.
        """
        return self._max_work_item_dims

    cpdef get_max_work_item_sizes(self):
        """ Returns the maximum number of work-items
            that are permitted in each dimension of the
            work-group of the nd_range. The minimum
            value is (1; 1; 1) for devices that are not of
            device type ``info::device_type::custom``.
        """
        max_work_item_sizes = []
        for n in range(3):
            max_work_item_sizes.append(self._max_work_item_sizes[n])
        return tuple(max_work_item_sizes)

    cpdef get_max_work_group_size(self):
        """ Returns the maximum number of work-items
            that are permitted in a work-group executing a
            kernel on a single compute unit. The minimum
            value is 1.
        """
        return self._max_work_group_size

    cpdef get_max_num_sub_groups(self):
        """ Returns the maximum number of sub-groups
            in a work-group for any kernel executed on the
            device. The minimum value is 1.
        """
        return self._max_num_sub_groups

    cpdef is_accelerator(self):
        """ Returns True if the SyclDevice instance is a SYCL accelerator
        device.

        Returns:
            bool: True if the SyclDevice is a SYCL accelerator device,
            else False.
        """
        return self._accelerator_device

    cpdef is_cpu(self):
        """ Returns True if the SyclDevice instance is a SYCL CPU device.

        Returns:
            bool: True if the SyclDevice is a SYCL CPU device, else False.
        """
        return self._cpu_device

    cpdef is_gpu(self):
        """ Returns True if the SyclDevice instance is a SYCL GPU device.

        Returns:
            bool: True if the SyclDevice is a SYCL GPU device, else False.
        """
        return self._gpu_device

    cpdef is_host(self):
        """ Returns True if the SyclDevice instance is a SYCL host device.

        Returns:
            bool: True if the SyclDevice is a SYCL host device, else False.
        """
        return self._host_device

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
    def __name__(self):
        return "SyclDevice"

    def __repr__(self):
        return "<dpctl." + self.__name__ + " at {}>".format(hex(id(self)))


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
                level_zero_gpu.dump_device_info()

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
                gpu.dump_device_info()

    """
    @staticmethod
    cdef void _init_helper(SyclDevice device, DPCTLSyclDeviceRef DRef):
        device._device_ref = DRef
        device._device_name = DPCTLDevice_GetName(DRef)
        device._driver_version = DPCTLDevice_GetDriverInfo(DRef)
        device._int64_base_atomics = DPCTLDevice_HasInt64BaseAtomics(DRef)
        device._int64_extended_atomics = (
            DPCTLDevice_HasInt64ExtendedAtomics(DRef)
        )
        device._max_compute_units = DPCTLDevice_GetMaxComputeUnits(DRef)
        device._max_num_sub_groups = DPCTLDevice_GetMaxNumSubGroups(DRef)
        device._max_work_group_size = DPCTLDevice_GetMaxWorkGroupSize(DRef)
        device._max_work_item_dims = DPCTLDevice_GetMaxWorkItemDims(DRef)
        device._max_work_item_sizes = DPCTLDevice_GetMaxWorkItemSizes(DRef)
        device._vendor_name = DPCTLDevice_GetVendorName(DRef)
        device._accelerator_device = DPCTLDevice_IsAccelerator(DRef)
        device._cpu_device = DPCTLDevice_IsCPU(DRef)
        device._gpu_device = DPCTLDevice_IsGPU(DRef)
        device._host_device = DPCTLDevice_IsHost(DRef)

    @staticmethod
    cdef SyclDevice _create(DPCTLSyclDeviceRef dref):
        cdef SyclDevice ret = <SyclDevice>_SyclDevice.__new__(_SyclDevice)
        # Initialize the attributes of the SyclDevice object
        SyclDevice._init_helper(ret, dref)
        return SyclDevice(ret)

    cdef void _init_from__SyclDevice(self, _SyclDevice other):
        self._device_ref = DPCTLDevice_Copy(other._device_ref)
        self._device_name = DPCTLDevice_GetName(self._device_ref)
        self._driver_version = DPCTLDevice_GetDriverInfo(self._device_ref)
        self._int64_base_atomics = other._int64_base_atomics
        self._int64_extended_atomics = other._int64_extended_atomics
        self._max_compute_units = other._max_compute_units
        self._max_num_sub_groups = other._max_num_sub_groups
        self._max_work_group_size = other._max_work_group_size
        self._max_work_item_dims = other._max_work_item_dims
        self._max_work_item_sizes =  (
            DPCTLDevice_GetMaxWorkItemSizes(self._device_ref)
        )
        self._vendor_name = DPCTLDevice_GetVendorName(self._device_ref)
        self._accelerator_device = other._accelerator_device
        self._cpu_device = other._cpu_device
        self._gpu_device = other._gpu_device
        self._host_device = other._host_device

    cdef int _init_from_selector(self, DPCTLSyclDeviceSelectorRef DSRef):
        # Initialize the attributes of the SyclDevice object
        DRef = DPCTLDevice_CreateFromSelector(DSRef)
        if DRef is NULL:
            return -1
        else:
            SyclDevice._init_helper(self, DRef)
            return 0

    def __cinit__(self, arg=None):
        cdef DPCTLSyclDeviceSelectorRef DSRef = NULL
        cdef DPCTLSyclDeviceRef DRef = NULL
        cdef const char *filter_c_str = NULL
        cdef int ret = 0

        if type(arg) is unicode:
            string = bytes(<unicode>arg, "utf-8")
            filter_c_str = string
            DSRef = DPCTLFilterSelector_Create(filter_c_str)
            ret = self._init_from_selector(DSRef)
            if ret == -1:
                raise ValueError("Could not create a Device with the selector")
            # Free up the device selector
            DPCTLDeviceSelector_Delete(DSRef)
        elif isinstance(arg, unicode):
            string = bytes(unicode(arg), "utf-8")
            filter_c_str = <unicode>string
            DSRef = DPCTLFilterSelector_Create(filter_c_str)
            if ret == -1:
                raise ValueError("Could not create a Device with the selector")
            # Free up the device selector
            DPCTLDeviceSelector_Delete(DSRef)
        elif isinstance(arg, _SyclDevice):
            self._init_from__SyclDevice(arg)
        elif arg is None:
            DSRef = DPCTLDefaultSelector_Create()
            self._init_from_selector(DSRef)
        else:
            raise ValueError(
                "Invalid argument. Argument should be a str object specifying "
                "a SYCL filter selector string."
            )

    @property
    def __name__(self):
        return "SyclDevice"

    def __repr__(self):
        return "<dpctl." + self.__name__ + " at {}>".format(hex(id(self)))


cpdef select_accelerator_device():
    """ A wrapper for SYCL's `accelerator_selector` device_selector class.

    Returns:
        A new SyclDevice object containing the SYCL device returned by the
        `accelerator_selector`.
    Raises:
        A ValueError is raised if the SYCL `accelerator_selector` is unable to
        select a device.
    """
    cdef DPCTLSyclDeviceSelectorRef DSRef = DPCTLAcceleratorSelector_Create()
    cdef DPCTLSyclDeviceRef DRef = DPCTLDevice_CreateFromSelector(DSRef)
    # Free up the device selector
    DPCTLDeviceSelector_Delete(DSRef)
    if DRef is NULL:
        raise ValueError("Device unavailable.")
    # The _create call frees DSRef and DRef
    Device = SyclDevice._create(DRef)
    return Device


cpdef select_cpu_device():
    """ A wrapper for SYCL's `cpu_selector` device_selector class.

    Returns:
        A new SyclDevice object containing the SYCL device returned by the
        `cpu_selector`.
    Raises:
        A ValueError is raised if the SYCL `cpu_seector` is unable to select a
        device.
    """
    cdef DPCTLSyclDeviceSelectorRef DSRef = DPCTLCPUSelector_Create()
    cdef DPCTLSyclDeviceRef DRef = DPCTLDevice_CreateFromSelector(DSRef)
    # Free up the device selector
    DPCTLDeviceSelector_Delete(DSRef)
    if DRef is NULL:
        raise ValueError("Device unavailable.")
    # The _create call frees DSRef and DRef
    Device = SyclDevice._create(DRef)
    return Device


cpdef select_default_device():
    """ A wrapper for SYCL's `default_selector` device_selector class.

    Returns:
        A new SyclDevice object containing the SYCL device returned by the
        `default_selector`.
    Raises:
        A ValueError is raised if the SYCL `default_seector` is unable to
        select a device.
    """
    cdef DPCTLSyclDeviceSelectorRef DSRef = DPCTLDefaultSelector_Create()
    cdef DPCTLSyclDeviceRef DRef = DPCTLDevice_CreateFromSelector(DSRef)
    # Free up the device selector
    DPCTLDeviceSelector_Delete(DSRef)
    if DRef is NULL:
        raise ValueError("Device unavailable.")
    # The _create call frees DSRef and DRef
    Device = SyclDevice._create(DRef)
    return Device


cpdef select_gpu_device():
    """ A wrapper for SYCL's `gpu_selector` device_selector class.

    Returns:
        A new SyclDevice object containing the SYCL device returned by the
        `gpu_selector`.
    Raises:
        A ValueError is raised if the SYCL `gpu_seector` is unable to select a
        device.
    """
    cdef DPCTLSyclDeviceSelectorRef DSRef = DPCTLGPUSelector_Create()
    cdef DPCTLSyclDeviceRef DRef = DPCTLDevice_CreateFromSelector(DSRef)
    # Free up the device selector
    DPCTLDeviceSelector_Delete(DSRef)
    if DRef is NULL:
        raise ValueError("Device unavailable.")
    # The _create call frees DSRef and DRef
    Device = SyclDevice._create(DRef)
    return Device


cpdef select_host_device():
    """ A wrapper for SYCL's `host_selector` device_selector class.

    Returns:
        A new SyclDevice object containing the SYCL device returned by the
        `host_selector`.
    Raises:
        A ValueError is raised if the SYCL `host_seector` is unable to select a
        device.
    """
    cdef DPCTLSyclDeviceSelectorRef DSRef = DPCTLHostSelector_Create()
    cdef DPCTLSyclDeviceRef DRef = DPCTLDevice_CreateFromSelector(DSRef)
    # Free up the device selector
    DPCTLDeviceSelector_Delete(DSRef)
    if DRef is NULL:
        raise ValueError("Device unavailable.")
    # The _create call frees DSRef and DRef
    Device = SyclDevice._create(DRef)
    return Device
