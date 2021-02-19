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

from ._backend cimport *
from . import device_type

__all__ = [
    "SyclDevice",
]


cdef class SyclDevice:
    """ Python wrapper class for cl::sycl::device.
    """

    @staticmethod
    cdef SyclDevice _create (DPCTLSyclDeviceRef dref):
        cdef SyclDevice ret = SyclDevice.__new__(SyclDevice)
        ret._device_ref = dref
        ret._vendor_name = DPCTLDevice_GetVendorName(dref)
        ret._device_name = DPCTLDevice_GetName(dref)
        ret._driver_version = DPCTLDevice_GetDriverInfo(dref)
        ret._max_compute_units = DPCTLDevice_GetMaxComputeUnits(dref)
        ret._max_work_item_dims = DPCTLDevice_GetMaxWorkItemDims(dref)
        ret._max_work_item_sizes = DPCTLDevice_GetMaxWorkItemSizes(dref)
        ret._max_work_group_size = DPCTLDevice_GetMaxWorkGroupSize(dref)
        ret._max_num_sub_groups = DPCTLDevice_GetMaxNumSubGroups(dref)
        ret._int64_base_atomics = DPCTLDevice_HasInt64BaseAtomics(dref)
        ret._int64_extended_atomics = DPCTLDevice_HasInt64ExtendedAtomics(dref)
        return ret

    def __dealloc__ (self):
        DPCTLDevice_Delete(self._device_ref)
        DPCTLCString_Delete(self._device_name)
        DPCTLCString_Delete(self._vendor_name)
        DPCTLCString_Delete(self._driver_version)
        DPCTLSize_t_Array_Delete(self._max_work_item_sizes)

    def dump_device_info (self):
        """ Print information about the SYCL device.
        """
        DPCTLDevice_DumpInfo(self._device_ref)

    cpdef get_device_name (self):
        """ Returns the name of the device as a string
        """
        return self._device_name.decode()

    cpdef get_device_type (self):
        """ Returns the type of the device as a `device_type` enum
        """
        if DPCTLDevice_IsGPU(self._device_ref):
            return device_type.gpu
        elif DPCTLDevice_IsCPU(self._device_ref):
            return device_type.cpu
        else:
            raise ValueError("Unknown device type.")

    cpdef get_vendor_name (self):
        """ Returns the device vendor name as a string
        """
        return self._vendor_name.decode()

    cpdef get_driver_version (self):
        """ Returns the OpenCL software driver version as a string
            in the form: major number.minor number, if this SYCL
            device is an OpenCL device. Returns a string class
            with the value "1.2" if this SYCL device is a host device.
        """
        return self._driver_version.decode()

    cpdef has_int64_base_atomics (self):
        """ Returns true if device has int64_base_atomics else returns false.
        """
        return self._int64_base_atomics

    cpdef has_int64_extended_atomics (self):
        """ Returns true if device has int64_extended_atomics else returns false.
        """
        return self._int64_extended_atomics

    cpdef get_max_compute_units (self):
        """ Returns the number of parallel compute units
            available to the device. The minimum value is 1.
        """
        return self._max_compute_units

    cpdef get_max_work_item_dims (self):
        """ Returns the maximum dimensions that specify
            the global and local work-item IDs used by the
            data parallel execution model. The minimum
            value is 3 if this SYCL device is not of device
            type ``info::device_type::custom``.
        """
        return self._max_work_item_dims

    cpdef get_max_work_item_sizes (self):
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

    cpdef get_max_work_group_size (self):
        """ Returns the maximum number of work-items
            that are permitted in a work-group executing a
            kernel on a single compute unit. The minimum
            value is 1.
        """
        return self._max_work_group_size

    cpdef get_max_num_sub_groups (self):
        """ Returns the maximum number of sub-groups
            in a work-group for any kernel executed on the
            device. The minimum value is 1.
        """
        return self._max_num_sub_groups

    cdef DPCTLSyclDeviceRef get_device_ref (self):
        """ Returns the DPCTLSyclDeviceRef pointer for this class.
        """
        return self._device_ref

    def addressof_ref (self):
        """
        Returns the address of the DPCTLSyclDeviceRef pointer as a size_t.

        Returns:
            The address of the DPCTLSyclDeviceRef object used to create this
            SyclDevice cast to a size_t.
        """
        return int(<size_t>self._device_ref)
