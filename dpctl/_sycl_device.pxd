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

""" This file declares the SyclDevice extension type.
"""

from libcpp cimport bool
from libc.stdint cimport uint32_t
from ._backend cimport (
    DPCTLSyclDeviceRef,
    DPCTLSyclDeviceSelectorRef,
)


cdef class _SyclDevice:
    ''' Wrapper class for a Sycl Device
    '''
    cdef DPCTLSyclDeviceRef _device_ref
    cdef bool _accelerator_device
    cdef bool _cpu_device
    cdef bool _gpu_device
    cdef bool _host_device
    cdef const char *_vendor_name
    cdef const char *_device_name
    cdef const char *_driver_version
    cdef uint32_t _max_compute_units
    cdef uint32_t _max_work_item_dims
    cdef size_t *_max_work_item_sizes
    cdef size_t _max_work_group_size
    cdef uint32_t _max_num_sub_groups
    cdef size_t _image_2d_max_width
    cdef size_t _image_2d_max_height
    cdef size_t _image_3d_max_width
    cdef size_t _image_3d_max_height
    cdef size_t _image_3d_max_depth
    cdef DPCTLSyclDeviceRef get_device_ref(self)
    cpdef get_backend(self)
    cpdef get_device_name(self)
    cpdef get_device_type(self)
    cpdef get_vendor_name(self)
    cpdef get_driver_version(self)
    cpdef get_max_compute_units(self)
    cpdef get_max_work_item_dims(self)
    cpdef get_max_work_item_sizes(self)
    cpdef get_max_work_group_size(self)
    cpdef get_max_num_sub_groups(self)
    cpdef get_image_2d_max_width(self)
    cpdef get_image_2d_max_height(self)
    cpdef get_image_3d_max_width(self)
    cpdef get_image_3d_max_height(self)
    cpdef get_image_3d_max_depth(self)
    cpdef is_accelerator(self)
    cpdef is_cpu(self)
    cpdef is_gpu(self)
    cpdef is_host(self)


cdef class SyclDevice(_SyclDevice):
    @staticmethod
    cdef SyclDevice _create(DPCTLSyclDeviceRef dref)
    @staticmethod
    cdef void _init_helper(SyclDevice device, DPCTLSyclDeviceRef DRef)
    cdef void _init_from__SyclDevice(self, _SyclDevice other)
    cdef int _init_from_selector(self, DPCTLSyclDeviceSelectorRef DSRef)

