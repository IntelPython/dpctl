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

""" This file declares the SyclDevice extension type.
"""

from ._backend cimport (
    DPCTLSyclDeviceRef,
    DPCTLSyclDeviceSelectorRef,
)


cdef class _SyclDevice:
    ''' Wrapper class for a Sycl Device
    '''
    cdef DPCTLSyclDeviceRef _device_ref
    cdef const char *_vendor_name
    cdef const char *_device_name
    cdef const char *_driver_version
    cdef size_t *_max_work_item_sizes


cdef class SyclDevice(_SyclDevice):
    @staticmethod
    cdef SyclDevice _create(DPCTLSyclDeviceRef dref)
    @staticmethod
    cdef void _init_helper(_SyclDevice device, DPCTLSyclDeviceRef DRef)
    cdef int _init_from__SyclDevice(self, _SyclDevice other)
    cdef int _init_from_selector(self, DPCTLSyclDeviceSelectorRef DSRef)
    cdef DPCTLSyclDeviceRef get_device_ref(self)
