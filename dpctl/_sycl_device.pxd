#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2024 Intel Corporation
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

from libcpp cimport bool as cpp_bool

from ._backend cimport (
    DPCTLSyclDeviceRef,
    DPCTLSyclDeviceSelectorRef,
    _partition_affinity_domain_type,
)


cdef public api class _SyclDevice [
    object Py_SyclDeviceObject,
    type Py_SyclDeviceType
]:
    """ A helper data-owner class to abstract a `sycl::device` instance.
    """
    cdef DPCTLSyclDeviceRef _device_ref
    cdef const char *_vendor
    cdef const char *_name
    cdef const char *_driver_version
    cdef size_t *_max_work_item_sizes


cdef public api class SyclDevice(_SyclDevice) [
    object PySyclDeviceObject,
    type PySyclDeviceType
]:
    @staticmethod
    cdef SyclDevice _create(DPCTLSyclDeviceRef dref)
    cdef int _init_from__SyclDevice(self, _SyclDevice other)
    cdef int _init_from_selector(self, DPCTLSyclDeviceSelectorRef DSRef)
    cdef DPCTLSyclDeviceRef get_device_ref(self)
    cdef list create_sub_devices_equally(self, size_t count)
    cdef list create_sub_devices_by_counts(self, object counts)
    cdef list create_sub_devices_by_affinity(self, _partition_affinity_domain_type domain)
    cdef cpp_bool equals(self, SyclDevice q)
    cdef int get_device_type_ordinal(self)
    cdef int get_overall_ordinal(self)
    cdef int get_backend_ordinal(self)
    cdef int get_backend_and_device_type_ordinal(self)
