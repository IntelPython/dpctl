#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2022 Intel Corporation
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

cimport libcpp.string

cimport dpctl
cimport dpctl.sycl


cdef extern from "utils.hpp":
    cdef libcpp.string.string get_device_name(dpctl.sycl.device)
    cdef libcpp.string.string get_device_driver_version(dpctl.sycl.device)
    cdef dpctl.sycl.device *copy_device(dpctl.sycl.device)


def device_name(dpctl.SyclDevice dev):
    cdef dpctl.DPCTLSyclDeviceRef d_ref = dev.get_device_ref()
    cdef const dpctl.sycl.device *dpcpp_device = dpctl.sycl.unwrap_device(d_ref)

    return get_device_name(dpcpp_device[0])


def device_driver_version(dpctl.SyclDevice dev):
    cdef dpctl.DPCTLSyclDeviceRef d_ref = dev.get_device_ref()
    cdef const dpctl.sycl.device *dpcpp_device = dpctl.sycl.unwrap_device(d_ref)

    return get_device_driver_version(dpcpp_device[0])


cpdef dpctl.SyclDevice device_copy(dpctl.SyclDevice dev):
    cdef dpctl.DPCTLSyclDeviceRef d_ref = dev.get_device_ref()
    cdef const dpctl.sycl.device *dpcpp_device = dpctl.sycl.unwrap_device(d_ref)
    cdef dpctl.sycl.device *copied_device = copy_device(dpcpp_device[0])
    cdef dpctl.DPCTLSyclDeviceRef copied_d_ref = dpctl.sycl.wrap_device(copied_device)

    return dpctl.SyclDevice._create(copied_d_ref)
