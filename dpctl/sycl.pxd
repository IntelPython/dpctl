#  Copyright 2022-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# distutils: language = c++
# cython: language_level=3

# SYCL static imports for Cython

from . cimport _backend as dpctl_backend


cdef extern from "sycl/sycl.hpp" namespace "sycl":
    cdef cppclass queue "sycl::queue":
        pass

    cdef cppclass device "sycl::device":
        pass

    cdef cppclass context "sycl::context":
        pass

    cdef cppclass event "sycl::event":
        pass

    cdef cppclass kernel "sycl::kernel":
        pass

    cdef cppclass executable_kernel_bundle \
        "sycl::kernel_bundle<sycl::bundle_state::executable>":
        pass

cdef extern from "syclinterface/dpctl_sycl_type_casters.hpp" \
    namespace "dpctl::syclinterface":
    # queue
    cdef dpctl_backend.DPCTLSyclQueueRef wrap_queue \
        "dpctl::syclinterface::wrap<sycl::queue>" (const queue *)
    cdef queue * unwrap_queue "dpctl::syclinterface::unwrap<sycl::queue>" (
        dpctl_backend.DPCTLSyclQueueRef)

    # device
    cdef dpctl_backend.DPCTLSyclDeviceRef wrap_device \
        "dpctl::syclinterface::wrap<sycl::device>" (const device *)
    cdef device * unwrap_device "dpctl::syclinterface::unwrap<sycl::device>" (
        dpctl_backend.DPCTLSyclDeviceRef)

    # context
    cdef dpctl_backend.DPCTLSyclContextRef wrap_context \
        "dpctl::syclinterface::wrap<sycl::context>" (const context *)
    cdef context * unwrap_context "dpctl::syclinterface::unwrap<sycl::context>" (
        dpctl_backend.DPCTLSyclContextRef)

    # event
    cdef dpctl_backend.DPCTLSyclEventRef wrap_event \
        "dpctl::syclinterface::wrap<sycl::event>" (const event *)
    cdef event * unwrap_event "dpctl::syclinterface::unwrap<sycl::event>" (
        dpctl_backend.DPCTLSyclEventRef)
