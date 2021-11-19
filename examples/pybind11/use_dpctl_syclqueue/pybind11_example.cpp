//==- pybind11_example.cpp - Example of Pybind11 extension working with  -===//
//  dpctl Python objects.
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements Pybind11-generated extension exposing functions that
/// take dpctl Python objects, such as dpctl.SyclQueue, dpctl.SyclDevice as
/// arguments.
///
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <cstdint>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

// clang-format off
// Ordering of includes is important here. dpctl_sycl_types defines types
// used by dpctl's Python C-API headers.
#include "dpctl_sycl_types.h"
#include "../_sycl_queue.h"
#include "../_sycl_queue_api.h"
#include "../_sycl_device.h"
#include "../_sycl_device_api.h"
// clang-format on

namespace py = pybind11;

size_t get_max_compute_units(py::object queue)
{
    PyObject *queue_ptr = queue.ptr();
    if (PyObject_TypeCheck(queue_ptr, &PySyclQueueType)) {
        DPCTLSyclQueueRef QRef =
            get_queue_ref(reinterpret_cast<PySyclQueueObject *>(queue_ptr));
        sycl::queue *q = reinterpret_cast<sycl::queue *>(QRef);

        return q->get_device()
            .get_info<sycl::info::device::max_compute_units>();
    }
    else {
        throw std::runtime_error("expected dpctl.SyclQueue as argument");
    }
}

uint64_t get_device_global_mem_size(py::object device)
{
    PyObject *device_pycapi = device.ptr();
    if (PyObject_TypeCheck(device_pycapi, &PySyclDeviceType)) {
        DPCTLSyclDeviceRef DRef = get_device_ref(
            reinterpret_cast<PySyclDeviceObject *>(device_pycapi));
        sycl::device *d_ptr = reinterpret_cast<sycl::device *>(DRef);
        return d_ptr->get_info<sycl::info::device::global_mem_size>();
    }
    else {
        throw std::runtime_error("expected dpctl.SyclDevice as argument");
    }
}

uint64_t get_device_local_mem_size(py::object device)
{
    PyObject *device_pycapi = device.ptr();
    if (PyObject_TypeCheck(device_pycapi, &PySyclDeviceType)) {
        DPCTLSyclDeviceRef DRef = get_device_ref(
            reinterpret_cast<PySyclDeviceObject *>(device_pycapi));
        sycl::device *d_ptr = reinterpret_cast<sycl::device *>(DRef);
        return d_ptr->get_info<sycl::info::device::local_mem_size>();
    }
    else {
        throw std::runtime_error("expected dpctl.SyclDevice as argument");
    }
}

py::array_t<int64_t>
offloaded_array_mod(py::object queue,
                    py::array_t<int64_t, py::array::c_style> array,
                    int64_t mod)
{
    sycl::queue *q_ptr;

    PyObject *queue_pycapi = queue.ptr();
    if (PyObject_TypeCheck(queue_pycapi, &PySyclQueueType)) {
        DPCTLSyclQueueRef QRef =
            get_queue_ref(reinterpret_cast<PySyclQueueObject *>(queue_pycapi));
        q_ptr = reinterpret_cast<sycl::queue *>(QRef);
    }
    else {
        throw std::runtime_error("expected dpctl.SyclQueue as argument");
    }

    py::buffer_info arg_pybuf = array.request();
    if (arg_pybuf.ndim != 1) {
        throw std::runtime_error("Expecting a vector");
    }
    if (mod <= 0) {
        throw std::runtime_error("Modulus must be non-negative");
    }

    size_t n = arg_pybuf.size;

    auto res = py::array_t<int64_t>(n);
    py::buffer_info res_pybuf = res.request();

    int64_t *a = static_cast<int64_t *>(arg_pybuf.ptr);
    int64_t *r = static_cast<int64_t *>(res_pybuf.ptr);

    {
        const sycl::property_list props = {
            sycl::property::buffer::use_host_ptr()};
        sycl::buffer<int64_t, 1> a_buf(a, sycl::range<1>(n), props);
        sycl::buffer<int64_t, 1> r_buf(r, sycl::range<1>(n), props);

        q_ptr
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor a_acc(a_buf, cgh, sycl::read_only);
                sycl::accessor r_acc(r_buf, cgh, sycl::write_only,
                                     sycl::no_init);

                cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                    r_acc[idx] = a_acc[idx] % mod;
                });
            })
            .wait_and_throw();
    }

    return res;
}

PYBIND11_MODULE(pybind11_example, m)
{
    // Import the dpctl._sycl_queue, dpctl._sycl_device extensions
    import_dpctl___sycl_device();
    import_dpctl___sycl_queue();
    m.def("get_max_compute_units", &get_max_compute_units,
          "Computes max_compute_units property of the device underlying given "
          "dpctl.SyclQueue");
    m.def("get_device_global_mem_size", &get_device_global_mem_size,
          "Computes amount of global memory of the given dpctl.SyclDevice");
    m.def("get_device_local_mem_size", &get_device_local_mem_size,
          "Computes amount of local memory of the given dpctl.SyclDevice");
    m.def("offloaded_array_mod", &offloaded_array_mod,
          "Compute offloaded modular reduction of integer-valued NumPy array");
}
