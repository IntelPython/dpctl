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

#include "dpctl4pybind11.hpp"
#include <CL/sycl.hpp>
#include <cstdint>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

uint64_t get_device_global_mem_size(sycl::device &d)
{
    return d.get_info<sycl::info::device::global_mem_size>();
}

uint64_t get_device_local_mem_size(sycl::device &d)
{
    return d.get_info<sycl::info::device::local_mem_size>();
}

py::array_t<int64_t>
offloaded_array_mod(sycl::queue q,
                    py::array_t<int64_t, py::array::c_style> array,
                    int64_t mod)
{
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

        q.submit([&](sycl::handler &cgh) {
             sycl::accessor a_acc(a_buf, cgh, sycl::read_only);
             sycl::accessor r_acc(r_buf, cgh, sycl::write_only, sycl::no_init);

             cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                 r_acc[idx] = a_acc[idx] % mod;
             });
         }).wait_and_throw();
    }

    return res;
}

std::vector<std::size_t> get_sub_group_sizes(const sycl::device &d)
{
    return d.get_info<sycl::info::device::sub_group_sizes>();
}

PYBIND11_MODULE(_use_queue_device, m)
{
    m.def(
        "get_max_compute_units",
        [=](sycl::queue q) -> size_t {
            sycl::device d = q.get_device();
            return d.get_info<sycl::info::device::max_compute_units>();
        },
        "Computes max_compute_units property of the device underlying given "
        "dpctl.SyclQueue");
    m.def("get_device_global_mem_size", &get_device_global_mem_size,
          "Computes amount of global memory of the given dpctl.SyclDevice");
    m.def("get_device_local_mem_size", &get_device_local_mem_size,
          "Computes amount of local memory of the given dpctl.SyclDevice");
    m.def("offloaded_array_mod", &offloaded_array_mod,
          "Compute offloaded modular reduction of integer-valued NumPy array");
    m.def("get_sub_group_sizes", &get_sub_group_sizes,
          "Gets info::device::sub_group_sizes property of given device");
}
