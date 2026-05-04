//==- _example.cpp - Example of Pybind11 extension working with  =---------===//
//  dpctl Python objects.
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2025 Intel Corporation
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
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sycl/sycl.hpp>
#include <vector>

namespace py = pybind11;

void submit_custom_kernel(sycl::queue &q,
                          sycl::kernel &krn,
                          dpctl::memory::usm_memory x,
                          dpctl::memory::usm_memory y,
                          const std::vector<sycl::event> &depends = {})
{
    const std::size_t nbytes_x = x.get_nbytes();
    const std::size_t nbytes_y = y.get_nbytes();

    if (nbytes_x != nbytes_y) {
        throw py::value_error("src and dst arguments must have equal nbytes.");
    }
    if (nbytes_x % sizeof(std::int32_t) != 0) {
        throw py::value_error("src and dst must be interpretable as int32 "
                              "(nbytes must be a multiple of 4).");
    }

    auto *x_data = reinterpret_cast<std::int32_t *>(x.get_pointer());
    auto *y_data = reinterpret_cast<std::int32_t *>(y.get_pointer());

    const std::size_t n_elems = nbytes_x / sizeof(std::int32_t);

    if (!dpctl::utils::queues_are_compatible(q, {x.get_queue(), y.get_queue()}))
    {
        throw std::runtime_error(
            "Execution queue is not compatible with allocation queues");
    }

    sycl::event e = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.set_arg(0, x_data);
        cgh.set_arg(1, y_data);
        cgh.parallel_for(sycl::range<1>(n_elems), krn);
    });

    e.wait();

    return;
}

PYBIND11_MODULE(_use_kernel, m)
{
    m.def("submit_custom_kernel", &submit_custom_kernel,
          "Submit given kernel with arguments (int *, int *) to queue",
          py::arg("queue"), py::arg("kernel"), py::arg("src"), py::arg("dst"),
          py::arg("depends") = py::list());
}
