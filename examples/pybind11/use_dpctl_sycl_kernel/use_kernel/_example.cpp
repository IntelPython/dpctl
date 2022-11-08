//==- _example.cpp - Example of Pybind11 extension working with  =---------===//
//  dpctl Python objects.
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2022 Intel Corporation
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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

void submit_custom_kernel(sycl::queue q,
                          sycl::kernel krn,
                          dpctl::tensor::usm_ndarray x,
                          dpctl::tensor::usm_ndarray y,
                          const std::vector<sycl::event> &depends = {})
{
    if (x.get_ndim() != 1 || !x.is_c_contiguous() || y.get_ndim() != 1 ||
        !y.is_c_contiguous())
    {
        throw py::value_error(
            "src and dst arguments must be 1D and contiguous.");
    }

    auto const &api = dpctl::detail::dpctl_capi::get();
    if (x.get_typenum() != api.UAR_INT32_ || y.get_typenum() != api.UAR_INT32_)
    {
        throw py::value_error(
            "src and dst arguments must have int32 element data types.");
    }

    size_t n_x = x.get_size();
    size_t n_y = y.get_size();

    if (n_x != n_y) {
        throw py::value_error("src and dst arguments must have equal size.");
    }

    if (!dpctl::utils::queues_are_compatible(q, {x.get_queue(), y.get_queue()}))
    {
        throw std::runtime_error(
            "Execution queue is not compatible with allocation queues");
    }

    void *x_data = x.get_data<void>();
    void *y_data = y.get_data<void>();

    sycl::event e = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.set_arg(0, x_data);
        cgh.set_arg(1, y_data);
        cgh.parallel_for(sycl::range<1>(n_x), krn);
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
