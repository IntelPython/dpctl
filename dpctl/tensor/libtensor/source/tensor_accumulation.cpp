//===-- tensor_accumulation.cpp -                              --*-C++-*-/===//
//   Implementation of _tensor_accumulation_impl module
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
/// This file defines functions of dpctl.tensor._tensor_impl extensions
//===----------------------------------------------------------------------===//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "accumulators.hpp"
#include "accumulators/accumulators_common.hpp"

namespace py = pybind11;

namespace py_int = dpctl::tensor::py_internal;

using py_int::py_cumsum_1d;
using py_int::py_mask_positions;

PYBIND11_MODULE(_tensor_accumulation_impl, m)
{
    py_int::populate_mask_positions_dispatch_vectors();
    py_int::populate_cumsum_1d_dispatch_vectors();

    dpctl::tensor::py_internal::init_accumulator_functions(m);

    m.def("mask_positions", &py_mask_positions, "", py::arg("mask"),
          py::arg("cumsum"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());

    m.def("_cumsum_1d", &py_cumsum_1d, "", py::arg("src"), py::arg("cumsum"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());
}
