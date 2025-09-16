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

#include "accumulators/accumulators_common.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_tensor_accumulation_impl, m, py::mod_gil_not_used())
{
    dpctl::tensor::py_internal::init_accumulator_functions(m);
}
