//===-- ------------ Implementation of _tensor_impl module  ----*-C++-*-/===//
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
//===--------------------------------------------------------------------===//
///
/// \file
/// This file defines functions of dpctl.tensor._tensor_impl extensions
//===--------------------------------------------------------------------===//

#pragma once
#include "utils/strided_iters.hpp"
#include <pybind11/pybind11.h>
#include <vector>

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

namespace py = pybind11;

void simplify_iteration_space(int &,
                              const py::ssize_t *&,
                              const py::ssize_t *&,
                              py::ssize_t,
                              bool,
                              bool,
                              const py::ssize_t *&,
                              py::ssize_t,
                              bool,
                              bool,
                              std::vector<py::ssize_t> &,
                              std::vector<py::ssize_t> &,
                              std::vector<py::ssize_t> &,
                              py::ssize_t &,
                              py::ssize_t &);

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
