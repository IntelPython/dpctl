//===-- ------------ Implementation of _tensor_impl module  ----*-C++-*-/===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2024 Intel Corporation
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
#include <pybind11/pybind11.h>
#include <vector>

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

namespace py = pybind11;

void simplify_iteration_space_1(int &,
                                const py::ssize_t *const &,
                                std::vector<py::ssize_t> const &,
                                std::vector<py::ssize_t> &,
                                std::vector<py::ssize_t> &,
                                py::ssize_t &);

void simplify_iteration_space(int &,
                              const py::ssize_t *const &,
                              std::vector<py::ssize_t> const &,
                              std::vector<py::ssize_t> const &,
                              std::vector<py::ssize_t> &,
                              std::vector<py::ssize_t> &,
                              std::vector<py::ssize_t> &,
                              py::ssize_t &,
                              py::ssize_t &);

void simplify_iteration_space_3(int &,
                                const py::ssize_t *const &,
                                // src1
                                std::vector<py::ssize_t> const &,
                                // src2
                                std::vector<py::ssize_t> const &,
                                // dst
                                std::vector<py::ssize_t> const &,
                                // output
                                std::vector<py::ssize_t> &,
                                std::vector<py::ssize_t> &,
                                std::vector<py::ssize_t> &,
                                std::vector<py::ssize_t> &,
                                py::ssize_t &,
                                py::ssize_t &,
                                py::ssize_t &);

void simplify_iteration_space_4(int &,
                                const py::ssize_t *const &,
                                // src1
                                std::vector<py::ssize_t> const &,
                                // src2
                                std::vector<py::ssize_t> const &,
                                // src3
                                std::vector<py::ssize_t> const &,
                                // dst
                                std::vector<py::ssize_t> const &,
                                // output
                                std::vector<py::ssize_t> &,
                                std::vector<py::ssize_t> &,
                                std::vector<py::ssize_t> &,
                                std::vector<py::ssize_t> &,
                                std::vector<py::ssize_t> &,
                                py::ssize_t &,
                                py::ssize_t &,
                                py::ssize_t &,
                                py::ssize_t &);

void compact_iteration_space(int &,
                             const py::ssize_t *const &,
                             std::vector<py::ssize_t> const &,
                             // output
                             std::vector<py::ssize_t> &,
                             std::vector<py::ssize_t> &);

void split_iteration_space(const std::vector<py::ssize_t> &,
                           const std::vector<py::ssize_t> &,
                           int,
                           int,
                           // output
                           std::vector<py::ssize_t> &,
                           std::vector<py::ssize_t> &,
                           std::vector<py::ssize_t> &,
                           std::vector<py::ssize_t> &);

py::ssize_t _ravel_multi_index_c(std::vector<py::ssize_t> const &,
                                 std::vector<py::ssize_t> const &);
py::ssize_t _ravel_multi_index_f(std::vector<py::ssize_t> const &,
                                 std::vector<py::ssize_t> const &);
std::vector<py::ssize_t> _unravel_index_c(py::ssize_t,
                                          std::vector<py::ssize_t> const &);
std::vector<py::ssize_t> _unravel_index_f(py::ssize_t,
                                          std::vector<py::ssize_t> const &);
} // namespace py_internal
} // namespace tensor
} // namespace dpctl
