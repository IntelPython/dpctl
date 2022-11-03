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
#include <string>

#include "dpctl4pybind11.hpp"
#include <CL/sycl.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

extern std::string default_device_fp_type(py::object);
extern std::string default_device_int_type(py::object);
extern std::string default_device_bool_type(py::object);
extern std::string default_device_complex_type(py::object);

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
