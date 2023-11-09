//===----------- Implementation of _tensor_impl module  ---------*-C++-*-/===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2023 Intel Corporation
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
/// This file declares functions for looking of supported types in elementwise
/// functions.
//===----------------------------------------------------------------------===//

#pragma once
#include "dpctl4pybind11.hpp"
#include <CL/sycl.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "utils/type_dispatch.hpp"

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

namespace dpctl
{
namespace tensor
{
namespace py_internal
{
namespace type_utils
{

/*! @brief Produce dtype from a type number */
extern py::dtype _dtype_from_typenum(td_ns::typenum_t);

/*! @brief Lookup typeid of the result from typeid of
 *         argument and the mapping table */
extern int _result_typeid(int, const int *);

} // namespace type_utils
} // namespace py_internal
} // namespace tensor
} // namespace dpctl
