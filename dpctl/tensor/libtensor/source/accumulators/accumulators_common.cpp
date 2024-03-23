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

#include <pybind11/pybind11.h>

#include "cumulative_logsumexp.hpp"
#include "cumulative_prod.hpp"
#include "cumulative_sum.hpp"

namespace py = pybind11;

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

/*! @brief Add accumulators to Python module */
void init_accumulator_functions(py::module_ m)
{
    init_cumulative_logsumexp(m);
    init_cumulative_prod(m);
    init_cumulative_sum(m);
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
