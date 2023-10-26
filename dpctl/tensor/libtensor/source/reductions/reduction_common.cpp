//===-- ------------ Implementation of _tensor_impl module  ----*-C++-*-/===//
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
//===--------------------------------------------------------------------===//
///
/// \file
/// This file defines functions of dpctl.tensor._tensor_impl extensions
//===--------------------------------------------------------------------===//

#include <pybind11/pybind11.h>

#include "argmax.hpp"
#include "argmin.hpp"
#include "logsumexp.hpp"
#include "max.hpp"
#include "min.hpp"
#include "prod.hpp"
#include "reduce_hypot.hpp"
#include "sum.hpp"

namespace py = pybind11;

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

/*! @brief Add reduction functions to Python module */
void init_reduction_functions(py::module_ m)
{
    init_argmax(m);
    init_argmin(m);
    init_logsumexp(m);
    init_max(m);
    init_min(m);
    init_prod(m);
    init_reduce_hypot(m);
    init_sum(m);
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
