//===----------- Implementation of _tensor_impl module  ---------*-C++-*-/===//
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
/// This file defines functions of dpctl.tensor._tensor_impl extensions
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl.hpp>
#include <utility>
#include <vector>

#include "dpctl4pybind11.hpp"
#include <pybind11/pybind11.h>

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

extern std::pair<sycl::event, sycl::event>
copy_usm_ndarray_for_reshape(dpctl::tensor::usm_ndarray src,
                             dpctl::tensor::usm_ndarray dst,
                             py::ssize_t shift,
                             sycl::queue exec_q,
                             const std::vector<sycl::event> &depends = {});

extern void init_copy_for_reshape_dispatch_vectors();

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
