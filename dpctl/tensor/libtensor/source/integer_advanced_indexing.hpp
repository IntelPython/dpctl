//===-- integer_advanced_indexing.hpp -                         --*-C++-*-/===//
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
/// This file declares Python API for implementation functions of
/// dpctl.tensor.take and dpctl.tensor.put
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
usm_ndarray_take(dpctl::tensor::usm_ndarray,
                 py::object,
                 dpctl::tensor::usm_ndarray,
                 int,
                 uint8_t,
                 sycl::queue,
                 const std::vector<sycl::event> & = {});

extern std::pair<sycl::event, sycl::event>
usm_ndarray_put(dpctl::tensor::usm_ndarray,
                py::object,
                dpctl::tensor::usm_ndarray,
                int,
                uint8_t,
                sycl::queue,
                const std::vector<sycl::event> & = {});

extern void init_advanced_indexing_dispatch_tables(void);

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
