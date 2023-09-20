//===-- boolean_advanced_indexing.hpp -                       --*-C++-*-/===//
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
/// dpctl.tensor.place, dpctl.tensor.extract, and dpctl.tensor.nonzero
//===----------------------------------------------------------------------===//

#pragma once
#include <sycl/sycl.hpp>
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
py_extract(const dpctl::tensor::usm_ndarray &src,
           const dpctl::tensor::usm_ndarray &cumsum,
           int axis_start, // axis_start <= mask_i < axis_end
           int axis_end,
           const dpctl::tensor::usm_ndarray &dst,
           sycl::queue &exec_q,
           const std::vector<sycl::event> &depends = {});

extern void populate_masked_extract_dispatch_vectors(void);

extern std::pair<sycl::event, sycl::event>
py_place(const dpctl::tensor::usm_ndarray &dst,
         const dpctl::tensor::usm_ndarray &cumsum,
         int axis_start, // axis_start <= mask_i < axis_end
         int axis_end,
         const dpctl::tensor::usm_ndarray &rhs,
         sycl::queue &exec_q,
         const std::vector<sycl::event> &depends = {});

extern void populate_masked_place_dispatch_vectors(void);

extern std::pair<sycl::event, sycl::event>
py_nonzero(const dpctl::tensor::usm_ndarray
               &cumsum, // int32 input array, 1D, C-contiguous
           const dpctl::tensor::usm_ndarray
               &indexes, // int32 2D output array, C-contiguous
           const std::vector<py::ssize_t>
               &mask_shape, // shape of array from which cumsum was computed
           sycl::queue &exec_q,
           const std::vector<sycl::event> &depends = {});

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
