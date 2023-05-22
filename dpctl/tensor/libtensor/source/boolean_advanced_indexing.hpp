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

extern void populate_mask_positions_dispatch_vectors(void);

extern size_t py_mask_positions(dpctl::tensor::usm_ndarray mask,
                                dpctl::tensor::usm_ndarray cumsum,
                                sycl::queue exec_q,
                                std::vector<sycl::event> const &depends = {});

extern std::pair<sycl::event, sycl::event>
py_extract(dpctl::tensor::usm_ndarray src,
           dpctl::tensor::usm_ndarray cumsum,
           int axis_start, // axis_start <= mask_i < axis_end
           int axis_end,
           dpctl::tensor::usm_ndarray dst,
           sycl::queue exec_q,
           std::vector<sycl::event> const &depends = {});

extern void populate_masked_extract_dispatch_vectors(void);

extern std::pair<sycl::event, sycl::event>
py_place(dpctl::tensor::usm_ndarray dst,
         dpctl::tensor::usm_ndarray cumsum,
         int axis_start, // axis_start <= mask_i < axis_end
         int axis_end,
         dpctl::tensor::usm_ndarray rhs,
         sycl::queue exec_q,
         std::vector<sycl::event> const &depends = {});

extern void populate_masked_place_dispatch_vectors(void);

extern std::pair<sycl::event, sycl::event> py_nonzero(
    dpctl::tensor::usm_ndarray cumsum,  // int64 input array, 1D, C-contiguous
    dpctl::tensor::usm_ndarray indexes, // int64 2D output array, C-contiguous
    std::vector<py::ssize_t>
        mask_shape, // shape of array from which cumsum was computed
    sycl::queue exec_q,
    std::vector<sycl::event> const &depends = {});

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
