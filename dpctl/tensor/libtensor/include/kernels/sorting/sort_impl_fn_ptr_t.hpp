//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2025 Intel Corporation
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
/// This file defines functions of dpctl.tensor._tensor_sorting_impl
/// extension.
//===--------------------------------------------------------------------===//

#pragma once

#include <cstddef>
#include <sycl/sycl.hpp>
#include <vector>

#include "kernels/dpctl_tensor_types.hpp"

namespace dpctl
{
namespace tensor
{
namespace kernels
{

using dpctl::tensor::ssize_t;

typedef sycl::event (*sort_contig_fn_ptr_t)(sycl::queue &,
                                            std::size_t,
                                            std::size_t,
                                            const char *,
                                            char *,
                                            ssize_t,
                                            ssize_t,
                                            ssize_t,
                                            ssize_t,
                                            const std::vector<sycl::event> &);

} // namespace kernels
} // namespace tensor
} // namespace dpctl
