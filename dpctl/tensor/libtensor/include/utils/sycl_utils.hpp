//=== sycl_utils.hpp - Implementation of utilities         ------- *-C++-*/===//
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
/// This file defines utilities used for kernel submission.
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl.hpp>
#include <algorithm>
#include <cstddef>
#include <vector>

namespace dpctl
{
namespace tensor
{
namespace sycl_utils
{

/*! @brief Find the smallest multiple of supported sub-group size larger than
 * nelems */
template <size_t f = 4>
size_t choose_workgroup_size(const size_t nelems,
                             const std::vector<size_t> &sg_sizes)
{
    std::vector<size_t> wg_choices;
    wg_choices.reserve(f * sg_sizes.size());

    for (const auto &sg_size : sg_sizes) {
#pragma unroll
        for (size_t i = 1; i <= f; ++i) {
            wg_choices.push_back(sg_size * i);
        }
    }
    std::sort(std::begin(wg_choices), std::end(wg_choices));

    size_t wg = 1;
    for (size_t i = 0; i < wg_choices.size(); ++i) {
        if (wg_choices[i] == wg) {
            continue;
        }
        wg = wg_choices[i];
        size_t n_groups = ((nelems + wg - 1) / wg);
        if (n_groups == 1)
            break;
    }

    return wg;
}

} // namespace sycl_utils
} // namespace tensor
} // namespace dpctl
