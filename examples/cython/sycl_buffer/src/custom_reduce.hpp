//===--- custom_reduce.hpp - work-group reduction for AdaptiveCpp ---------===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2026 Intel Corporation
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
/// This file implements a reduction over a work-group through local memory, to
/// be used in place of `sycl::reduce_over_group`.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

template <typename T, typename GroupTy, typename LocAccT, typename OpT>
T custom_reduce_over_group(const GroupTy &wg,
                           LocAccT local_mem_acc,
                           const T &local_val,
                           const OpT &op)
{
    const std::uint32_t lid = wg.get_local_linear_id();

    local_mem_acc[lid] = local_val;
    sycl::group_barrier(wg, sycl::memory_scope::work_group);

    // fold the upper half of the active range onto the lower one, until a
    // single element is left
    for (std::uint32_t n = wg.get_local_linear_range(); n > 1;) {
        const std::uint32_t half = (n + 1) >> 1;
        if (lid + half < n) {
            local_mem_acc[lid] =
                op(local_mem_acc[lid], local_mem_acc[lid + half]);
        }
        sycl::group_barrier(wg, sycl::memory_scope::work_group);
        n = half;
    }

    return local_mem_acc[0];
}
