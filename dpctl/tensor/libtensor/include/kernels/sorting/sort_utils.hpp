//=== sorting.hpp -  Implementation of sorting kernels       ---*-C++-*--/===//
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
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines kernels for tensor sort/argsort operations.
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <sycl/sycl.hpp>

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace sort_utils_detail
{

namespace syclexp = sycl::ext::oneapi::experimental;

template <class KernelName, typename T>
sycl::event iota_impl(sycl::queue &exec_q,
                      T *data,
                      std::size_t nelems,
                      const std::vector<sycl::event> &dependent_events)
{
    constexpr std::uint32_t lws = 256;
    constexpr std::uint32_t n_wi = 4;
    const std::size_t n_groups = (nelems + n_wi * lws - 1) / (n_wi * lws);

    sycl::range<1> gRange{n_groups * lws};
    sycl::range<1> lRange{lws};
    sycl::nd_range<1> ndRange{gRange, lRange};

    sycl::event e = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependent_events);
        cgh.parallel_for<KernelName>(ndRange, [=](sycl::nd_item<1> it) {
            const std::size_t gid = it.get_global_linear_id();
            const auto &sg = it.get_sub_group();
            const std::uint32_t lane_id = sg.get_local_id()[0];

            const std::size_t offset = (gid - lane_id) * n_wi;
            const std::uint32_t max_sgSize = sg.get_max_local_range()[0];

            std::array<T, n_wi> stripe{};
#pragma unroll
            for (std::uint32_t i = 0; i < n_wi; ++i) {
                stripe[i] = T(offset + lane_id + i * max_sgSize);
            }

            if (offset + n_wi * max_sgSize < nelems) {
                constexpr auto group_ls_props = syclexp::properties{
                    syclexp::data_placement_striped
                    // , syclexp::full_group
                };

                auto out_multi_ptr = sycl::address_space_cast<
                    sycl::access::address_space::global_space,
                    sycl::access::decorated::yes>(&data[offset]);

                syclexp::group_store(sg, sycl::span<T, n_wi>{&stripe[0], n_wi},
                                     out_multi_ptr, group_ls_props);
            }
            else {
                for (std::size_t idx = offset + lane_id; idx < nelems;
                     idx += max_sgSize)
                {
                    data[idx] = T(idx);
                }
            }
        });
    });

    return e;
}

template <class KernelName, typename IndexTy>
sycl::event map_back_impl(sycl::queue &exec_q,
                          std::size_t nelems,
                          const IndexTy *flat_index_data,
                          IndexTy *reduced_index_data,
                          std::size_t row_size,
                          const std::vector<sycl::event> &dependent_events)
{
    constexpr std::uint32_t lws = 64;
    constexpr std::uint32_t n_wi = 4;
    const std::size_t n_groups = (nelems + lws * n_wi - 1) / (n_wi * lws);

    sycl::range<1> lRange{lws};
    sycl::range<1> gRange{n_groups * lws};
    sycl::nd_range<1> ndRange{gRange, lRange};

    sycl::event map_back_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependent_events);

        cgh.parallel_for<KernelName>(ndRange, [=](sycl::nd_item<1> it) {
            const std::size_t gid = it.get_global_linear_id();
            const auto &sg = it.get_sub_group();
            const std::uint32_t lane_id = sg.get_local_id()[0];
            const std::uint32_t sg_size = sg.get_max_local_range()[0];

            const std::size_t start_id = (gid - lane_id) * n_wi + lane_id;

#pragma unroll
            for (std::uint32_t i = 0; i < n_wi; ++i) {
                const std::size_t data_id = start_id + i * sg_size;

                if (data_id < nelems) {
                    const IndexTy linear_index = flat_index_data[data_id];
                    reduced_index_data[data_id] = (linear_index % row_size);
                }
            }
        });
    });

    return map_back_ev;
}

} // end of namespace sort_utils_detail
} // end of namespace kernels
} // end of namespace tensor
} // end of namespace dpctl
