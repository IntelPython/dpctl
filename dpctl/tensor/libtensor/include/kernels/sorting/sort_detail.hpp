//=== searchsorted.hpp -                                      ---*-C++-*--/===//
//    Implementation of searching in sorted array
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
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines kernels for tensor sort/argsort operations.
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef>
#include <cstdint>

namespace dpctl
{
namespace tensor
{
namespace kernels
{

namespace sort_detail
{

template <typename T> T quotient_ceil(T n, T m)
{
    return (n + m - 1) / m;
}

template <typename Acc, typename Value, typename Compare>
std::size_t lower_bound_impl(const Acc acc,
                             const std::size_t first,
                             const std::size_t last,
                             const Value &value,
                             const Compare &comp)
{
    std::size_t n = last - first;
    std::size_t cur = n, start = first;
    std::size_t it;
    while (n > 0) {
        it = start;
        cur = n / 2;
        it += cur;
        if (comp(acc[it], value)) {
            n -= cur + 1, start = ++it;
        }
        else
            n = cur;
    }
    return start;
}

template <typename Acc, typename Value, typename Compare>
std::size_t upper_bound_impl(const Acc acc,
                             const std::size_t first,
                             const std::size_t last,
                             const Value &value,
                             const Compare &comp)
{
    const auto &op_comp = [comp](auto x, auto y) { return !comp(y, x); };
    return lower_bound_impl(acc, first, last, value, op_comp);
}

template <typename Acc, typename Value, typename Compare, typename IndexerT>
std::size_t lower_bound_indexed_impl(const Acc acc,
                                     std::size_t first,
                                     std::size_t last,
                                     const Value &value,
                                     const Compare &comp,
                                     const IndexerT &acc_indexer)
{
    std::size_t n = last - first;
    std::size_t cur = n, start = first;
    std::size_t it;
    while (n > 0) {
        it = start;
        cur = n / 2;
        it += cur;
        if (comp(acc[acc_indexer(it)], value)) {
            n -= cur + 1, start = ++it;
        }
        else
            n = cur;
    }
    return start;
}

template <typename Acc, typename Value, typename Compare, typename IndexerT>
std::size_t upper_bound_indexed_impl(const Acc acc,
                                     const std::size_t first,
                                     const std::size_t last,
                                     const Value &value,
                                     const Compare &comp,
                                     const IndexerT &acc_indexer)
{
    const auto &op_comp = [comp](auto x, auto y) { return !comp(y, x); };
    return lower_bound_indexed_impl(acc, first, last, value, op_comp,
                                    acc_indexer);
}

} // namespace sort_detail

} // namespace kernels
} // namespace tensor
} // namespace dpctl
