//=== isin.hpp -                                      ---*-C++-*--/===//
//    Implementation of searching for membership in sorted array
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
/// This file defines kernels for tensor membership operations.
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <vector>

#include "kernels/dpctl_tensor_types.hpp"
#include "kernels/sorting/search_sorted_detail.hpp"
#include "utils/offset_utils.hpp"

namespace dpctl
{
namespace tensor
{
namespace kernels
{

using dpctl::tensor::ssize_t;

template <typename T,
          typename HayIndexerT,
          typename NeedlesIndexerT,
          typename OutIndexerT,
          typename Compare>
struct IsinFunctor
{
private:
    bool invert;
    const T *hay_tp;
    const T *needles_tp;
    bool *out_tp;
    std::size_t hay_nelems;
    HayIndexerT hay_indexer;
    NeedlesIndexerT needles_indexer;
    OutIndexerT out_indexer;

public:
    IsinFunctor(const bool invert_,
                const T *hay_,
                const T *needles_,
                bool *out_,
                const std::size_t hay_nelems_,
                const HayIndexerT &hay_indexer_,
                const NeedlesIndexerT &needles_indexer_,
                const OutIndexerT &out_indexer_)
        : invert(invert_), hay_tp(hay_), needles_tp(needles_), out_tp(out_),
          hay_nelems(hay_nelems_), hay_indexer(hay_indexer_),
          needles_indexer(needles_indexer_), out_indexer(out_indexer_)
    {
    }

    void operator()(sycl::id<1> id) const
    {
        const Compare comp{};

        const std::size_t i = id[0];
        const T needle_v = needles_tp[needles_indexer(i)];

        // position of the needle_v in the hay array
        std::size_t pos{};

        static constexpr std::size_t zero(0);
        // search in hay in left-closed interval, give `pos` such that
        // hay[pos - 1] < needle_v <= hay[pos]

        // lower_bound returns the first pos such that bool(hay[pos] <
        // needle_v) is false, i.e. needle_v <= hay[pos]
        pos = static_cast<std::size_t>(
            search_sorted_detail::lower_bound_indexed_impl(
                hay_tp, zero, hay_nelems, needle_v, comp, hay_indexer));
        bool out = (pos == hay_nelems ? false : hay_tp[pos] == needle_v);
        out_tp[out_indexer(i)] = (invert) ? !out : out;
    }
};

typedef sycl::event (*isin_contig_impl_fp_ptr_t)(
    sycl::queue &,
    const bool,
    const std::size_t,
    const std::size_t,
    const char *,
    const ssize_t,
    const char *,
    const ssize_t,
    char *,
    const ssize_t,
    const std::vector<sycl::event> &);

template <typename T> class isin_contig_impl_krn;

template <typename T, typename Compare>
sycl::event isin_contig_impl(sycl::queue &exec_q,
                             const bool invert,
                             const std::size_t hay_nelems,
                             const std::size_t needles_nelems,
                             const char *hay_cp,
                             const ssize_t hay_offset,
                             const char *needles_cp,
                             const ssize_t needles_offset,
                             char *out_cp,
                             const ssize_t out_offset,
                             const std::vector<sycl::event> &depends)
{
    const T *hay_tp = reinterpret_cast<const T *>(hay_cp) + hay_offset;
    const T *needles_tp =
        reinterpret_cast<const T *>(needles_cp) + needles_offset;

    bool *out_tp = reinterpret_cast<bool *>(out_cp) + out_offset;

    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        using KernelName = class isin_contig_impl_krn<T>;

        sycl::range<1> gRange(needles_nelems);

        using TrivialIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;

        constexpr TrivialIndexerT hay_indexer{};
        constexpr TrivialIndexerT needles_indexer{};
        constexpr TrivialIndexerT out_indexer{};

        const auto fnctr =
            IsinFunctor<T, TrivialIndexerT, TrivialIndexerT, TrivialIndexerT,
                        Compare>(invert, hay_tp, needles_tp, out_tp, hay_nelems,
                                 hay_indexer, needles_indexer, out_indexer);

        cgh.parallel_for<KernelName>(gRange, fnctr);
    });

    return comp_ev;
}

typedef sycl::event (*isin_strided_impl_fp_ptr_t)(
    sycl::queue &,
    const bool,
    const std::size_t,
    const std::size_t,
    const char *,
    const ssize_t,
    const ssize_t,
    const char *,
    const ssize_t,
    char *,
    const ssize_t,
    int,
    const ssize_t *,
    const std::vector<sycl::event> &);

template <typename T> class isin_strided_impl_krn;

template <typename T, typename Compare>
sycl::event isin_strided_impl(
    sycl::queue &exec_q,
    const bool invert,
    const std::size_t hay_nelems,
    const std::size_t needles_nelems,
    const char *hay_cp,
    const ssize_t hay_offset,
    // hay is 1D, so hay_nelems, hay_offset, hay_stride describe strided array
    const ssize_t hay_stride,
    const char *needles_cp,
    const ssize_t needles_offset,
    char *out_cp,
    const ssize_t out_offset,
    const int needles_nd,
    // packed_shape_strides is [needles_shape, needles_strides,
    // out_strides] has length of 3*needles_nd
    const ssize_t *packed_shape_strides,
    const std::vector<sycl::event> &depends)
{
    const T *hay_tp = reinterpret_cast<const T *>(hay_cp);
    const T *needles_tp = reinterpret_cast<const T *>(needles_cp);

    bool *out_tp = reinterpret_cast<bool *>(out_cp);

    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        sycl::range<1> gRange(needles_nelems);

        using HayIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
        const HayIndexerT hay_indexer(
            /* offset */ hay_offset,
            /* size   */ hay_nelems,
            /* step   */ hay_stride);

        using NeedlesIndexerT = dpctl::tensor::offset_utils::StridedIndexer;
        const ssize_t *needles_shape_strides = packed_shape_strides;
        const NeedlesIndexerT needles_indexer(needles_nd, needles_offset,
                                              needles_shape_strides);
        using OutIndexerT = dpctl::tensor::offset_utils::UnpackedStridedIndexer;

        const ssize_t *out_shape = packed_shape_strides;
        const ssize_t *out_strides = packed_shape_strides + 2 * needles_nd;
        const OutIndexerT out_indexer(needles_nd, out_offset, out_shape,
                                      out_strides);

        const auto fnctr =
            IsinFunctor<T, HayIndexerT, NeedlesIndexerT, OutIndexerT, Compare>(
                invert, hay_tp, needles_tp, out_tp, hay_nelems, hay_indexer,
                needles_indexer, out_indexer);
        using KernelName = class isin_strided_impl_krn<T>;

        cgh.parallel_for<KernelName>(gRange, fnctr);
    });

    return comp_ev;
}

} // namespace kernels
} // namespace tensor
} // namespace dpctl
