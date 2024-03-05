//=== boolean_advanced_indexing.hpp -                      ------*-C++-*--/===//
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
//===---------------------------------------------------------------------===//
///
/// \file
/// This file defines kernels for advanced tensor index operations.
//===---------------------------------------------------------------------===//

#pragma once
#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

#include "dpctl_tensor_types.hpp"
#include "utils/offset_utils.hpp"
#include "utils/type_dispatch_building.hpp"

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace indexing
{

using namespace dpctl::tensor::offset_utils;

template <typename OrthogIndexerT,
          typename MaskedSrcIndexerT,
          typename MaskedDstIndexerT,
          typename dataT,
          typename indT>
struct MaskedExtractStridedFunctor
{
    MaskedExtractStridedFunctor(const char *src_data_p,
                                const char *cumsum_data_p,
                                char *dst_data_p,
                                size_t orthog_iter_size,
                                size_t masked_iter_size,
                                const OrthogIndexerT &orthog_src_dst_indexer_,
                                const MaskedSrcIndexerT &masked_src_indexer_,
                                const MaskedDstIndexerT &masked_dst_indexer_)
        : src_cp(src_data_p), cumsum_cp(cumsum_data_p), dst_cp(dst_data_p),
          orthog_nelems(orthog_iter_size), masked_nelems(masked_iter_size),
          orthog_src_dst_indexer(orthog_src_dst_indexer_),
          masked_src_indexer(masked_src_indexer_),
          masked_dst_indexer(masked_dst_indexer_)
    {
    }

    void operator()(sycl::id<1> idx) const
    {
        const dataT *src_data = reinterpret_cast<const dataT *>(src_cp);
        dataT *dst_data = reinterpret_cast<dataT *>(dst_cp);
        const indT *cumsum_data = reinterpret_cast<const indT *>(cumsum_cp);

        size_t global_i = idx[0];
        size_t orthog_i = global_i / masked_nelems;
        size_t masked_i = global_i - masked_nelems * orthog_i;

        indT current_running_count = cumsum_data[masked_i];
        bool mask_set =
            (masked_i == 0)
                ? (current_running_count == 1)
                : (current_running_count == cumsum_data[masked_i - 1] + 1);

        // dst[cumsum[i], j] - 1 = src[i, j] if cumsum[i] == ((i > 0) ?
        // cumsum[i-1]
        // + 1 : 1)
        if (mask_set) {
            auto orthog_offsets =
                orthog_src_dst_indexer(static_cast<ssize_t>(orthog_i));

            size_t total_src_offset = masked_src_indexer(masked_i) +
                                      orthog_offsets.get_first_offset();
            size_t total_dst_offset =
                masked_dst_indexer(current_running_count - 1) +
                orthog_offsets.get_second_offset();

            dst_data[total_dst_offset] = src_data[total_src_offset];
        }
    }

private:
    const char *src_cp = nullptr;
    const char *cumsum_cp = nullptr;
    char *dst_cp = nullptr;
    size_t orthog_nelems = 0;
    size_t masked_nelems = 0;
    // has nd, shape, src_strides, dst_strides for
    // dimensions that ARE NOT masked
    const OrthogIndexerT orthog_src_dst_indexer;
    // has nd, shape, src_strides for
    // dimensions that ARE masked
    const MaskedSrcIndexerT masked_src_indexer;
    // has 1, dst_strides for dimensions that ARE masked
    const MaskedDstIndexerT masked_dst_indexer;
};

template <typename OrthogIndexerT,
          typename MaskedDstIndexerT,
          typename MaskedRhsIndexerT,
          typename dataT,
          typename indT>
struct MaskedPlaceStridedFunctor
{
    MaskedPlaceStridedFunctor(char *dst_data_p,
                              const char *cumsum_data_p,
                              const char *rhs_data_p,
                              size_t orthog_iter_size,
                              size_t masked_iter_size,
                              const OrthogIndexerT &orthog_dst_rhs_indexer_,
                              const MaskedDstIndexerT &masked_dst_indexer_,
                              const MaskedRhsIndexerT &masked_rhs_indexer_)
        : dst_cp(dst_data_p), cumsum_cp(cumsum_data_p), rhs_cp(rhs_data_p),
          orthog_nelems(orthog_iter_size), masked_nelems(masked_iter_size),
          orthog_dst_rhs_indexer(orthog_dst_rhs_indexer_),
          masked_dst_indexer(masked_dst_indexer_),
          masked_rhs_indexer(masked_rhs_indexer_)
    {
    }

    void operator()(sycl::id<1> idx) const
    {
        dataT *dst_data = reinterpret_cast<dataT *>(dst_cp);
        const indT *cumsum_data = reinterpret_cast<const indT *>(cumsum_cp);
        const dataT *rhs_data = reinterpret_cast<const dataT *>(rhs_cp);

        size_t global_i = idx[0];
        size_t orthog_i = global_i / masked_nelems;
        size_t masked_i = global_i - masked_nelems * orthog_i;

        indT current_running_count = cumsum_data[masked_i];
        bool mask_set =
            (masked_i == 0)
                ? (current_running_count == 1)
                : (current_running_count == cumsum_data[masked_i - 1] + 1);

        // src[i, j] = rhs[cumsum[i] - 1, j] if cumsum[i] == ((i > 0) ?
        // cumsum[i-1]
        // + 1 : 1)
        if (mask_set) {
            auto orthog_offsets =
                orthog_dst_rhs_indexer(static_cast<ssize_t>(orthog_i));

            size_t total_dst_offset = masked_dst_indexer(masked_i) +
                                      orthog_offsets.get_first_offset();
            size_t total_rhs_offset =
                masked_rhs_indexer(current_running_count - 1) +
                orthog_offsets.get_second_offset();

            dst_data[total_dst_offset] = rhs_data[total_rhs_offset];
        }
    }

private:
    char *dst_cp = nullptr;
    const char *cumsum_cp = nullptr;
    const char *rhs_cp = nullptr;
    size_t orthog_nelems = 0;
    size_t masked_nelems = 0;
    // has nd, shape, dst_strides, rhs_strides for
    // dimensions that ARE NOT masked
    const OrthogIndexerT orthog_dst_rhs_indexer;
    // has nd, shape, dst_strides for
    // dimensions that ARE masked
    const MaskedDstIndexerT masked_dst_indexer;
    // has 1, rhs_strides for dimensions that ARE masked
    const MaskedRhsIndexerT masked_rhs_indexer;
};

// ======= Masked extraction ================================

template <typename OrthoIndexerT,
          typename MaskedSrcIndexerT,
          typename MaskedDstIndexerT,
          typename dataT,
          typename indT>
class masked_extract_all_slices_strided_impl_krn;

typedef sycl::event (*masked_extract_all_slices_strided_impl_fn_ptr_t)(
    sycl::queue &,
    ssize_t,
    const char *,
    const char *,
    char *,
    int,
    ssize_t const *,
    ssize_t,
    ssize_t,
    const std::vector<sycl::event> &);

template <typename dataT, typename indT>
sycl::event masked_extract_all_slices_strided_impl(
    sycl::queue &exec_q,
    ssize_t iteration_size,
    const char *src_p,
    const char *cumsum_p,
    char *dst_p,
    int nd,
    const ssize_t
        *packed_src_shape_strides, // [src_shape, src_strides], length 2*nd
    ssize_t dst_size,              // dst is 1D
    ssize_t dst_stride,
    const std::vector<sycl::event> &depends = {})
{
    //  using MaskedExtractStridedFunctor;
    //  using Strided1DIndexer;
    //  using StridedIndexer;
    //  using TwoZeroOffsets_Indexer;

    constexpr TwoZeroOffsets_Indexer orthog_src_dst_indexer{};

    /* StridedIndexer(int _nd, ssize_t _offset, ssize_t const
     * *_packed_shape_strides) */
    const StridedIndexer masked_src_indexer(nd, 0, packed_src_shape_strides);
    const Strided1DIndexer masked_dst_indexer(0, dst_size, dst_stride);

    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        cgh.parallel_for<class masked_extract_all_slices_strided_impl_krn<
            TwoZeroOffsets_Indexer, StridedIndexer, Strided1DIndexer, dataT,
            indT>>(
            sycl::range<1>(static_cast<size_t>(iteration_size)),
            MaskedExtractStridedFunctor<TwoZeroOffsets_Indexer, StridedIndexer,
                                        Strided1DIndexer, dataT, indT>(
                src_p, cumsum_p, dst_p, 1, iteration_size,
                orthog_src_dst_indexer, masked_src_indexer,
                masked_dst_indexer));
    });

    return comp_ev;
}

typedef sycl::event (*masked_extract_some_slices_strided_impl_fn_ptr_t)(
    sycl::queue &,
    ssize_t,
    ssize_t,
    const char *,
    const char *,
    char *,
    int,
    ssize_t const *,
    ssize_t,
    ssize_t,
    int,
    ssize_t const *,
    ssize_t,
    ssize_t,
    const std::vector<sycl::event> &);

template <typename OrthoIndexerT,
          typename MaskedSrcIndexerT,
          typename MaskedDstIndexerT,
          typename dataT,
          typename indT>
class masked_extract_some_slices_strided_impl_krn;

template <typename dataT, typename indT>
sycl::event masked_extract_some_slices_strided_impl(
    sycl::queue &exec_q,
    ssize_t orthog_nelems,
    ssize_t masked_nelems,
    const char *src_p,
    const char *cumsum_p,
    char *dst_p,
    int orthog_nd,
    // [ortho_shape, ortho_src_strides, // ortho_dst_strides],
    // length 3*ortho_nd
    const ssize_t *packed_ortho_src_dst_shape_strides,
    ssize_t ortho_src_offset,
    ssize_t ortho_dst_offset,
    int masked_nd,
    // [masked_src_shape, masked_src_strides],
    // length 2*masked_nd, mask_dst is 1D
    const ssize_t *packed_masked_src_shape_strides,
    ssize_t masked_dst_size,
    ssize_t masked_dst_stride,
    const std::vector<sycl::event> &depends = {})
{
    //  using MaskedExtractStridedFunctor;
    //  using Strided1DIndexer;
    //  using StridedIndexer;
    //  using TwoOffsets_StridedIndexer;

    const TwoOffsets_StridedIndexer orthog_src_dst_indexer{
        orthog_nd, ortho_src_offset, ortho_dst_offset,
        packed_ortho_src_dst_shape_strides};

    const StridedIndexer masked_src_indexer{masked_nd, 0,
                                            packed_masked_src_shape_strides};
    const Strided1DIndexer masked_dst_indexer{0, masked_dst_size,
                                              masked_dst_stride};

    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        cgh.parallel_for<class masked_extract_some_slices_strided_impl_krn<
            TwoOffsets_StridedIndexer, StridedIndexer, Strided1DIndexer, dataT,
            indT>>(
            sycl::range<1>(static_cast<size_t>(orthog_nelems * masked_nelems)),
            MaskedExtractStridedFunctor<TwoOffsets_StridedIndexer,
                                        StridedIndexer, Strided1DIndexer, dataT,
                                        indT>(
                src_p, cumsum_p, dst_p, orthog_nelems, masked_nelems,
                orthog_src_dst_indexer, masked_src_indexer,
                masked_dst_indexer));
    });

    return comp_ev;
}

template <typename fnT, typename T>
struct MaskExtractAllSlicesStridedFactoryForInt32
{
    fnT get()
    {
        fnT fn = masked_extract_all_slices_strided_impl<T, std::int32_t>;
        return fn;
    }
};

template <typename fnT, typename T>
struct MaskExtractAllSlicesStridedFactoryForInt64
{
    fnT get()
    {
        fnT fn = masked_extract_all_slices_strided_impl<T, std::int64_t>;
        return fn;
    }
};

template <typename fnT, typename T>
struct MaskExtractSomeSlicesStridedFactoryForInt32
{
    fnT get()
    {
        fnT fn = masked_extract_some_slices_strided_impl<T, std::int32_t>;
        return fn;
    }
};

template <typename fnT, typename T>
struct MaskExtractSomeSlicesStridedFactoryForInt64
{
    fnT get()
    {
        fnT fn = masked_extract_some_slices_strided_impl<T, std::int64_t>;
        return fn;
    }
};

// Masked placement

template <typename OrthoIndexerT,
          typename MaskedDstIndexerT,
          typename MaskedRhsIndexerT,
          typename dataT,
          typename indT>
class masked_place_all_slices_strided_impl_krn;

typedef sycl::event (*masked_place_all_slices_strided_impl_fn_ptr_t)(
    sycl::queue &,
    ssize_t,
    char *,
    const char *,
    const char *,
    int,
    ssize_t const *,
    ssize_t,
    ssize_t,
    const std::vector<sycl::event> &);

template <typename dataT, typename indT>
sycl::event masked_place_all_slices_strided_impl(
    sycl::queue &exec_q,
    ssize_t iteration_size,
    char *dst_p,
    const char *cumsum_p,
    const char *rhs_p,
    int nd,
    const ssize_t
        *packed_dst_shape_strides, // [dst_shape, dst_strides], length 2*nd
    ssize_t rhs_size,              // rhs is 1D
    ssize_t rhs_stride,
    const std::vector<sycl::event> &depends = {})
{
    constexpr TwoZeroOffsets_Indexer orthog_dst_rhs_indexer{};

    /* StridedIndexer(int _nd, ssize_t _offset, ssize_t const
     * *_packed_shape_strides) */
    const StridedIndexer masked_dst_indexer(nd, 0, packed_dst_shape_strides);
    const Strided1DCyclicIndexer masked_rhs_indexer(0, rhs_size, rhs_stride);

    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        cgh.parallel_for<class masked_place_all_slices_strided_impl_krn<
            TwoZeroOffsets_Indexer, StridedIndexer, Strided1DCyclicIndexer,
            dataT, indT>>(
            sycl::range<1>(static_cast<size_t>(iteration_size)),
            MaskedPlaceStridedFunctor<TwoZeroOffsets_Indexer, StridedIndexer,
                                      Strided1DCyclicIndexer, dataT, indT>(
                dst_p, cumsum_p, rhs_p, 1, iteration_size,
                orthog_dst_rhs_indexer, masked_dst_indexer,
                masked_rhs_indexer));
    });

    return comp_ev;
}

typedef sycl::event (*masked_place_some_slices_strided_impl_fn_ptr_t)(
    sycl::queue &,
    ssize_t,
    ssize_t,
    char *,
    const char *,
    const char *,
    int,
    ssize_t const *,
    ssize_t,
    ssize_t,
    int,
    ssize_t const *,
    ssize_t,
    ssize_t,
    const std::vector<sycl::event> &);

template <typename OrthoIndexerT,
          typename MaskedSrcIndexerT,
          typename MaskedDstIndexerT,
          typename dataT,
          typename indT>
class masked_place_some_slices_strided_impl_krn;

template <typename dataT, typename indT>
sycl::event masked_place_some_slices_strided_impl(
    sycl::queue &exec_q,
    ssize_t orthog_nelems,
    ssize_t masked_nelems,
    char *dst_p,
    const char *cumsum_p,
    const char *rhs_p,
    int orthog_nd,
    // [ortho_shape, ortho_dst_strides, ortho_rhs_strides],
    // length 3*ortho_nd
    const ssize_t *packed_ortho_dst_rhs_shape_strides,
    ssize_t ortho_dst_offset,
    ssize_t ortho_rhs_offset,
    int masked_nd,
    // [masked_dst_shape, masked_dst_strides],
    // length 2*masked_nd, mask_dst is 1D
    const ssize_t *packed_masked_dst_shape_strides,
    ssize_t masked_rhs_size,
    ssize_t masked_rhs_stride,
    const std::vector<sycl::event> &depends = {})
{
    const TwoOffsets_StridedIndexer orthog_dst_rhs_indexer{
        orthog_nd, ortho_dst_offset, ortho_rhs_offset,
        packed_ortho_dst_rhs_shape_strides};

    /* StridedIndexer(int _nd, ssize_t _offset, ssize_t const
     * *_packed_shape_strides) */
    const StridedIndexer masked_dst_indexer{masked_nd, 0,
                                            packed_masked_dst_shape_strides};
    const Strided1DCyclicIndexer masked_rhs_indexer{0, masked_rhs_size,
                                                    masked_rhs_stride};

    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        cgh.parallel_for<class masked_place_some_slices_strided_impl_krn<
            TwoOffsets_StridedIndexer, StridedIndexer, Strided1DCyclicIndexer,
            dataT, indT>>(
            sycl::range<1>(static_cast<size_t>(orthog_nelems * masked_nelems)),
            MaskedPlaceStridedFunctor<TwoOffsets_StridedIndexer, StridedIndexer,
                                      Strided1DCyclicIndexer, dataT, indT>(
                dst_p, cumsum_p, rhs_p, orthog_nelems, masked_nelems,
                orthog_dst_rhs_indexer, masked_dst_indexer,
                masked_rhs_indexer));
    });

    return comp_ev;
}

template <typename fnT, typename T>
struct MaskPlaceAllSlicesStridedFactoryForInt32
{
    fnT get()
    {
        fnT fn = masked_place_all_slices_strided_impl<T, std::int32_t>;
        return fn;
    }
};

template <typename fnT, typename T>
struct MaskPlaceAllSlicesStridedFactoryForInt64
{
    fnT get()
    {
        fnT fn = masked_place_all_slices_strided_impl<T, std::int64_t>;
        return fn;
    }
};

template <typename fnT, typename T>
struct MaskPlaceSomeSlicesStridedFactoryForInt32
{
    fnT get()
    {
        fnT fn = masked_place_some_slices_strided_impl<T, std::int32_t>;
        return fn;
    }
};

template <typename fnT, typename T>
struct MaskPlaceSomeSlicesStridedFactoryForInt64
{
    fnT get()
    {
        fnT fn = masked_place_some_slices_strided_impl<T, std::int64_t>;
        return fn;
    }
};

// Non-zero

template <typename T1, typename T2> class non_zero_indexes_krn;

typedef sycl::event (*non_zero_indexes_fn_ptr_t)(
    sycl::queue &,
    ssize_t,
    ssize_t,
    int,
    const char *,
    char *,
    const ssize_t *,
    std::vector<sycl::event> const &);

template <typename indT1, typename indT2>
sycl::event non_zero_indexes_impl(sycl::queue &exec_q,
                                  ssize_t iter_size,
                                  ssize_t nz_elems,
                                  int nd,
                                  const char *cumsum_cp,
                                  char *indexes_cp,
                                  const ssize_t *mask_shape,
                                  std::vector<sycl::event> const &depends)
{
    const indT1 *cumsum_data = reinterpret_cast<const indT1 *>(cumsum_cp);
    indT2 *indexes_data = reinterpret_cast<indT2 *>(indexes_cp);

    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.parallel_for<class non_zero_indexes_krn<indT1, indT2>>(
            sycl::range<1>(iter_size), [=](sycl::id<1> idx)
        {
            auto i = idx[0];

            auto cs_curr_val = cumsum_data[i] - 1;
            auto cs_prev_val = (i > 0) ? cumsum_data[i - 1] : indT1(0);
            bool cond = (cs_curr_val == cs_prev_val);

            ssize_t i_ = static_cast<ssize_t>(i);
            for (int dim = nd; --dim > 0;) {
                auto sd = mask_shape[dim];
                ssize_t q = i_ / sd;
                ssize_t r = (i_ - q * sd);
                if (cond) {
                    indexes_data[cs_curr_val + dim * nz_elems] =
                        static_cast<indT2>(r);
                }
                i_ = q;
            }
            if (cond) {
                indexes_data[cs_curr_val] = static_cast<indT2>(i_);
            }
            });
    });

    return comp_ev;
}

} // namespace indexing
} // namespace kernels
} // namespace tensor
} // namespace dpctl
