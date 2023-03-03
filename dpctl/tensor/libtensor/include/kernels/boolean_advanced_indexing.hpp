//=== boolean_advance_indexing.hpp -                       ---*-C++-*--/===//
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
/// This file defines kernels for advanced tensor index operations.
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl.hpp>
#include <array>
#include <cstdint>
#include <limits>
#include <pybind11/pybind11.h>
#include <utility>
#include <vector>

#include "utils/strided_iters.hpp"
#include "utils/type_dispatch.hpp"

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace indexing
{

namespace py = pybind11;

template <typename T> T ceiling_quotient(T n, T m)
{
    return (n + m - 1) / m;
}
template <typename T1, typename T2> T1 ceiling_quotient(T1 n, T2 m)
{
    return ceiling_quotient<T1>(n, static_cast<T1>(m));
}

template <typename inputT, typename outputT, typename IndexerT, size_t n_wi>
class inclusive_scan_rec_local_scan_krn;

template <typename inputT, typename outputT, typename IndexerT>
class inclusive_scan_rec_chunk_update_krn;

struct NoOpIndexer
{
    size_t operator()(size_t gid) const
    {
        return gid;
    }
};

struct StridedIndexer
{
    StridedIndexer(int _nd,
                   py::ssize_t _offset,
                   py::ssize_t const *_packed_shape_strides)
        : nd(_nd), starting_offset(_offset),
          shape_strides(_packed_shape_strides)
    {
    }

    size_t operator()(size_t gid) const
    {
        CIndexer_vector _ind(nd);
        py::ssize_t relative_offset(0);
        _ind.get_displacement<const py::ssize_t *, const py::ssize_t *>(
            static_cast<py::ssize_t>(gid),
            shape_strides,      // shape ptr
            shape_strides + nd, // strides ptr
            relative_offset);
        return starting_offset + relative_offset;
    }

private:
    int nd;
    py::ssize_t starting_offset;
    py::ssize_t const *shape_strides;
};

struct Strided1DIndexer
{
    Strided1DIndexer(py::ssize_t _offset, py::ssize_t _size, py::ssize_t _step)
        : offset(_offset), size(static_cast<size_t>(_size)), step(_step)
    {
    }

    size_t operator()(size_t gid) const
    {
        return static_cast<size_t>(offset + std::min<size_t>(gid, size) * step);
    }

private:
    py::ssize_t offset = 0;
    size_t size = 1;
    py::ssize_t step = 1;
};

template <typename _IndexerFn> struct ZeroChecker
{

    ZeroChecker(_IndexerFn _indexer) : indexer_fn(_indexer) {}

    template <typename dataT>
    bool operator()(dataT const *data, size_t gid) const
    {
        constexpr dataT _zero(0);

        return data[indexer_fn(gid)] == _zero;
    }

private:
    _IndexerFn indexer_fn;
};

/*
 * for integer type maskT,
 *       output[j] = sum( input[s0 + i * s1], 0 <= i <= j)
 * for 0 <= j < n_elems
 */
template <typename inputT, typename outputT, typename IndexerT, size_t n_wi>
sycl::event inclusive_scan_rec(sycl::queue exec_q,
                               size_t n_elems,
                               size_t wg_size,
                               const inputT *input,
                               outputT *output,
                               size_t s0,
                               size_t s1,
                               IndexerT indexer,
                               std::vector<sycl::event> const &depends = {})
{
    size_t n_groups = ceiling_quotient(n_elems, n_wi * wg_size);

    sycl::event inc_scan_phase1_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        using slmT = sycl::local_accessor<size_t, 1>;

        auto lws = sycl::range<1>(wg_size);
        auto gws = sycl::range<1>(n_groups * wg_size);

        slmT slm_iscan_tmp(lws, cgh);

        ZeroChecker<IndexerT> is_zero_fn(indexer);

        cgh.parallel_for<class inclusive_scan_rec_local_scan_krn<inputT, outputT, ZeroChecker<IndexerT>, n_wi>>(
            sycl::nd_range<1>(gws, lws),
            [=](sycl::nd_item<1> it)
        {
            auto chunk_gid = it.get_global_id(0);
            auto lid = it.get_local_id(0);

            std::array<size_t, n_wi> local_isum;

            size_t i = chunk_gid * n_wi;
            for (size_t m_wi = 0; m_wi < n_wi; ++m_wi) {
                constexpr outputT out_zero(0);
                constexpr outputT out_one(1);
                local_isum[m_wi] =
                    (i + m_wi < n_elems)
                        ? (is_zero_fn(input, s0 + s1 * (i + m_wi)) ? out_zero
                                                                   : out_one)
                        : out_zero;
            }

// local_isum is now result of
// inclusive scan of locally stored mask indicators
#pragma unroll
            for (size_t m_wi = 1; m_wi < n_wi; ++m_wi) {
                local_isum[m_wi] += local_isum[m_wi - 1];
            }

            size_t wg_iscan_val =
                sycl::inclusive_scan_over_group(it.get_group(),
                                                local_isum.back(),
                                                sycl::plus<size_t>(),
                                                size_t(0));

            slm_iscan_tmp[(lid + 1) % wg_size] = wg_iscan_val;
            it.barrier(sycl::access::fence_space::local_space);
            size_t addand = (lid == 0) ? 0 : slm_iscan_tmp[lid];
            it.barrier(sycl::access::fence_space::local_space);

#pragma unroll
            for (size_t m_wi = 0; m_wi < n_wi; ++m_wi) {
                local_isum[m_wi] += addand;
            }

            for (size_t m_wi = 0; m_wi < n_wi && i + m_wi < n_elems; ++m_wi) {
                output[i + m_wi] = local_isum[m_wi];
            }
            }
        );
    });

    sycl::event out_event = inc_scan_phase1_ev;
    if (n_groups > 1) {
        outputT *temp = sycl::malloc_device<outputT>(n_groups - 1, exec_q);

        auto chunk_size = wg_size * n_wi;

        NoOpIndexer _no_op_indexer{};
        auto e2 = inclusive_scan_rec<outputT, outputT, NoOpIndexer, n_wi>(
            exec_q, n_groups - 1, wg_size, output, temp, chunk_size - 1,
            chunk_size, _no_op_indexer, {inc_scan_phase1_ev});

        // output[ chunk_size * (i + 1) + j] += temp[i]
        auto e3 = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(e2);
            cgh.parallel_for<class inclusive_scan_rec_chunk_update_krn<inputT, outputT, IndexerT>>(
                {n_elems},
                [=](auto wiid)
            {
                auto gid = wiid[0];
                auto i = (gid / chunk_size);
                output[gid] += (i > 0) ? temp[i - 1] : 0;
                }
            );
        });

        // dangling task to free the temporary
        exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(e3);
            auto ctx = exec_q.get_context();
            cgh.host_task([ctx, temp]() { sycl::free(temp, ctx); });
        });

        out_event = e3;
    }

    return out_event;
}

template <typename displacementT> struct TwoOffsets
{
    TwoOffsets() : first_offset(0), second_offset(0) {}
    TwoOffsets(displacementT first_offset_, displacementT second_offset_)
        : first_offset(first_offset_), second_offset(second_offset_)
    {
    }

    displacementT get_first_offset() const
    {
        return first_offset;
    }
    displacementT get_second_offset() const
    {
        return second_offset;
    }

private:
    displacementT first_offset = 0;
    displacementT second_offset = 0;
};

struct TwoOffsets_StridedIndexer
{
    TwoOffsets_StridedIndexer(int common_nd,
                              py::ssize_t first_offset_,
                              py::ssize_t second_offset_,
                              py::ssize_t const *_packed_shape_strides)
        : nd(common_nd), starting_first_offset(first_offset_),
          starting_second_offset(second_offset_),
          shape_strides(_packed_shape_strides)
    {
    }

    TwoOffsets<py::ssize_t> operator()(py::ssize_t gid) const
    {
        CIndexer_vector _ind(nd);
        py::ssize_t relative_first_offset(0);
        py::ssize_t relative_second_offset(0);
        _ind.get_displacement<const py::ssize_t *, const py::ssize_t *>(
            gid,
            shape_strides,          // shape ptr
            shape_strides + nd,     // src strides ptr
            shape_strides + 2 * nd, // src strides ptr
            relative_first_offset, relative_second_offset);
        return TwoOffsets<py::ssize_t>(
            starting_first_offset + relative_first_offset,
            starting_second_offset + relative_second_offset);
    }

private:
    int nd;
    py::ssize_t starting_first_offset;
    py::ssize_t starting_second_offset;
    py::ssize_t const *shape_strides;
};

struct TwoZeroOffsets_Indexer
{
    TwoZeroOffsets_Indexer() {}

    TwoOffsets<py::ssize_t> operator()(py::ssize_t) const
    {
        return TwoOffsets<py::ssize_t>();
    }
};

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
                                OrthogIndexerT orthog_src_dst_indexer_,
                                MaskedSrcIndexerT masked_src_indexer_,
                                MaskedDstIndexerT masked_dst_indexer_)
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
                orthog_src_dst_indexer(static_cast<py::ssize_t>(orthog_i));

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
    OrthogIndexerT
        orthog_src_dst_indexer; // has nd, shape, src_strides, dst_strides for
                                // dimensions that ARE NOT masked
    MaskedSrcIndexerT masked_src_indexer; // has nd, shape, src_strides for
                                          // dimensions that ARE     masked
    MaskedDstIndexerT
        masked_dst_indexer; // has 1, dst_strides for dimensions that ARE masked
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
                              OrthogIndexerT orthog_dst_rhs_indexer_,
                              MaskedDstIndexerT masked_dst_indexer_,
                              MaskedRhsIndexerT masked_rhs_indexer_)
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
                orthog_dst_rhs_indexer(static_cast<py::ssize_t>(orthog_i));

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
    OrthogIndexerT
        orthog_dst_rhs_indexer; // has nd, shape, dst_strides, rhs_strides for
                                // dimensions that ARE NOT masked
    MaskedDstIndexerT masked_dst_indexer; // has nd, shape, dst_strides for
                                          // dimensions that ARE     masked
    MaskedRhsIndexerT
        masked_rhs_indexer; // has 1, rhs_strides for dimensions that ARE masked
};

// mask positions

typedef size_t (*mask_positions_contig_impl_fn_ptr_t)(
    sycl::queue,
    size_t,
    const char *,
    char *,
    std::vector<sycl::event> const &);

template <typename maskT, typename cumsumT>
size_t mask_positions_contig_impl(sycl::queue q,
                                  size_t n_elems,
                                  const char *mask,
                                  char *cumsum,
                                  std::vector<sycl::event> const &depends = {})
{
    constexpr int n_wi = 8;
    const maskT *mask_data_ptr = reinterpret_cast<const maskT *>(mask);
    cumsumT *cumsum_data_ptr = reinterpret_cast<cumsumT *>(cumsum);
    size_t wg_size = 128;

    NoOpIndexer flat_indexer{};

    sycl::event comp_ev = inclusive_scan_rec<maskT, cumsumT, NoOpIndexer, n_wi>(
        q, n_elems, wg_size, mask_data_ptr, cumsum_data_ptr, 0, 1, flat_indexer,
        depends);

    cumsumT *last_elem = cumsum_data_ptr + (n_elems - 1);

    cumsumT *last_elem_host_usm = sycl::malloc_host<cumsumT>(1, q);

    if (last_elem_host_usm == nullptr) {
        throw std::bad_alloc();
    }
    sycl::event copy_e =
        q.copy<std::int64_t>(last_elem, last_elem_host_usm, 1, {comp_ev});
    copy_e.wait();
    size_t return_val = static_cast<size_t>(*last_elem_host_usm);
    sycl::free(last_elem_host_usm, q);

    return return_val;
}

template <typename fnT, typename T> struct MaskPositionsContigFactory
{
    fnT get()
    {
        fnT fn = mask_positions_contig_impl<T, std::int64_t>;
        return fn;
    }
};

typedef size_t (*mask_positions_strided_impl_fn_ptr_t)(
    sycl::queue,
    size_t,
    const char *,
    int,
    py::ssize_t,
    const py::ssize_t *,
    char *,
    std::vector<sycl::event> const &);

template <typename maskT, typename cumsumT>
size_t mask_positions_strided_impl(sycl::queue q,
                                   size_t n_elems,
                                   const char *mask,
                                   int nd,
                                   py::ssize_t input_offset,
                                   const py::ssize_t *shape_strides,
                                   char *cumsum,
                                   std::vector<sycl::event> const &depends = {})
{
    constexpr int n_wi = 8;
    const maskT *mask_data_ptr = reinterpret_cast<const maskT *>(mask);
    cumsumT *cumsum_data_ptr = reinterpret_cast<cumsumT *>(cumsum);
    size_t wg_size = 128;

    StridedIndexer strided_indexer{nd, input_offset, shape_strides};

    sycl::event comp_ev =
        inclusive_scan_rec<maskT, cumsumT, StridedIndexer, n_wi>(
            q, n_elems, wg_size, mask_data_ptr, cumsum_data_ptr, 0, 1,
            strided_indexer, depends);

    cumsumT *last_elem = cumsum_data_ptr + (n_elems - 1);

    cumsumT *last_elem_host_usm = sycl::malloc_host<cumsumT>(1, q);

    if (last_elem_host_usm == nullptr) {
        throw std::bad_alloc();
    }
    sycl::event copy_e =
        q.copy<std::int64_t>(last_elem, last_elem_host_usm, 1, {comp_ev});
    copy_e.wait();
    size_t return_val = static_cast<size_t>(*last_elem_host_usm);
    sycl::free(last_elem_host_usm, q);

    return return_val;
}

template <typename fnT, typename T> struct MaskPositionsStridedFactory
{
    fnT get()
    {
        fnT fn = mask_positions_strided_impl<T, std::int64_t>;
        return fn;
    }
};

// ======= Masked extraction ================================

template <typename OrthoIndexerT,
          typename MaskedSrcIndexerT,
          typename MaskedDstIndexerT,
          typename dataT,
          typename indT>
class masked_extract_all_slices_strided_impl_krn;

typedef sycl::event (*masked_extract_all_slices_strided_impl_fn_ptr_t)(
    sycl::queue,
    py::ssize_t,
    const char *,
    const char *,
    char *,
    int,
    py::ssize_t const *,
    py::ssize_t,
    py::ssize_t,
    const std::vector<sycl::event> &);

template <typename dataT, typename indT>
sycl::event masked_extract_all_slices_strided_impl(
    sycl::queue exec_q,
    py::ssize_t iteration_size,
    const char *src_p,
    const char *cumsum_p,
    char *dst_p,
    int nd,
    const py::ssize_t
        *packed_src_shape_strides, // [src_shape, src_strides], length 2*nd
    py::ssize_t dst_size,          // dst is 1D
    py::ssize_t dst_stride,
    const std::vector<sycl::event> &depends = {})
{
    //  using MaskedExtractStridedFunctor;
    //  using Strided1DIndexer;
    //  using StridedIndexer;
    //  using TwoZeroOffsets_Indexer;

    TwoZeroOffsets_Indexer orthog_src_dst_indexer{};

    /* StridedIndexer(int _nd, py::ssize_t _offset, py::ssize_t const
     * *_packed_shape_strides) */
    StridedIndexer masked_src_indexer(nd, 0, packed_src_shape_strides);
    Strided1DIndexer masked_dst_indexer(0, dst_size, dst_stride);

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
    sycl::queue,
    py::ssize_t,
    py::ssize_t,
    const char *,
    const char *,
    char *,
    int,
    py::ssize_t const *,
    py::ssize_t,
    py::ssize_t,
    int,
    py::ssize_t const *,
    py::ssize_t,
    py::ssize_t,
    const std::vector<sycl::event> &);

template <typename OrthoIndexerT,
          typename MaskedSrcIndexerT,
          typename MaskedDstIndexerT,
          typename dataT,
          typename indT>
class masked_extract_some_slices_strided_impl_krn;

template <typename dataT, typename indT>
sycl::event masked_extract_some_slices_strided_impl(
    sycl::queue exec_q,
    py::ssize_t orthog_nelems,
    py::ssize_t masked_nelems,
    const char *src_p,
    const char *cumsum_p,
    char *dst_p,
    int orthog_nd,
    const py::ssize_t
        *packed_ortho_src_dst_shape_strides, // [ortho_shape, ortho_src_strides,
                                             // ortho_dst_strides], length
                                             // 3*ortho_nd
    py::ssize_t ortho_src_offset,
    py::ssize_t ortho_dst_offset,
    int masked_nd,
    const py::ssize_t *packed_masked_src_shape_strides, // [masked_src_shape,
                                                        // masked_src_strides],
                                                        // length 2*masked_nd
    py::ssize_t masked_dst_size,                        // mask_dst is 1D
    py::ssize_t masked_dst_stride,
    const std::vector<sycl::event> &depends = {})
{
    //  using MaskedExtractStridedFunctor;
    //  using Strided1DIndexer;
    //  using StridedIndexer;
    //  using TwoOffsets_StridedIndexer;

    TwoOffsets_StridedIndexer orthog_src_dst_indexer{
        orthog_nd, ortho_src_offset, ortho_dst_offset,
        packed_ortho_src_dst_shape_strides};

    StridedIndexer masked_src_indexer{masked_nd, 0,
                                      packed_masked_src_shape_strides};
    Strided1DIndexer masked_dst_indexer{0, masked_dst_size, masked_dst_stride};

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

template <typename fnT, typename T> struct MaskExtractAllSlicesStridedFactory
{
    fnT get()
    {
        fnT fn = masked_extract_all_slices_strided_impl<T, std::int64_t>;
        return fn;
    }
};

template <typename fnT, typename T> struct MaskExtractSomeSlicesStridedFactory
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
    sycl::queue,
    py::ssize_t,
    char *,
    const char *,
    const char *,
    int,
    py::ssize_t const *,
    py::ssize_t,
    py::ssize_t,
    const std::vector<sycl::event> &);

template <typename dataT, typename indT>
sycl::event masked_place_all_slices_strided_impl(
    sycl::queue exec_q,
    py::ssize_t iteration_size,
    char *dst_p,
    const char *cumsum_p,
    const char *rhs_p,
    int nd,
    const py::ssize_t
        *packed_dst_shape_strides, // [dst_shape, dst_strides], length 2*nd
    py::ssize_t rhs_size,          // rhs is 1D
    py::ssize_t rhs_stride,
    const std::vector<sycl::event> &depends = {})
{
    //  using MaskedPlaceStridedFunctor;
    //  using Strided1DIndexer;
    //  using StridedIndexer;
    //  using TwoZeroOffsets_Indexer;

    TwoZeroOffsets_Indexer orthog_dst_rhs_indexer{};

    /* StridedIndexer(int _nd, py::ssize_t _offset, py::ssize_t const
     * *_packed_shape_strides) */
    StridedIndexer masked_dst_indexer(nd, 0, packed_dst_shape_strides);
    Strided1DIndexer masked_rhs_indexer(0, rhs_size, rhs_stride);

    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        cgh.parallel_for<class masked_place_all_slices_strided_impl_krn<
            TwoZeroOffsets_Indexer, StridedIndexer, Strided1DIndexer, dataT,
            indT>>(
            sycl::range<1>(static_cast<size_t>(iteration_size)),
            MaskedPlaceStridedFunctor<TwoZeroOffsets_Indexer, StridedIndexer,
                                      Strided1DIndexer, dataT, indT>(
                dst_p, cumsum_p, rhs_p, 1, iteration_size,
                orthog_dst_rhs_indexer, masked_dst_indexer,
                masked_rhs_indexer));
    });

    return comp_ev;
}

typedef sycl::event (*masked_place_some_slices_strided_impl_fn_ptr_t)(
    sycl::queue,
    py::ssize_t,
    py::ssize_t,
    char *,
    const char *,
    const char *,
    int,
    py::ssize_t const *,
    py::ssize_t,
    py::ssize_t,
    int,
    py::ssize_t const *,
    py::ssize_t,
    py::ssize_t,
    const std::vector<sycl::event> &);

template <typename OrthoIndexerT,
          typename MaskedSrcIndexerT,
          typename MaskedDstIndexerT,
          typename dataT,
          typename indT>
class masked_place_some_slices_strided_impl_krn;

template <typename dataT, typename indT>
sycl::event masked_place_some_slices_strided_impl(
    sycl::queue exec_q,
    py::ssize_t orthog_nelems,
    py::ssize_t masked_nelems,
    char *dst_p,
    const char *cumsum_p,
    const char *rhs_p,
    int orthog_nd,
    const py::ssize_t
        *packed_ortho_dst_rhs_shape_strides, // [ortho_shape, ortho_dst_strides,
                                             // ortho_rhs_strides], length
                                             // 3*ortho_nd
    py::ssize_t ortho_dst_offset,
    py::ssize_t ortho_rhs_offset,
    int masked_nd,
    const py::ssize_t *packed_masked_dst_shape_strides, // [masked_dst_shape,
                                                        // masked_dst_strides],
                                                        // length 2*masked_nd
    py::ssize_t masked_rhs_size,                        // mask_dst is 1D
    py::ssize_t masked_rhs_stride,
    const std::vector<sycl::event> &depends = {})
{
    //  using MaskedPlaceStridedFunctor;
    //  using Strided1DIndexer;
    //  using StridedIndexer;
    //  using TwoOffsets_StridedIndexer;

    TwoOffsets_StridedIndexer orthog_dst_rhs_indexer{
        orthog_nd, ortho_dst_offset, ortho_rhs_offset,
        packed_ortho_dst_rhs_shape_strides};

    /* StridedIndexer(int _nd, py::ssize_t _offset, py::ssize_t const
     * *_packed_shape_strides) */
    StridedIndexer masked_dst_indexer{masked_nd, 0,
                                      packed_masked_dst_shape_strides};
    Strided1DIndexer masked_rhs_indexer{0, masked_rhs_size, masked_rhs_stride};

    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        cgh.parallel_for<class masked_place_some_slices_strided_impl_krn<
            TwoOffsets_StridedIndexer, StridedIndexer, Strided1DIndexer, dataT,
            indT>>(
            sycl::range<1>(static_cast<size_t>(orthog_nelems * masked_nelems)),
            MaskedPlaceStridedFunctor<TwoOffsets_StridedIndexer, StridedIndexer,
                                      Strided1DIndexer, dataT, indT>(
                dst_p, cumsum_p, rhs_p, orthog_nelems, masked_nelems,
                orthog_dst_rhs_indexer, masked_dst_indexer,
                masked_rhs_indexer));
    });

    return comp_ev;
}

static masked_place_all_slices_strided_impl_fn_ptr_t
    masked_place_all_slices_strided_impl_dispatch_vector
        [dpctl::tensor::detail::num_types];

template <typename fnT, typename T> struct MaskPlaceAllSlicesStridedFactory
{
    fnT get()
    {
        fnT fn = masked_place_all_slices_strided_impl<T, std::int64_t>;
        return fn;
    }
};

static masked_place_some_slices_strided_impl_fn_ptr_t
    masked_place_some_slices_strided_impl_dispatch_vector
        [dpctl::tensor::detail::num_types];

template <typename fnT, typename T> struct MaskPlaceSomeSlicesStridedFactory
{
    fnT get()
    {
        fnT fn = masked_place_some_slices_strided_impl<T, std::int64_t>;
        return fn;
    }
};

// Non-zero

class non_zero_indexes_krn;

template <typename indT1, typename indT2>
sycl::event non_zero_indexes_impl(sycl::queue exec_q,
                                  py::ssize_t iter_size,
                                  py::ssize_t nz_elems,
                                  int nd,
                                  const char *cumsum_cp,
                                  char *indexes_cp,
                                  const py::ssize_t *mask_shape,
                                  std::vector<sycl::event> const &depends)
{
    const indT1 *cumsum_data = reinterpret_cast<const indT1 *>(cumsum_cp);
    indT2 *indexes_data = reinterpret_cast<indT2 *>(indexes_cp);

    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.parallel_for<class non_zero_indexes_krn>(
            sycl::range<1>(iter_size), [=](sycl::id<1> idx) {
                auto i = idx[0];

                auto cs_curr_val = cumsum_data[i] - 1;
                auto cs_prev_val = (i > 0) ? cumsum_data[i - 1] : indT1(0);
                bool cond = (cs_curr_val == cs_prev_val);

                py::ssize_t i_ = static_cast<py::ssize_t>(i);
                for (int dim = nd; --dim > 0;) {
                    auto sd = mask_shape[dim];
                    py::ssize_t q = i_ / sd;
                    py::ssize_t r = (i_ - q * sd);
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
