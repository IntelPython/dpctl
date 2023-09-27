//===  accumulators.hpp - Implementation of accumulator kernels --*-C++-*-/===//
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
//===---------------------------------------------------------------------===//
///
/// \file
/// This file defines kernels for accumulators (cumulative sum, prod, etc.).
//===---------------------------------------------------------------------===//

#pragma once
#include <CL/sycl.hpp>
#include <array>
#include <cstdint>
#include <limits>
#include <pybind11/pybind11.h>
#include <utility>
#include <vector>

#include "utils/offset_utils.hpp"
#include "utils/type_dispatch.hpp"

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace accumulators
{

namespace py = pybind11;

using namespace dpctl::tensor::offset_utils;

template <typename T> T ceiling_quotient(T n, T m)
{
    return (n + m - 1) / m;
}
template <typename T1, typename T2> T1 ceiling_quotient(T1 n, T2 m)
{
    return ceiling_quotient<T1>(n, static_cast<T1>(m));
}

template <typename inputT,
          typename outputT,
          size_t n_wi,
          typename IndexerT,
          typename TransformerT>
class inclusive_scan_rec_local_scan_krn;

template <typename inputT,
          typename outputT,
          typename IndexerT,
          typename TransformerT>
class inclusive_scan_rec_chunk_update_krn;

template <typename inputT, typename outputT> struct NonZeroIndicator
{
    NonZeroIndicator() {}

    outputT operator()(const inputT &val) const
    {
        constexpr outputT out_one(1);
        constexpr outputT out_zero(0);
        constexpr inputT val_zero(0);

        return (val == val_zero) ? out_zero : out_one;
    }
};

template <typename T> struct NoOpTransformer
{
    NoOpTransformer() {}

    T operator()(const T &val) const
    {
        return val;
    }
};

/*
 * for integer type maskT,
 *       output[j] = sum( input[s0 + i * s1], 0 <= i <= j)
 * for 0 <= j < n_elems
 */
template <typename inputT,
          typename outputT,
          size_t n_wi,
          typename IndexerT,
          typename TransformerT>
sycl::event inclusive_scan_rec(sycl::queue &exec_q,
                               size_t n_elems,
                               size_t wg_size,
                               const inputT *input,
                               outputT *output,
                               size_t s0,
                               size_t s1,
                               IndexerT indexer,
                               TransformerT transformer,
                               std::vector<sycl::event> const &depends = {})
{
    size_t n_groups = ceiling_quotient(n_elems, n_wi * wg_size);

    const sycl::event &inc_scan_phase1_ev =
        exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            using slmT = sycl::local_accessor<size_t, 1>;

            auto lws = sycl::range<1>(wg_size);
            auto gws = sycl::range<1>(n_groups * wg_size);

            slmT slm_iscan_tmp(lws, cgh);

        cgh.parallel_for<class inclusive_scan_rec_local_scan_krn<
            inputT, outputT, n_wi, IndexerT, decltype(transformer)>>(
            sycl::nd_range<1>(gws, lws), [=, slm_iscan_tmp = std::move(slm_iscan_tmp)](sycl::nd_item<1> it)
        {
            auto chunk_gid = it.get_global_id(0);
            auto lid = it.get_local_id(0);

            std::array<size_t, n_wi> local_isum;

            size_t i = chunk_gid * n_wi;
            for (size_t m_wi = 0; m_wi < n_wi; ++m_wi) {
                constexpr outputT out_zero(0);

                local_isum[m_wi] =
                    (i + m_wi < n_elems)
                        ? transformer(input[indexer(s0 + s1 * (i + m_wi))])
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
        });
        });

    sycl::event out_event = inc_scan_phase1_ev;
    if (n_groups > 1) {
        outputT *temp = sycl::malloc_device<outputT>(n_groups - 1, exec_q);

        auto chunk_size = wg_size * n_wi;

        NoOpIndexer _no_op_indexer{};
        NoOpTransformer<outputT> _no_op_transformer{};
        auto e2 = inclusive_scan_rec<outputT, outputT, n_wi, NoOpIndexer,
                                     decltype(_no_op_transformer)>(
            exec_q, n_groups - 1, wg_size, output, temp, chunk_size - 1,
            chunk_size, _no_op_indexer, _no_op_transformer,
            {inc_scan_phase1_ev});

        // output[ chunk_size * (i + 1) + j] += temp[i]
        auto e3 = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(e2);
            cgh.parallel_for<class inclusive_scan_rec_chunk_update_krn<
                inputT, outputT, IndexerT, decltype(transformer)>>(
                {n_elems}, [=](auto wiid)
            {
                auto gid = wiid[0];
                auto i = (gid / chunk_size);
                output[gid] += (i > 0) ? temp[i - 1] : 0;
            });
        });

        sycl::event e4 = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(e3);
            const auto &ctx = exec_q.get_context();
            cgh.host_task([ctx, temp]() { sycl::free(temp, ctx); });
        });

        out_event = std::move(e4);
    }

    return out_event;
}

typedef size_t (*accumulate_contig_impl_fn_ptr_t)(
    sycl::queue &,
    size_t,
    const char *,
    char *,
    std::vector<sycl::event> const &);

template <typename maskT, typename cumsumT, typename transformerT>
size_t accumulate_contig_impl(sycl::queue &q,
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
    transformerT non_zero_indicator{};

    const sycl::event &comp_ev =
        inclusive_scan_rec<maskT, cumsumT, n_wi, decltype(flat_indexer),
                           decltype(non_zero_indicator)>(
            q, n_elems, wg_size, mask_data_ptr, cumsum_data_ptr, 0, 1,
            flat_indexer, non_zero_indicator, depends);

    cumsumT *last_elem = cumsum_data_ptr + (n_elems - 1);

    cumsumT *last_elem_host_usm = sycl::malloc_host<cumsumT>(1, q);

    if (last_elem_host_usm == nullptr) {
        throw std::bad_alloc();
    }
    sycl::event copy_e =
        q.copy<cumsumT>(last_elem, last_elem_host_usm, 1, {comp_ev});
    copy_e.wait();
    size_t return_val = static_cast<size_t>(*last_elem_host_usm);
    sycl::free(last_elem_host_usm, q);

    return return_val;
}

template <typename fnT, typename T> struct MaskPositionsContigFactoryForInt32
{
    fnT get()
    {
        using cumsumT = std::int32_t;
        fnT fn =
            accumulate_contig_impl<T, cumsumT, NonZeroIndicator<T, cumsumT>>;
        return fn;
    }
};

template <typename fnT, typename T> struct MaskPositionsContigFactoryForInt64
{
    fnT get()
    {
        using cumsumT = std::int64_t;
        fnT fn =
            accumulate_contig_impl<T, cumsumT, NonZeroIndicator<T, cumsumT>>;
        return fn;
    }
};

template <typename fnT, typename T> struct Cumsum1DContigFactory
{
    fnT get()
    {
        if constexpr (std::is_integral_v<T>) {
            using cumsumT = std::int64_t;
            fnT fn =
                accumulate_contig_impl<T, cumsumT, NoOpTransformer<cumsumT>>;
            return fn;
        }
        else {
            return nullptr;
        }
    }
};

typedef size_t (*accumulate_strided_impl_fn_ptr_t)(
    sycl::queue &,
    size_t,
    const char *,
    int,
    const py::ssize_t *,
    char *,
    std::vector<sycl::event> const &);

template <typename maskT, typename cumsumT, typename transformerT>
size_t accumulate_strided_impl(sycl::queue &q,
                               size_t n_elems,
                               const char *mask,
                               int nd,
                               const py::ssize_t *shape_strides,
                               char *cumsum,
                               std::vector<sycl::event> const &depends = {})
{
    constexpr int n_wi = 8;
    const maskT *mask_data_ptr = reinterpret_cast<const maskT *>(mask);
    cumsumT *cumsum_data_ptr = reinterpret_cast<cumsumT *>(cumsum);
    size_t wg_size = 128;

    StridedIndexer strided_indexer{nd, 0, shape_strides};
    transformerT non_zero_indicator{};

    const sycl::event &comp_ev =
        inclusive_scan_rec<maskT, cumsumT, n_wi, decltype(strided_indexer),
                           decltype(non_zero_indicator)>(
            q, n_elems, wg_size, mask_data_ptr, cumsum_data_ptr, 0, 1,
            strided_indexer, non_zero_indicator, depends);

    cumsumT *last_elem = cumsum_data_ptr + (n_elems - 1);

    cumsumT *last_elem_host_usm = sycl::malloc_host<cumsumT>(1, q);

    if (last_elem_host_usm == nullptr) {
        throw std::bad_alloc();
    }
    sycl::event copy_e =
        q.copy<cumsumT>(last_elem, last_elem_host_usm, 1, {comp_ev});
    copy_e.wait();
    size_t return_val = static_cast<size_t>(*last_elem_host_usm);
    sycl::free(last_elem_host_usm, q);

    return return_val;
}

template <typename fnT, typename T> struct MaskPositionsStridedFactoryForInt32
{
    fnT get()
    {
        using cumsumT = std::int32_t;
        fnT fn =
            accumulate_strided_impl<T, cumsumT, NonZeroIndicator<T, cumsumT>>;
        return fn;
    }
};

template <typename fnT, typename T> struct MaskPositionsStridedFactoryForInt64
{
    fnT get()
    {
        using cumsumT = std::int64_t;
        fnT fn =
            accumulate_strided_impl<T, cumsumT, NonZeroIndicator<T, cumsumT>>;
        return fn;
    }
};

template <typename fnT, typename T> struct Cumsum1DStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_integral_v<T>) {
            using cumsumT = std::int64_t;
            fnT fn =
                accumulate_strided_impl<T, cumsumT, NoOpTransformer<cumsumT>>;
            return fn;
        }
        else {
            return nullptr;
        }
    }
};

} // namespace accumulators
} // namespace kernels
} // namespace tensor
} // namespace dpctl
