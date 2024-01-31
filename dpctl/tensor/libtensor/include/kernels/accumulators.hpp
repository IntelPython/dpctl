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
#include <array>
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
namespace accumulators
{

using namespace dpctl::tensor::offset_utils;

template <typename T> T ceiling_quotient(T n, T m)
{
    return (n + m - 1) / m;
}

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

// Iterative cumulative summation

using nwiT = std::uint16_t;

template <typename inputT,
          typename outputT,
          nwiT n_wi,
          typename IndexerT,
          typename TransformerT>
class inclusive_scan_iter_local_scan_krn;

template <typename inputT,
          typename outputT,
          nwiT n_wi,
          typename IndexerT,
          typename TransformerT,
          typename OtherIndexerT,
          typename OtherTransformerT>
class inclusive_scan_iter_chunk_update_krn;

template <typename inputT,
          typename outputT,
          nwiT n_wi,
          typename IndexerT,
          typename TransformerT>
sycl::event
inclusive_scan_base_step(sycl::queue &exec_q,
                         const size_t wg_size,
                         const size_t n_elems,
                         const inputT *input,
                         outputT *output,
                         const size_t s0,
                         const size_t s1,
                         IndexerT indexer,
                         TransformerT transformer,
                         size_t &n_groups,
                         const std::vector<sycl::event> &depends = {})
{
    n_groups = ceiling_quotient<size_t>(n_elems, n_wi * wg_size);

    sycl::event inc_scan_phase1_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        using slmT = sycl::local_accessor<outputT, 1>;

        auto lws = sycl::range<1>(wg_size);
        auto gws = sycl::range<1>(n_groups * wg_size);

        auto ndRange = sycl::nd_range<1>(gws, lws);

        slmT slm_iscan_tmp(lws, cgh);

        using KernelName =
            inclusive_scan_iter_local_scan_krn<inputT, outputT, n_wi, IndexerT,
                                               TransformerT>;

        cgh.parallel_for<KernelName>(ndRange, [=, slm_iscan_tmp =
                                                      std::move(slm_iscan_tmp)](
                                                  sycl::nd_item<1> it) {
            auto chunk_gid = it.get_global_id(0);
            auto lid = it.get_local_id(0);

            std::array<outputT, n_wi> local_isum;

            size_t i = chunk_gid * n_wi;

#pragma unroll
            for (nwiT m_wi = 0; m_wi < n_wi; ++m_wi) {
                constexpr outputT out_zero(0);

                local_isum[m_wi] =
                    (i + m_wi < n_elems)
                        ? transformer(input[indexer(s0 + s1 * (i + m_wi))])
                        : out_zero;
            }

#pragma unroll
            for (nwiT m_wi = 1; m_wi < n_wi; ++m_wi) {
                local_isum[m_wi] += local_isum[m_wi - 1];
            }
            // local_isum is now result of
            // inclusive scan of locally stored inputs

            outputT wg_iscan_val = sycl::inclusive_scan_over_group(
                it.get_group(), local_isum.back(), sycl::plus<outputT>(),
                outputT(0));

            slm_iscan_tmp[(lid + 1) % wg_size] = wg_iscan_val;
            it.barrier(sycl::access::fence_space::local_space);
            outputT addand = (lid == 0) ? outputT(0) : slm_iscan_tmp[lid];

#pragma unroll
            for (nwiT m_wi = 0; m_wi < n_wi; ++m_wi) {
                local_isum[m_wi] += addand;
            }

            for (nwiT m_wi = 0; (m_wi < n_wi) && (i + m_wi < n_elems); ++m_wi) {
                output[i + m_wi] = local_isum[m_wi];
            }
        });
    });

    return inc_scan_phase1_ev;
}

namespace
{
template <typename T> class stack_t
{
    T *src_;
    size_t size_;
    T *local_scans_;

public:
    stack_t() : src_{}, size_{}, local_scans_{} {}
    stack_t(T *src, size_t sz, T *local_scans)
        : src_(src), size_(sz), local_scans_(local_scans)
    {
    }
    ~stack_t(){};

    T *get_src_ptr() const
    {
        return src_;
    }

    const T *get_src_const_ptr() const
    {
        return src_;
    }

    size_t get_size() const
    {
        return size_;
    }

    T *get_local_scans_ptr() const
    {
        return local_scans_;
    }
};
} // end of anonymous namespace

/*
 * for integer type maskT,
 *       output[j] = sum( input[s0 + i * s1], 0 <= i <= j)
 * for 0 <= j < n_elems
 */
template <typename inputT,
          typename outputT,
          nwiT n_wi,
          typename IndexerT,
          typename TransformerT>
sycl::event inclusive_scan_iter(sycl::queue &exec_q,
                                const size_t wg_size,
                                const size_t n_elems,
                                const inputT *input,
                                outputT *output,
                                const size_t s0,
                                const size_t s1,
                                IndexerT indexer,
                                TransformerT transformer,
                                std::vector<sycl::event> &host_tasks,
                                const std::vector<sycl::event> &depends = {})
{
    size_t n_groups;
    sycl::event inc_scan_phase1_ev =
        inclusive_scan_base_step<inputT, outputT, n_wi, IndexerT, TransformerT>(
            exec_q, wg_size, n_elems, input, output, s0, s1, indexer,
            transformer, n_groups, depends);

    sycl::event dependent_event = inc_scan_phase1_ev;
    if (n_groups > 1) {
        auto chunk_size = wg_size * n_wi;

        // how much of temporary allocation do we need
        size_t n_groups_ = n_groups;
        size_t temp_size = 0;
        while (n_groups_ > 1) {
            const auto this_size = (n_groups_ - 1);
            temp_size += this_size;
            n_groups_ = ceiling_quotient<size_t>(this_size, chunk_size);
        }

        // allocate
        outputT *temp = sycl::malloc_device<outputT>(temp_size, exec_q);

        if (!temp) {
            throw std::bad_alloc();
        }

        std::vector<stack_t<outputT>> stack{};

        // inclusive scans over blocks
        n_groups_ = n_groups;
        outputT *src = output;
        outputT *local_scans = temp;

        NoOpIndexer _no_op_indexer{};
        using NoOpTransformerT = NoOpTransformer<outputT>;
        NoOpTransformerT _no_op_transformer{};
        size_t size_to_update = n_elems;
        while (n_groups_ > 1) {

            size_t src_size = n_groups_ - 1;
            dependent_event =
                inclusive_scan_base_step<outputT, outputT, n_wi, NoOpIndexer,
                                         NoOpTransformerT>(
                    exec_q, wg_size, src_size, src, local_scans, chunk_size - 1,
                    chunk_size, _no_op_indexer, _no_op_transformer,
                    n_groups_, // n_groups_ is modified in place
                    {dependent_event});
            stack.push_back({src, size_to_update, local_scans});
            src = local_scans;
            local_scans += src_size;
            size_to_update = src_size;
        }

        for (size_t reverse_stack_id = 0; reverse_stack_id < stack.size();
             ++reverse_stack_id)
        {
            auto stack_id = stack.size() - 1 - reverse_stack_id;

            auto stack_elem = stack[stack_id];
            outputT *src = stack_elem.get_src_ptr();
            size_t src_size = stack_elem.get_size();
            outputT *local_scans = stack_elem.get_local_scans_ptr();

            // output[ chunk_size * (i + 1) + j] += temp[i]
            dependent_event = exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(dependent_event);

                using UpdateKernelName =
                    class inclusive_scan_iter_chunk_update_krn<
                        inputT, outputT, n_wi, IndexerT, TransformerT,
                        NoOpIndexer, NoOpTransformerT>;

                cgh.parallel_for<UpdateKernelName>(
                    {src_size}, [chunk_size, src, local_scans](auto wiid) {
                        auto gid = wiid[0];
                        auto i = (gid / chunk_size);
                        src[gid] += (i > 0) ? local_scans[i - 1] : outputT(0);
                    });
            });
        }

        sycl::event e4 = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(dependent_event);
            const auto &ctx = exec_q.get_context();
            cgh.host_task([ctx, temp]() { sycl::free(temp, ctx); });
        });
        host_tasks.push_back(e4);
    }

    return dependent_event;
}

typedef size_t (*accumulate_contig_impl_fn_ptr_t)(
    sycl::queue &,
    size_t,
    const char *,
    char *,
    std::vector<sycl::event> &,
    const std::vector<sycl::event> &);

template <typename maskT, typename cumsumT, typename transformerT>
size_t accumulate_contig_impl(sycl::queue &q,
                              size_t n_elems,
                              const char *mask,
                              char *cumsum,
                              std::vector<sycl::event> &host_tasks,
                              const std::vector<sycl::event> &depends = {})
{
    const maskT *mask_data_ptr = reinterpret_cast<const maskT *>(mask);
    cumsumT *cumsum_data_ptr = reinterpret_cast<cumsumT *>(cumsum);

    NoOpIndexer flat_indexer{};
    transformerT non_zero_indicator{};

    sycl::event comp_ev;

    const sycl::device &dev = q.get_device();
    if (dev.has(sycl::aspect::cpu)) {
        constexpr nwiT n_wi_for_cpu = 8;
        size_t wg_size = 256;
        comp_ev = inclusive_scan_iter<maskT, cumsumT, n_wi_for_cpu,
                                      decltype(flat_indexer),
                                      decltype(non_zero_indicator)>(
            q, wg_size, n_elems, mask_data_ptr, cumsum_data_ptr, 0, 1,
            flat_indexer, non_zero_indicator, host_tasks, depends);
    }
    else {
        constexpr nwiT n_wi_for_gpu = 4;
        size_t wg_size = 256;
        comp_ev = inclusive_scan_iter<maskT, cumsumT, n_wi_for_gpu,
                                      decltype(flat_indexer),
                                      decltype(non_zero_indicator)>(
            q, wg_size, n_elems, mask_data_ptr, cumsum_data_ptr, 0, 1,
            flat_indexer, non_zero_indicator, host_tasks, depends);
    }
    cumsumT *last_elem = cumsum_data_ptr + (n_elems - 1);

    cumsumT *last_elem_host_usm = sycl::malloc_host<cumsumT>(1, q);

    if (last_elem_host_usm == nullptr) {
        throw std::bad_alloc();
    }
    sycl::event copy_e = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(comp_ev);
        cgh.copy<cumsumT>(last_elem, last_elem_host_usm, 1);
    });
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
    const ssize_t *,
    char *,
    std::vector<sycl::event> &,
    const std::vector<sycl::event> &);

template <typename maskT, typename cumsumT, typename transformerT>
size_t accumulate_strided_impl(sycl::queue &q,
                               size_t n_elems,
                               const char *mask,
                               int nd,
                               const ssize_t *shape_strides,
                               char *cumsum,
                               std::vector<sycl::event> &host_tasks,
                               const std::vector<sycl::event> &depends = {})
{
    const maskT *mask_data_ptr = reinterpret_cast<const maskT *>(mask);
    cumsumT *cumsum_data_ptr = reinterpret_cast<cumsumT *>(cumsum);

    StridedIndexer strided_indexer{nd, 0, shape_strides};
    transformerT non_zero_indicator{};

    const sycl::device &dev = q.get_device();
    sycl::event comp_ev;
    if (dev.has(sycl::aspect::cpu)) {
        constexpr nwiT n_wi_for_cpu = 8;
        size_t wg_size = 256;
        comp_ev = inclusive_scan_iter<maskT, cumsumT, n_wi_for_cpu,
                                      decltype(strided_indexer),
                                      decltype(non_zero_indicator)>(
            q, wg_size, n_elems, mask_data_ptr, cumsum_data_ptr, 0, 1,
            strided_indexer, non_zero_indicator, host_tasks, depends);
    }
    else {
        constexpr nwiT n_wi_for_gpu = 4;
        size_t wg_size = 256;
        comp_ev = inclusive_scan_iter<maskT, cumsumT, n_wi_for_gpu,
                                      decltype(strided_indexer),
                                      decltype(non_zero_indicator)>(
            q, wg_size, n_elems, mask_data_ptr, cumsum_data_ptr, 0, 1,
            strided_indexer, non_zero_indicator, host_tasks, depends);
    }

    cumsumT *last_elem = cumsum_data_ptr + (n_elems - 1);

    cumsumT *last_elem_host_usm = sycl::malloc_host<cumsumT>(1, q);

    if (last_elem_host_usm == nullptr) {
        throw std::bad_alloc();
    }
    sycl::event copy_e = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(comp_ev);
        cgh.copy<cumsumT>(last_elem, last_elem_host_usm, 1);
    });
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
