//===  accumulators.hpp - Implementation of accumulator kernels --*-C++-*-/===//
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
#include "utils/sycl_utils.hpp"
#include "utils/type_dispatch_building.hpp"
#include "utils/type_utils.hpp"

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
    constexpr NonZeroIndicator() {}

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
    constexpr NoOpTransformer() {}

    T operator()(const T &val) const
    {
        return val;
    }
};

template <typename srcTy, typename dstTy> struct CastTransformer
{
    constexpr CastTransformer() {}

    dstTy operator()(const srcTy &val) const
    {
        using dpctl::tensor::type_utils::convert_impl;
        return convert_impl<dstTy, srcTy>(val);
    }
};

template <typename BinOpT, typename T> struct can_use_inclusive_scan_over_group
{
    static constexpr bool value = sycl::has_known_identity<BinOpT, T>::value;
};

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

    size_t get_size() const
    {
        return size_;
    }

    T *get_local_scans_ptr() const
    {
        return local_scans_;
    }
};

template <typename T> class stack_strided_t
{
    T *src_;
    size_t size_;
    T *local_scans_;
    size_t local_stride_;

public:
    stack_strided_t() : src_{}, size_{}, local_scans_{}, local_stride_{} {}
    stack_strided_t(T *src, size_t sz, T *local_scans, size_t local_stride)
        : src_(src), size_(sz), local_scans_(local_scans),
          local_stride_(local_stride)
    {
    }
    ~stack_strided_t(){};

    T *get_src_ptr() const
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

    size_t get_local_stride() const
    {
        return local_stride_;
    }
};

} // end of anonymous namespace

// Iterative cumulative summation

namespace su_ns = dpctl::tensor::sycl_utils;

using nwiT = std::uint16_t;

template <typename inputT,
          typename outputT,
          nwiT n_wi,
          typename IterIndexerT,
          typename InpIndexerT,
          typename OutIndexerT,
          typename TransformerT,
          typename ScanOpT,
          bool include_initial>
class inclusive_scan_iter_local_scan_krn;

template <typename inputT,
          typename outputT,
          nwiT n_wi,
          typename TransformerT,
          typename OtherTransformerT,
          typename ScanOpT,
          bool include_initial>
class inclusive_scan_iter_chunk_update_krn;

template <typename inputT,
          typename outputT,
          nwiT n_wi,
          typename IterIndexerT,
          typename InpIndexerT,
          typename OutIndexerT,
          typename TransformerT,
          typename ScanOpT,
          bool include_initial = false>
sycl::event
inclusive_scan_base_step(sycl::queue &exec_q,
                         const size_t wg_size,
                         const size_t iter_nelems,
                         const size_t acc_nelems,
                         const inputT *input,
                         outputT *output,
                         const size_t s0,
                         const size_t s1,
                         const IterIndexerT &iter_indexer,
                         const InpIndexerT &inp_indexer,
                         const OutIndexerT &out_indexer,
                         TransformerT transformer,
                         const ScanOpT &scan_op,
                         outputT identity,
                         size_t &acc_groups,
                         const std::vector<sycl::event> &depends = {})
{
    acc_groups = ceiling_quotient<size_t>(acc_nelems, n_wi * wg_size);

    sycl::event inc_scan_phase1_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        using slmT = sycl::local_accessor<outputT, 1>;

        auto gws = sycl::range<1>(iter_nelems * acc_groups * wg_size);
        auto lws = sycl::range<1>(wg_size);

        auto ndRange = sycl::nd_range<1>(gws, lws);

        slmT slm_iscan_tmp(lws, cgh);

        using KernelName = inclusive_scan_iter_local_scan_krn<
            inputT, outputT, n_wi, IterIndexerT, InpIndexerT, OutIndexerT,
            TransformerT, ScanOpT, include_initial>;

        cgh.parallel_for<KernelName>(ndRange, [=, slm_iscan_tmp =
                                                      std::move(slm_iscan_tmp)](
                                                  sycl::nd_item<1> it) {
            const size_t gid = it.get_global_id(0);
            const size_t lid = it.get_local_id(0);

            const size_t iter_gid = gid / (acc_groups * wg_size);
            const size_t chunk_gid = gid - (iter_gid * acc_groups * wg_size);

            std::array<outputT, n_wi> local_iscan;

            size_t i = chunk_gid * n_wi;
            const auto &iter_offsets = iter_indexer(iter_gid);
            const auto &inp_iter_offset = iter_offsets.get_first_offset();
            const auto &out_iter_offset = iter_offsets.get_second_offset();

#pragma unroll
            for (nwiT m_wi = 0; m_wi < n_wi; ++m_wi) {
                if constexpr (!include_initial) {
                    local_iscan[m_wi] =
                        (i + m_wi < acc_nelems)
                            ? transformer(
                                  input[inp_iter_offset +
                                        inp_indexer(s0 + s1 * (i + m_wi))])
                            : identity;
                }
                else {
                    // shift input to the left by a single element relative to
                    // output
                    local_iscan[m_wi] =
                        (i + m_wi < acc_nelems && i + m_wi > 0)
                            ? transformer(
                                  input[inp_iter_offset +
                                        inp_indexer((s0 + s1 * (i + m_wi)) -
                                                    1)])
                            : identity;
                }
            }

#pragma unroll
            for (nwiT m_wi = 1; m_wi < n_wi; ++m_wi) {
                local_iscan[m_wi] =
                    scan_op(local_iscan[m_wi], local_iscan[m_wi - 1]);
            }
            // local_iscan is now result of
            // inclusive scan of locally stored inputs

            outputT wg_iscan_val;
            if constexpr (can_use_inclusive_scan_over_group<ScanOpT,
                                                            outputT>::value) {
                wg_iscan_val = sycl::inclusive_scan_over_group(
                    it.get_group(), local_iscan.back(), scan_op, identity);
            }
            else {
                wg_iscan_val = su_ns::custom_inclusive_scan_over_group(
                    it.get_group(), slm_iscan_tmp, local_iscan.back(), scan_op);
                // ensure all finished reading from SLM, to avoid race condition
                // with subsequent writes into SLM
                it.barrier(sycl::access::fence_space::local_space);
            }

            slm_iscan_tmp[(lid + 1) % wg_size] = wg_iscan_val;
            it.barrier(sycl::access::fence_space::local_space);
            outputT addand = (lid == 0) ? identity : slm_iscan_tmp[lid];

#pragma unroll
            for (nwiT m_wi = 0; m_wi < n_wi; ++m_wi) {
                local_iscan[m_wi] = scan_op(local_iscan[m_wi], addand);
            }

            for (nwiT m_wi = 0; (m_wi < n_wi) && (i + m_wi < acc_nelems);
                 ++m_wi) {
                output[out_iter_offset + out_indexer(i + m_wi)] =
                    local_iscan[m_wi];
            }
        });
    });

    return inc_scan_phase1_ev;
}

template <typename inputT,
          typename outputT,
          nwiT n_wi,
          typename TransformerT,
          typename OtherIndexerT,
          typename OtherTransformerT,
          typename ScanOpT,
          bool include_initial>
class inclusive_scan_1d_iter_chunk_update_krn;

/*
 * output[j] = sum( input[s0 + i * s1], 0 <= i <= j)
 * for 0 <= j < n_elems
 */
template <typename inputT,
          typename outputT,
          nwiT n_wi,
          typename IndexerT,
          typename TransformerT,
          typename ScanOpT,
          bool include_initial>
sycl::event inclusive_scan_iter_1d(sycl::queue &exec_q,
                                   const size_t wg_size,
                                   const size_t n_elems,
                                   const inputT *input,
                                   outputT *output,
                                   const size_t s0,
                                   const size_t s1,
                                   const IndexerT &indexer,
                                   const TransformerT &transformer,
                                   std::vector<sycl::event> &host_tasks,
                                   const std::vector<sycl::event> &depends = {})
{
    ScanOpT scan_op = ScanOpT();
    constexpr outputT identity = su_ns::Identity<ScanOpT, outputT>::value;

    constexpr size_t _iter_nelems = 1;

    using IterIndexerT = dpctl::tensor::offset_utils::TwoZeroOffsets_Indexer;
    constexpr IterIndexerT _no_op_iter_indexer{};

    using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
    constexpr NoOpIndexerT _no_op_indexer{};

    size_t n_groups;
    sycl::event inc_scan_phase1_ev =
        inclusive_scan_base_step<inputT, outputT, n_wi, IterIndexerT, IndexerT,
                                 NoOpIndexerT, TransformerT, ScanOpT,
                                 include_initial>(
            exec_q, wg_size, _iter_nelems, n_elems, input, output, s0, s1,
            _no_op_iter_indexer, indexer, _no_op_indexer, transformer, scan_op,
            identity, n_groups, depends);

    sycl::event dependent_event = inc_scan_phase1_ev;
    if (n_groups > 1) {
        const size_t chunk_size = wg_size * n_wi;

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

        using NoOpTransformerT = NoOpTransformer<outputT>;
        constexpr NoOpTransformerT _no_op_transformer{};
        size_t size_to_update = n_elems;
        while (n_groups_ > 1) {

            const size_t src_size = n_groups_ - 1;
            dependent_event =
                inclusive_scan_base_step<outputT, outputT, n_wi, IterIndexerT,
                                         NoOpIndexerT, NoOpIndexerT,
                                         NoOpTransformerT, ScanOpT>(
                    exec_q, wg_size, _iter_nelems, src_size, src, local_scans,
                    chunk_size - 1, chunk_size, _no_op_iter_indexer,
                    _no_op_indexer, _no_op_indexer, _no_op_transformer, scan_op,
                    identity, n_groups_, // n_groups_ is modified in place
                    {dependent_event});
            stack.push_back({src, size_to_update, local_scans});
            src = local_scans;
            local_scans += src_size;
            size_to_update = src_size;
        }

        for (size_t reverse_stack_id = 0; reverse_stack_id < stack.size();
             ++reverse_stack_id)
        {
            const size_t stack_id = stack.size() - 1 - reverse_stack_id;

            const auto &stack_elem = stack[stack_id];
            outputT *src = stack_elem.get_src_ptr();
            const size_t src_size = stack_elem.get_size();
            outputT *local_scans = stack_elem.get_local_scans_ptr();

            // output[ chunk_size * (i + 1) + j] += temp[i]
            dependent_event = exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(dependent_event);

                constexpr nwiT updates_per_wi = n_wi;
                const size_t n_items = ceiling_quotient<size_t>(src_size, n_wi);

                using UpdateKernelName =
                    class inclusive_scan_1d_iter_chunk_update_krn<
                        inputT, outputT, n_wi, IndexerT, TransformerT,
                        NoOpTransformerT, ScanOpT, include_initial>;

                cgh.parallel_for<UpdateKernelName>(
                    {n_items}, [chunk_size, src, src_size, local_scans, scan_op,
                                identity](auto wiid) {
                        const size_t gid = n_wi * wiid[0];
#pragma unroll
                        for (size_t i = 0; i < updates_per_wi; ++i) {
                            const size_t src_id = gid + i;
                            if (src_id < src_size) {
                                const size_t scan_id = (src_id / chunk_size);
                                src[src_id] =
                                    (scan_id > 0)
                                        ? scan_op(src[src_id],
                                                  local_scans[scan_id - 1])
                                        : scan_op(src[src_id], identity);
                            }
                        }
                    });
            });
        }

        sycl::event free_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(dependent_event);
            const auto &ctx = exec_q.get_context();
            cgh.host_task([ctx, temp]() { sycl::free(temp, ctx); });
        });
        host_tasks.push_back(free_ev);
    }

    return dependent_event;
}

typedef sycl::event (*accumulate_1d_contig_impl_fn_ptr_t)(
    sycl::queue &,
    size_t,
    const char *,
    char *,
    std::vector<sycl::event> &,
    const std::vector<sycl::event> &);

template <typename srcT,
          typename dstT,
          typename transformerT,
          typename AccumulateOpT,
          bool include_initial>
sycl::event
accumulate_1d_contig_impl(sycl::queue &q,
                          size_t n_elems,
                          const char *src,
                          char *dst,
                          std::vector<sycl::event> &host_tasks,
                          const std::vector<sycl::event> &depends = {})
{
    const srcT *src_data_ptr = reinterpret_cast<const srcT *>(src);
    dstT *dst_data_ptr = reinterpret_cast<dstT *>(dst);

    using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
    constexpr NoOpIndexerT flat_indexer{};
    constexpr transformerT transformer{};

    constexpr size_t s0 = 0;
    constexpr size_t s1 = 1;

    sycl::event comp_ev;
    const sycl::device &dev = q.get_device();
    if (dev.has(sycl::aspect::cpu)) {
        constexpr nwiT n_wi_for_cpu = 8;
        const size_t wg_size = 256;
        comp_ev = inclusive_scan_iter_1d<srcT, dstT, n_wi_for_cpu, NoOpIndexerT,
                                         transformerT, AccumulateOpT,
                                         include_initial>(
            q, wg_size, n_elems, src_data_ptr, dst_data_ptr, s0, s1,
            flat_indexer, transformer, host_tasks, depends);
    }
    else {
        constexpr nwiT n_wi_for_gpu = 4;
        const size_t wg_size = 256;
        comp_ev = inclusive_scan_iter_1d<srcT, dstT, n_wi_for_gpu, NoOpIndexerT,
                                         transformerT, AccumulateOpT,
                                         include_initial>(
            q, wg_size, n_elems, src_data_ptr, dst_data_ptr, s0, s1,
            flat_indexer, transformer, host_tasks, depends);
    }
    return comp_ev;
}

template <typename inputT,
          typename outputT,
          nwiT n_wi,
          typename IterIndexerT,
          typename IndexerT,
          typename TransformerT,
          typename OtherTransformerT,
          typename ScanOpT,
          bool include_initial>
class inclusive_scan_final_chunk_update_krn;

template <typename inputT,
          typename outputT,
          nwiT n_wi,
          typename InpIterIndexerT,
          typename OutIterIndexerT,
          typename InpIndexerT,
          typename OutIndexerT,
          typename TransformerT,
          typename ScanOpT,
          bool include_initial>
sycl::event inclusive_scan_iter(sycl::queue &exec_q,
                                const size_t wg_size,
                                const size_t iter_nelems,
                                const size_t acc_nelems,
                                const inputT *input,
                                outputT *output,
                                const size_t s0,
                                const size_t s1,
                                const InpIterIndexerT &inp_iter_indexer,
                                const OutIterIndexerT &out_iter_indexer,
                                const InpIndexerT &inp_indexer,
                                const OutIndexerT &out_indexer,
                                const TransformerT &transformer,
                                std::vector<sycl::event> &host_tasks,
                                const std::vector<sycl::event> &depends = {})
{
    ScanOpT scan_op = ScanOpT();
    constexpr outputT identity = su_ns::Identity<ScanOpT, outputT>::value;

    using IterIndexerT =
        dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
            InpIterIndexerT, OutIterIndexerT>;
    const IterIndexerT iter_indexer{inp_iter_indexer, out_iter_indexer};

    size_t acc_groups;
    sycl::event inc_scan_phase1_ev =
        inclusive_scan_base_step<inputT, outputT, n_wi, IterIndexerT,
                                 InpIndexerT, OutIndexerT, TransformerT,
                                 ScanOpT, include_initial>(
            exec_q, wg_size, iter_nelems, acc_nelems, input, output, s0, s1,
            iter_indexer, inp_indexer, out_indexer, transformer, scan_op,
            identity, acc_groups, depends);

    sycl::event dependent_event = inc_scan_phase1_ev;
    if (acc_groups > 1) {
        const size_t chunk_size = wg_size * n_wi;

        // how much of temporary allocation do we need
        size_t acc_groups_ = acc_groups;
        size_t temp_size = 0;
        while (acc_groups_ > 1) {
            const size_t this_size = (acc_groups_ - 1);
            temp_size += this_size;
            acc_groups_ = ceiling_quotient<size_t>(this_size, chunk_size);
        }

        // allocate
        outputT *temp =
            sycl::malloc_device<outputT>(iter_nelems * temp_size, exec_q);

        if (!temp) {
            throw std::bad_alloc();
        }

        std::vector<stack_strided_t<outputT>> stack{};

        // inclusive scans over blocks
        acc_groups_ = acc_groups;
        outputT *src = output;
        outputT *local_scans = temp;

        using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        constexpr NoOpIndexerT _no_op_indexer{};
        using NoOpTransformerT = NoOpTransformer<outputT>;
        constexpr NoOpTransformerT _no_op_transformer{};
        size_t size_to_update = acc_nelems;

        {
            size_t src_size = acc_groups - 1;
            using LocalScanIndexerT =
                dpctl::tensor::offset_utils::Strided1DIndexer;
            const LocalScanIndexerT scan_iter_indexer{
                0, static_cast<ssize_t>(iter_nelems),
                static_cast<ssize_t>(src_size)};

            using IterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    OutIterIndexerT, LocalScanIndexerT>;
            const IterIndexerT iter_indexer_{out_iter_indexer,
                                             scan_iter_indexer};

            dependent_event =
                inclusive_scan_base_step<outputT, outputT, n_wi, IterIndexerT,
                                         OutIndexerT, NoOpIndexerT,
                                         NoOpTransformerT, ScanOpT>(
                    exec_q, wg_size, iter_nelems, src_size, src, local_scans,
                    chunk_size - 1, chunk_size, iter_indexer_, out_indexer,
                    _no_op_indexer, _no_op_transformer, scan_op, identity,
                    acc_groups_, // acc_groups_ is modified in place
                    {dependent_event});
            stack.push_back({src, size_to_update, local_scans, src_size});
            src = local_scans;
            local_scans += src_size * iter_nelems;
            size_to_update = src_size;
        }

        while (acc_groups_ > 1) {
            size_t src_size = acc_groups_ - 1;

            using LocalScanIndexerT =
                dpctl::tensor::offset_utils::Strided1DIndexer;
            const LocalScanIndexerT scan1_iter_indexer{
                0, static_cast<ssize_t>(iter_nelems),
                static_cast<ssize_t>(size_to_update)};
            const LocalScanIndexerT scan2_iter_indexer{
                0, static_cast<ssize_t>(iter_nelems),
                static_cast<ssize_t>(src_size)};

            using IterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    LocalScanIndexerT, LocalScanIndexerT>;
            const IterIndexerT iter_indexer_{scan1_iter_indexer,
                                             scan2_iter_indexer};

            dependent_event =
                inclusive_scan_base_step<outputT, outputT, n_wi, IterIndexerT,
                                         NoOpIndexerT, NoOpIndexerT,
                                         NoOpTransformerT, ScanOpT>(
                    exec_q, wg_size, iter_nelems, src_size, src, local_scans,
                    chunk_size - 1, chunk_size, iter_indexer_, _no_op_indexer,
                    _no_op_indexer, _no_op_transformer, scan_op, identity,
                    acc_groups_, // acc_groups_ is modified in place
                    {dependent_event});
            stack.push_back({src, size_to_update, local_scans, src_size});
            src = local_scans;
            local_scans += src_size * iter_nelems;
            size_to_update = src_size;
        }

        for (size_t reverse_stack_id = 0; reverse_stack_id < stack.size() - 1;
             ++reverse_stack_id)
        {
            const size_t stack_id = stack.size() - 1 - reverse_stack_id;

            const auto &stack_elem = stack[stack_id];
            outputT *src = stack_elem.get_src_ptr();
            size_t src_size = stack_elem.get_size();
            outputT *local_scans = stack_elem.get_local_scans_ptr();
            size_t local_stride = stack_elem.get_local_stride();

            constexpr nwiT updates_per_wi = n_wi;
            const size_t update_nelems =
                ceiling_quotient<size_t>(src_size, updates_per_wi);

            dependent_event = exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(dependent_event);

                using UpdateKernelName =
                    class inclusive_scan_iter_chunk_update_krn<
                        inputT, outputT, n_wi, TransformerT, NoOpTransformerT,
                        ScanOpT, include_initial>;

                cgh.parallel_for<UpdateKernelName>(
                    {iter_nelems * update_nelems},
                    [chunk_size, update_nelems, src_size, local_stride, src,
                     local_scans, scan_op, identity](auto wiid) {
                        const size_t gid = wiid[0];

                        const size_t iter_gid = gid / update_nelems;
                        const size_t axis_gid =
                            gid - (iter_gid * update_nelems);

                        const size_t src_axis_id0 = axis_gid * updates_per_wi;
                        const size_t src_iter_id = iter_gid * src_size;
#pragma unroll
                        for (nwiT i = 0; i < updates_per_wi; ++i) {
                            const size_t src_axis_id = src_axis_id0 + i;
                            const size_t src_id = src_axis_id + src_iter_id;

                            if (src_axis_id < src_size) {
                                const size_t scan_axis_id =
                                    src_axis_id / chunk_size;
                                const size_t scan_id =
                                    scan_axis_id + iter_gid * local_stride;

                                src[src_id] =
                                    (scan_axis_id > 0)
                                        ? scan_op(src[src_id],
                                                  local_scans[scan_id - 1])
                                        : scan_op(src[src_id], identity);
                            }
                        }
                    });
            });
        }

        // last stack element is always directly to output
        {
            const auto &stack_elem = stack[0];
            outputT *src = stack_elem.get_src_ptr();
            const size_t src_size = stack_elem.get_size();
            outputT *local_scans = stack_elem.get_local_scans_ptr();
            const size_t local_stride = stack_elem.get_local_stride();

            constexpr nwiT updates_per_wi = n_wi;
            const size_t update_nelems =
                ceiling_quotient<size_t>(src_size, updates_per_wi);

            dependent_event = exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(dependent_event);

                using UpdateKernelName =
                    class inclusive_scan_final_chunk_update_krn<
                        inputT, outputT, n_wi, OutIterIndexerT, OutIndexerT,
                        TransformerT, NoOpTransformerT, ScanOpT,
                        include_initial>;

                cgh.parallel_for<UpdateKernelName>(
                    {iter_nelems * update_nelems},
                    [chunk_size, update_nelems, src_size, local_stride, src,
                     local_scans, scan_op, identity, out_iter_indexer,
                     out_indexer](auto wiid) {
                        const size_t gid = wiid[0];

                        const size_t iter_gid = gid / update_nelems;
                        const size_t axis_gid =
                            gid - (iter_gid * update_nelems);

                        const size_t src_axis_id0 = axis_gid * updates_per_wi;
                        const size_t src_iter_id = out_iter_indexer(iter_gid);
#pragma unroll
                        for (nwiT i = 0; i < updates_per_wi; ++i) {
                            const size_t src_axis_id = src_axis_id0 + i;
                            const size_t src_id =
                                out_indexer(src_axis_id) + src_iter_id;

                            if (src_axis_id < src_size) {
                                const size_t scan_axis_id =
                                    src_axis_id / chunk_size;
                                const size_t scan_id =
                                    scan_axis_id + iter_gid * local_stride;

                                src[src_id] =
                                    (scan_axis_id > 0)
                                        ? scan_op(src[src_id],
                                                  local_scans[scan_id - 1])
                                        : scan_op(src[src_id], identity);
                            }
                        }
                    });
            });
        }

        sycl::event free_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(dependent_event);
            const auto &ctx = exec_q.get_context();
            cgh.host_task([ctx, temp]() { sycl::free(temp, ctx); });
        });
        host_tasks.push_back(free_ev);
    }

    return dependent_event;
}

typedef sycl::event (*accumulate_strided_impl_fn_ptr_t)(
    sycl::queue &,
    size_t,
    size_t,
    const char *,
    int,
    const ssize_t *,
    ssize_t,
    ssize_t,
    int,
    const ssize_t *,
    char *,
    std::vector<sycl::event> &,
    const std::vector<sycl::event> &);

template <typename srcT,
          typename dstT,
          typename transformerT,
          typename AccumulateOpT,
          bool include_initial>
sycl::event
accumulate_strided_impl(sycl::queue &q,
                        size_t iter_nelems,
                        size_t acc_nelems,
                        const char *src,
                        int iter_nd,
                        const ssize_t *iter_shape_strides,
                        ssize_t inp_iter_offset,
                        ssize_t out_iter_offset,
                        int acc_nd,
                        const ssize_t *acc_shape_strides,
                        char *dst,
                        std::vector<sycl::event> &host_tasks,
                        const std::vector<sycl::event> &depends = {})
{
    const srcT *src_data_ptr = reinterpret_cast<const srcT *>(src);
    dstT *dst_data_ptr = reinterpret_cast<dstT *>(dst);

    using InpIndexerT = dpctl::tensor::offset_utils::StridedIndexer;
    const InpIndexerT inp_axis_indexer{acc_nd, 0, acc_shape_strides};
    const InpIndexerT inp_iter_indexer{iter_nd, inp_iter_offset,
                                       iter_shape_strides};

    using OutIndexerT = dpctl::tensor::offset_utils::UnpackedStridedIndexer;
    const OutIndexerT out_axis_indexer{acc_nd, 0, acc_shape_strides,
                                       acc_shape_strides + 2 * acc_nd};
    const OutIndexerT out_iter_indexer{iter_nd, out_iter_offset,
                                       iter_shape_strides,
                                       iter_shape_strides + 2 * iter_nd};

    constexpr transformerT transformer{};

    constexpr size_t s0 = 0;
    constexpr size_t s1 = 1;

    const sycl::device &dev = q.get_device();
    sycl::event comp_ev;
    if (dev.has(sycl::aspect::cpu)) {
        constexpr nwiT n_wi_for_cpu = 8;
        const size_t wg_size = 256;
        comp_ev =
            inclusive_scan_iter<srcT, dstT, n_wi_for_cpu, InpIndexerT,
                                OutIndexerT, InpIndexerT, OutIndexerT,
                                transformerT, AccumulateOpT, include_initial>(
                q, wg_size, iter_nelems, acc_nelems, src_data_ptr, dst_data_ptr,
                s0, s1, inp_iter_indexer, out_iter_indexer, inp_axis_indexer,
                out_axis_indexer, transformer, host_tasks, depends);
    }
    else {
        constexpr nwiT n_wi_for_gpu = 4;
        const size_t wg_size = 256;
        comp_ev =
            inclusive_scan_iter<srcT, dstT, n_wi_for_gpu, InpIndexerT,
                                OutIndexerT, InpIndexerT, OutIndexerT,
                                transformerT, AccumulateOpT, include_initial>(
                q, wg_size, iter_nelems, acc_nelems, src_data_ptr, dst_data_ptr,
                s0, s1, inp_iter_indexer, out_iter_indexer, inp_axis_indexer,
                out_axis_indexer, transformer, host_tasks, depends);
    }

    return comp_ev;
}

typedef size_t (*cumsum_val_contig_impl_fn_ptr_t)(
    sycl::queue &,
    size_t,
    const char *,
    char *,
    std::vector<sycl::event> &,
    const std::vector<sycl::event> &);

template <typename maskT, typename cumsumT, typename transformerT>
size_t cumsum_val_contig_impl(sycl::queue &q,
                              size_t n_elems,
                              const char *mask,
                              char *cumsum,
                              std::vector<sycl::event> &host_tasks,
                              const std::vector<sycl::event> &depends = {})
{
    const maskT *mask_data_ptr = reinterpret_cast<const maskT *>(mask);
    cumsumT *cumsum_data_ptr = reinterpret_cast<cumsumT *>(cumsum);

    using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
    constexpr NoOpIndexerT flat_indexer{};
    constexpr transformerT transformer{};

    constexpr size_t s0 = 0;
    constexpr size_t s1 = 1;
    constexpr bool include_initial = false;
    using AccumulateOpT = sycl::plus<cumsumT>;

    sycl::event comp_ev;
    const sycl::device &dev = q.get_device();
    if (dev.has(sycl::aspect::cpu)) {
        constexpr nwiT n_wi_for_cpu = 8;
        const size_t wg_size = 256;
        comp_ev = inclusive_scan_iter_1d<maskT, cumsumT, n_wi_for_cpu,
                                         NoOpIndexerT, transformerT,
                                         AccumulateOpT, include_initial>(
            q, wg_size, n_elems, mask_data_ptr, cumsum_data_ptr, s0, s1,
            flat_indexer, transformer, host_tasks, depends);
    }
    else {
        constexpr nwiT n_wi_for_gpu = 4;
        const size_t wg_size = 256;
        comp_ev = inclusive_scan_iter_1d<maskT, cumsumT, n_wi_for_gpu,
                                         NoOpIndexerT, transformerT,
                                         AccumulateOpT, include_initial>(
            q, wg_size, n_elems, mask_data_ptr, cumsum_data_ptr, s0, s1,
            flat_indexer, transformer, host_tasks, depends);
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
            cumsum_val_contig_impl<T, cumsumT, NonZeroIndicator<T, cumsumT>>;
        return fn;
    }
};

template <typename fnT, typename T> struct MaskPositionsContigFactoryForInt64
{
    fnT get()
    {
        using cumsumT = std::int64_t;
        fnT fn =
            cumsum_val_contig_impl<T, cumsumT, NonZeroIndicator<T, cumsumT>>;
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
                cumsum_val_contig_impl<T, cumsumT, NoOpTransformer<cumsumT>>;
            return fn;
        }
        else {
            return nullptr;
        }
    }
};

typedef size_t (*cumsum_val_strided_impl_fn_ptr_t)(
    sycl::queue &,
    size_t,
    const char *,
    int,
    const ssize_t *,
    char *,
    std::vector<sycl::event> &,
    const std::vector<sycl::event> &);

template <typename maskT, typename cumsumT, typename transformerT>
size_t cumsum_val_strided_impl(sycl::queue &q,
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

    using StridedIndexerT = dpctl::tensor::offset_utils::StridedIndexer;
    const StridedIndexerT strided_indexer{nd, 0, shape_strides};
    constexpr transformerT transformer{};

    constexpr size_t s0 = 0;
    constexpr size_t s1 = 1;
    constexpr bool include_initial = false;
    using AccumulateOpT = sycl::plus<cumsumT>;

    const sycl::device &dev = q.get_device();
    sycl::event comp_ev;
    if (dev.has(sycl::aspect::cpu)) {
        constexpr nwiT n_wi_for_cpu = 8;
        const size_t wg_size = 256;
        comp_ev = inclusive_scan_iter_1d<maskT, cumsumT, n_wi_for_cpu,
                                         StridedIndexerT, transformerT,
                                         AccumulateOpT, include_initial>(
            q, wg_size, n_elems, mask_data_ptr, cumsum_data_ptr, s0, s1,
            strided_indexer, transformer, host_tasks, depends);
    }
    else {
        constexpr nwiT n_wi_for_gpu = 4;
        const size_t wg_size = 256;
        comp_ev = inclusive_scan_iter_1d<maskT, cumsumT, n_wi_for_gpu,
                                         StridedIndexerT, transformerT,
                                         AccumulateOpT, include_initial>(
            q, wg_size, n_elems, mask_data_ptr, cumsum_data_ptr, s0, s1,
            strided_indexer, transformer, host_tasks, depends);
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
            cumsum_val_strided_impl<T, cumsumT, NonZeroIndicator<T, cumsumT>>;
        return fn;
    }
};

template <typename fnT, typename T> struct MaskPositionsStridedFactoryForInt64
{
    fnT get()
    {
        using cumsumT = std::int64_t;
        fnT fn =
            cumsum_val_strided_impl<T, cumsumT, NonZeroIndicator<T, cumsumT>>;
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
                cumsum_val_strided_impl<T, cumsumT, NoOpTransformer<cumsumT>>;
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
