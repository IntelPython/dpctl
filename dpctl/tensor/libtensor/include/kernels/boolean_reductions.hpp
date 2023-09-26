//=== boolean_reductions.hpp - Implementation of boolean reduction kernels    //
//                                                            ---*-C++-*--/===//
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
/// This file defines kernels for dpctl.tensor.any and dpctl.tensor.all
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl.hpp>

#include <complex>
#include <cstdint>
#include <utility>
#include <vector>

#include "pybind11/pybind11.h"

#include "utils/offset_utils.hpp"
#include "utils/sycl_utils.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

namespace py = pybind11;

namespace dpctl
{
namespace tensor
{
namespace kernels
{

template <typename T> struct boolean_predicate
{
    bool operator()(const T &v) const
    {
        using dpctl::tensor::type_utils::convert_impl;
        return convert_impl<bool, T>(v);
    }
};

template <typename inpT, typename outT, typename PredicateT>
struct all_reduce_wg_contig
{
    void operator()(sycl::nd_item<1> &ndit,
                    outT *out,
                    const size_t &out_idx,
                    const inpT *start,
                    const inpT *end) const
    {
        PredicateT pred{};
        auto wg = ndit.get_group();
        outT red_val_over_wg =
            static_cast<outT>(sycl::joint_all_of(wg, start, end, pred));

        if (wg.leader()) {
            sycl::atomic_ref<outT, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                res_ref(out[out_idx]);
            res_ref.fetch_and(red_val_over_wg);
        }
    }
};

template <typename inpT, typename outT, typename PredicateT>
struct any_reduce_wg_contig
{
    void operator()(sycl::nd_item<1> &ndit,
                    outT *out,
                    const size_t &out_idx,
                    const inpT *start,
                    const inpT *end) const
    {
        PredicateT pred{};
        auto wg = ndit.get_group();
        outT red_val_over_wg =
            static_cast<outT>(sycl::joint_any_of(wg, start, end, pred));

        if (wg.leader()) {
            sycl::atomic_ref<outT, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                res_ref(out[out_idx]);
            res_ref.fetch_or(red_val_over_wg);
        }
    }
};

template <typename T> struct all_reduce_wg_strided
{
    void operator()(sycl::nd_item<1> &ndit,
                    T *out,
                    const size_t &out_idx,
                    const T &local_val) const
    {
        auto wg = ndit.get_group();
        T red_val_over_wg = static_cast<T>(sycl::all_of_group(wg, local_val));

        if (wg.leader()) {
            sycl::atomic_ref<T, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                res_ref(out[out_idx]);
            res_ref.fetch_and(red_val_over_wg);
        }
    }
};

template <typename T> struct any_reduce_wg_strided
{
    void operator()(sycl::nd_item<1> &ndit,
                    T *out,
                    const size_t &out_idx,
                    const T &local_val) const
    {
        auto wg = ndit.get_group();
        T red_val_over_wg = static_cast<T>(sycl::any_of_group(wg, local_val));

        if (wg.leader()) {
            sycl::atomic_ref<T, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                res_ref(out[out_idx]);
            res_ref.fetch_or(red_val_over_wg);
        }
    }
};

template <typename argT,
          typename outT,
          typename ReductionOp,
          typename InputOutputIterIndexerT,
          typename InputRedIndexerT>
struct SequentialBooleanReduction
{
private:
    const argT *inp_ = nullptr;
    outT *out_ = nullptr;
    ReductionOp reduction_op_;
    outT identity_;
    InputOutputIterIndexerT inp_out_iter_indexer_;
    InputRedIndexerT inp_reduced_dims_indexer_;
    size_t reduction_max_gid_ = 0;

public:
    SequentialBooleanReduction(const argT *inp,
                               outT *res,
                               ReductionOp reduction_op,
                               const outT &identity_val,
                               InputOutputIterIndexerT arg_res_iter_indexer,
                               InputRedIndexerT arg_reduced_dims_indexer,
                               size_t reduction_size)
        : inp_(inp), out_(res), reduction_op_(reduction_op),
          identity_(identity_val), inp_out_iter_indexer_(arg_res_iter_indexer),
          inp_reduced_dims_indexer_(arg_reduced_dims_indexer),
          reduction_max_gid_(reduction_size)
    {
    }

    void operator()(sycl::id<1> id) const
    {

        auto const &inp_out_iter_offsets_ = inp_out_iter_indexer_(id[0]);
        const py::ssize_t &inp_iter_offset =
            inp_out_iter_offsets_.get_first_offset();
        const py::ssize_t &out_iter_offset =
            inp_out_iter_offsets_.get_second_offset();

        outT red_val(identity_);
        for (size_t m = 0; m < reduction_max_gid_; ++m) {
            py::ssize_t inp_reduction_offset =
                static_cast<py::ssize_t>(inp_reduced_dims_indexer_(m));
            py::ssize_t inp_offset = inp_iter_offset + inp_reduction_offset;

            // must convert to boolean first to handle nans
            using dpctl::tensor::type_utils::convert_impl;
            outT val = convert_impl<bool, argT>(inp_[inp_offset]);
            ReductionOp op = reduction_op_;

            red_val = op(red_val, val);
        }

        out_[out_iter_offset] = red_val;
    }
};

template <typename argT, typename outT, typename GroupOp>
struct ContigBooleanReduction
{
private:
    const argT *inp_ = nullptr;
    outT *out_ = nullptr;
    GroupOp group_op_;
    size_t reduction_max_gid_ = 0;
    size_t iter_gws_ = 1;
    size_t reductions_per_wi = 16;

public:
    ContigBooleanReduction(const argT *inp,
                           outT *res,
                           GroupOp group_op,
                           size_t reduction_size,
                           size_t iteration_size,
                           size_t reduction_size_per_wi)
        : inp_(inp), out_(res), group_op_(group_op),
          reduction_max_gid_(reduction_size), iter_gws_(iteration_size),
          reductions_per_wi(reduction_size_per_wi)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        const size_t reduction_id = it.get_group(0) % iter_gws_;
        const size_t reduction_batch_id = it.get_group(0) / iter_gws_;
        const size_t wg_size = it.get_local_range(0);

        const size_t base = reduction_id * reduction_max_gid_;
        const size_t start =
            base + reduction_batch_id * wg_size * reductions_per_wi;
        const size_t end = std::min((start + (reductions_per_wi * wg_size)),
                                    base + reduction_max_gid_);
        // reduction and atomic operations are performed
        // in group_op_
        group_op_(it, out_, reduction_id, inp_ + start, inp_ + end);
    }
};

typedef sycl::event (*boolean_reduction_contig_impl_fn_ptr)(
    sycl::queue &,
    size_t,
    size_t,
    const char *,
    char *,
    py::ssize_t,
    py::ssize_t,
    py::ssize_t,
    const std::vector<sycl::event> &);

template <typename T1, typename T2, typename T3>
class boolean_reduction_contig_krn;

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class boolean_reduction_seq_contig_krn;

using dpctl::tensor::sycl_utils::choose_workgroup_size;

template <typename argTy, typename resTy, typename RedOpT, typename GroupOpT>
sycl::event
boolean_reduction_axis1_contig_impl(sycl::queue &exec_q,
                                    size_t iter_nelems,
                                    size_t reduction_nelems,
                                    const char *arg_cp,
                                    char *res_cp,
                                    py::ssize_t iter_arg_offset,
                                    py::ssize_t iter_res_offset,
                                    py::ssize_t red_arg_offset,
                                    const std::vector<sycl::event> &depends)
{
    const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_cp) +
                          iter_arg_offset + red_arg_offset;
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp) + iter_res_offset;

    constexpr resTy identity_val = sycl::known_identity<RedOpT, resTy>::value;

    const sycl::device &d = exec_q.get_device();
    const auto &sg_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

    sycl::event red_ev;
    if (reduction_nelems < wg) {
        red_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            using InputIterIndexerT =
                dpctl::tensor::offset_utils::Strided1DIndexer;
            using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            using InputOutputIterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    InputIterIndexerT, NoOpIndexerT>;
            using ReductionIndexerT = NoOpIndexerT;

            InputOutputIterIndexerT in_out_iter_indexer{
                InputIterIndexerT{0, static_cast<py::ssize_t>(iter_nelems),
                                  static_cast<py::ssize_t>(reduction_nelems)},
                NoOpIndexerT{}};
            ReductionIndexerT reduction_indexer{};

            cgh.parallel_for<class boolean_reduction_seq_contig_krn<
                argTy, resTy, RedOpT, InputOutputIterIndexerT,
                ReductionIndexerT>>(
                sycl::range<1>(iter_nelems),
                SequentialBooleanReduction<argTy, resTy, RedOpT,
                                           InputOutputIterIndexerT,
                                           ReductionIndexerT>(
                    arg_tp, res_tp, RedOpT(), identity_val, in_out_iter_indexer,
                    reduction_indexer, reduction_nelems));
        });
    }
    else {
        sycl::event init_ev = exec_q.fill<resTy>(res_tp, resTy(identity_val),
                                                 iter_nelems, depends);
        red_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(init_ev);

            constexpr std::uint8_t dim = 1;

            constexpr size_t preferred_reductions_per_wi = 4;
            size_t reductions_per_wi =
                (reduction_nelems < preferred_reductions_per_wi * wg)
                    ? ((reduction_nelems + wg - 1) / wg)
                    : preferred_reductions_per_wi;

            size_t reduction_groups =
                (reduction_nelems + reductions_per_wi * wg - 1) /
                (reductions_per_wi * wg);

            auto gws = sycl::range<dim>{iter_nelems * reduction_groups * wg};
            auto lws = sycl::range<dim>{wg};

            cgh.parallel_for<
                class boolean_reduction_contig_krn<argTy, resTy, GroupOpT>>(
                sycl::nd_range<dim>(gws, lws),
                ContigBooleanReduction<argTy, resTy, GroupOpT>(
                    arg_tp, res_tp, GroupOpT(), reduction_nelems, iter_nelems,
                    reductions_per_wi));
        });
    }
    return red_ev;
}

template <typename fnT, typename srcTy> struct AllAxis1ContigFactory
{
    fnT get() const
    {
        using resTy = std::int32_t;
        using RedOpT = sycl::logical_and<resTy>;
        using GroupOpT =
            all_reduce_wg_contig<srcTy, resTy, boolean_predicate<srcTy>>;

        return dpctl::tensor::kernels::boolean_reduction_axis1_contig_impl<
            srcTy, resTy, RedOpT, GroupOpT>;
    }
};

template <typename fnT, typename srcTy> struct AnyAxis1ContigFactory
{
    fnT get() const
    {
        using resTy = std::int32_t;
        using RedOpT = sycl::logical_or<resTy>;
        using GroupOpT =
            any_reduce_wg_contig<srcTy, resTy, boolean_predicate<srcTy>>;

        return dpctl::tensor::kernels::boolean_reduction_axis1_contig_impl<
            srcTy, resTy, RedOpT, GroupOpT>;
    }
};

template <typename argT,
          typename outT,
          typename ReductionOp,
          typename GroupOp,
          typename InputOutputIterIndexerT,
          typename InputRedIndexerT>
struct StridedBooleanReduction
{
private:
    const argT *inp_ = nullptr;
    outT *out_ = nullptr;
    ReductionOp reduction_op_;
    GroupOp group_op_;
    outT identity_;
    InputOutputIterIndexerT inp_out_iter_indexer_;
    InputRedIndexerT inp_reduced_dims_indexer_;
    size_t reduction_max_gid_ = 0;
    size_t iter_gws_ = 1;
    size_t reductions_per_wi = 16;

public:
    StridedBooleanReduction(const argT *inp,
                            outT *res,
                            ReductionOp reduction_op,
                            GroupOp group_op,
                            const outT &identity_val,
                            InputOutputIterIndexerT arg_res_iter_indexer,
                            InputRedIndexerT arg_reduced_dims_indexer,
                            size_t reduction_size,
                            size_t iteration_size,
                            size_t reduction_size_per_wi)
        : inp_(inp), out_(res), reduction_op_(reduction_op),
          group_op_(group_op), identity_(identity_val),
          inp_out_iter_indexer_(arg_res_iter_indexer),
          inp_reduced_dims_indexer_(arg_reduced_dims_indexer),
          reduction_max_gid_(reduction_size), iter_gws_(iteration_size),
          reductions_per_wi(reduction_size_per_wi)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        const size_t reduction_id = it.get_group(0) % iter_gws_;
        const size_t reduction_batch_id = it.get_group(0) / iter_gws_;

        const size_t reduction_lid = it.get_local_id(0);
        const size_t wg_size = it.get_local_range(0);

        auto inp_out_iter_offsets_ = inp_out_iter_indexer_(reduction_id);
        const py::ssize_t &inp_iter_offset =
            inp_out_iter_offsets_.get_first_offset();
        const py::ssize_t &out_iter_offset =
            inp_out_iter_offsets_.get_second_offset();

        outT local_red_val(identity_);
        size_t arg_reduce_gid0 =
            reduction_lid + reduction_batch_id * wg_size * reductions_per_wi;
        size_t arg_reduce_gid_max = std::min(
            reduction_max_gid_, arg_reduce_gid0 + reductions_per_wi * wg_size);
        for (size_t arg_reduce_gid = arg_reduce_gid0;
             arg_reduce_gid < arg_reduce_gid_max; arg_reduce_gid += wg_size)
        {
            py::ssize_t inp_reduction_offset = static_cast<py::ssize_t>(
                inp_reduced_dims_indexer_(arg_reduce_gid));
            py::ssize_t inp_offset = inp_iter_offset + inp_reduction_offset;

            // must convert to boolean first to handle nans
            using dpctl::tensor::type_utils::convert_impl;
            bool val = convert_impl<bool, argT>(inp_[inp_offset]);
            ReductionOp op = reduction_op_;

            local_red_val = op(local_red_val, static_cast<outT>(val));
        }
        // reduction and atomic operations are performed
        // in group_op_
        group_op_(it, out_, out_iter_offset, local_red_val);
    }
};

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6>
class boolean_reduction_axis0_contig_krn;

template <typename argTy, typename resTy, typename RedOpT, typename GroupOpT>
sycl::event
boolean_reduction_axis0_contig_impl(sycl::queue &exec_q,
                                    size_t iter_nelems,
                                    size_t reduction_nelems,
                                    const char *arg_cp,
                                    char *res_cp,
                                    py::ssize_t iter_arg_offset,
                                    py::ssize_t iter_res_offset,
                                    py::ssize_t red_arg_offset,
                                    const std::vector<sycl::event> &depends)
{
    const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_cp) +
                          iter_arg_offset + red_arg_offset;
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp) + iter_res_offset;

    constexpr resTy identity_val = sycl::known_identity<RedOpT, resTy>::value;

    const sycl::device &d = exec_q.get_device();
    const auto &sg_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

    {
        sycl::event init_ev = exec_q.fill<resTy>(res_tp, resTy(identity_val),
                                                 iter_nelems, depends);
        sycl::event red_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(init_ev);

            constexpr std::uint8_t dim = 1;

            using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            using ColsIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
            using InputOutputIterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    NoOpIndexerT, NoOpIndexerT>;
            using ReductionIndexerT = ColsIndexerT;

            NoOpIndexerT columns_indexer{};
            NoOpIndexerT result_indexer{};
            InputOutputIterIndexerT in_out_iter_indexer{columns_indexer,
                                                        result_indexer};
            ReductionIndexerT reduction_indexer{
                0, static_cast<py::ssize_t>(reduction_nelems),
                static_cast<py::ssize_t>(iter_nelems)};

            constexpr size_t preferred_reductions_per_wi = 4;
            size_t reductions_per_wi =
                (reduction_nelems < preferred_reductions_per_wi * wg)
                    ? ((reduction_nelems + wg - 1) / wg)
                    : preferred_reductions_per_wi;

            size_t reduction_groups =
                (reduction_nelems + reductions_per_wi * wg - 1) /
                (reductions_per_wi * wg);

            auto gws = sycl::range<dim>{iter_nelems * reduction_groups * wg};
            auto lws = sycl::range<dim>{wg};

            cgh.parallel_for<class boolean_reduction_axis0_contig_krn<
                argTy, resTy, RedOpT, GroupOpT, InputOutputIterIndexerT,
                ReductionIndexerT>>(
                sycl::nd_range<dim>(gws, lws),
                StridedBooleanReduction<argTy, resTy, RedOpT, GroupOpT,
                                        InputOutputIterIndexerT,
                                        ReductionIndexerT>(
                    arg_tp, res_tp, RedOpT(), GroupOpT(), identity_val,
                    in_out_iter_indexer, reduction_indexer, reduction_nelems,
                    iter_nelems, reductions_per_wi));
        });
        return red_ev;
    }
}

template <typename fnT, typename srcTy> struct AllAxis0ContigFactory
{
    fnT get() const
    {
        using resTy = std::int32_t;
        using RedOpT = sycl::logical_and<resTy>;
        using GroupOpT = all_reduce_wg_strided<resTy>;

        return dpctl::tensor::kernels::boolean_reduction_axis0_contig_impl<
            srcTy, resTy, RedOpT, GroupOpT>;
    }
};

template <typename fnT, typename srcTy> struct AnyAxis0ContigFactory
{
    fnT get() const
    {
        using resTy = std::int32_t;
        using RedOpT = sycl::logical_or<resTy>;
        using GroupOpT = any_reduce_wg_strided<resTy>;

        return dpctl::tensor::kernels::boolean_reduction_axis0_contig_impl<
            srcTy, resTy, RedOpT, GroupOpT>;
    }
};

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6>
class boolean_reduction_strided_krn;

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class boolean_reduction_seq_strided_krn;

typedef sycl::event (*boolean_reduction_strided_impl_fn_ptr)(
    sycl::queue &,
    size_t,
    size_t,
    const char *,
    char *,
    int,
    const py::ssize_t *,
    py::ssize_t,
    py::ssize_t,
    int,
    const py::ssize_t *,
    py::ssize_t,
    const std::vector<sycl::event> &);

template <typename argTy, typename resTy, typename RedOpT, typename GroupOpT>
sycl::event
boolean_reduction_strided_impl(sycl::queue &exec_q,
                               size_t iter_nelems,
                               size_t reduction_nelems,
                               const char *arg_cp,
                               char *res_cp,
                               int iter_nd,
                               const py::ssize_t *iter_shape_and_strides,
                               py::ssize_t iter_arg_offset,
                               py::ssize_t iter_res_offset,
                               int red_nd,
                               const py::ssize_t *reduction_shape_stride,
                               py::ssize_t reduction_arg_offset,
                               const std::vector<sycl::event> &depends)
{
    const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_cp);
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp);

    constexpr resTy identity_val = sycl::known_identity<RedOpT, resTy>::value;

    const sycl::device &d = exec_q.get_device();
    const auto &sg_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

    sycl::event red_ev;
    if (reduction_nelems < wg) {
        red_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            using InputOutputIterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;
            using ReductionIndexerT =
                dpctl::tensor::offset_utils::StridedIndexer;

            InputOutputIterIndexerT in_out_iter_indexer{
                iter_nd, iter_arg_offset, iter_res_offset,
                iter_shape_and_strides};
            ReductionIndexerT reduction_indexer{red_nd, reduction_arg_offset,
                                                reduction_shape_stride};

            cgh.parallel_for<class boolean_reduction_seq_strided_krn<
                argTy, resTy, RedOpT, InputOutputIterIndexerT,
                ReductionIndexerT>>(
                sycl::range<1>(iter_nelems),
                SequentialBooleanReduction<argTy, resTy, RedOpT,
                                           InputOutputIterIndexerT,
                                           ReductionIndexerT>(
                    arg_tp, res_tp, RedOpT(), identity_val, in_out_iter_indexer,
                    reduction_indexer, reduction_nelems));
        });
    }
    else {
        sycl::event init_ev = exec_q.submit([&](sycl::handler &cgh) {
            using IndexerT =
                dpctl::tensor::offset_utils::UnpackedStridedIndexer;

            const py::ssize_t *const &res_shape = iter_shape_and_strides;
            const py::ssize_t *const &res_strides =
                iter_shape_and_strides + 2 * iter_nd;
            IndexerT res_indexer(iter_nd, iter_res_offset, res_shape,
                                 res_strides);

            cgh.depends_on(depends);

            cgh.parallel_for(sycl::range<1>(iter_nelems), [=](sycl::id<1> id) {
                auto res_offset = res_indexer(id[0]);
                res_tp[res_offset] = identity_val;
            });
        });
        red_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(init_ev);

            constexpr std::uint8_t dim = 1;

            using InputOutputIterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;
            using ReductionIndexerT =
                dpctl::tensor::offset_utils::StridedIndexer;

            InputOutputIterIndexerT in_out_iter_indexer{
                iter_nd, iter_arg_offset, iter_res_offset,
                iter_shape_and_strides};
            ReductionIndexerT reduction_indexer{red_nd, reduction_arg_offset,
                                                reduction_shape_stride};

            constexpr size_t preferred_reductions_per_wi = 4;
            size_t reductions_per_wi =
                (reduction_nelems < preferred_reductions_per_wi * wg)
                    ? ((reduction_nelems + wg - 1) / wg)
                    : preferred_reductions_per_wi;

            size_t reduction_groups =
                (reduction_nelems + reductions_per_wi * wg - 1) /
                (reductions_per_wi * wg);

            auto gws = sycl::range<dim>{iter_nelems * reduction_groups * wg};
            auto lws = sycl::range<dim>{wg};

            cgh.parallel_for<class boolean_reduction_strided_krn<
                argTy, resTy, RedOpT, GroupOpT, InputOutputIterIndexerT,
                ReductionIndexerT>>(
                sycl::nd_range<dim>(gws, lws),
                StridedBooleanReduction<argTy, resTy, RedOpT, GroupOpT,
                                        InputOutputIterIndexerT,
                                        ReductionIndexerT>(
                    arg_tp, res_tp, RedOpT(), GroupOpT(), identity_val,
                    in_out_iter_indexer, reduction_indexer, reduction_nelems,
                    iter_nelems, reductions_per_wi));
        });
    }
    return red_ev;
}

template <typename fnT, typename srcTy> struct AllStridedFactory
{
    fnT get() const
    {
        using resTy = std::int32_t;
        using RedOpT = sycl::logical_and<resTy>;
        using GroupOpT = all_reduce_wg_strided<resTy>;

        return dpctl::tensor::kernels::boolean_reduction_strided_impl<
            srcTy, resTy, RedOpT, GroupOpT>;
    }
};

template <typename fnT, typename srcTy> struct AnyStridedFactory
{
    fnT get() const
    {
        using resTy = std::int32_t;
        using RedOpT = sycl::logical_or<resTy>;
        using GroupOpT = any_reduce_wg_strided<resTy>;

        return dpctl::tensor::kernels::boolean_reduction_strided_impl<
            srcTy, resTy, RedOpT, GroupOpT>;
    }
};

} // namespace kernels
} // namespace tensor
} // namespace dpctl
