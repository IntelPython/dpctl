//=== reductions.hpp - Implementation of reduction kernels ------- *-C++-*/===//
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
/// This file defines kernels for tensor reduction along axis.
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl.hpp>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include "pybind11/pybind11.h"
#include "utils/math_utils.hpp"
#include "utils/offset_utils.hpp"
#include "utils/sycl_utils.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;
namespace su_ns = dpctl::tensor::sycl_utils;

namespace dpctl
{
namespace tensor
{
namespace kernels
{

template <typename ReductionOpT, typename T> struct needs_workaround
{
    static constexpr bool value =
        std::is_same_v<ReductionOpT, sycl::multiplies<T>> &&
        (std::is_same_v<T, std::int64_t> || std::is_same_v<T, std::uint64_t>);
};

template <typename ReductionOpT, typename T> struct can_use_reduce_over_group
{
    static constexpr bool value =
        sycl::has_known_identity<ReductionOpT, T>::value &&
        !needs_workaround<ReductionOpT, T>::value;
};

template <typename argT,
          typename outT,
          typename ReductionOp,
          typename InputOutputIterIndexerT,
          typename InputRedIndexerT>
struct SequentialReduction
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
    SequentialReduction(const argT *inp,
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
            const py::ssize_t inp_reduction_offset =
                inp_reduced_dims_indexer_(m);
            const py::ssize_t inp_offset =
                inp_iter_offset + inp_reduction_offset;

            using dpctl::tensor::type_utils::convert_impl;
            outT val = convert_impl<outT, argT>(inp_[inp_offset]);
            red_val = reduction_op_(red_val, val);
        }

        out_[out_iter_offset] = red_val;
    }
};

/* === Reduction, using sycl::reduce_over_group, and sycl::atomic_ref === */

/*
  This kernel only works for outT with sizeof(outT) == 4, or sizeof(outT) == 8
  if the device has aspect atomic64 and only with those supported by
  sycl::atomic_ref
*/
template <typename argT,
          typename outT,
          typename ReductionOp,
          typename InputOutputIterIndexerT,
          typename InputRedIndexerT>
struct ReductionOverGroupWithAtomicFunctor
{
private:
    const argT *inp_ = nullptr;
    outT *out_ = nullptr;
    ReductionOp reduction_op_;
    outT identity_;
    InputOutputIterIndexerT inp_out_iter_indexer_;
    InputRedIndexerT inp_reduced_dims_indexer_;
    size_t reduction_max_gid_ = 0;
    size_t iter_gws_ = 1;
    size_t reductions_per_wi = 16;

public:
    ReductionOverGroupWithAtomicFunctor(
        const argT *data,
        outT *res,
        ReductionOp reduction_op,
        const outT &identity_val,
        InputOutputIterIndexerT arg_res_iter_indexer,
        InputRedIndexerT arg_reduced_dims_indexer,
        size_t reduction_size,
        size_t iteration_size,
        size_t reduction_size_per_wi)
        : inp_(data), out_(res), reduction_op_(reduction_op),
          identity_(identity_val), inp_out_iter_indexer_(arg_res_iter_indexer),
          inp_reduced_dims_indexer_(arg_reduced_dims_indexer),
          reduction_max_gid_(reduction_size), iter_gws_(iteration_size),
          reductions_per_wi(reduction_size_per_wi)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        const size_t iter_gid = it.get_group(0) % iter_gws_;
        const size_t reduction_batch_id = it.get_group(0) / iter_gws_;

        const size_t reduction_lid = it.get_local_id(0);
        const size_t wg = it.get_local_range(0); //   0 <= reduction_lid < wg

        // work-items operate over input with indices
        //   inp_data_id = reduction_batch_id * wg * reductions_per_wi + m * wg
        //   + reduction_lid
        // for 0 <= m < reductions_per_wi

        auto inp_out_iter_offsets_ = inp_out_iter_indexer_(iter_gid);
        const auto &inp_iter_offset = inp_out_iter_offsets_.get_first_offset();
        const auto &out_iter_offset = inp_out_iter_offsets_.get_second_offset();

        outT local_red_val(identity_);
        size_t arg_reduce_gid0 =
            reduction_lid + reduction_batch_id * wg * reductions_per_wi;
        size_t arg_reduce_gid_max = std::min(
            reduction_max_gid_, arg_reduce_gid0 + reductions_per_wi * wg);

        for (size_t arg_reduce_gid = arg_reduce_gid0;
             arg_reduce_gid < arg_reduce_gid_max; arg_reduce_gid += wg)
        {
            auto inp_reduction_offset =
                inp_reduced_dims_indexer_(arg_reduce_gid);
            auto inp_offset = inp_iter_offset + inp_reduction_offset;

            using dpctl::tensor::type_utils::convert_impl;
            outT val = convert_impl<outT, argT>(inp_[inp_offset]);

            local_red_val = reduction_op_(local_red_val, val);
        }

        auto work_group = it.get_group();
        // This only works if reduction_op_ is from small set of operators
        outT red_val_over_wg = sycl::reduce_over_group(
            work_group, local_red_val, identity_, reduction_op_);

        if (work_group.leader()) {
            sycl::atomic_ref<outT, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                res_ref(out_[out_iter_offset]);
            if constexpr (su_ns::IsPlus<outT, ReductionOp>::value) {
                res_ref += red_val_over_wg;
            }
            else if constexpr (std::is_same_v<ReductionOp, sycl::maximum<outT>>)
            {
                res_ref.fetch_max(red_val_over_wg);
            }
            else if constexpr (std::is_same_v<ReductionOp, sycl::minimum<outT>>)
            {
                res_ref.fetch_min(red_val_over_wg);
            }
            else {
                outT read_val = res_ref.load();
                outT new_val{};
                do {
                    new_val = reduction_op_(read_val, red_val_over_wg);
                } while (!res_ref.compare_exchange_strong(read_val, new_val));
            }
        }
    }
};

/* === Reduction, using custom_reduce_over_group, and sycl::atomic_ref === */

template <typename argT,
          typename outT,
          typename ReductionOp,
          typename InputOutputIterIndexerT,
          typename InputRedIndexerT,
          typename SlmT>
struct CustomReductionOverGroupWithAtomicFunctor
{
private:
    const argT *inp_ = nullptr;
    outT *out_ = nullptr;
    ReductionOp reduction_op_;
    outT identity_;
    InputOutputIterIndexerT inp_out_iter_indexer_;
    InputRedIndexerT inp_reduced_dims_indexer_;
    SlmT local_mem_;
    size_t reduction_max_gid_ = 0;
    size_t iter_gws_ = 1;
    size_t reductions_per_wi = 16;

public:
    CustomReductionOverGroupWithAtomicFunctor(
        const argT *data,
        outT *res,
        ReductionOp reduction_op,
        const outT &identity_val,
        InputOutputIterIndexerT arg_res_iter_indexer,
        InputRedIndexerT arg_reduced_dims_indexer,
        SlmT local_mem,
        size_t reduction_size,
        size_t iteration_size,
        size_t reduction_size_per_wi)
        : inp_(data), out_(res), reduction_op_(reduction_op),
          identity_(identity_val), inp_out_iter_indexer_(arg_res_iter_indexer),
          inp_reduced_dims_indexer_(arg_reduced_dims_indexer),
          local_mem_(local_mem), reduction_max_gid_(reduction_size),
          iter_gws_(iteration_size), reductions_per_wi(reduction_size_per_wi)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        const size_t iter_gid = it.get_group(0) % iter_gws_;
        const size_t reduction_batch_id = it.get_group(0) / iter_gws_;

        const size_t reduction_lid = it.get_local_id(0);
        const size_t wg = it.get_local_range(0); //   0 <= reduction_lid < wg

        // work-items operate over input with indices
        //   inp_data_id = reduction_batch_id * wg * reductions_per_wi + m * wg
        //   + reduction_lid
        // for 0 <= m < reductions_per_wi

        auto inp_out_iter_offsets_ = inp_out_iter_indexer_(iter_gid);
        const auto &inp_iter_offset = inp_out_iter_offsets_.get_first_offset();
        const auto &out_iter_offset = inp_out_iter_offsets_.get_second_offset();

        outT local_red_val(identity_);
        size_t arg_reduce_gid0 =
            reduction_lid + reduction_batch_id * wg * reductions_per_wi;
        size_t arg_reduce_gid_max = std::min(
            reduction_max_gid_, arg_reduce_gid0 + reductions_per_wi * wg);

        for (size_t arg_reduce_gid = arg_reduce_gid0;
             arg_reduce_gid < arg_reduce_gid_max; arg_reduce_gid += wg)
        {
            auto inp_reduction_offset =
                inp_reduced_dims_indexer_(arg_reduce_gid);
            auto inp_offset = inp_iter_offset + inp_reduction_offset;

            using dpctl::tensor::type_utils::convert_impl;
            outT val = convert_impl<outT, argT>(inp_[inp_offset]);

            local_red_val = reduction_op_(local_red_val, val);
        }

        auto work_group = it.get_group();
        outT red_val_over_wg = su_ns::custom_reduce_over_group(
            work_group, local_mem_, local_red_val, reduction_op_);

        if (work_group.leader()) {
            sycl::atomic_ref<outT, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                res_ref(out_[out_iter_offset]);
            outT read_val = res_ref.load();
            outT new_val{};
            do {
                new_val = reduction_op_(read_val, red_val_over_wg);
            } while (!res_ref.compare_exchange_strong(read_val, new_val));
        }
    }
};

typedef sycl::event (*reduction_strided_impl_fn_ptr)(
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

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class reduction_over_group_with_atomics_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6>
class custom_reduction_over_group_with_atomics_krn;

template <typename T1, typename T2, typename T3>
class reduction_over_group_with_atomics_init_krn;

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class reduction_seq_strided_krn;

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class reduction_seq_contig_krn;

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class reduction_axis0_over_group_with_atomics_contig_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6>
class custom_reduction_axis0_over_group_with_atomics_contig_krn;

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class reduction_axis1_over_group_with_atomics_contig_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6>
class custom_reduction_axis1_over_group_with_atomics_contig_krn;

using dpctl::tensor::sycl_utils::choose_workgroup_size;

template <typename argTy, typename resTy, typename ReductionOpT>
sycl::event reduction_over_group_with_atomics_strided_impl(
    sycl::queue &exec_q,
    size_t iter_nelems, // number of reductions    (num. of rows in a matrix
                        // when reducing over rows)
    size_t reduction_nelems, // size of each reduction  (length of rows, i.e.
                             // number of columns)
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

    constexpr resTy identity_val = su_ns::Identity<ReductionOpT, resTy>::value;

    const sycl::device &d = exec_q.get_device();
    const auto &sg_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

    if (reduction_nelems < wg) {
        sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
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

            cgh.parallel_for<class reduction_seq_strided_krn<
                argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                ReductionIndexerT>>(
                sycl::range<1>(iter_nelems),
                SequentialReduction<argTy, resTy, ReductionOpT,
                                    InputOutputIterIndexerT, ReductionIndexerT>(
                    arg_tp, res_tp, ReductionOpT(), identity_val,
                    in_out_iter_indexer, reduction_indexer, reduction_nelems));
        });

        return comp_ev;
    }
    else {
        sycl::event res_init_ev = exec_q.submit([&](sycl::handler &cgh) {
            using IndexerT =
                dpctl::tensor::offset_utils::UnpackedStridedIndexer;

            const py::ssize_t *const &res_shape = iter_shape_and_strides;
            const py::ssize_t *const &res_strides =
                iter_shape_and_strides + 2 * iter_nd;
            IndexerT res_indexer(iter_nd, iter_res_offset, res_shape,
                                 res_strides);
            using InitKernelName =
                class reduction_over_group_with_atomics_init_krn<resTy, argTy,
                                                                 ReductionOpT>;
            cgh.depends_on(depends);

            cgh.parallel_for<InitKernelName>(
                sycl::range<1>(iter_nelems), [=](sycl::id<1> id) {
                    auto res_offset = res_indexer(id[0]);
                    res_tp[res_offset] = identity_val;
                });
        });

        sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(res_init_ev);

            using InputOutputIterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;
            using ReductionIndexerT =
                dpctl::tensor::offset_utils::StridedIndexer;

            InputOutputIterIndexerT in_out_iter_indexer{
                iter_nd, iter_arg_offset, iter_res_offset,
                iter_shape_and_strides};
            ReductionIndexerT reduction_indexer{red_nd, reduction_arg_offset,
                                                reduction_shape_stride};

            constexpr size_t preferred_reductions_per_wi = 8;
            size_t reductions_per_wi =
                (reduction_nelems < preferred_reductions_per_wi * wg)
                    ? std::max<size_t>(1, (reduction_nelems + wg - 1) / wg)
                    : preferred_reductions_per_wi;

            size_t reduction_groups =
                (reduction_nelems + reductions_per_wi * wg - 1) /
                (reductions_per_wi * wg);

            auto globalRange =
                sycl::range<1>{iter_nelems * reduction_groups * wg};
            auto localRange = sycl::range<1>{wg};

            if constexpr (can_use_reduce_over_group<ReductionOpT, resTy>::value)
            {
                using KernelName = class reduction_over_group_with_atomics_krn<
                    argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                    ReductionIndexerT>;

                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    ReductionOverGroupWithAtomicFunctor<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT>(arg_tp, res_tp, ReductionOpT(),
                                           identity_val, in_out_iter_indexer,
                                           reduction_indexer, reduction_nelems,
                                           iter_nelems, reductions_per_wi));
            }
            else {
                using SlmT = sycl::local_accessor<resTy, 1>;
                SlmT local_memory = SlmT(localRange, cgh);
                using KernelName =
                    class custom_reduction_over_group_with_atomics_krn<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT, SlmT>;

                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    CustomReductionOverGroupWithAtomicFunctor<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT, SlmT>(
                        arg_tp, res_tp, ReductionOpT(), identity_val,
                        in_out_iter_indexer, reduction_indexer, local_memory,
                        reduction_nelems, iter_nelems, reductions_per_wi));
            }
        });

        return comp_ev;
    }
}

// Contig

typedef sycl::event (*reduction_contig_impl_fn_ptr)(
    sycl::queue &,
    size_t,
    size_t,
    const char *,
    char *,
    py::ssize_t,
    py::ssize_t,
    py::ssize_t,
    const std::vector<sycl::event> &);

/* @brief Reduce rows in a matrix */
template <typename argTy, typename resTy, typename ReductionOpT>
sycl::event reduction_axis1_over_group_with_atomics_contig_impl(
    sycl::queue &exec_q,
    size_t iter_nelems, // number of reductions    (num. of rows in a matrix
                        // when reducing over rows)
    size_t reduction_nelems, // size of each reduction  (length of rows, i.e.
                             // number of columns)
    const char *arg_cp,
    char *res_cp,
    py::ssize_t iter_arg_offset,
    py::ssize_t iter_res_offset,
    py::ssize_t reduction_arg_offset,
    const std::vector<sycl::event> &depends)
{
    const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_cp) +
                          iter_arg_offset + reduction_arg_offset;
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp) + iter_res_offset;

    constexpr resTy identity_val = su_ns::Identity<ReductionOpT, resTy>::value;

    const sycl::device &d = exec_q.get_device();
    const auto &sg_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

    if (reduction_nelems < wg) {
        sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
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

            cgh.parallel_for<class reduction_seq_contig_krn<
                argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                ReductionIndexerT>>(
                sycl::range<1>(iter_nelems),
                SequentialReduction<argTy, resTy, ReductionOpT,
                                    InputOutputIterIndexerT, ReductionIndexerT>(
                    arg_tp, res_tp, ReductionOpT(), identity_val,
                    in_out_iter_indexer, reduction_indexer, reduction_nelems));
        });

        return comp_ev;
    }
    else {
        sycl::event res_init_ev = exec_q.fill<resTy>(
            res_tp, resTy(identity_val), iter_nelems, depends);

        sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(res_init_ev);

            using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            using RowsIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
            using InputOutputIterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    RowsIndexerT, NoOpIndexerT>;
            using ReductionIndexerT = NoOpIndexerT;

            RowsIndexerT rows_indexer{
                0, static_cast<py::ssize_t>(iter_nelems),
                static_cast<py::ssize_t>(reduction_nelems)};
            NoOpIndexerT result_indexer{};
            InputOutputIterIndexerT in_out_iter_indexer{rows_indexer,
                                                        result_indexer};
            ReductionIndexerT reduction_indexer{};

            constexpr size_t preferred_reductions_per_wi = 8;
            size_t reductions_per_wi =
                (reduction_nelems < preferred_reductions_per_wi * wg)
                    ? std::max<size_t>(1, (reduction_nelems + wg - 1) / wg)
                    : preferred_reductions_per_wi;

            size_t reduction_groups =
                (reduction_nelems + reductions_per_wi * wg - 1) /
                (reductions_per_wi * wg);

            auto globalRange =
                sycl::range<1>{iter_nelems * reduction_groups * wg};
            auto localRange = sycl::range<1>{wg};

            if constexpr (can_use_reduce_over_group<ReductionOpT, resTy>::value)
            {
                using KernelName =
                    class reduction_axis1_over_group_with_atomics_contig_krn<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT>;

                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    ReductionOverGroupWithAtomicFunctor<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT>(arg_tp, res_tp, ReductionOpT(),
                                           identity_val, in_out_iter_indexer,
                                           reduction_indexer, reduction_nelems,
                                           iter_nelems, reductions_per_wi));
            }
            else {
                using SlmT = sycl::local_accessor<resTy, 1>;
                SlmT local_memory = SlmT(localRange, cgh);
                using KernelName = class
                    custom_reduction_axis1_over_group_with_atomics_contig_krn<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT, SlmT>;

                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    CustomReductionOverGroupWithAtomicFunctor<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT, SlmT>(
                        arg_tp, res_tp, ReductionOpT(), identity_val,
                        in_out_iter_indexer, reduction_indexer, local_memory,
                        reduction_nelems, iter_nelems, reductions_per_wi));
            }
        });
        return comp_ev;
    }
}

/* @brief Reduce rows in a matrix */
template <typename argTy, typename resTy, typename ReductionOpT>
sycl::event reduction_axis0_over_group_with_atomics_contig_impl(
    sycl::queue &exec_q,
    size_t iter_nelems, // number of reductions    (num. of cols in a matrix
                        // when reducing over cols)
    size_t reduction_nelems, // size of each reduction  (length of cols, i.e.
                             // number of rows)
    const char *arg_cp,
    char *res_cp,
    py::ssize_t iter_arg_offset,
    py::ssize_t iter_res_offset,
    py::ssize_t reduction_arg_offset,
    const std::vector<sycl::event> &depends)
{
    const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_cp) +
                          iter_arg_offset + reduction_arg_offset;
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp) + iter_res_offset;

    constexpr resTy identity_val = su_ns::Identity<ReductionOpT, resTy>::value;

    const sycl::device &d = exec_q.get_device();
    const auto &sg_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

    if (reduction_nelems < wg) {
        sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            using InputOutputIterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    NoOpIndexerT, NoOpIndexerT>;
            using ReductionIndexerT =
                dpctl::tensor::offset_utils::Strided1DIndexer;

            InputOutputIterIndexerT in_out_iter_indexer{NoOpIndexerT{},
                                                        NoOpIndexerT{}};
            ReductionIndexerT reduction_indexer{
                0, static_cast<py::ssize_t>(reduction_nelems),
                static_cast<py::ssize_t>(iter_nelems)};

            using KernelName =
                class reduction_seq_contig_krn<argTy, resTy, ReductionOpT,
                                               InputOutputIterIndexerT,
                                               ReductionIndexerT>;

            sycl::range<1> iter_range{iter_nelems};

            cgh.parallel_for<KernelName>(
                iter_range,
                SequentialReduction<argTy, resTy, ReductionOpT,
                                    InputOutputIterIndexerT, ReductionIndexerT>(
                    arg_tp, res_tp, ReductionOpT(), identity_val,
                    in_out_iter_indexer, reduction_indexer, reduction_nelems));
        });

        return comp_ev;
    }
    else {
        sycl::event res_init_ev = exec_q.fill<resTy>(
            res_tp, resTy(identity_val), iter_nelems, depends);

        sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(res_init_ev);

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
                0, /* size */ static_cast<py::ssize_t>(reduction_nelems),
                /* step */ static_cast<py::ssize_t>(iter_nelems)};

            constexpr size_t preferred_reductions_per_wi = 8;
            size_t reductions_per_wi =
                (reduction_nelems < preferred_reductions_per_wi * wg)
                    ? std::max<size_t>(1, (reduction_nelems + wg - 1) / wg)
                    : preferred_reductions_per_wi;

            size_t reduction_groups =
                (reduction_nelems + reductions_per_wi * wg - 1) /
                (reductions_per_wi * wg);

            auto globalRange =
                sycl::range<1>{iter_nelems * reduction_groups * wg};
            auto localRange = sycl::range<1>{wg};

            if constexpr (can_use_reduce_over_group<ReductionOpT, resTy>::value)
            {
                using KernelName =
                    class reduction_axis0_over_group_with_atomics_contig_krn<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT>;

                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    ReductionOverGroupWithAtomicFunctor<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT>(arg_tp, res_tp, ReductionOpT(),
                                           identity_val, in_out_iter_indexer,
                                           reduction_indexer, reduction_nelems,
                                           iter_nelems, reductions_per_wi));
            }
            else {
                using SlmT = sycl::local_accessor<resTy, 1>;
                SlmT local_memory = SlmT(localRange, cgh);
                using KernelName = class
                    custom_reduction_axis0_over_group_with_atomics_contig_krn<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT, SlmT>;

                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    CustomReductionOverGroupWithAtomicFunctor<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT, SlmT>(
                        arg_tp, res_tp, ReductionOpT(), identity_val,
                        in_out_iter_indexer, reduction_indexer, local_memory,
                        reduction_nelems, iter_nelems, reductions_per_wi));
            }
        });
        return comp_ev;
    }
}

/* = Reduction, using sycl::reduce_over_group, but not using atomic_ref = */

template <typename argT,
          typename outT,
          typename ReductionOp,
          typename InputOutputIterIndexerT,
          typename InputRedIndexerT>
struct ReductionOverGroupNoAtomicFunctor
{
private:
    const argT *inp_ = nullptr;
    outT *out_ = nullptr;
    ReductionOp reduction_op_;
    outT identity_;
    InputOutputIterIndexerT inp_out_iter_indexer_;
    InputRedIndexerT inp_reduced_dims_indexer_;
    size_t reduction_max_gid_ = 0;
    size_t iter_gws_ = 1;
    size_t reductions_per_wi = 16;

public:
    ReductionOverGroupNoAtomicFunctor(
        const argT *data,
        outT *res,
        ReductionOp reduction_op,
        const outT &identity_val,
        InputOutputIterIndexerT arg_res_iter_indexer,
        InputRedIndexerT arg_reduced_dims_indexer,
        size_t reduction_size,
        size_t iteration_size,
        size_t reduction_size_per_wi)
        : inp_(data), out_(res), reduction_op_(reduction_op),
          identity_(identity_val), inp_out_iter_indexer_(arg_res_iter_indexer),
          inp_reduced_dims_indexer_(arg_reduced_dims_indexer),
          reduction_max_gid_(reduction_size), iter_gws_(iteration_size),
          reductions_per_wi(reduction_size_per_wi)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        const size_t reduction_lid = it.get_local_id(0);
        const size_t wg = it.get_local_range(0); //   0 <= reduction_lid < wg

        const size_t iter_gid = it.get_group(0) % iter_gws_;
        const size_t reduction_batch_id = it.get_group(0) / iter_gws_;
        const size_t n_reduction_groups = it.get_group_range(0) / iter_gws_;

        // work-items operates over input with indices
        //   inp_data_id = reduction_batch_id * wg * reductions_per_wi + m * wg
        //   + reduction_lid
        // for 0 <= m < reductions_per_wi

        auto inp_out_iter_offsets_ = inp_out_iter_indexer_(iter_gid);
        const auto &inp_iter_offset = inp_out_iter_offsets_.get_first_offset();
        const auto &out_iter_offset = inp_out_iter_offsets_.get_second_offset();

        outT local_red_val(identity_);
        size_t arg_reduce_gid0 =
            reduction_lid + reduction_batch_id * wg * reductions_per_wi;
        for (size_t m = 0; m < reductions_per_wi; ++m) {
            size_t arg_reduce_gid = arg_reduce_gid0 + m * wg;

            if (arg_reduce_gid < reduction_max_gid_) {
                auto inp_reduction_offset =
                    inp_reduced_dims_indexer_(arg_reduce_gid);
                auto inp_offset = inp_iter_offset + inp_reduction_offset;

                using dpctl::tensor::type_utils::convert_impl;
                outT val = convert_impl<outT, argT>(inp_[inp_offset]);

                local_red_val = reduction_op_(local_red_val, val);
            }
        }

        auto work_group = it.get_group();
        // This only works if reduction_op_ is from small set of operators
        outT red_val_over_wg = sycl::reduce_over_group(
            work_group, local_red_val, identity_, reduction_op_);

        if (work_group.leader()) {
            // each group writes to a different memory location
            out_[out_iter_offset * n_reduction_groups + reduction_batch_id] =
                red_val_over_wg;
        }
    }
};

/* = Reduction, using custom_reduce_over_group and not using atomic_ref*/

template <typename argT,
          typename outT,
          typename ReductionOp,
          typename InputOutputIterIndexerT,
          typename InputRedIndexerT,
          typename SlmT>
struct CustomReductionOverGroupNoAtomicFunctor
{
private:
    const argT *inp_ = nullptr;
    outT *out_ = nullptr;
    ReductionOp reduction_op_;
    outT identity_;
    InputOutputIterIndexerT inp_out_iter_indexer_;
    InputRedIndexerT inp_reduced_dims_indexer_;
    SlmT local_mem_;
    size_t reduction_max_gid_ = 0;
    size_t iter_gws_ = 1;
    size_t reductions_per_wi = 16;

public:
    CustomReductionOverGroupNoAtomicFunctor(
        const argT *data,
        outT *res,
        ReductionOp reduction_op,
        const outT &identity_val,
        InputOutputIterIndexerT arg_res_iter_indexer,
        InputRedIndexerT arg_reduced_dims_indexer,
        SlmT local_mem,
        size_t reduction_size,
        size_t iteration_size,
        size_t reduction_size_per_wi)
        : inp_(data), out_(res), reduction_op_(reduction_op),
          identity_(identity_val), inp_out_iter_indexer_(arg_res_iter_indexer),
          inp_reduced_dims_indexer_(arg_reduced_dims_indexer),
          local_mem_(local_mem), reduction_max_gid_(reduction_size),
          iter_gws_(iteration_size), reductions_per_wi(reduction_size_per_wi)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        const size_t reduction_lid = it.get_local_id(0);
        const size_t wg = it.get_local_range(0); //   0 <= reduction_lid < wg

        const size_t iter_gid = it.get_group(0) % iter_gws_;
        const size_t reduction_batch_id = it.get_group(0) / iter_gws_;
        const size_t n_reduction_groups = it.get_group_range(0) / iter_gws_;

        // work-items operates over input with indices
        //   inp_data_id = reduction_batch_id * wg * reductions_per_wi + m * wg
        //   + reduction_lid
        // for 0 <= m < reductions_per_wi

        auto inp_out_iter_offsets_ = inp_out_iter_indexer_(iter_gid);
        const auto &inp_iter_offset = inp_out_iter_offsets_.get_first_offset();
        const auto &out_iter_offset = inp_out_iter_offsets_.get_second_offset();

        outT local_red_val(identity_);
        size_t arg_reduce_gid0 =
            reduction_lid + reduction_batch_id * wg * reductions_per_wi;
        for (size_t m = 0; m < reductions_per_wi; ++m) {
            size_t arg_reduce_gid = arg_reduce_gid0 + m * wg;

            if (arg_reduce_gid < reduction_max_gid_) {
                auto inp_reduction_offset =
                    inp_reduced_dims_indexer_(arg_reduce_gid);
                auto inp_offset = inp_iter_offset + inp_reduction_offset;

                using dpctl::tensor::type_utils::convert_impl;
                outT val = convert_impl<outT, argT>(inp_[inp_offset]);

                local_red_val = reduction_op_(local_red_val, val);
            }
        }

        auto work_group = it.get_group();
        // This only works if reduction_op_ is from small set of operators
        outT red_val_over_wg = su_ns::custom_reduce_over_group(
            work_group, local_mem_, local_red_val, reduction_op_);

        if (work_group.leader()) {
            // each group writes to a different memory location
            out_[out_iter_offset * n_reduction_groups + reduction_batch_id] =
                red_val_over_wg;
        }
    }
};

typedef sycl::event (*reduction_strided_impl_fn_ptr)(
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

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class reduction_over_group_temps_strided_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6>
class custom_reduction_over_group_temps_strided_krn;

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class single_reduction_axis0_temps_contig_krn;

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class first_reduction_axis0_temps_contig_krn;

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class middle_reduction_axis0_temps_contig_krn;

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class final_reduction_axis0_temps_contig_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6>
class single_custom_reduction_axis0_temps_contig_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6>
class first_custom_reduction_axis0_temps_contig_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6>
class middle_custom_reduction_axis0_temps_contig_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6>
class final_custom_reduction_axis0_temps_contig_krn;

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class single_reduction_axis1_temps_contig_krn;

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class first_reduction_axis1_temps_contig_krn;

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class middle_reduction_axis1_temps_contig_krn;

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class final_reduction_axis1_temps_contig_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6>
class single_custom_reduction_axis1_temps_contig_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6>
class first_custom_reduction_axis1_temps_contig_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6>
class middle_custom_reduction_axis1_temps_contig_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6>
class final_custom_reduction_axis1_temps_contig_krn;

template <typename argTy, typename resTy, typename ReductionOpT>
sycl::event reduction_over_group_temps_strided_impl(
    sycl::queue &exec_q,
    size_t iter_nelems, // number of reductions    (num. of rows in a matrix
                        // when reducing over rows)
    size_t reduction_nelems, // size of each reduction  (length of rows, i.e.
                             // number of columns)
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

    constexpr resTy identity_val = su_ns::Identity<ReductionOpT, resTy>::value;

    const sycl::device &d = exec_q.get_device();
    const auto &sg_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

    if (reduction_nelems < wg) {
        sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
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

            cgh.parallel_for<class reduction_seq_strided_krn<
                argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                ReductionIndexerT>>(
                sycl::range<1>(iter_nelems),
                SequentialReduction<argTy, resTy, ReductionOpT,
                                    InputOutputIterIndexerT, ReductionIndexerT>(
                    arg_tp, res_tp, ReductionOpT(), identity_val,
                    in_out_iter_indexer, reduction_indexer, reduction_nelems));
        });

        return comp_ev;
    }

    constexpr size_t preferred_reductions_per_wi = 8;
    // max_max_wg prevents running out of resources on CPU
    constexpr size_t max_max_wg = 2048;
    size_t max_wg = std::min(
        max_max_wg, d.get_info<sycl::info::device::max_work_group_size>() / 2);

    size_t reductions_per_wi(preferred_reductions_per_wi);
    if (reduction_nelems <= preferred_reductions_per_wi * max_wg) {
        // Perform reduction using one 1 work-group per iteration,
        // can output directly to res
        sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
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

            if (iter_nelems == 1) {
                // increase GPU occupancy
                wg = max_wg;
            }
            reductions_per_wi =
                std::max<size_t>(1, (reduction_nelems + wg - 1) / wg);

            size_t reduction_groups =
                (reduction_nelems + reductions_per_wi * wg - 1) /
                (reductions_per_wi * wg);
            assert(reduction_groups == 1);

            auto globalRange =
                sycl::range<1>{iter_nelems * reduction_groups * wg};
            auto localRange = sycl::range<1>{wg};

            if constexpr (can_use_reduce_over_group<ReductionOpT, resTy>::value)
            {
                using KernelName = class reduction_over_group_temps_strided_krn<
                    argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                    ReductionIndexerT>;

                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    ReductionOverGroupNoAtomicFunctor<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT>(arg_tp, res_tp, ReductionOpT(),
                                           identity_val, in_out_iter_indexer,
                                           reduction_indexer, reduction_nelems,
                                           iter_nelems, reductions_per_wi));
            }
            else {
                using SlmT = sycl::local_accessor<resTy, 1>;
                SlmT local_memory = SlmT(localRange, cgh);
                using KernelName =
                    class custom_reduction_over_group_temps_strided_krn<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT, SlmT>;

                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    CustomReductionOverGroupNoAtomicFunctor<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT, SlmT>(
                        arg_tp, res_tp, ReductionOpT(), identity_val,
                        in_out_iter_indexer, reduction_indexer, local_memory,
                        reduction_nelems, iter_nelems, reductions_per_wi));
            }
        });
        return comp_ev;
    }
    else {
        // more than one work-groups is needed, requires a temporary
        size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);
        assert(reduction_groups > 1);

        size_t second_iter_reduction_groups_ =
            (reduction_groups + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);

        resTy *partially_reduced_tmp = sycl::malloc_device<resTy>(
            iter_nelems * (reduction_groups + second_iter_reduction_groups_),
            exec_q);
        resTy *partially_reduced_tmp2 = nullptr;

        if (partially_reduced_tmp == nullptr) {
            throw std::runtime_error("Unabled to allocate device_memory");
        }
        else {
            partially_reduced_tmp2 =
                partially_reduced_tmp + reduction_groups * iter_nelems;
        }

        const sycl::event &first_reduction_ev = exec_q.submit([&](sycl::handler
                                                                      &cgh) {
            cgh.depends_on(depends);

            using InputIndexerT = dpctl::tensor::offset_utils::StridedIndexer;
            using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            using InputOutputIterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    InputIndexerT, ResIndexerT>;
            using ReductionIndexerT =
                dpctl::tensor::offset_utils::StridedIndexer;

            // Only 2*iter_nd entries describing shape and strides of
            // iterated dimensions of input array from
            // iter_shape_and_strides are going to be accessed by
            // inp_indexer
            InputIndexerT inp_indexer(iter_nd, iter_arg_offset,
                                      iter_shape_and_strides);
            ResIndexerT noop_tmp_indexer{};

            InputOutputIterIndexerT in_out_iter_indexer{inp_indexer,
                                                        noop_tmp_indexer};
            ReductionIndexerT reduction_indexer{red_nd, reduction_arg_offset,
                                                reduction_shape_stride};

            auto globalRange =
                sycl::range<1>{iter_nelems * reduction_groups * wg};
            auto localRange = sycl::range<1>{wg};

            if constexpr (can_use_reduce_over_group<ReductionOpT, resTy>::value)
            {
                using KernelName = class reduction_over_group_temps_strided_krn<
                    argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                    ReductionIndexerT>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    ReductionOverGroupNoAtomicFunctor<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT>(
                        arg_tp, partially_reduced_tmp, ReductionOpT(),
                        identity_val, in_out_iter_indexer, reduction_indexer,
                        reduction_nelems, iter_nelems,
                        preferred_reductions_per_wi));
            }
            else {
                using SlmT = sycl::local_accessor<resTy, 1>;
                SlmT local_memory = SlmT(localRange, cgh);
                using KernelName =
                    class custom_reduction_over_group_temps_strided_krn<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT, SlmT>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    CustomReductionOverGroupNoAtomicFunctor<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT, SlmT>(
                        arg_tp, partially_reduced_tmp, ReductionOpT(),
                        identity_val, in_out_iter_indexer, reduction_indexer,
                        local_memory, reduction_nelems, iter_nelems,
                        preferred_reductions_per_wi));
            }
        });

        size_t remaining_reduction_nelems = reduction_groups;

        resTy *temp_arg = partially_reduced_tmp;
        resTy *temp2_arg = partially_reduced_tmp2;
        sycl::event dependent_ev = first_reduction_ev;

        while (remaining_reduction_nelems >
               preferred_reductions_per_wi * max_wg) {
            size_t reduction_groups_ = (remaining_reduction_nelems +
                                        preferred_reductions_per_wi * wg - 1) /
                                       (preferred_reductions_per_wi * wg);
            assert(reduction_groups_ > 1);

            // keep reducing
            sycl::event partial_reduction_ev = exec_q.submit([&](sycl::handler
                                                                     &cgh) {
                cgh.depends_on(dependent_ev);

                using InputIndexerT =
                    dpctl::tensor::offset_utils::Strided1DIndexer;
                using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
                using InputOutputIterIndexerT =
                    dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                        InputIndexerT, ResIndexerT>;
                using ReductionIndexerT =
                    dpctl::tensor::offset_utils::NoOpIndexer;

                InputIndexerT inp_indexer{
                    0, static_cast<py::ssize_t>(iter_nelems),
                    static_cast<py::ssize_t>(reduction_groups_)};
                ResIndexerT res_iter_indexer{};

                InputOutputIterIndexerT in_out_iter_indexer{inp_indexer,
                                                            res_iter_indexer};
                ReductionIndexerT reduction_indexer{};

                auto globalRange =
                    sycl::range<1>{iter_nelems * reduction_groups_ * wg};
                auto localRange = sycl::range<1>{wg};
                if constexpr (can_use_reduce_over_group<ReductionOpT,
                                                        resTy>::value) {
                    using KernelName =
                        class reduction_over_group_temps_strided_krn<
                            resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                            ReductionIndexerT>;
                    cgh.parallel_for<KernelName>(
                        sycl::nd_range<1>(globalRange, localRange),
                        ReductionOverGroupNoAtomicFunctor<
                            resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                            ReductionIndexerT>(
                            temp_arg, temp2_arg, ReductionOpT(), identity_val,
                            in_out_iter_indexer, reduction_indexer,
                            remaining_reduction_nelems, iter_nelems,
                            preferred_reductions_per_wi));
                }
                else {
                    using SlmT = sycl::local_accessor<resTy, 1>;
                    SlmT local_memory = SlmT(localRange, cgh);
                    using KernelName =
                        class custom_reduction_over_group_temps_strided_krn<
                            resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                            ReductionIndexerT, SlmT>;
                    cgh.parallel_for<KernelName>(
                        sycl::nd_range<1>(globalRange, localRange),
                        CustomReductionOverGroupNoAtomicFunctor<
                            resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                            ReductionIndexerT, SlmT>(
                            temp_arg, temp2_arg, ReductionOpT(), identity_val,
                            in_out_iter_indexer, reduction_indexer,
                            local_memory, remaining_reduction_nelems,
                            iter_nelems, preferred_reductions_per_wi));
                }
            });

            remaining_reduction_nelems = reduction_groups_;
            std::swap(temp_arg, temp2_arg);
            dependent_ev = std::move(partial_reduction_ev);
        }

        // final reduction to res
        sycl::event final_reduction_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(dependent_ev);

            using InputIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
            using ResIndexerT =
                dpctl::tensor::offset_utils::UnpackedStridedIndexer;
            using InputOutputIterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    InputIndexerT, ResIndexerT>;
            using ReductionIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;

            InputIndexerT inp_indexer{
                0, static_cast<py::ssize_t>(iter_nelems),
                static_cast<py::ssize_t>(remaining_reduction_nelems)};
            ResIndexerT res_iter_indexer{iter_nd, iter_res_offset,
                                         /* shape */ iter_shape_and_strides,
                                         /* strides */ iter_shape_and_strides +
                                             2 * iter_nd};

            InputOutputIterIndexerT in_out_iter_indexer{inp_indexer,
                                                        res_iter_indexer};
            ReductionIndexerT reduction_indexer{};

            wg = max_wg;
            reductions_per_wi =
                std::max<size_t>(1, (remaining_reduction_nelems + wg - 1) / wg);

            size_t reduction_groups =
                (remaining_reduction_nelems + reductions_per_wi * wg - 1) /
                (reductions_per_wi * wg);
            assert(reduction_groups == 1);

            auto globalRange =
                sycl::range<1>{iter_nelems * reduction_groups * wg};
            auto localRange = sycl::range<1>{wg};

            if constexpr (can_use_reduce_over_group<ReductionOpT, resTy>::value)
            {
                using KernelName = class reduction_over_group_temps_strided_krn<
                    argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                    ReductionIndexerT>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    ReductionOverGroupNoAtomicFunctor<
                        resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT>(temp_arg, res_tp, ReductionOpT(),
                                           identity_val, in_out_iter_indexer,
                                           reduction_indexer,
                                           remaining_reduction_nelems,
                                           iter_nelems, reductions_per_wi));
            }
            else {
                using SlmT = sycl::local_accessor<resTy, 1>;
                SlmT local_memory = SlmT(localRange, cgh);
                using KernelName =
                    class custom_reduction_over_group_temps_strided_krn<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT, SlmT>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    CustomReductionOverGroupNoAtomicFunctor<
                        resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT, SlmT>(
                        temp_arg, res_tp, ReductionOpT(), identity_val,
                        in_out_iter_indexer, reduction_indexer, local_memory,
                        remaining_reduction_nelems, iter_nelems,
                        reductions_per_wi));
            }
        });

        sycl::event cleanup_host_task_event =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(final_reduction_ev);
                const sycl::context &ctx = exec_q.get_context();

                cgh.host_task([ctx, partially_reduced_tmp] {
                    sycl::free(partially_reduced_tmp, ctx);
                });
            });

        // FIXME: do not return host-task event
        //   Instead collect all host-tasks to a list

        return cleanup_host_task_event;
    }
}

template <typename argTy, typename resTy, typename ReductionOpT>
sycl::event reduction_axis1_over_group_temps_contig_impl(
    sycl::queue &exec_q,
    size_t iter_nelems, // number of reductions    (num. of rows in a matrix
                        // when reducing over rows)
    size_t reduction_nelems, // size of each reduction  (length of rows, i.e.
                             // number of columns)
    const char *arg_cp,
    char *res_cp,
    py::ssize_t iter_arg_offset,
    py::ssize_t iter_res_offset,
    py::ssize_t reduction_arg_offset,
    const std::vector<sycl::event> &depends)
{
    const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_cp) +
                          iter_arg_offset + reduction_arg_offset;
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp) + iter_res_offset;

    constexpr resTy identity_val = su_ns::Identity<ReductionOpT, resTy>::value;

    const sycl::device &d = exec_q.get_device();
    const auto &sg_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

    if (reduction_nelems < wg) {
        sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
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

            cgh.parallel_for<class reduction_seq_contig_krn<
                argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                ReductionIndexerT>>(
                sycl::range<1>(iter_nelems),
                SequentialReduction<argTy, resTy, ReductionOpT,
                                    InputOutputIterIndexerT, ReductionIndexerT>(
                    arg_tp, res_tp, ReductionOpT(), identity_val,
                    in_out_iter_indexer, reduction_indexer, reduction_nelems));
        });

        return comp_ev;
    }

    constexpr size_t preferred_reductions_per_wi = 8;
    // max_max_wg prevents running out of resources on CPU
    constexpr size_t max_max_wg = 2048;
    size_t max_wg = std::min(
        max_max_wg, d.get_info<sycl::info::device::max_work_group_size>() / 2);

    size_t reductions_per_wi(preferred_reductions_per_wi);
    if (reduction_nelems <= preferred_reductions_per_wi * max_wg) {
        // Perform reduction using one 1 work-group per iteration,
        // can output directly to res
        sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
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

            if (iter_nelems == 1) {
                // increase GPU occupancy
                wg = max_wg;
            }
            reductions_per_wi =
                std::max<size_t>(1, (reduction_nelems + wg - 1) / wg);

            size_t reduction_groups =
                (reduction_nelems + reductions_per_wi * wg - 1) /
                (reductions_per_wi * wg);
            assert(reduction_groups == 1);

            auto globalRange =
                sycl::range<1>{iter_nelems * reduction_groups * wg};
            auto localRange = sycl::range<1>{wg};

            if constexpr (can_use_reduce_over_group<ReductionOpT, resTy>::value)
            {
                using KernelName =
                    class single_reduction_axis1_temps_contig_krn<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT>;

                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    ReductionOverGroupNoAtomicFunctor<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT>(arg_tp, res_tp, ReductionOpT(),
                                           identity_val, in_out_iter_indexer,
                                           reduction_indexer, reduction_nelems,
                                           iter_nelems, reductions_per_wi));
            }
            else {
                using SlmT = sycl::local_accessor<resTy, 1>;
                SlmT local_memory = SlmT(localRange, cgh);
                using KernelName =
                    class single_custom_reduction_axis1_temps_contig_krn<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT, SlmT>;

                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    CustomReductionOverGroupNoAtomicFunctor<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT, SlmT>(
                        arg_tp, res_tp, ReductionOpT(), identity_val,
                        in_out_iter_indexer, reduction_indexer, local_memory,
                        reduction_nelems, iter_nelems, reductions_per_wi));
            }
        });
        return comp_ev;
    }
    else {
        // more than one work-groups is needed, requires a temporary
        size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);
        assert(reduction_groups > 1);

        size_t second_iter_reduction_groups_ =
            (reduction_groups + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);

        resTy *partially_reduced_tmp = sycl::malloc_device<resTy>(
            iter_nelems * (reduction_groups + second_iter_reduction_groups_),
            exec_q);
        resTy *partially_reduced_tmp2 = nullptr;

        if (partially_reduced_tmp == nullptr) {
            throw std::runtime_error("Unabled to allocate device_memory");
        }
        else {
            partially_reduced_tmp2 =
                partially_reduced_tmp + reduction_groups * iter_nelems;
        }

        const sycl::event &first_reduction_ev = exec_q.submit([&](sycl::handler
                                                                      &cgh) {
            cgh.depends_on(depends);

            using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            using RowsIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
            using InputOutputIterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    RowsIndexerT, NoOpIndexerT>;
            using ReductionIndexerT = NoOpIndexerT;

            RowsIndexerT rows_indexer{
                0, static_cast<py::ssize_t>(iter_nelems),
                static_cast<py::ssize_t>(reduction_nelems)};
            NoOpIndexerT noop_tmp_indexer{};
            InputOutputIterIndexerT in_out_iter_indexer{rows_indexer,
                                                        noop_tmp_indexer};
            ReductionIndexerT reduction_indexer{};

            auto globalRange =
                sycl::range<1>{iter_nelems * reduction_groups * wg};
            auto localRange = sycl::range<1>{wg};

            if constexpr (can_use_reduce_over_group<ReductionOpT, resTy>::value)
            {
                using KernelName = class first_reduction_axis1_temps_contig_krn<
                    argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                    ReductionIndexerT>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    ReductionOverGroupNoAtomicFunctor<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT>(
                        arg_tp, partially_reduced_tmp, ReductionOpT(),
                        identity_val, in_out_iter_indexer, reduction_indexer,
                        reduction_nelems, iter_nelems,
                        preferred_reductions_per_wi));
            }
            else {
                using SlmT = sycl::local_accessor<resTy, 1>;
                SlmT local_memory = SlmT(localRange, cgh);
                using KernelName =
                    class first_custom_reduction_axis1_temps_contig_krn<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT, SlmT>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    CustomReductionOverGroupNoAtomicFunctor<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT, SlmT>(
                        arg_tp, partially_reduced_tmp, ReductionOpT(),
                        identity_val, in_out_iter_indexer, reduction_indexer,
                        local_memory, reduction_nelems, iter_nelems,
                        preferred_reductions_per_wi));
            }
        });

        size_t remaining_reduction_nelems = reduction_groups;

        resTy *temp_arg = partially_reduced_tmp;
        resTy *temp2_arg = partially_reduced_tmp2;
        sycl::event dependent_ev = first_reduction_ev;

        while (remaining_reduction_nelems >
               preferred_reductions_per_wi * max_wg) {
            size_t reduction_groups_ = (remaining_reduction_nelems +
                                        preferred_reductions_per_wi * wg - 1) /
                                       (preferred_reductions_per_wi * wg);
            assert(reduction_groups_ > 1);

            // keep reducing
            sycl::event partial_reduction_ev = exec_q.submit([&](sycl::handler
                                                                     &cgh) {
                cgh.depends_on(dependent_ev);

                using InputIndexerT =
                    dpctl::tensor::offset_utils::Strided1DIndexer;
                using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
                using InputOutputIterIndexerT =
                    dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                        InputIndexerT, ResIndexerT>;
                using ReductionIndexerT =
                    dpctl::tensor::offset_utils::NoOpIndexer;

                InputIndexerT inp_indexer{
                    0, static_cast<py::ssize_t>(iter_nelems),
                    static_cast<py::ssize_t>(reduction_groups_)};
                ResIndexerT res_iter_indexer{};

                InputOutputIterIndexerT in_out_iter_indexer{inp_indexer,
                                                            res_iter_indexer};
                ReductionIndexerT reduction_indexer{};

                auto globalRange =
                    sycl::range<1>{iter_nelems * reduction_groups_ * wg};
                auto localRange = sycl::range<1>{wg};
                if constexpr (can_use_reduce_over_group<ReductionOpT,
                                                        resTy>::value) {
                    using KernelName =
                        class middle_reduction_axis1_temps_contig_krn<
                            resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                            ReductionIndexerT>;
                    cgh.parallel_for<KernelName>(
                        sycl::nd_range<1>(globalRange, localRange),
                        ReductionOverGroupNoAtomicFunctor<
                            resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                            ReductionIndexerT>(
                            temp_arg, temp2_arg, ReductionOpT(), identity_val,
                            in_out_iter_indexer, reduction_indexer,
                            remaining_reduction_nelems, iter_nelems,
                            preferred_reductions_per_wi));
                }
                else {
                    using SlmT = sycl::local_accessor<resTy, 1>;
                    SlmT local_memory = SlmT(localRange, cgh);
                    using KernelName =
                        class middle_custom_reduction_axis1_temps_contig_krn<
                            resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                            ReductionIndexerT, SlmT>;
                    cgh.parallel_for<KernelName>(
                        sycl::nd_range<1>(globalRange, localRange),
                        CustomReductionOverGroupNoAtomicFunctor<
                            resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                            ReductionIndexerT, SlmT>(
                            temp_arg, temp2_arg, ReductionOpT(), identity_val,
                            in_out_iter_indexer, reduction_indexer,
                            local_memory, remaining_reduction_nelems,
                            iter_nelems, preferred_reductions_per_wi));
                }
            });

            remaining_reduction_nelems = reduction_groups_;
            std::swap(temp_arg, temp2_arg);
            dependent_ev = std::move(partial_reduction_ev);
        }

        // final reduction to res
        sycl::event final_reduction_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(dependent_ev);

            using InputIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
            using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            using InputOutputIterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    InputIndexerT, ResIndexerT>;
            using ReductionIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;

            InputIndexerT inp_indexer{
                0, static_cast<py::ssize_t>(iter_nelems),
                static_cast<py::ssize_t>(remaining_reduction_nelems)};
            ResIndexerT res_iter_indexer{};

            InputOutputIterIndexerT in_out_iter_indexer{inp_indexer,
                                                        res_iter_indexer};
            ReductionIndexerT reduction_indexer{};

            wg = max_wg;
            reductions_per_wi =
                std::max<size_t>(1, (remaining_reduction_nelems + wg - 1) / wg);

            size_t reduction_groups =
                (remaining_reduction_nelems + reductions_per_wi * wg - 1) /
                (reductions_per_wi * wg);
            assert(reduction_groups == 1);

            auto globalRange =
                sycl::range<1>{iter_nelems * reduction_groups * wg};
            auto localRange = sycl::range<1>{wg};

            if constexpr (can_use_reduce_over_group<ReductionOpT, resTy>::value)
            {
                using KernelName = class final_reduction_axis1_temps_contig_krn<
                    argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                    ReductionIndexerT>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    ReductionOverGroupNoAtomicFunctor<
                        resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT>(temp_arg, res_tp, ReductionOpT(),
                                           identity_val, in_out_iter_indexer,
                                           reduction_indexer,
                                           remaining_reduction_nelems,
                                           iter_nelems, reductions_per_wi));
            }
            else {
                using SlmT = sycl::local_accessor<resTy, 1>;
                SlmT local_memory = SlmT(localRange, cgh);
                using KernelName =
                    class final_custom_reduction_axis1_temps_contig_krn<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT, SlmT>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    CustomReductionOverGroupNoAtomicFunctor<
                        resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT, SlmT>(
                        temp_arg, res_tp, ReductionOpT(), identity_val,
                        in_out_iter_indexer, reduction_indexer, local_memory,
                        remaining_reduction_nelems, iter_nelems,
                        reductions_per_wi));
            }
        });

        sycl::event cleanup_host_task_event =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(final_reduction_ev);
                const sycl::context &ctx = exec_q.get_context();

                cgh.host_task([ctx, partially_reduced_tmp] {
                    sycl::free(partially_reduced_tmp, ctx);
                });
            });

        // FIXME: do not return host-task event
        //   Instead collect all host-tasks to a list

        return cleanup_host_task_event;
    }
}

template <typename argTy, typename resTy, typename ReductionOpT>
sycl::event reduction_axis0_over_group_temps_contig_impl(
    sycl::queue &exec_q,
    size_t iter_nelems, // number of reductions    (num. of rows in a matrix
                        // when reducing over rows)
    size_t reduction_nelems, // size of each reduction  (length of rows, i.e.
                             // number of columns)
    const char *arg_cp,
    char *res_cp,
    py::ssize_t iter_arg_offset,
    py::ssize_t iter_res_offset,
    py::ssize_t reduction_arg_offset,
    const std::vector<sycl::event> &depends)
{
    const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_cp) +
                          iter_arg_offset + reduction_arg_offset;
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp) + iter_res_offset;

    constexpr resTy identity_val = su_ns::Identity<ReductionOpT, resTy>::value;

    const sycl::device &d = exec_q.get_device();
    const auto &sg_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

    if (reduction_nelems < wg) {
        sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            using InputOutputIterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    NoOpIndexerT, NoOpIndexerT>;
            using ReductionIndexerT =
                dpctl::tensor::offset_utils::Strided1DIndexer;

            InputOutputIterIndexerT in_out_iter_indexer{NoOpIndexerT{},
                                                        NoOpIndexerT{}};
            ReductionIndexerT reduction_indexer{
                0, static_cast<py::ssize_t>(reduction_nelems),
                static_cast<py::ssize_t>(iter_nelems)};

            using KernelName =
                class reduction_seq_contig_krn<argTy, resTy, ReductionOpT,
                                               InputOutputIterIndexerT,
                                               ReductionIndexerT>;

            sycl::range<1> iter_range{iter_nelems};

            cgh.parallel_for<KernelName>(
                iter_range,
                SequentialReduction<argTy, resTy, ReductionOpT,
                                    InputOutputIterIndexerT, ReductionIndexerT>(
                    arg_tp, res_tp, ReductionOpT(), identity_val,
                    in_out_iter_indexer, reduction_indexer, reduction_nelems));
        });

        return comp_ev;
    }

    constexpr size_t preferred_reductions_per_wi = 8;
    // max_max_wg prevents running out of resources on CPU
    constexpr size_t max_max_wg = 2048;
    size_t max_wg = std::min(
        max_max_wg, d.get_info<sycl::info::device::max_work_group_size>() / 2);

    size_t reductions_per_wi(preferred_reductions_per_wi);
    if (reduction_nelems <= preferred_reductions_per_wi * max_wg) {
        // Perform reduction using one 1 work-group per iteration,
        // can output directly to res
        sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

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
                0, /* size */ static_cast<py::ssize_t>(reduction_nelems),
                /* step */ static_cast<py::ssize_t>(iter_nelems)};

            if (iter_nelems == 1) {
                // increase GPU occupancy
                wg = max_wg;
            }
            reductions_per_wi =
                std::max<size_t>(1, (reduction_nelems + wg - 1) / wg);

            size_t reduction_groups =
                (reduction_nelems + reductions_per_wi * wg - 1) /
                (reductions_per_wi * wg);
            assert(reduction_groups == 1);

            auto globalRange =
                sycl::range<1>{iter_nelems * reduction_groups * wg};
            auto localRange = sycl::range<1>{wg};

            if constexpr (can_use_reduce_over_group<ReductionOpT, resTy>::value)
            {
                using KernelName =
                    class single_reduction_axis0_temps_contig_krn<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT>;

                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    ReductionOverGroupNoAtomicFunctor<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT>(arg_tp, res_tp, ReductionOpT(),
                                           identity_val, in_out_iter_indexer,
                                           reduction_indexer, reduction_nelems,
                                           iter_nelems, reductions_per_wi));
            }
            else {
                using SlmT = sycl::local_accessor<resTy, 1>;
                SlmT local_memory = SlmT(localRange, cgh);
                using KernelName =
                    class single_custom_reduction_axis0_temps_contig_krn<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT, SlmT>;

                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    CustomReductionOverGroupNoAtomicFunctor<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT, SlmT>(
                        arg_tp, res_tp, ReductionOpT(), identity_val,
                        in_out_iter_indexer, reduction_indexer, local_memory,
                        reduction_nelems, iter_nelems, reductions_per_wi));
            }
        });
        return comp_ev;
    }
    else {
        // more than one work-groups is needed, requires a temporary
        size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);
        assert(reduction_groups > 1);

        size_t second_iter_reduction_groups_ =
            (reduction_groups + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);

        resTy *partially_reduced_tmp = sycl::malloc_device<resTy>(
            iter_nelems * (reduction_groups + second_iter_reduction_groups_),
            exec_q);
        resTy *partially_reduced_tmp2 = nullptr;

        if (partially_reduced_tmp == nullptr) {
            throw std::runtime_error("Unabled to allocate device_memory");
        }
        else {
            partially_reduced_tmp2 =
                partially_reduced_tmp + reduction_groups * iter_nelems;
        }

        const sycl::event &first_reduction_ev = exec_q.submit([&](sycl::handler
                                                                      &cgh) {
            cgh.depends_on(depends);

            using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            using ColsIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
            using InputOutputIterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    NoOpIndexerT, NoOpIndexerT>;
            using ReductionIndexerT = ColsIndexerT;

            NoOpIndexerT columns_indexer{};
            NoOpIndexerT noop_tmp_indexer{};
            InputOutputIterIndexerT in_out_iter_indexer{columns_indexer,
                                                        noop_tmp_indexer};
            ReductionIndexerT reduction_indexer{
                0, /* size */ static_cast<py::ssize_t>(reduction_nelems),
                /* step */ static_cast<py::ssize_t>(iter_nelems)};

            auto globalRange =
                sycl::range<1>{iter_nelems * reduction_groups * wg};
            auto localRange = sycl::range<1>{wg};

            if constexpr (can_use_reduce_over_group<ReductionOpT, resTy>::value)
            {
                using KernelName = class first_reduction_axis0_temps_contig_krn<
                    argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                    ReductionIndexerT>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    ReductionOverGroupNoAtomicFunctor<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT>(
                        arg_tp, partially_reduced_tmp, ReductionOpT(),
                        identity_val, in_out_iter_indexer, reduction_indexer,
                        reduction_nelems, iter_nelems,
                        preferred_reductions_per_wi));
            }
            else {
                using SlmT = sycl::local_accessor<resTy, 1>;
                SlmT local_memory = SlmT(localRange, cgh);
                using KernelName =
                    class first_custom_reduction_axis0_temps_contig_krn<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT, SlmT>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    CustomReductionOverGroupNoAtomicFunctor<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT, SlmT>(
                        arg_tp, partially_reduced_tmp, ReductionOpT(),
                        identity_val, in_out_iter_indexer, reduction_indexer,
                        local_memory, reduction_nelems, iter_nelems,
                        preferred_reductions_per_wi));
            }
        });

        size_t remaining_reduction_nelems = reduction_groups;

        resTy *temp_arg = partially_reduced_tmp;
        resTy *temp2_arg = partially_reduced_tmp2;
        sycl::event dependent_ev = first_reduction_ev;

        while (remaining_reduction_nelems >
               preferred_reductions_per_wi * max_wg) {
            size_t reduction_groups_ = (remaining_reduction_nelems +
                                        preferred_reductions_per_wi * wg - 1) /
                                       (preferred_reductions_per_wi * wg);
            assert(reduction_groups_ > 1);

            // keep reducing
            sycl::event partial_reduction_ev = exec_q.submit([&](sycl::handler
                                                                     &cgh) {
                cgh.depends_on(dependent_ev);

                using InputIndexerT =
                    dpctl::tensor::offset_utils::Strided1DIndexer;
                using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
                using InputOutputIterIndexerT =
                    dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                        InputIndexerT, ResIndexerT>;
                using ReductionIndexerT =
                    dpctl::tensor::offset_utils::NoOpIndexer;

                InputIndexerT inp_indexer{
                    0, static_cast<py::ssize_t>(iter_nelems),
                    static_cast<py::ssize_t>(reduction_groups_)};
                ResIndexerT res_iter_indexer{};

                InputOutputIterIndexerT in_out_iter_indexer{inp_indexer,
                                                            res_iter_indexer};
                ReductionIndexerT reduction_indexer{};

                auto globalRange =
                    sycl::range<1>{iter_nelems * reduction_groups_ * wg};
                auto localRange = sycl::range<1>{wg};
                if constexpr (can_use_reduce_over_group<ReductionOpT,
                                                        resTy>::value) {
                    using KernelName =
                        class middle_reduction_axis0_temps_contig_krn<
                            resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                            ReductionIndexerT>;
                    cgh.parallel_for<KernelName>(
                        sycl::nd_range<1>(globalRange, localRange),
                        ReductionOverGroupNoAtomicFunctor<
                            resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                            ReductionIndexerT>(
                            temp_arg, temp2_arg, ReductionOpT(), identity_val,
                            in_out_iter_indexer, reduction_indexer,
                            remaining_reduction_nelems, iter_nelems,
                            preferred_reductions_per_wi));
                }
                else {
                    using SlmT = sycl::local_accessor<resTy, 1>;
                    SlmT local_memory = SlmT(localRange, cgh);
                    using KernelName =
                        class middle_custom_reduction_axis0_temps_contig_krn<
                            resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                            ReductionIndexerT, SlmT>;
                    cgh.parallel_for<KernelName>(
                        sycl::nd_range<1>(globalRange, localRange),
                        CustomReductionOverGroupNoAtomicFunctor<
                            resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                            ReductionIndexerT, SlmT>(
                            temp_arg, temp2_arg, ReductionOpT(), identity_val,
                            in_out_iter_indexer, reduction_indexer,
                            local_memory, remaining_reduction_nelems,
                            iter_nelems, preferred_reductions_per_wi));
                }
            });

            remaining_reduction_nelems = reduction_groups_;
            std::swap(temp_arg, temp2_arg);
            dependent_ev = std::move(partial_reduction_ev);
        }

        // final reduction to res
        sycl::event final_reduction_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(dependent_ev);

            using InputIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
            using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            using InputOutputIterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    InputIndexerT, ResIndexerT>;
            using ReductionIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;

            InputIndexerT inp_indexer{
                0, static_cast<py::ssize_t>(iter_nelems),
                static_cast<py::ssize_t>(remaining_reduction_nelems)};
            ResIndexerT res_iter_indexer{};

            InputOutputIterIndexerT in_out_iter_indexer{inp_indexer,
                                                        res_iter_indexer};
            ReductionIndexerT reduction_indexer{};

            wg = max_wg;
            reductions_per_wi =
                std::max<size_t>(1, (remaining_reduction_nelems + wg - 1) / wg);

            size_t reduction_groups =
                (remaining_reduction_nelems + reductions_per_wi * wg - 1) /
                (reductions_per_wi * wg);
            assert(reduction_groups == 1);

            auto globalRange =
                sycl::range<1>{iter_nelems * reduction_groups * wg};
            auto localRange = sycl::range<1>{wg};

            if constexpr (can_use_reduce_over_group<ReductionOpT, resTy>::value)
            {
                using KernelName = class final_reduction_axis0_temps_contig_krn<
                    argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                    ReductionIndexerT>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    ReductionOverGroupNoAtomicFunctor<
                        resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT>(temp_arg, res_tp, ReductionOpT(),
                                           identity_val, in_out_iter_indexer,
                                           reduction_indexer,
                                           remaining_reduction_nelems,
                                           iter_nelems, reductions_per_wi));
            }
            else {
                using SlmT = sycl::local_accessor<resTy, 1>;
                SlmT local_memory = SlmT(localRange, cgh);
                using KernelName =
                    class final_custom_reduction_axis0_temps_contig_krn<
                        argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT, SlmT>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    CustomReductionOverGroupNoAtomicFunctor<
                        resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT, SlmT>(
                        temp_arg, res_tp, ReductionOpT(), identity_val,
                        in_out_iter_indexer, reduction_indexer, local_memory,
                        remaining_reduction_nelems, iter_nelems,
                        reductions_per_wi));
            }
        });

        sycl::event cleanup_host_task_event =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(final_reduction_ev);
                const sycl::context &ctx = exec_q.get_context();

                cgh.host_task([ctx, partially_reduced_tmp] {
                    sycl::free(partially_reduced_tmp, ctx);
                });
            });

        // FIXME: do not return host-task event
        //   Instead collect all host-tasks to a list

        return cleanup_host_task_event;
    }
}

/* @brief Types supported by comparison-reduction code based on atomic_ref */
template <typename argTy, typename outTy>
struct TypePairSupportDataForCompReductionAtomic
{

    /* value if true a kernel for <argTy, outTy> must be instantiated, false
     * otherwise */
    static constexpr bool is_defined = std::disjunction< // disjunction is C++17
                                                         // feature, supported
                                                         // by DPC++
        // input int32
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, std::int32_t>,
        // input uint32
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, std::uint32_t>,
        // input int64
        td_ns::TypePairDefinedEntry<argTy, std::int64_t, outTy, std::int64_t>,
        // input uint64
        td_ns::TypePairDefinedEntry<argTy, std::uint64_t, outTy, std::uint64_t>,
        // fall-through
        td_ns::NotDefinedEntry>::is_defined;
};

template <typename argTy, typename outTy>
struct TypePairSupportDataForCompReductionTemps
{

    static constexpr bool is_defined = std::disjunction< // disjunction is C++17
                                                         // feature, supported
                                                         // by DPC++ input bool
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, bool>,
        // input int8_t
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, std::int8_t>,

        // input uint8_t
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::uint8_t>,

        // input int16_t
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, std::int16_t>,

        // input uint16_t
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::uint16_t>,

        // input int32_t
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, std::int32_t>,
        // input uint32_t
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, std::uint32_t>,

        // input int64_t
        td_ns::TypePairDefinedEntry<argTy, std::int64_t, outTy, std::int64_t>,

        // input uint32_t
        td_ns::TypePairDefinedEntry<argTy, std::uint64_t, outTy, std::uint64_t>,

        // input half
        td_ns::TypePairDefinedEntry<argTy, sycl::half, outTy, sycl::half>,

        // input float
        td_ns::TypePairDefinedEntry<argTy, float, outTy, float>,

        // input double
        td_ns::TypePairDefinedEntry<argTy, double, outTy, double>,

        // input std::complex
        td_ns::TypePairDefinedEntry<argTy,
                                    std::complex<float>,
                                    outTy,
                                    std::complex<float>>,

        td_ns::TypePairDefinedEntry<argTy,
                                    std::complex<double>,
                                    outTy,
                                    std::complex<double>>,

        // fall-through
        td_ns::NotDefinedEntry>::is_defined;
};

template <typename fnT, typename srcTy, typename dstTy>
struct MaxOverAxisAtomicStridedFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForCompReductionAtomic<
                          srcTy, dstTy>::is_defined)
        {
            if constexpr (std::is_floating_point<dstTy>::value) {
                using ReductionOpT = su_ns::Maximum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_over_group_with_atomics_strided_impl<
                        srcTy, dstTy, ReductionOpT>;
            }
            else {
                using ReductionOpT = sycl::maximum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_over_group_with_atomics_strided_impl<
                        srcTy, dstTy, ReductionOpT>;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct MaxOverAxisTempsStridedFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForCompReductionTemps<
                          srcTy, dstTy>::is_defined)
        {
            if constexpr (std::is_integral_v<dstTy> &&
                          !std::is_same_v<dstTy, bool>) {
                using ReductionOpT = sycl::maximum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_over_group_temps_strided_impl<srcTy, dstTy,
                                                            ReductionOpT>;
            }
            else {
                using ReductionOpT = su_ns::Maximum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_over_group_temps_strided_impl<srcTy, dstTy,
                                                            ReductionOpT>;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct MaxOverAxis1AtomicContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForCompReductionAtomic<
                          srcTy, dstTy>::is_defined)
        {
            if constexpr (std::is_floating_point<dstTy>::value) {
                using ReductionOpT = su_ns::Maximum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_axis1_over_group_with_atomics_contig_impl<
                        srcTy, dstTy, ReductionOpT>;
            }
            else {
                using ReductionOpT = sycl::maximum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_axis1_over_group_with_atomics_contig_impl<
                        srcTy, dstTy, ReductionOpT>;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct MaxOverAxis0AtomicContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForCompReductionAtomic<
                          srcTy, dstTy>::is_defined)
        {
            if constexpr (std::is_floating_point<dstTy>::value) {
                using ReductionOpT = su_ns::Maximum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_axis0_over_group_with_atomics_contig_impl<
                        srcTy, dstTy, ReductionOpT>;
            }
            else {
                using ReductionOpT = sycl::maximum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_axis0_over_group_with_atomics_contig_impl<
                        srcTy, dstTy, ReductionOpT>;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct MaxOverAxis1TempsContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForCompReductionTemps<
                          srcTy, dstTy>::is_defined)
        {
            if constexpr (std::is_integral_v<dstTy> &&
                          !std::is_same_v<dstTy, bool>) {
                using ReductionOpT = sycl::maximum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_axis1_over_group_temps_contig_impl<srcTy, dstTy,
                                                                 ReductionOpT>;
            }
            else {
                using ReductionOpT = su_ns::Maximum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_axis1_over_group_temps_contig_impl<srcTy, dstTy,
                                                                 ReductionOpT>;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct MaxOverAxis0TempsContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForCompReductionTemps<
                          srcTy, dstTy>::is_defined)
        {
            if constexpr (std::is_integral_v<dstTy> &&
                          !std::is_same_v<dstTy, bool>) {
                using ReductionOpT = sycl::maximum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_axis0_over_group_temps_contig_impl<srcTy, dstTy,
                                                                 ReductionOpT>;
            }
            else {
                using ReductionOpT = su_ns::Maximum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_axis0_over_group_temps_contig_impl<srcTy, dstTy,
                                                                 ReductionOpT>;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct MinOverAxisAtomicStridedFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForCompReductionAtomic<
                          srcTy, dstTy>::is_defined)
        {
            if constexpr (std::is_floating_point<dstTy>::value) {
                using ReductionOpT = su_ns::Minimum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_over_group_with_atomics_strided_impl<
                        srcTy, dstTy, ReductionOpT>;
            }
            else {
                using ReductionOpT = sycl::minimum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_over_group_with_atomics_strided_impl<
                        srcTy, dstTy, ReductionOpT>;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct MinOverAxisTempsStridedFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForCompReductionTemps<
                          srcTy, dstTy>::is_defined)
        {
            if constexpr (std::is_integral_v<dstTy> &&
                          !std::is_same_v<dstTy, bool>) {
                using ReductionOpT = sycl::minimum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_over_group_temps_strided_impl<srcTy, dstTy,
                                                            ReductionOpT>;
            }
            else {
                using ReductionOpT = su_ns::Minimum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_over_group_temps_strided_impl<srcTy, dstTy,
                                                            ReductionOpT>;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct MinOverAxis1AtomicContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForCompReductionAtomic<
                          srcTy, dstTy>::is_defined)
        {
            if constexpr (std::is_floating_point<dstTy>::value) {
                using ReductionOpT = su_ns::Minimum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_axis1_over_group_with_atomics_contig_impl<
                        srcTy, dstTy, ReductionOpT>;
            }
            else {
                using ReductionOpT = sycl::minimum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_axis1_over_group_with_atomics_contig_impl<
                        srcTy, dstTy, ReductionOpT>;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct MinOverAxis0AtomicContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForCompReductionAtomic<
                          srcTy, dstTy>::is_defined)
        {
            if constexpr (std::is_floating_point<dstTy>::value) {
                using ReductionOpT = su_ns::Minimum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_axis0_over_group_with_atomics_contig_impl<
                        srcTy, dstTy, ReductionOpT>;
            }
            else {
                using ReductionOpT = sycl::minimum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_axis0_over_group_with_atomics_contig_impl<
                        srcTy, dstTy, ReductionOpT>;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct MinOverAxis1TempsContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForCompReductionTemps<
                          srcTy, dstTy>::is_defined)
        {
            if constexpr (std::is_integral_v<dstTy> &&
                          !std::is_same_v<dstTy, bool>) {
                using ReductionOpT = sycl::minimum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_axis1_over_group_temps_contig_impl<srcTy, dstTy,
                                                                 ReductionOpT>;
            }
            else {
                using ReductionOpT = su_ns::Minimum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_axis1_over_group_temps_contig_impl<srcTy, dstTy,
                                                                 ReductionOpT>;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct MinOverAxis0TempsContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForCompReductionTemps<
                          srcTy, dstTy>::is_defined)
        {
            if constexpr (std::is_integral_v<dstTy> &&
                          !std::is_same_v<dstTy, bool>) {
                using ReductionOpT = sycl::minimum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_axis0_over_group_temps_contig_impl<srcTy, dstTy,
                                                                 ReductionOpT>;
            }
            else {
                using ReductionOpT = su_ns::Minimum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_axis0_over_group_temps_contig_impl<srcTy, dstTy,
                                                                 ReductionOpT>;
            }
        }
        else {
            return nullptr;
        }
    }
};

// Sum

/* @brief Types supported by plus-reduction code based on atomic_ref */
template <typename argTy, typename outTy>
struct TypePairSupportDataForSumReductionAtomic
{

    /* value if true a kernel for <argTy, outTy> must be instantiated, false
     * otherwise */
    static constexpr bool is_defined = std::disjunction< // disjunction is C++17
                                                         // feature, supported
                                                         // by DPC++ input bool
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::uint64_t>,
        // input int8
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, std::int64_t>,
        // input uint8
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::uint64_t>,
        // input int16
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, std::int64_t>,
        // input uint16
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::uint64_t>,
        // input int32
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, std::int64_t>,
        // input uint32
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, std::uint64_t>,
        // input int64
        td_ns::TypePairDefinedEntry<argTy, std::int64_t, outTy, std::int64_t>,
        // input uint64
        td_ns::TypePairDefinedEntry<argTy, std::uint64_t, outTy, std::uint64_t>,
        // fall-through
        td_ns::NotDefinedEntry>::is_defined;
};

template <typename argTy, typename outTy>
struct TypePairSupportDataForSumReductionTemps
{

    static constexpr bool is_defined = std::disjunction< // disjunction is C++17
                                                         // feature, supported
                                                         // by DPC++ input bool
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::int8_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::uint8_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::int16_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::uint16_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::uint64_t>,

        // input int8_t
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, std::int8_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, std::int16_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, std::int64_t>,

        // input uint8_t
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::uint8_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::int16_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::uint16_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::uint64_t>,

        // input int16_t
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, std::int16_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, std::int64_t>,

        // input uint16_t
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::uint16_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::uint64_t>,

        // input int32_t
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, std::int64_t>,

        // input uint32_t
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, std::uint64_t>,

        // input int64_t
        td_ns::TypePairDefinedEntry<argTy, std::int64_t, outTy, std::int64_t>,

        // input uint32_t
        td_ns::TypePairDefinedEntry<argTy, std::uint64_t, outTy, std::uint64_t>,

        // input half
        td_ns::TypePairDefinedEntry<argTy, sycl::half, outTy, sycl::half>,
        td_ns::TypePairDefinedEntry<argTy, sycl::half, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, sycl::half, outTy, double>,
        td_ns::
            TypePairDefinedEntry<argTy, sycl::half, outTy, std::complex<float>>,
        td_ns::TypePairDefinedEntry<argTy,
                                    sycl::half,
                                    outTy,
                                    std::complex<double>>,

        // input float
        td_ns::TypePairDefinedEntry<argTy, float, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, float, outTy, double>,
        td_ns::TypePairDefinedEntry<argTy, float, outTy, std::complex<float>>,
        td_ns::TypePairDefinedEntry<argTy, float, outTy, std::complex<double>>,

        // input double
        td_ns::TypePairDefinedEntry<argTy, double, outTy, double>,
        td_ns::TypePairDefinedEntry<argTy, double, outTy, std::complex<double>>,

        // input std::complex
        td_ns::TypePairDefinedEntry<argTy,
                                    std::complex<float>,
                                    outTy,
                                    std::complex<float>>,
        td_ns::TypePairDefinedEntry<argTy,
                                    std::complex<float>,
                                    outTy,
                                    std::complex<double>>,

        td_ns::TypePairDefinedEntry<argTy,
                                    std::complex<double>,
                                    outTy,
                                    std::complex<double>>,

        // fall-throug
        td_ns::NotDefinedEntry>::is_defined;
};

template <typename fnT, typename srcTy, typename dstTy>
struct SumOverAxisAtomicStridedFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForSumReductionAtomic<
                          srcTy, dstTy>::is_defined)
        {
            using ReductionOpT = sycl::plus<dstTy>;
            return dpctl::tensor::kernels::
                reduction_over_group_with_atomics_strided_impl<srcTy, dstTy,
                                                               ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct SumOverAxisTempsStridedFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForSumReductionTemps<
                          srcTy, dstTy>::is_defined) {
            using ReductionOpT = sycl::plus<dstTy>;
            return dpctl::tensor::kernels::
                reduction_over_group_temps_strided_impl<srcTy, dstTy,
                                                        ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct SumOverAxis1AtomicContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForSumReductionAtomic<
                          srcTy, dstTy>::is_defined)
        {
            using ReductionOpT = sycl::plus<dstTy>;
            return dpctl::tensor::kernels::
                reduction_axis1_over_group_with_atomics_contig_impl<
                    srcTy, dstTy, ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct SumOverAxis0AtomicContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForSumReductionAtomic<
                          srcTy, dstTy>::is_defined)
        {
            using ReductionOpT = sycl::plus<dstTy>;
            return dpctl::tensor::kernels::
                reduction_axis0_over_group_with_atomics_contig_impl<
                    srcTy, dstTy, ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct SumOverAxis1TempsContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForSumReductionTemps<
                          srcTy, dstTy>::is_defined) {
            using ReductionOpT = sycl::plus<dstTy>;
            return dpctl::tensor::kernels::
                reduction_axis1_over_group_temps_contig_impl<srcTy, dstTy,
                                                             ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct SumOverAxis0TempsContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForSumReductionTemps<
                          srcTy, dstTy>::is_defined) {
            using ReductionOpT = sycl::plus<dstTy>;
            return dpctl::tensor::kernels::
                reduction_axis0_over_group_temps_contig_impl<srcTy, dstTy,
                                                             ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

// Product

/* @brief Types supported by plus-reduction code based on atomic_ref */
template <typename argTy, typename outTy>
struct TypePairSupportDataForProductReductionAtomic
{

    /* value if true a kernel for <argTy, outTy> must be instantiated, false
     * otherwise */
    static constexpr bool is_defined = std::disjunction< // disjunction is C++17
                                                         // feature, supported
                                                         // by DPC++ input bool
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::uint64_t>,
        // input int8
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, std::int64_t>,
        // input uint8
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::uint64_t>,
        // input int16
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, std::int64_t>,
        // input uint16
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::uint64_t>,
        // input int32
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, std::int64_t>,
        // input uint32
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, std::uint64_t>,
        // input int64
        td_ns::TypePairDefinedEntry<argTy, std::int64_t, outTy, std::int64_t>,
        // input uint64
        td_ns::TypePairDefinedEntry<argTy, std::uint64_t, outTy, std::uint64_t>,
        // fall-through
        td_ns::NotDefinedEntry>::is_defined;
};

template <typename argTy, typename outTy>
struct TypePairSupportDataForProductReductionTemps
{

    static constexpr bool is_defined = std::disjunction< // disjunction is C++17
                                                         // feature, supported
                                                         // by DPC++ input bool
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::int8_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::uint8_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::int16_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::uint16_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::uint64_t>,

        // input int8_t
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, std::int8_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, std::int16_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, std::int64_t>,

        // input uint8_t
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::uint8_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::int16_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::uint16_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::uint64_t>,

        // input int16_t
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, std::int16_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, std::int64_t>,

        // input uint16_t
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::uint16_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::uint64_t>,

        // input int32_t
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, std::int64_t>,

        // input uint32_t
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, std::uint64_t>,

        // input int64_t
        td_ns::TypePairDefinedEntry<argTy, std::int64_t, outTy, std::int64_t>,

        // input uint32_t
        td_ns::TypePairDefinedEntry<argTy, std::uint64_t, outTy, std::uint64_t>,

        // input half
        td_ns::TypePairDefinedEntry<argTy, sycl::half, outTy, sycl::half>,
        td_ns::TypePairDefinedEntry<argTy, sycl::half, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, sycl::half, outTy, double>,
        td_ns::
            TypePairDefinedEntry<argTy, sycl::half, outTy, std::complex<float>>,
        td_ns::TypePairDefinedEntry<argTy,
                                    sycl::half,
                                    outTy,
                                    std::complex<double>>,

        // input float
        td_ns::TypePairDefinedEntry<argTy, float, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, float, outTy, double>,
        td_ns::TypePairDefinedEntry<argTy, float, outTy, std::complex<float>>,
        td_ns::TypePairDefinedEntry<argTy, float, outTy, std::complex<double>>,

        // input double
        td_ns::TypePairDefinedEntry<argTy, double, outTy, double>,
        td_ns::TypePairDefinedEntry<argTy, double, outTy, std::complex<double>>,

        // input std::complex
        td_ns::TypePairDefinedEntry<argTy,
                                    std::complex<float>,
                                    outTy,
                                    std::complex<float>>,
        td_ns::TypePairDefinedEntry<argTy,
                                    std::complex<float>,
                                    outTy,
                                    std::complex<double>>,

        td_ns::TypePairDefinedEntry<argTy,
                                    std::complex<double>,
                                    outTy,
                                    std::complex<double>>,

        // fall-throug
        td_ns::NotDefinedEntry>::is_defined;
};

template <typename fnT, typename srcTy, typename dstTy>
struct ProductOverAxisAtomicStridedFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForProductReductionAtomic<
                          srcTy, dstTy>::is_defined)
        {
            using ReductionOpT = sycl::multiplies<dstTy>;
            return dpctl::tensor::kernels::
                reduction_over_group_with_atomics_strided_impl<srcTy, dstTy,
                                                               ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct ProductOverAxisTempsStridedFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForProductReductionTemps<
                          srcTy, dstTy>::is_defined)
        {
            using ReductionOpT = sycl::multiplies<dstTy>;
            return dpctl::tensor::kernels::
                reduction_over_group_temps_strided_impl<srcTy, dstTy,
                                                        ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct ProductOverAxis1AtomicContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForProductReductionAtomic<
                          srcTy, dstTy>::is_defined)
        {
            using ReductionOpT = sycl::multiplies<dstTy>;
            return dpctl::tensor::kernels::
                reduction_axis1_over_group_with_atomics_contig_impl<
                    srcTy, dstTy, ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct ProductOverAxis0AtomicContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForProductReductionAtomic<
                          srcTy, dstTy>::is_defined)
        {
            using ReductionOpT = sycl::multiplies<dstTy>;
            return dpctl::tensor::kernels::
                reduction_axis0_over_group_with_atomics_contig_impl<
                    srcTy, dstTy, ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct ProductOverAxis1TempsContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForProductReductionTemps<
                          srcTy, dstTy>::is_defined)
        {
            using ReductionOpT = sycl::multiplies<dstTy>;
            return dpctl::tensor::kernels::
                reduction_axis1_over_group_temps_contig_impl<srcTy, dstTy,
                                                             ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct ProductOverAxis0TempsContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForProductReductionTemps<
                          srcTy, dstTy>::is_defined)
        {
            using ReductionOpT = sycl::multiplies<dstTy>;
            return dpctl::tensor::kernels::
                reduction_axis0_over_group_temps_contig_impl<srcTy, dstTy,
                                                             ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename argTy, typename outTy>
struct TypePairSupportDataForHypotReductionTemps
{

    static constexpr bool is_defined = std::disjunction< // disjunction is C++17
                                                         // feature, supported
                                                         // by DPC++ input bool
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, sycl::half>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, double>,

        // input int8_t
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, sycl::half>,
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, double>,

        // input uint8_t
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, sycl::half>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, double>,

        // input int16_t
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, double>,

        // input uint16_t
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, double>,

        // input int32_t
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, double>,

        // input uint32_t
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, double>,

        // input int64_t
        td_ns::TypePairDefinedEntry<argTy, std::int64_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::int64_t, outTy, double>,

        // input uint64_t
        td_ns::TypePairDefinedEntry<argTy, std::uint64_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::uint64_t, outTy, double>,

        // input half
        td_ns::TypePairDefinedEntry<argTy, sycl::half, outTy, sycl::half>,
        td_ns::TypePairDefinedEntry<argTy, sycl::half, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, sycl::half, outTy, double>,

        // input float
        td_ns::TypePairDefinedEntry<argTy, float, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, float, outTy, double>,

        // input double
        td_ns::TypePairDefinedEntry<argTy, double, outTy, double>,

        // fall-through
        td_ns::NotDefinedEntry>::is_defined;
};

template <typename fnT, typename srcTy, typename dstTy>
struct HypotOverAxisTempsStridedFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForHypotReductionTemps<
                          srcTy, dstTy>::is_defined)
        {
            using ReductionOpT = su_ns::Hypot<dstTy>;
            return dpctl::tensor::kernels::
                reduction_over_group_temps_strided_impl<srcTy, dstTy,
                                                        ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct HypotOverAxis1TempsContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForHypotReductionTemps<
                          srcTy, dstTy>::is_defined)
        {
            using ReductionOpT = su_ns::Hypot<dstTy>;
            return dpctl::tensor::kernels::
                reduction_axis1_over_group_temps_contig_impl<srcTy, dstTy,
                                                             ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct HypotOverAxis0TempsContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForHypotReductionTemps<
                          srcTy, dstTy>::is_defined)
        {
            using ReductionOpT = su_ns::Hypot<dstTy>;
            return dpctl::tensor::kernels::
                reduction_axis0_over_group_temps_contig_impl<srcTy, dstTy,
                                                             ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename argTy, typename outTy>
struct TypePairSupportDataForLogSumExpReductionTemps
{

    static constexpr bool is_defined = std::disjunction< // disjunction is C++17
                                                         // feature, supported
                                                         // by DPC++ input bool
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, sycl::half>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, double>,

        // input int8_t
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, sycl::half>,
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, double>,

        // input uint8_t
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, sycl::half>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, double>,

        // input int16_t
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, double>,

        // input uint16_t
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, double>,

        // input int32_t
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, double>,

        // input uint32_t
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, double>,

        // input int64_t
        td_ns::TypePairDefinedEntry<argTy, std::int64_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::int64_t, outTy, double>,

        // input uint64_t
        td_ns::TypePairDefinedEntry<argTy, std::uint64_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::uint64_t, outTy, double>,

        // input half
        td_ns::TypePairDefinedEntry<argTy, sycl::half, outTy, sycl::half>,
        td_ns::TypePairDefinedEntry<argTy, sycl::half, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, sycl::half, outTy, double>,

        // input float
        td_ns::TypePairDefinedEntry<argTy, float, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, float, outTy, double>,

        // input double
        td_ns::TypePairDefinedEntry<argTy, double, outTy, double>,

        // fall-through
        td_ns::NotDefinedEntry>::is_defined;
};

template <typename fnT, typename srcTy, typename dstTy>
struct LogSumExpOverAxisTempsStridedFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForLogSumExpReductionTemps<
                          srcTy, dstTy>::is_defined)
        {
            using ReductionOpT = su_ns::LogSumExp<dstTy>;
            return dpctl::tensor::kernels::
                reduction_over_group_temps_strided_impl<srcTy, dstTy,
                                                        ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct LogSumExpOverAxis1TempsContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForLogSumExpReductionTemps<
                          srcTy, dstTy>::is_defined)
        {
            using ReductionOpT = su_ns::LogSumExp<dstTy>;
            return dpctl::tensor::kernels::
                reduction_axis1_over_group_temps_contig_impl<srcTy, dstTy,
                                                             ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct LogSumExpOverAxis0TempsContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForLogSumExpReductionTemps<
                          srcTy, dstTy>::is_defined)
        {
            using ReductionOpT = su_ns::LogSumExp<dstTy>;
            return dpctl::tensor::kernels::
                reduction_axis0_over_group_temps_contig_impl<srcTy, dstTy,
                                                             ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

// Argmax and Argmin

/* = Search reduction using reduce_over_group*/

template <typename argT,
          typename outT,
          typename ReductionOp,
          typename IdxReductionOp,
          typename InputOutputIterIndexerT,
          typename InputRedIndexerT,
          bool First,
          bool Last>
struct SearchReduction
{
private:
    const argT *inp_ = nullptr;
    argT *vals_ = nullptr;
    const outT *inds_ = nullptr;
    outT *out_ = nullptr;
    ReductionOp reduction_op_;
    argT identity_;
    IdxReductionOp idx_reduction_op_;
    outT idx_identity_;
    InputOutputIterIndexerT inp_out_iter_indexer_;
    InputRedIndexerT inp_reduced_dims_indexer_;
    size_t reduction_max_gid_ = 0;
    size_t iter_gws_ = 1;
    size_t reductions_per_wi = 16;

public:
    SearchReduction(const argT *data,
                    argT *vals,
                    const outT *inds,
                    outT *res,
                    ReductionOp reduction_op,
                    const argT &identity_val,
                    IdxReductionOp idx_reduction_op,
                    const outT &idx_identity_val,
                    InputOutputIterIndexerT arg_res_iter_indexer,
                    InputRedIndexerT arg_reduced_dims_indexer,
                    size_t reduction_size,
                    size_t iteration_size,
                    size_t reduction_size_per_wi)
        : inp_(data), vals_(vals), inds_(inds), out_(res),
          reduction_op_(reduction_op), identity_(identity_val),
          idx_reduction_op_(idx_reduction_op), idx_identity_(idx_identity_val),
          inp_out_iter_indexer_(arg_res_iter_indexer),
          inp_reduced_dims_indexer_(arg_reduced_dims_indexer),
          reduction_max_gid_(reduction_size), iter_gws_(iteration_size),
          reductions_per_wi(reduction_size_per_wi)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        const size_t reduction_lid = it.get_local_id(0);
        const size_t wg = it.get_local_range(0); //   0 <= reduction_lid < wg

        const size_t iter_gid = it.get_group(0) % iter_gws_;
        const size_t reduction_batch_id = it.get_group(0) / iter_gws_;
        const size_t n_reduction_groups = it.get_group_range(0) / iter_gws_;

        // work-items operates over input with indices
        //   inp_data_id = reduction_batch_id * wg * reductions_per_wi + m * wg
        //   + reduction_lid
        // for 0 <= m < reductions_per_wi

        auto inp_out_iter_offsets_ = inp_out_iter_indexer_(iter_gid);
        const auto &inp_iter_offset = inp_out_iter_offsets_.get_first_offset();
        const auto &out_iter_offset = inp_out_iter_offsets_.get_second_offset();

        argT local_red_val(identity_);
        outT local_idx(idx_identity_);
        size_t arg_reduce_gid0 =
            reduction_lid + reduction_batch_id * wg * reductions_per_wi;
        for (size_t m = 0; m < reductions_per_wi; ++m) {
            size_t arg_reduce_gid = arg_reduce_gid0 + m * wg;

            if (arg_reduce_gid < reduction_max_gid_) {
                auto inp_reduction_offset =
                    inp_reduced_dims_indexer_(arg_reduce_gid);
                auto inp_offset = inp_iter_offset + inp_reduction_offset;

                argT val = inp_[inp_offset];
                if (val == local_red_val) {
                    if constexpr (!First) {
                        local_idx =
                            idx_reduction_op_(local_idx, inds_[inp_offset]);
                    }
                    else {
                        local_idx = idx_reduction_op_(
                            local_idx, static_cast<outT>(arg_reduce_gid));
                    }
                }
                else {
                    if constexpr (su_ns::IsMinimum<argT, ReductionOp>::value) {
                        if (val < local_red_val) {
                            local_red_val = val;
                            if constexpr (!First) {
                                local_idx = inds_[inp_offset];
                            }
                            else {
                                local_idx = static_cast<outT>(arg_reduce_gid);
                            }
                        }
                    }
                    else if constexpr (su_ns::IsMaximum<argT,
                                                        ReductionOp>::value) {
                        if (val > local_red_val) {
                            local_red_val = val;
                            if constexpr (!First) {
                                local_idx = inds_[inp_offset];
                            }
                            else {
                                local_idx = static_cast<outT>(arg_reduce_gid);
                            }
                        }
                    }
                }
            }
        }

        auto work_group = it.get_group();
        // This only works if reduction_op_ is from small set of operators
        argT red_val_over_wg = sycl::reduce_over_group(
            work_group, local_red_val, identity_, reduction_op_);

        if constexpr (std::is_integral_v<argT>) {
            local_idx =
                (red_val_over_wg == local_red_val) ? local_idx : idx_identity_;
        }
        else {
            local_idx =
                (red_val_over_wg == local_red_val ||
                 std::isnan(red_val_over_wg) || std::isnan(local_red_val))
                    ? local_idx
                    : idx_identity_;
        }
        outT idx_over_wg = sycl::reduce_over_group(
            work_group, local_idx, idx_identity_, idx_reduction_op_);

        if (work_group.leader()) {
            // each group writes to a different memory location
            if constexpr (!Last) {
                // if not the final reduction, write value corresponding to
                // an index to a temporary
                vals_[out_iter_offset * n_reduction_groups +
                      reduction_batch_id] = red_val_over_wg;
            }
            out_[out_iter_offset * n_reduction_groups + reduction_batch_id] =
                idx_over_wg;
        }
    }
};

/* = Search reduction using custom_reduce_over_group*/

template <typename argT,
          typename outT,
          typename ReductionOp,
          typename IdxReductionOp,
          typename InputOutputIterIndexerT,
          typename InputRedIndexerT,
          typename SlmT,
          bool First,
          bool Last>
struct CustomSearchReduction
{
private:
    const argT *inp_ = nullptr;
    argT *vals_ = nullptr;
    const outT *inds_ = nullptr;
    outT *out_ = nullptr;
    ReductionOp reduction_op_;
    argT identity_;
    IdxReductionOp idx_reduction_op_;
    outT idx_identity_;
    InputOutputIterIndexerT inp_out_iter_indexer_;
    InputRedIndexerT inp_reduced_dims_indexer_;
    SlmT local_mem_;
    size_t reduction_max_gid_ = 0;
    size_t iter_gws_ = 1;
    size_t reductions_per_wi = 16;

public:
    CustomSearchReduction(const argT *data,
                          argT *vals,
                          outT *inds,
                          outT *res,
                          ReductionOp reduction_op,
                          const argT &identity_val,
                          IdxReductionOp idx_reduction_op,
                          const outT &idx_identity_val,
                          InputOutputIterIndexerT arg_res_iter_indexer,
                          InputRedIndexerT arg_reduced_dims_indexer,
                          SlmT local_mem,
                          size_t reduction_size,
                          size_t iteration_size,
                          size_t reduction_size_per_wi)
        : inp_(data), vals_(vals), inds_(inds), out_(res),
          reduction_op_(reduction_op), identity_(identity_val),
          idx_reduction_op_(idx_reduction_op), idx_identity_(idx_identity_val),
          inp_out_iter_indexer_(arg_res_iter_indexer),
          inp_reduced_dims_indexer_(arg_reduced_dims_indexer),
          local_mem_(local_mem), reduction_max_gid_(reduction_size),
          iter_gws_(iteration_size), reductions_per_wi(reduction_size_per_wi)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        const size_t reduction_lid = it.get_local_id(0);
        const size_t wg = it.get_local_range(0); //   0 <= reduction_lid < wg

        const size_t iter_gid = it.get_group(0) % iter_gws_;
        const size_t reduction_batch_id = it.get_group(0) / iter_gws_;
        const size_t n_reduction_groups = it.get_group_range(0) / iter_gws_;

        // work-items operates over input with indices
        //   inp_data_id = reduction_batch_id * wg * reductions_per_wi + m * wg
        //   + reduction_lid
        // for 0 <= m < reductions_per_wi

        auto inp_out_iter_offsets_ = inp_out_iter_indexer_(iter_gid);
        const auto &inp_iter_offset = inp_out_iter_offsets_.get_first_offset();
        const auto &out_iter_offset = inp_out_iter_offsets_.get_second_offset();

        argT local_red_val(identity_);
        outT local_idx(idx_identity_);
        size_t arg_reduce_gid0 =
            reduction_lid + reduction_batch_id * wg * reductions_per_wi;
        for (size_t m = 0; m < reductions_per_wi; ++m) {
            size_t arg_reduce_gid = arg_reduce_gid0 + m * wg;

            if (arg_reduce_gid < reduction_max_gid_) {
                auto inp_reduction_offset =
                    inp_reduced_dims_indexer_(arg_reduce_gid);
                auto inp_offset = inp_iter_offset + inp_reduction_offset;

                argT val = inp_[inp_offset];
                if (val == local_red_val) {
                    if constexpr (!First) {
                        local_idx =
                            idx_reduction_op_(local_idx, inds_[inp_offset]);
                    }
                    else {
                        local_idx = idx_reduction_op_(
                            local_idx, static_cast<outT>(arg_reduce_gid));
                    }
                }
                else {
                    if constexpr (su_ns::IsMinimum<argT, ReductionOp>::value) {
                        using dpctl::tensor::type_utils::is_complex;
                        if constexpr (is_complex<argT>::value) {
                            using dpctl::tensor::math_utils::less_complex;
                            // less_complex always returns false for NaNs, so
                            // check
                            if (less_complex<argT>(val, local_red_val) ||
                                std::isnan(std::real(val)) ||
                                std::isnan(std::imag(val)))
                            {
                                local_red_val = val;
                                if constexpr (!First) {
                                    local_idx = inds_[inp_offset];
                                }
                                else {
                                    local_idx =
                                        static_cast<outT>(arg_reduce_gid);
                                }
                            }
                        }
                        else if constexpr (std::is_floating_point_v<argT>) {
                            if (val < local_red_val || std::isnan(val)) {
                                local_red_val = val;
                                if constexpr (!First) {
                                    local_idx = inds_[inp_offset];
                                }
                                else {
                                    local_idx =
                                        static_cast<outT>(arg_reduce_gid);
                                }
                            }
                        }
                        else {
                            if (val < local_red_val) {
                                local_red_val = val;
                                if constexpr (!First) {
                                    local_idx = inds_[inp_offset];
                                }
                                else {
                                    local_idx =
                                        static_cast<outT>(arg_reduce_gid);
                                }
                            }
                        }
                    }
                    else if constexpr (su_ns::IsMaximum<argT,
                                                        ReductionOp>::value) {
                        using dpctl::tensor::type_utils::is_complex;
                        if constexpr (is_complex<argT>::value) {
                            using dpctl::tensor::math_utils::greater_complex;
                            if (greater_complex<argT>(val, local_red_val) ||
                                std::isnan(std::real(val)) ||
                                std::isnan(std::imag(val)))
                            {
                                local_red_val = val;
                                if constexpr (!First) {
                                    local_idx = inds_[inp_offset];
                                }
                                else {
                                    local_idx =
                                        static_cast<outT>(arg_reduce_gid);
                                }
                            }
                        }
                        else if constexpr (std::is_floating_point_v<argT>) {
                            if (val > local_red_val || std::isnan(val)) {
                                local_red_val = val;
                                if constexpr (!First) {
                                    local_idx = inds_[inp_offset];
                                }
                                else {
                                    local_idx =
                                        static_cast<outT>(arg_reduce_gid);
                                }
                            }
                        }
                        else {
                            if (val > local_red_val) {
                                local_red_val = val;
                                if constexpr (!First) {
                                    local_idx = inds_[inp_offset];
                                }
                                else {
                                    local_idx =
                                        static_cast<outT>(arg_reduce_gid);
                                }
                            }
                        }
                    }
                }
            }
        }

        auto work_group = it.get_group();
        // This only works if reduction_op_ is from small set of operators
        argT red_val_over_wg = su_ns::custom_reduce_over_group(
            work_group, local_mem_, local_red_val, reduction_op_);

        using dpctl::tensor::type_utils::is_complex;
        if constexpr (is_complex<argT>::value) {
            // equality does not hold for NaNs, so check here
            local_idx = (red_val_over_wg == local_red_val ||
                         std::isnan(std::real(local_red_val)) ||
                         std::isnan(std::imag(local_red_val)))
                            ? local_idx
                            : idx_identity_;
        }
        else if constexpr (std::is_floating_point_v<argT>) {
            // equality does not hold for NaNs, so check here
            local_idx =
                (red_val_over_wg == local_red_val || std::isnan(local_red_val))
                    ? local_idx
                    : idx_identity_;
        }
        else {
            local_idx =
                red_val_over_wg == local_red_val ? local_idx : idx_identity_;
        }
        outT idx_over_wg = sycl::reduce_over_group(
            work_group, local_idx, idx_identity_, idx_reduction_op_);
        if (work_group.leader()) {
            // each group writes to a different memory location
            if constexpr (!Last) {
                // if not the final reduction, write value corresponding to
                // an index to a temporary
                vals_[out_iter_offset * n_reduction_groups +
                      reduction_batch_id] = red_val_over_wg;
            }
            out_[out_iter_offset * n_reduction_groups + reduction_batch_id] =
                idx_over_wg;
        }
    }
};

typedef sycl::event (*search_strided_impl_fn_ptr)(
    sycl::queue,
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

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6,
          bool b1,
          bool b2>
class search_over_group_temps_strided_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6,
          typename T7,
          bool b1,
          bool b2>
class custom_search_over_group_temps_strided_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6,
          bool b1,
          bool b2>
class single_search_axis0_temps_contig_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6,
          bool b1,
          bool b2>
class first_search_axis0_temps_contig_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6,
          bool b1,
          bool b2>
class middle_search_axis0_temps_contig_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6,
          bool b1,
          bool b2>
class final_search_axis0_temps_contig_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6,
          typename T7,
          bool b1,
          bool b2>
class single_custom_search_axis0_temps_contig_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6,
          typename T7,
          bool b1,
          bool b2>
class first_custom_search_axis0_temps_contig_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6,
          typename T7,
          bool b1,
          bool b2>
class middle_custom_search_axis0_temps_contig_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6,
          typename T7,
          bool b1,
          bool b2>
class final_custom_search_axis0_temps_contig_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6,
          bool b1,
          bool b2>
class single_search_axis1_temps_contig_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6,
          bool b1,
          bool b2>
class first_search_axis1_temps_contig_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6,
          bool b1,
          bool b2>
class middle_search_axis1_temps_contig_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6,
          bool b1,
          bool b2>
class final_search_axis1_temps_contig_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6,
          typename T7,
          bool b1,
          bool b2>
class single_custom_search_axis1_temps_contig_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6,
          typename T7,
          bool b1,
          bool b2>
class first_custom_search_axis1_temps_contig_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6,
          typename T7,
          bool b1,
          bool b2>
class middle_custom_search_axis1_temps_contig_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6,
          typename T7,
          bool b1,
          bool b2>
class final_custom_search_axis1_temps_contig_krn;

template <typename argTy,
          typename resTy,
          typename ReductionOpT,
          typename IndexOpT>
sycl::event search_over_group_temps_strided_impl(
    sycl::queue exec_q,
    size_t iter_nelems, // number of reductions    (num. of rows in a matrix
                        // when reducing over rows)
    size_t reduction_nelems, // size of each reduction  (length of rows, i.e.
                             // number of columns)
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

    constexpr argTy identity_val = su_ns::Identity<ReductionOpT, argTy>::value;
    constexpr resTy idx_identity_val = su_ns::Identity<IndexOpT, resTy>::value;

    const sycl::device &d = exec_q.get_device();
    const auto &sg_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

    constexpr size_t preferred_reductions_per_wi = 4;
    // max_max_wg prevents running out of resources on CPU
    size_t max_wg =
        std::min(size_t(2048),
                 d.get_info<sycl::info::device::max_work_group_size>() / 2);

    size_t reductions_per_wi(preferred_reductions_per_wi);
    if (reduction_nelems <= preferred_reductions_per_wi * max_wg) {
        // Perform reduction using one 1 work-group per iteration,
        // can output directly to res
        sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
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

            if (iter_nelems == 1) {
                // increase GPU occupancy
                wg = max_wg;
            }
            reductions_per_wi =
                std::max<size_t>(1, (reduction_nelems + wg - 1) / wg);

            size_t reduction_groups =
                (reduction_nelems + reductions_per_wi * wg - 1) /
                (reductions_per_wi * wg);
            assert(reduction_groups == 1);

            auto globalRange =
                sycl::range<1>{iter_nelems * reduction_groups * wg};
            auto localRange = sycl::range<1>{wg};

            if constexpr (can_use_reduce_over_group<ReductionOpT, resTy>::value)
            {
                using KernelName = class search_over_group_temps_strided_krn<
                    argTy, resTy, ReductionOpT, IndexOpT,
                    InputOutputIterIndexerT, ReductionIndexerT, true, true>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    SearchReduction<argTy, resTy, ReductionOpT, IndexOpT,
                                    InputOutputIterIndexerT, ReductionIndexerT,
                                    true, true>(
                        arg_tp, nullptr, nullptr, res_tp, ReductionOpT(),
                        identity_val, IndexOpT(), idx_identity_val,
                        in_out_iter_indexer, reduction_indexer,
                        reduction_nelems, iter_nelems, reductions_per_wi));
            }
            else {
                using SlmT = sycl::local_accessor<argTy, 1>;
                SlmT local_memory = SlmT(localRange, cgh);
                using KernelName =
                    class custom_search_over_group_temps_strided_krn<
                        argTy, resTy, ReductionOpT, IndexOpT,
                        InputOutputIterIndexerT, ReductionIndexerT, SlmT, true,
                        true>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    CustomSearchReduction<argTy, resTy, ReductionOpT, IndexOpT,
                                          InputOutputIterIndexerT,
                                          ReductionIndexerT, SlmT, true, true>(
                        arg_tp, nullptr, nullptr, res_tp, ReductionOpT(),
                        identity_val, IndexOpT(), idx_identity_val,
                        in_out_iter_indexer, reduction_indexer, local_memory,
                        reduction_nelems, iter_nelems, reductions_per_wi));
            }
        });
        return comp_ev;
    }
    else {
        // more than one work-groups is needed, requires a temporary
        size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);
        assert(reduction_groups > 1);

        size_t second_iter_reduction_groups_ =
            (reduction_groups + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);

        resTy *partially_reduced_tmp = sycl::malloc_device<resTy>(
            iter_nelems * (reduction_groups + second_iter_reduction_groups_),
            exec_q);
        resTy *partially_reduced_tmp2 = nullptr;

        if (partially_reduced_tmp == nullptr) {
            throw std::runtime_error("Unable to allocate device_memory");
        }
        else {
            partially_reduced_tmp2 =
                partially_reduced_tmp + reduction_groups * iter_nelems;
        }

        argTy *partially_reduced_vals_tmp = sycl::malloc_device<argTy>(
            iter_nelems * (reduction_groups + second_iter_reduction_groups_),
            exec_q);
        argTy *partially_reduced_vals_tmp2 = nullptr;

        if (partially_reduced_vals_tmp == nullptr) {
            throw std::runtime_error("Unable to allocate device_memory");
        }
        else {
            partially_reduced_vals_tmp2 =
                partially_reduced_vals_tmp + reduction_groups * iter_nelems;
        }

        sycl::event first_reduction_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            using InputIndexerT = dpctl::tensor::offset_utils::StridedIndexer;
            using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            using InputOutputIterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    InputIndexerT, ResIndexerT>;
            using ReductionIndexerT =
                dpctl::tensor::offset_utils::StridedIndexer;

            // Only 2*iter_nd entries describing shape and strides of iterated
            // dimensions of input array from iter_shape_and_strides are going
            // to be accessed by inp_indexer
            InputIndexerT inp_indexer(iter_nd, iter_arg_offset,
                                      iter_shape_and_strides);
            ResIndexerT noop_tmp_indexer{};

            InputOutputIterIndexerT in_out_iter_indexer{inp_indexer,
                                                        noop_tmp_indexer};
            ReductionIndexerT reduction_indexer{red_nd, reduction_arg_offset,
                                                reduction_shape_stride};

            auto globalRange =
                sycl::range<1>{iter_nelems * reduction_groups * wg};
            auto localRange = sycl::range<1>{wg};

            if constexpr (can_use_reduce_over_group<ReductionOpT, resTy>::value)
            {
                using KernelName = class search_over_group_temps_strided_krn<
                    argTy, resTy, ReductionOpT, IndexOpT,
                    InputOutputIterIndexerT, ReductionIndexerT, true, false>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    SearchReduction<argTy, resTy, ReductionOpT, IndexOpT,
                                    InputOutputIterIndexerT, ReductionIndexerT,
                                    true, false>(
                        arg_tp, partially_reduced_vals_tmp, nullptr,
                        partially_reduced_tmp, ReductionOpT(), identity_val,
                        IndexOpT(), idx_identity_val, in_out_iter_indexer,
                        reduction_indexer, reduction_nelems, iter_nelems,
                        preferred_reductions_per_wi));
            }
            else {
                using SlmT = sycl::local_accessor<argTy, 1>;
                SlmT local_memory = SlmT(localRange, cgh);
                using KernelName =
                    class custom_search_over_group_temps_strided_krn<
                        argTy, resTy, ReductionOpT, IndexOpT,
                        InputOutputIterIndexerT, ReductionIndexerT, SlmT, true,
                        false>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    CustomSearchReduction<argTy, resTy, ReductionOpT, IndexOpT,
                                          InputOutputIterIndexerT,
                                          ReductionIndexerT, SlmT, true, false>(
                        arg_tp, partially_reduced_vals_tmp, nullptr,
                        partially_reduced_tmp, ReductionOpT(), identity_val,
                        IndexOpT(), idx_identity_val, in_out_iter_indexer,
                        reduction_indexer, local_memory, reduction_nelems,
                        iter_nelems, preferred_reductions_per_wi));
            }
        });

        size_t remaining_reduction_nelems = reduction_groups;

        resTy *temp_arg = partially_reduced_tmp;
        resTy *temp2_arg = partially_reduced_tmp2;

        argTy *vals_temp_arg = partially_reduced_vals_tmp;
        argTy *vals_temp2_arg = partially_reduced_vals_tmp2;

        sycl::event dependent_ev = first_reduction_ev;

        while (remaining_reduction_nelems >
               preferred_reductions_per_wi * max_wg) {
            size_t reduction_groups_ = (remaining_reduction_nelems +
                                        preferred_reductions_per_wi * wg - 1) /
                                       (preferred_reductions_per_wi * wg);
            assert(reduction_groups_ > 1);

            // keep reducing
            sycl::event partial_reduction_ev = exec_q.submit([&](sycl::handler
                                                                     &cgh) {
                cgh.depends_on(dependent_ev);

                using InputIndexerT =
                    dpctl::tensor::offset_utils::Strided1DIndexer;
                using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
                using InputOutputIterIndexerT =
                    dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                        InputIndexerT, ResIndexerT>;
                using ReductionIndexerT =
                    dpctl::tensor::offset_utils::NoOpIndexer;

                InputIndexerT inp_indexer{
                    0, static_cast<py::ssize_t>(iter_nelems),
                    static_cast<py::ssize_t>(reduction_groups_)};
                ResIndexerT res_iter_indexer{};

                InputOutputIterIndexerT in_out_iter_indexer{inp_indexer,
                                                            res_iter_indexer};
                ReductionIndexerT reduction_indexer{};

                auto globalRange =
                    sycl::range<1>{iter_nelems * reduction_groups_ * wg};
                auto localRange = sycl::range<1>{wg};
                if constexpr (can_use_reduce_over_group<ReductionOpT,
                                                        resTy>::value) {
                    using KernelName =
                        class search_over_group_temps_strided_krn<
                            argTy, resTy, ReductionOpT, IndexOpT,
                            InputOutputIterIndexerT, ReductionIndexerT, false,
                            false>;
                    cgh.parallel_for<KernelName>(
                        sycl::nd_range<1>(globalRange, localRange),
                        SearchReduction<argTy, resTy, ReductionOpT, IndexOpT,
                                        InputOutputIterIndexerT,
                                        ReductionIndexerT, false, false>(
                            vals_temp_arg, vals_temp2_arg, temp_arg, temp2_arg,
                            ReductionOpT(), identity_val, IndexOpT(),
                            idx_identity_val, in_out_iter_indexer,
                            reduction_indexer, remaining_reduction_nelems,
                            iter_nelems, preferred_reductions_per_wi));
                }
                else {
                    using SlmT = sycl::local_accessor<argTy, 1>;
                    SlmT local_memory = SlmT(localRange, cgh);
                    using KernelName =
                        class custom_search_over_group_temps_strided_krn<
                            argTy, resTy, ReductionOpT, IndexOpT,
                            InputOutputIterIndexerT, ReductionIndexerT, SlmT,
                            false, false>;
                    cgh.parallel_for<KernelName>(
                        sycl::nd_range<1>(globalRange, localRange),
                        CustomSearchReduction<argTy, resTy, ReductionOpT,
                                              IndexOpT, InputOutputIterIndexerT,
                                              ReductionIndexerT, SlmT, false,
                                              false>(
                            vals_temp_arg, vals_temp2_arg, temp_arg, temp2_arg,
                            ReductionOpT(), identity_val, IndexOpT(),
                            idx_identity_val, in_out_iter_indexer,
                            reduction_indexer, local_memory,
                            remaining_reduction_nelems, iter_nelems,
                            preferred_reductions_per_wi));
                }
            });

            remaining_reduction_nelems = reduction_groups_;
            std::swap(temp_arg, temp2_arg);
            std::swap(vals_temp_arg, vals_temp2_arg);
            dependent_ev = partial_reduction_ev;
        }

        // final reduction to res
        sycl::event final_reduction_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(dependent_ev);

            using InputIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
            using ResIndexerT =
                dpctl::tensor::offset_utils::UnpackedStridedIndexer;
            using InputOutputIterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    InputIndexerT, ResIndexerT>;
            using ReductionIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;

            InputIndexerT inp_indexer{
                0, static_cast<py::ssize_t>(iter_nelems),
                static_cast<py::ssize_t>(remaining_reduction_nelems)};
            ResIndexerT res_iter_indexer{iter_nd, iter_res_offset,
                                         /* shape */ iter_shape_and_strides,
                                         /* strides */ iter_shape_and_strides +
                                             2 * iter_nd};

            InputOutputIterIndexerT in_out_iter_indexer{inp_indexer,
                                                        res_iter_indexer};
            ReductionIndexerT reduction_indexer{};

            wg = max_wg;
            reductions_per_wi =
                std::max<size_t>(1, (remaining_reduction_nelems + wg - 1) / wg);

            size_t reduction_groups =
                (remaining_reduction_nelems + reductions_per_wi * wg - 1) /
                (reductions_per_wi * wg);
            assert(reduction_groups == 1);

            auto globalRange =
                sycl::range<1>{iter_nelems * reduction_groups * wg};
            auto localRange = sycl::range<1>{wg};

            if constexpr (can_use_reduce_over_group<ReductionOpT, resTy>::value)
            {
                using KernelName = class search_over_group_temps_strided_krn<
                    argTy, resTy, ReductionOpT, IndexOpT,
                    InputOutputIterIndexerT, ReductionIndexerT, false, true>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    SearchReduction<argTy, resTy, ReductionOpT, IndexOpT,
                                    InputOutputIterIndexerT, ReductionIndexerT,
                                    false, true>(
                        vals_temp_arg, nullptr, temp_arg, res_tp,
                        ReductionOpT(), identity_val, IndexOpT(),
                        idx_identity_val, in_out_iter_indexer,
                        reduction_indexer, remaining_reduction_nelems,
                        iter_nelems, reductions_per_wi));
            }
            else {
                using SlmT = sycl::local_accessor<argTy, 1>;
                SlmT local_memory = SlmT(localRange, cgh);
                using KernelName =
                    class custom_search_over_group_temps_strided_krn<
                        argTy, resTy, ReductionOpT, IndexOpT,
                        InputOutputIterIndexerT, ReductionIndexerT, SlmT, false,
                        true>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    CustomSearchReduction<argTy, resTy, ReductionOpT, IndexOpT,
                                          InputOutputIterIndexerT,
                                          ReductionIndexerT, SlmT, false, true>(
                        vals_temp_arg, nullptr, temp_arg, res_tp,
                        ReductionOpT(), identity_val, IndexOpT(),
                        idx_identity_val, in_out_iter_indexer,
                        reduction_indexer, local_memory,
                        remaining_reduction_nelems, iter_nelems,
                        reductions_per_wi));
            }
        });

        sycl::event cleanup_host_task_event =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(final_reduction_ev);
                sycl::context ctx = exec_q.get_context();

                cgh.host_task(
                    [ctx, partially_reduced_tmp, partially_reduced_vals_tmp] {
                        sycl::free(partially_reduced_tmp, ctx);
                        sycl::free(partially_reduced_vals_tmp, ctx);
                    });
            });

        // FIXME: do not return host-task event
        //   Instead collect all host-tasks to a list

        return cleanup_host_task_event;
    }
}

typedef sycl::event (*search_contig_impl_fn_ptr)(
    sycl::queue,
    size_t,
    size_t,
    const char *,
    char *,
    py::ssize_t,
    py::ssize_t,
    py::ssize_t,
    const std::vector<sycl::event> &);

template <typename argTy,
          typename resTy,
          typename ReductionOpT,
          typename IndexOpT>
sycl::event search_axis1_over_group_temps_contig_impl(
    sycl::queue exec_q,
    size_t iter_nelems, // number of reductions    (num. of rows in a matrix
                        // when reducing over rows)
    size_t reduction_nelems, // size of each reduction  (length of rows, i.e.
                             // number of columns)
    const char *arg_cp,
    char *res_cp,
    py::ssize_t iter_arg_offset,
    py::ssize_t iter_res_offset,
    py::ssize_t reduction_arg_offset,
    const std::vector<sycl::event> &depends)
{
    const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_cp) +
                          iter_arg_offset + reduction_arg_offset;
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp) + iter_res_offset;

    constexpr argTy identity_val = su_ns::Identity<ReductionOpT, argTy>::value;
    constexpr resTy idx_identity_val = su_ns::Identity<IndexOpT, resTy>::value;

    const sycl::device &d = exec_q.get_device();
    const auto &sg_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

    constexpr size_t preferred_reductions_per_wi = 8;
    // max_max_wg prevents running out of resources on CPU
    size_t max_wg =
        std::min(size_t(2048),
                 d.get_info<sycl::info::device::max_work_group_size>() / 2);

    size_t reductions_per_wi(preferred_reductions_per_wi);
    if (reduction_nelems <= preferred_reductions_per_wi * max_wg) {
        // Perform reduction using one 1 work-group per iteration,
        // can output directly to res
        sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
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

            if (iter_nelems == 1) {
                // increase GPU occupancy
                wg = max_wg;
            }
            reductions_per_wi =
                std::max<size_t>(1, (reduction_nelems + wg - 1) / wg);

            size_t reduction_groups =
                (reduction_nelems + reductions_per_wi * wg - 1) /
                (reductions_per_wi * wg);
            assert(reduction_groups == 1);

            auto globalRange =
                sycl::range<1>{iter_nelems * reduction_groups * wg};
            auto localRange = sycl::range<1>{wg};

            if constexpr (can_use_reduce_over_group<ReductionOpT, resTy>::value)
            {
                using KernelName = class single_search_axis1_temps_contig_krn<
                    argTy, resTy, ReductionOpT, IndexOpT,
                    InputOutputIterIndexerT, ReductionIndexerT, true, true>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    SearchReduction<argTy, resTy, ReductionOpT, IndexOpT,
                                    InputOutputIterIndexerT, ReductionIndexerT,
                                    true, true>(
                        arg_tp, nullptr, nullptr, res_tp, ReductionOpT(),
                        identity_val, IndexOpT(), idx_identity_val,
                        in_out_iter_indexer, reduction_indexer,
                        reduction_nelems, iter_nelems, reductions_per_wi));
            }
            else {
                using SlmT = sycl::local_accessor<argTy, 1>;
                SlmT local_memory = SlmT(localRange, cgh);
                using KernelName =
                    class single_custom_search_axis1_temps_contig_krn<
                        argTy, resTy, ReductionOpT, IndexOpT,
                        InputOutputIterIndexerT, ReductionIndexerT, SlmT, true,
                        true>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    CustomSearchReduction<argTy, resTy, ReductionOpT, IndexOpT,
                                          InputOutputIterIndexerT,
                                          ReductionIndexerT, SlmT, true, true>(
                        arg_tp, nullptr, nullptr, res_tp, ReductionOpT(),
                        identity_val, IndexOpT(), idx_identity_val,
                        in_out_iter_indexer, reduction_indexer, local_memory,
                        reduction_nelems, iter_nelems, reductions_per_wi));
            }
        });
        return comp_ev;
    }
    else {
        // more than one work-groups is needed, requires a temporary
        size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);
        assert(reduction_groups > 1);

        size_t second_iter_reduction_groups_ =
            (reduction_groups + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);

        resTy *partially_reduced_tmp = sycl::malloc_device<resTy>(
            iter_nelems * (reduction_groups + second_iter_reduction_groups_),
            exec_q);
        resTy *partially_reduced_tmp2 = nullptr;

        if (partially_reduced_tmp == nullptr) {
            throw std::runtime_error("Unable to allocate device_memory");
        }
        else {
            partially_reduced_tmp2 =
                partially_reduced_tmp + reduction_groups * iter_nelems;
        }

        argTy *partially_reduced_vals_tmp = sycl::malloc_device<argTy>(
            iter_nelems * (reduction_groups + second_iter_reduction_groups_),
            exec_q);
        argTy *partially_reduced_vals_tmp2 = nullptr;

        if (partially_reduced_vals_tmp == nullptr) {
            throw std::runtime_error("Unable to allocate device_memory");
        }
        else {
            partially_reduced_vals_tmp2 =
                partially_reduced_vals_tmp + reduction_groups * iter_nelems;
        }

        sycl::event first_reduction_ev = exec_q.submit([&](sycl::handler &cgh) {
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

            auto globalRange =
                sycl::range<1>{iter_nelems * reduction_groups * wg};
            auto localRange = sycl::range<1>{wg};

            if constexpr (can_use_reduce_over_group<ReductionOpT, resTy>::value)
            {
                using KernelName = class first_search_axis1_temps_contig_krn<
                    argTy, resTy, ReductionOpT, IndexOpT,
                    InputOutputIterIndexerT, ReductionIndexerT, true, false>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    SearchReduction<argTy, resTy, ReductionOpT, IndexOpT,
                                    InputOutputIterIndexerT, ReductionIndexerT,
                                    true, false>(
                        arg_tp, partially_reduced_vals_tmp, nullptr,
                        partially_reduced_tmp, ReductionOpT(), identity_val,
                        IndexOpT(), idx_identity_val, in_out_iter_indexer,
                        reduction_indexer, reduction_nelems, iter_nelems,
                        preferred_reductions_per_wi));
            }
            else {
                using SlmT = sycl::local_accessor<argTy, 1>;
                SlmT local_memory = SlmT(localRange, cgh);
                using KernelName =
                    class first_custom_search_axis1_temps_contig_krn<
                        argTy, resTy, ReductionOpT, IndexOpT,
                        InputOutputIterIndexerT, ReductionIndexerT, SlmT, true,
                        false>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    CustomSearchReduction<argTy, resTy, ReductionOpT, IndexOpT,
                                          InputOutputIterIndexerT,
                                          ReductionIndexerT, SlmT, true, false>(
                        arg_tp, partially_reduced_vals_tmp, nullptr,
                        partially_reduced_tmp, ReductionOpT(), identity_val,
                        IndexOpT(), idx_identity_val, in_out_iter_indexer,
                        reduction_indexer, local_memory, reduction_nelems,
                        iter_nelems, preferred_reductions_per_wi));
            }
        });

        size_t remaining_reduction_nelems = reduction_groups;

        resTy *temp_arg = partially_reduced_tmp;
        resTy *temp2_arg = partially_reduced_tmp2;

        argTy *vals_temp_arg = partially_reduced_vals_tmp;
        argTy *vals_temp2_arg = partially_reduced_vals_tmp2;

        sycl::event dependent_ev = first_reduction_ev;

        while (remaining_reduction_nelems >
               preferred_reductions_per_wi * max_wg) {
            size_t reduction_groups_ = (remaining_reduction_nelems +
                                        preferred_reductions_per_wi * wg - 1) /
                                       (preferred_reductions_per_wi * wg);
            assert(reduction_groups_ > 1);

            // keep reducing
            sycl::event partial_reduction_ev = exec_q.submit([&](sycl::handler
                                                                     &cgh) {
                cgh.depends_on(dependent_ev);

                using InputIndexerT =
                    dpctl::tensor::offset_utils::Strided1DIndexer;
                using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
                using InputOutputIterIndexerT =
                    dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                        InputIndexerT, ResIndexerT>;
                using ReductionIndexerT =
                    dpctl::tensor::offset_utils::NoOpIndexer;

                InputIndexerT inp_indexer{
                    0, static_cast<py::ssize_t>(iter_nelems),
                    static_cast<py::ssize_t>(reduction_groups_)};
                ResIndexerT res_iter_indexer{};

                InputOutputIterIndexerT in_out_iter_indexer{inp_indexer,
                                                            res_iter_indexer};
                ReductionIndexerT reduction_indexer{};

                auto globalRange =
                    sycl::range<1>{iter_nelems * reduction_groups_ * wg};
                auto localRange = sycl::range<1>{wg};
                if constexpr (can_use_reduce_over_group<ReductionOpT,
                                                        resTy>::value) {
                    using KernelName =
                        class middle_search_axis1_temps_contig_krn<
                            argTy, resTy, ReductionOpT, IndexOpT,
                            InputOutputIterIndexerT, ReductionIndexerT, false,
                            false>;
                    cgh.parallel_for<KernelName>(
                        sycl::nd_range<1>(globalRange, localRange),
                        SearchReduction<argTy, resTy, ReductionOpT, IndexOpT,
                                        InputOutputIterIndexerT,
                                        ReductionIndexerT, false, false>(
                            vals_temp_arg, vals_temp2_arg, temp_arg, temp2_arg,
                            ReductionOpT(), identity_val, IndexOpT(),
                            idx_identity_val, in_out_iter_indexer,
                            reduction_indexer, remaining_reduction_nelems,
                            iter_nelems, preferred_reductions_per_wi));
                }
                else {
                    using SlmT = sycl::local_accessor<argTy, 1>;
                    SlmT local_memory = SlmT(localRange, cgh);
                    using KernelName =
                        class middle_custom_search_axis1_temps_contig_krn<
                            argTy, resTy, ReductionOpT, IndexOpT,
                            InputOutputIterIndexerT, ReductionIndexerT, SlmT,
                            false, false>;
                    cgh.parallel_for<KernelName>(
                        sycl::nd_range<1>(globalRange, localRange),
                        CustomSearchReduction<argTy, resTy, ReductionOpT,
                                              IndexOpT, InputOutputIterIndexerT,
                                              ReductionIndexerT, SlmT, false,
                                              false>(
                            vals_temp_arg, vals_temp2_arg, temp_arg, temp2_arg,
                            ReductionOpT(), identity_val, IndexOpT(),
                            idx_identity_val, in_out_iter_indexer,
                            reduction_indexer, local_memory,
                            remaining_reduction_nelems, iter_nelems,
                            preferred_reductions_per_wi));
                }
            });

            remaining_reduction_nelems = reduction_groups_;
            std::swap(temp_arg, temp2_arg);
            std::swap(vals_temp_arg, vals_temp2_arg);
            dependent_ev = partial_reduction_ev;
        }

        // final reduction to res
        sycl::event final_reduction_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(dependent_ev);

            using InputIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
            using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            using InputOutputIterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    InputIndexerT, ResIndexerT>;
            using ReductionIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;

            InputIndexerT inp_indexer{
                0, static_cast<py::ssize_t>(iter_nelems),
                static_cast<py::ssize_t>(remaining_reduction_nelems)};
            ResIndexerT res_iter_indexer{};

            InputOutputIterIndexerT in_out_iter_indexer{inp_indexer,
                                                        res_iter_indexer};
            ReductionIndexerT reduction_indexer{};

            wg = max_wg;
            reductions_per_wi =
                std::max<size_t>(1, (remaining_reduction_nelems + wg - 1) / wg);

            size_t reduction_groups =
                (remaining_reduction_nelems + reductions_per_wi * wg - 1) /
                (reductions_per_wi * wg);
            assert(reduction_groups == 1);

            auto globalRange =
                sycl::range<1>{iter_nelems * reduction_groups * wg};
            auto localRange = sycl::range<1>{wg};

            if constexpr (can_use_reduce_over_group<ReductionOpT, resTy>::value)
            {
                using KernelName = class final_search_axis1_temps_contig_krn<
                    argTy, resTy, ReductionOpT, IndexOpT,
                    InputOutputIterIndexerT, ReductionIndexerT, false, true>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    SearchReduction<argTy, resTy, ReductionOpT, IndexOpT,
                                    InputOutputIterIndexerT, ReductionIndexerT,
                                    false, true>(
                        vals_temp_arg, nullptr, temp_arg, res_tp,
                        ReductionOpT(), identity_val, IndexOpT(),
                        idx_identity_val, in_out_iter_indexer,
                        reduction_indexer, remaining_reduction_nelems,
                        iter_nelems, reductions_per_wi));
            }
            else {
                using SlmT = sycl::local_accessor<argTy, 1>;
                SlmT local_memory = SlmT(localRange, cgh);
                using KernelName =
                    class final_custom_search_axis1_temps_contig_krn<
                        argTy, resTy, ReductionOpT, IndexOpT,
                        InputOutputIterIndexerT, ReductionIndexerT, SlmT, false,
                        true>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    CustomSearchReduction<argTy, resTy, ReductionOpT, IndexOpT,
                                          InputOutputIterIndexerT,
                                          ReductionIndexerT, SlmT, false, true>(
                        vals_temp_arg, nullptr, temp_arg, res_tp,
                        ReductionOpT(), identity_val, IndexOpT(),
                        idx_identity_val, in_out_iter_indexer,
                        reduction_indexer, local_memory,
                        remaining_reduction_nelems, iter_nelems,
                        reductions_per_wi));
            }
        });

        sycl::event cleanup_host_task_event =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(final_reduction_ev);
                sycl::context ctx = exec_q.get_context();

                cgh.host_task(
                    [ctx, partially_reduced_tmp, partially_reduced_vals_tmp] {
                        sycl::free(partially_reduced_tmp, ctx);
                        sycl::free(partially_reduced_vals_tmp, ctx);
                    });
            });

        // FIXME: do not return host-task event
        //   Instead collect all host-tasks to a list

        return cleanup_host_task_event;
    }
}

template <typename argTy,
          typename resTy,
          typename ReductionOpT,
          typename IndexOpT>
sycl::event search_axis0_over_group_temps_contig_impl(
    sycl::queue exec_q,
    size_t iter_nelems, // number of reductions    (num. of rows in a matrix
                        // when reducing over rows)
    size_t reduction_nelems, // size of each reduction  (length of rows, i.e.
                             // number of columns)
    const char *arg_cp,
    char *res_cp,
    py::ssize_t iter_arg_offset,
    py::ssize_t iter_res_offset,
    py::ssize_t reduction_arg_offset,
    const std::vector<sycl::event> &depends)
{
    const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_cp) +
                          iter_arg_offset + reduction_arg_offset;
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp) + iter_res_offset;

    constexpr argTy identity_val = su_ns::Identity<ReductionOpT, argTy>::value;
    constexpr resTy idx_identity_val = su_ns::Identity<IndexOpT, resTy>::value;

    const sycl::device &d = exec_q.get_device();
    const auto &sg_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

    constexpr size_t preferred_reductions_per_wi = 8;
    // max_max_wg prevents running out of resources on CPU
    size_t max_wg =
        std::min(size_t(2048),
                 d.get_info<sycl::info::device::max_work_group_size>() / 2);

    size_t reductions_per_wi(preferred_reductions_per_wi);
    if (reduction_nelems <= preferred_reductions_per_wi * max_wg) {
        // Perform reduction using one 1 work-group per iteration,
        // can output directly to res
        sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

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
                0, /* size */ static_cast<py::ssize_t>(reduction_nelems),
                /* step */ static_cast<py::ssize_t>(iter_nelems)};

            if (iter_nelems == 1) {
                // increase GPU occupancy
                wg = max_wg;
            }
            reductions_per_wi =
                std::max<size_t>(1, (reduction_nelems + wg - 1) / wg);

            size_t reduction_groups =
                (reduction_nelems + reductions_per_wi * wg - 1) /
                (reductions_per_wi * wg);
            assert(reduction_groups == 1);

            auto globalRange =
                sycl::range<1>{iter_nelems * reduction_groups * wg};
            auto localRange = sycl::range<1>{wg};

            if constexpr (can_use_reduce_over_group<ReductionOpT, resTy>::value)
            {
                using KernelName = class single_search_axis0_temps_contig_krn<
                    argTy, resTy, ReductionOpT, IndexOpT,
                    InputOutputIterIndexerT, ReductionIndexerT, true, true>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    SearchReduction<argTy, resTy, ReductionOpT, IndexOpT,
                                    InputOutputIterIndexerT, ReductionIndexerT,
                                    true, true>(
                        arg_tp, nullptr, nullptr, res_tp, ReductionOpT(),
                        identity_val, IndexOpT(), idx_identity_val,
                        in_out_iter_indexer, reduction_indexer,
                        reduction_nelems, iter_nelems, reductions_per_wi));
            }
            else {
                using SlmT = sycl::local_accessor<argTy, 1>;
                SlmT local_memory = SlmT(localRange, cgh);
                using KernelName =
                    class single_custom_search_axis0_temps_contig_krn<
                        argTy, resTy, ReductionOpT, IndexOpT,
                        InputOutputIterIndexerT, ReductionIndexerT, SlmT, true,
                        true>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    CustomSearchReduction<argTy, resTy, ReductionOpT, IndexOpT,
                                          InputOutputIterIndexerT,
                                          ReductionIndexerT, SlmT, true, true>(
                        arg_tp, nullptr, nullptr, res_tp, ReductionOpT(),
                        identity_val, IndexOpT(), idx_identity_val,
                        in_out_iter_indexer, reduction_indexer, local_memory,
                        reduction_nelems, iter_nelems, reductions_per_wi));
            }
        });
        return comp_ev;
    }
    else {
        // more than one work-groups is needed, requires a temporary
        size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);
        assert(reduction_groups > 1);

        size_t second_iter_reduction_groups_ =
            (reduction_groups + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);

        resTy *partially_reduced_tmp = sycl::malloc_device<resTy>(
            iter_nelems * (reduction_groups + second_iter_reduction_groups_),
            exec_q);
        resTy *partially_reduced_tmp2 = nullptr;

        if (partially_reduced_tmp == nullptr) {
            throw std::runtime_error("Unable to allocate device_memory");
        }
        else {
            partially_reduced_tmp2 =
                partially_reduced_tmp + reduction_groups * iter_nelems;
        }

        argTy *partially_reduced_vals_tmp = sycl::malloc_device<argTy>(
            iter_nelems * (reduction_groups + second_iter_reduction_groups_),
            exec_q);
        argTy *partially_reduced_vals_tmp2 = nullptr;

        if (partially_reduced_vals_tmp == nullptr) {
            throw std::runtime_error("Unable to allocate device_memory");
        }
        else {
            partially_reduced_vals_tmp2 =
                partially_reduced_vals_tmp + reduction_groups * iter_nelems;
        }

        sycl::event first_reduction_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

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
                0, /* size */ static_cast<py::ssize_t>(reduction_nelems),
                /* step */ static_cast<py::ssize_t>(iter_nelems)};

            auto globalRange =
                sycl::range<1>{iter_nelems * reduction_groups * wg};
            auto localRange = sycl::range<1>{wg};

            if constexpr (can_use_reduce_over_group<ReductionOpT, resTy>::value)
            {
                using KernelName = class first_search_axis0_temps_contig_krn<
                    argTy, resTy, ReductionOpT, IndexOpT,
                    InputOutputIterIndexerT, ReductionIndexerT, true, false>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    SearchReduction<argTy, resTy, ReductionOpT, IndexOpT,
                                    InputOutputIterIndexerT, ReductionIndexerT,
                                    true, false>(
                        arg_tp, partially_reduced_vals_tmp, nullptr,
                        partially_reduced_tmp, ReductionOpT(), identity_val,
                        IndexOpT(), idx_identity_val, in_out_iter_indexer,
                        reduction_indexer, reduction_nelems, iter_nelems,
                        preferred_reductions_per_wi));
            }
            else {
                using SlmT = sycl::local_accessor<argTy, 1>;
                SlmT local_memory = SlmT(localRange, cgh);
                using KernelName =
                    class first_custom_search_axis0_temps_contig_krn<
                        argTy, resTy, ReductionOpT, IndexOpT,
                        InputOutputIterIndexerT, ReductionIndexerT, SlmT, true,
                        false>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    CustomSearchReduction<argTy, resTy, ReductionOpT, IndexOpT,
                                          InputOutputIterIndexerT,
                                          ReductionIndexerT, SlmT, true, false>(
                        arg_tp, partially_reduced_vals_tmp, nullptr,
                        partially_reduced_tmp, ReductionOpT(), identity_val,
                        IndexOpT(), idx_identity_val, in_out_iter_indexer,
                        reduction_indexer, local_memory, reduction_nelems,
                        iter_nelems, preferred_reductions_per_wi));
            }
        });

        size_t remaining_reduction_nelems = reduction_groups;

        resTy *temp_arg = partially_reduced_tmp;
        resTy *temp2_arg = partially_reduced_tmp2;

        argTy *vals_temp_arg = partially_reduced_vals_tmp;
        argTy *vals_temp2_arg = partially_reduced_vals_tmp2;

        sycl::event dependent_ev = first_reduction_ev;

        while (remaining_reduction_nelems >
               preferred_reductions_per_wi * max_wg) {
            size_t reduction_groups_ = (remaining_reduction_nelems +
                                        preferred_reductions_per_wi * wg - 1) /
                                       (preferred_reductions_per_wi * wg);
            assert(reduction_groups_ > 1);

            // keep reducing
            sycl::event partial_reduction_ev = exec_q.submit([&](sycl::handler
                                                                     &cgh) {
                cgh.depends_on(dependent_ev);

                using InputIndexerT =
                    dpctl::tensor::offset_utils::Strided1DIndexer;
                using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
                using InputOutputIterIndexerT =
                    dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                        InputIndexerT, ResIndexerT>;
                using ReductionIndexerT =
                    dpctl::tensor::offset_utils::NoOpIndexer;

                InputIndexerT inp_indexer{
                    0, static_cast<py::ssize_t>(iter_nelems),
                    static_cast<py::ssize_t>(reduction_groups_)};
                ResIndexerT res_iter_indexer{};

                InputOutputIterIndexerT in_out_iter_indexer{inp_indexer,
                                                            res_iter_indexer};
                ReductionIndexerT reduction_indexer{};

                auto globalRange =
                    sycl::range<1>{iter_nelems * reduction_groups_ * wg};
                auto localRange = sycl::range<1>{wg};
                if constexpr (can_use_reduce_over_group<ReductionOpT,
                                                        resTy>::value) {
                    using KernelName =
                        class middle_search_axis0_temps_contig_krn<
                            argTy, resTy, ReductionOpT, IndexOpT,
                            InputOutputIterIndexerT, ReductionIndexerT, false,
                            false>;
                    cgh.parallel_for<KernelName>(
                        sycl::nd_range<1>(globalRange, localRange),
                        SearchReduction<argTy, resTy, ReductionOpT, IndexOpT,
                                        InputOutputIterIndexerT,
                                        ReductionIndexerT, false, false>(
                            vals_temp_arg, vals_temp2_arg, temp_arg, temp2_arg,
                            ReductionOpT(), identity_val, IndexOpT(),
                            idx_identity_val, in_out_iter_indexer,
                            reduction_indexer, remaining_reduction_nelems,
                            iter_nelems, preferred_reductions_per_wi));
                }
                else {
                    using SlmT = sycl::local_accessor<argTy, 1>;
                    SlmT local_memory = SlmT(localRange, cgh);
                    using KernelName =
                        class middle_custom_search_axis0_temps_contig_krn<
                            argTy, resTy, ReductionOpT, IndexOpT,
                            InputOutputIterIndexerT, ReductionIndexerT, SlmT,
                            false, false>;
                    cgh.parallel_for<KernelName>(
                        sycl::nd_range<1>(globalRange, localRange),
                        CustomSearchReduction<argTy, resTy, ReductionOpT,
                                              IndexOpT, InputOutputIterIndexerT,
                                              ReductionIndexerT, SlmT, false,
                                              false>(
                            vals_temp_arg, vals_temp2_arg, temp_arg, temp2_arg,
                            ReductionOpT(), identity_val, IndexOpT(),
                            idx_identity_val, in_out_iter_indexer,
                            reduction_indexer, local_memory,
                            remaining_reduction_nelems, iter_nelems,
                            preferred_reductions_per_wi));
                }
            });

            remaining_reduction_nelems = reduction_groups_;
            std::swap(temp_arg, temp2_arg);
            std::swap(vals_temp_arg, vals_temp2_arg);
            dependent_ev = partial_reduction_ev;
        }

        // final reduction to res
        sycl::event final_reduction_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(dependent_ev);

            using InputIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
            using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            using InputOutputIterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    InputIndexerT, ResIndexerT>;
            using ReductionIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;

            InputIndexerT inp_indexer{
                0, static_cast<py::ssize_t>(iter_nelems),
                static_cast<py::ssize_t>(remaining_reduction_nelems)};
            ResIndexerT res_iter_indexer{};

            InputOutputIterIndexerT in_out_iter_indexer{inp_indexer,
                                                        res_iter_indexer};
            ReductionIndexerT reduction_indexer{};

            wg = max_wg;
            reductions_per_wi =
                std::max<size_t>(1, (remaining_reduction_nelems + wg - 1) / wg);

            size_t reduction_groups =
                (remaining_reduction_nelems + reductions_per_wi * wg - 1) /
                (reductions_per_wi * wg);
            assert(reduction_groups == 1);

            auto globalRange =
                sycl::range<1>{iter_nelems * reduction_groups * wg};
            auto localRange = sycl::range<1>{wg};

            if constexpr (can_use_reduce_over_group<ReductionOpT, resTy>::value)
            {
                using KernelName = class final_search_axis0_temps_contig_krn<
                    argTy, resTy, ReductionOpT, IndexOpT,
                    InputOutputIterIndexerT, ReductionIndexerT, false, true>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    SearchReduction<argTy, resTy, ReductionOpT, IndexOpT,
                                    InputOutputIterIndexerT, ReductionIndexerT,
                                    false, true>(
                        vals_temp_arg, nullptr, temp_arg, res_tp,
                        ReductionOpT(), identity_val, IndexOpT(),
                        idx_identity_val, in_out_iter_indexer,
                        reduction_indexer, remaining_reduction_nelems,
                        iter_nelems, reductions_per_wi));
            }
            else {
                using SlmT = sycl::local_accessor<argTy, 1>;
                SlmT local_memory = SlmT(localRange, cgh);
                using KernelName =
                    class final_custom_search_axis0_temps_contig_krn<
                        argTy, resTy, ReductionOpT, IndexOpT,
                        InputOutputIterIndexerT, ReductionIndexerT, SlmT, false,
                        true>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    CustomSearchReduction<argTy, resTy, ReductionOpT, IndexOpT,
                                          InputOutputIterIndexerT,
                                          ReductionIndexerT, SlmT, false, true>(
                        vals_temp_arg, nullptr, temp_arg, res_tp,
                        ReductionOpT(), identity_val, IndexOpT(),
                        idx_identity_val, in_out_iter_indexer,
                        reduction_indexer, local_memory,
                        remaining_reduction_nelems, iter_nelems,
                        reductions_per_wi));
            }
        });

        sycl::event cleanup_host_task_event =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(final_reduction_ev);
                sycl::context ctx = exec_q.get_context();

                cgh.host_task(
                    [ctx, partially_reduced_tmp, partially_reduced_vals_tmp] {
                        sycl::free(partially_reduced_tmp, ctx);
                        sycl::free(partially_reduced_vals_tmp, ctx);
                    });
            });

        // FIXME: do not return host-task event
        //   Instead collect all host-tasks to a list

        return cleanup_host_task_event;
    }
}

template <typename argTy, typename outTy>
struct TypePairSupportDataForSearchReductionTemps
{

    static constexpr bool is_defined = std::disjunction< // disjunction is C++17
                                                         // feature, supported
                                                         // by DPC++ input bool
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::int64_t>,
        // input int8_t
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, std::int64_t>,

        // input uint8_t
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::int64_t>,

        // input int16_t
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, std::int64_t>,

        // input uint16_t
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::int64_t>,

        // input int32_t
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, std::int64_t>,
        // input uint32_t
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, std::int64_t>,

        // input int64_t
        td_ns::TypePairDefinedEntry<argTy, std::int64_t, outTy, std::int64_t>,

        // input uint32_t
        td_ns::TypePairDefinedEntry<argTy, std::uint64_t, outTy, std::int64_t>,

        // input half
        td_ns::TypePairDefinedEntry<argTy, sycl::half, outTy, std::int64_t>,

        // input float
        td_ns::TypePairDefinedEntry<argTy, float, outTy, std::int64_t>,

        // input double
        td_ns::TypePairDefinedEntry<argTy, double, outTy, std::int64_t>,

        // input std::complex
        td_ns::TypePairDefinedEntry<argTy,
                                    std::complex<float>,
                                    outTy,
                                    std::int64_t>,

        td_ns::TypePairDefinedEntry<argTy,
                                    std::complex<double>,
                                    outTy,
                                    std::int64_t>,

        // fall-through
        td_ns::NotDefinedEntry>::is_defined;
};

template <typename fnT, typename srcTy, typename dstTy>
struct ArgmaxOverAxisTempsStridedFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForSearchReductionTemps<
                          srcTy, dstTy>::is_defined)
        {
            if constexpr (std::is_integral_v<srcTy> &&
                          !std::is_same_v<srcTy, bool>) {
                // op for values
                using ReductionOpT = sycl::maximum<srcTy>;
                // op for indices
                using IndexOpT = sycl::minimum<dstTy>;
                return dpctl::tensor::kernels::
                    search_over_group_temps_strided_impl<
                        srcTy, dstTy, ReductionOpT, IndexOpT>;
            }
            else {
                // op for values
                using ReductionOpT = su_ns::Maximum<srcTy>;
                // op for indices
                using IndexOpT = sycl::minimum<dstTy>;
                return dpctl::tensor::kernels::
                    search_over_group_temps_strided_impl<
                        srcTy, dstTy, ReductionOpT, IndexOpT>;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct ArgmaxOverAxis1TempsContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForSearchReductionTemps<
                          srcTy, dstTy>::is_defined)
        {
            if constexpr (std::is_integral_v<srcTy> &&
                          !std::is_same_v<srcTy, bool>) {
                // op for values
                using ReductionOpT = sycl::maximum<srcTy>;
                // op for indices
                using IndexOpT = sycl::minimum<dstTy>;
                return dpctl::tensor::kernels::
                    search_axis1_over_group_temps_contig_impl<
                        srcTy, dstTy, ReductionOpT, IndexOpT>;
            }
            else {
                // op for values
                using ReductionOpT = su_ns::Maximum<srcTy>;
                // op for indices
                using IndexOpT = sycl::minimum<dstTy>;
                return dpctl::tensor::kernels::
                    search_axis1_over_group_temps_contig_impl<
                        srcTy, dstTy, ReductionOpT, IndexOpT>;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct ArgmaxOverAxis0TempsContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForSearchReductionTemps<
                          srcTy, dstTy>::is_defined)
        {
            if constexpr (std::is_integral_v<srcTy> &&
                          !std::is_same_v<srcTy, bool>) {
                // op for values
                using ReductionOpT = sycl::maximum<srcTy>;
                // op for indices
                using IndexOpT = sycl::minimum<dstTy>;
                return dpctl::tensor::kernels::
                    search_axis0_over_group_temps_contig_impl<
                        srcTy, dstTy, ReductionOpT, IndexOpT>;
            }
            else {
                // op for values
                using ReductionOpT = su_ns::Maximum<srcTy>;
                // op for indices
                using IndexOpT = sycl::minimum<dstTy>;
                return dpctl::tensor::kernels::
                    search_axis0_over_group_temps_contig_impl<
                        srcTy, dstTy, ReductionOpT, IndexOpT>;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct ArgminOverAxisTempsStridedFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForSearchReductionTemps<
                          srcTy, dstTy>::is_defined)
        {
            if constexpr (std::is_integral_v<srcTy> &&
                          !std::is_same_v<srcTy, bool>) {
                // op for values
                using ReductionOpT = sycl::minimum<srcTy>;
                // op for indices
                using IndexOpT = sycl::minimum<dstTy>;
                return dpctl::tensor::kernels::
                    search_over_group_temps_strided_impl<
                        srcTy, dstTy, ReductionOpT, IndexOpT>;
            }
            else {
                // op for values
                using ReductionOpT = su_ns::Minimum<srcTy>;
                // op for indices
                using IndexOpT = sycl::minimum<dstTy>;
                return dpctl::tensor::kernels::
                    search_over_group_temps_strided_impl<
                        srcTy, dstTy, ReductionOpT, IndexOpT>;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct ArgminOverAxis1TempsContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForSearchReductionTemps<
                          srcTy, dstTy>::is_defined)
        {
            if constexpr (std::is_integral_v<srcTy> &&
                          !std::is_same_v<srcTy, bool>) {
                // op for values
                using ReductionOpT = sycl::minimum<srcTy>;
                // op for indices
                using IndexOpT = sycl::minimum<dstTy>;
                return dpctl::tensor::kernels::
                    search_axis1_over_group_temps_contig_impl<
                        srcTy, dstTy, ReductionOpT, IndexOpT>;
            }
            else {
                // op for values
                using ReductionOpT = su_ns::Minimum<srcTy>;
                // op for indices
                using IndexOpT = sycl::minimum<dstTy>;
                return dpctl::tensor::kernels::
                    search_axis1_over_group_temps_contig_impl<
                        srcTy, dstTy, ReductionOpT, IndexOpT>;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct ArgminOverAxis0TempsContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForSearchReductionTemps<
                          srcTy, dstTy>::is_defined)
        {
            if constexpr (std::is_integral_v<srcTy> &&
                          !std::is_same_v<srcTy, bool>) {
                // op for values
                using ReductionOpT = sycl::minimum<srcTy>;
                // op for indices
                using IndexOpT = sycl::minimum<dstTy>;
                return dpctl::tensor::kernels::
                    search_axis0_over_group_temps_contig_impl<
                        srcTy, dstTy, ReductionOpT, IndexOpT>;
            }
            else {
                // op for values
                using ReductionOpT = su_ns::Minimum<srcTy>;
                // op for indices
                using IndexOpT = sycl::minimum<dstTy>;
                return dpctl::tensor::kernels::
                    search_axis0_over_group_temps_contig_impl<
                        srcTy, dstTy, ReductionOpT, IndexOpT>;
            }
        }
        else {
            return nullptr;
        }
    }
};

} // namespace kernels
} // namespace tensor
} // namespace dpctl
