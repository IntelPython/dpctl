#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <type_traits>
#include <utility>
#include <vector>

#include "kernels/dpctl_tensor_types.hpp"
#include "kernels/reductions.hpp"
#include "utils/offset_utils.hpp"
#include "utils/sycl_utils.hpp"
#include "utils/type_utils.hpp"

namespace dpctl
{
namespace tensor
{
namespace kernels
{

template <typename lhsT,
          typename rhsT,
          typename outT,
          typename BatchIndexerT,
          typename RedIndexerT>
struct SequentialDotProduct
{
private:
    const lhsT *lhs_ = nullptr;
    const rhsT *rhs_ = nullptr;
    outT *out_ = nullptr;
    BatchIndexerT batch_indexer_;
    RedIndexerT reduced_dims_indexer_;
    size_t reduction_max_gid_ = 0;

public:
    SequentialDotProduct(const lhsT *lhs,
                         const rhsT *rhs,
                         outT *out,
                         BatchIndexerT batch_indexer,
                         RedIndexerT reduced_dims_indexer,
                         size_t reduction_size)
        : lhs_(lhs), rhs_(rhs), out_(out), batch_indexer_(batch_indexer),
          reduced_dims_indexer_(reduced_dims_indexer),
          reduction_max_gid_(reduction_size)
    {
    }

    void operator()(sycl::id<1> id) const
    {

        auto const &batch_offsets = batch_indexer_(id[0]);
        const ssize_t &lhs_batch_offset = batch_offsets.get_first_offset();
        const ssize_t &rhs_batch_offset = batch_offsets.get_second_offset();
        const ssize_t &out_batch_offset = batch_offsets.get_third_offset();

        outT red_val(0);
        for (size_t m = 0; m < reduction_max_gid_; ++m) {
            auto reduction_offsets = reduced_dims_indexer_(m);
            auto lhs_reduction_offset = reduction_offsets.get_first_offset();
            auto rhs_reduction_offset = reduction_offsets.get_second_offset();

            using dpctl::tensor::type_utils::convert_impl;
            red_val += convert_impl<outT, lhsT>(
                           lhs_[lhs_batch_offset + lhs_reduction_offset]) *
                       convert_impl<outT, rhsT>(
                           rhs_[rhs_batch_offset + rhs_reduction_offset]);
        }

        out_[out_batch_offset] = red_val;
    }
};

template <typename lhsT,
          typename rhsT,
          typename outT,
          typename BatchIndexerT,
          typename RedIndexerT>
struct DotProductFunctor
{
private:
    const lhsT *lhs_ = nullptr;
    const rhsT *rhs_ = nullptr;
    outT *out_ = nullptr;
    BatchIndexerT batch_indexer_;
    RedIndexerT reduced_dims_indexer_;
    size_t reduction_max_gid_ = 0;
    size_t batches_ = 1;
    size_t reductions_per_wi = 16;

public:
    DotProductFunctor(const lhsT *lhs,
                      const rhsT *rhs,
                      outT *res,
                      BatchIndexerT batch_indexer,
                      RedIndexerT arg_reduced_dims_indexer,
                      size_t reduction_size,
                      size_t iteration_size,
                      size_t reduction_size_per_wi)
        : lhs_(lhs), rhs_(rhs), out_(res), batch_indexer_(batch_indexer),
          reduced_dims_indexer_(arg_reduced_dims_indexer),
          reduction_max_gid_(reduction_size), batches_(iteration_size),
          reductions_per_wi(reduction_size_per_wi)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        const size_t batch_id = it.get_group(0) % batches_;
        const size_t reduction_batch_id = it.get_group(0) / batches_;

        const size_t reduction_lid = it.get_local_id(0);
        const size_t wg = it.get_local_range(0); //   0 <= reduction_lid < wg

        // work-items operate over input with indices
        //   inp_data_id = reduction_batch_id * wg * reductions_per_wi + m * wg
        //   + reduction_lid
        // for 0 <= m < reductions_per_wi
        // for each input

        auto batch_offsets_ = batch_indexer_(batch_id);
        const auto &lhs_batch_offset = batch_offsets_.get_first_offset();
        const auto &rhs_batch_offset = batch_offsets_.get_second_offset();
        const auto &out_batch_offset = batch_offsets_.get_third_offset();

        outT local_red_val(0);
        size_t arg_reduce_gid0 =
            reduction_lid + reduction_batch_id * wg * reductions_per_wi;
        size_t arg_reduce_gid_max = std::min(
            reduction_max_gid_, arg_reduce_gid0 + reductions_per_wi * wg);

        for (size_t arg_reduce_gid = arg_reduce_gid0;
             arg_reduce_gid < arg_reduce_gid_max; arg_reduce_gid += wg)
        {
            auto reduction_offsets_ = reduced_dims_indexer_(arg_reduce_gid);
            const auto &lhs_reduction_offset =
                reduction_offsets_.get_first_offset();
            const auto &rhs_reduction_offset =
                reduction_offsets_.get_second_offset();

            using dpctl::tensor::type_utils::convert_impl;
            outT val = convert_impl<outT, lhsT>(
                           lhs_[lhs_batch_offset + lhs_reduction_offset]) *
                       convert_impl<outT, rhsT>(
                           rhs_[rhs_batch_offset + rhs_reduction_offset]);

            local_red_val += val;
        }

        auto work_group = it.get_group();
        outT red_val_over_wg = sycl::reduce_over_group(
            work_group, local_red_val, outT(0), sycl::plus<outT>());

        if (work_group.leader()) {
            sycl::atomic_ref<outT, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                res_ref(out_[out_batch_offset]);
            res_ref += red_val_over_wg;
        }
    }
};

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class dot_product_seq_krn;

template <typename T1, typename T2, typename T3> class dot_product_init_krn;

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class dot_product_krn;

typedef sycl::event (*dot_product_impl_fn_ptr_t)(
    sycl::queue &,
    size_t,
    size_t,
    const char *,
    const char *,
    char *,
    int,
    const ssize_t *,
    ssize_t,
    ssize_t,
    ssize_t,
    int,
    const ssize_t *,
    ssize_t,
    ssize_t,
    const std::vector<sycl::event> &);

template <typename lhsTy, typename rhsTy, typename resTy>
sycl::event dot_product_impl(sycl::queue &exec_q,
                             size_t batches,
                             size_t reduction_nelems,
                             const char *lhs_cp,
                             const char *rhs_cp,
                             char *res_cp,
                             int batch_nd,
                             const ssize_t *batch_shape_and_strides,
                             ssize_t batch_lhs_offset,
                             ssize_t batch_rhs_offset,
                             ssize_t batch_res_offset,
                             int red_nd,
                             const ssize_t *reduction_shape_stride,
                             ssize_t reduction_lhs_offset,
                             ssize_t reduction_rhs_offset,
                             const std::vector<sycl::event> &depends = {})
{
    const lhsTy *lhs_tp = reinterpret_cast<const lhsTy *>(lhs_cp);
    const rhsTy *rhs_tp = reinterpret_cast<const rhsTy *>(rhs_cp);
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp);

    const sycl::device &d = exec_q.get_device();
    const auto &sg_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

    if (reduction_nelems < wg) {
        sycl::event dot_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            using InputOutputBatchIndexerT =
                dpctl::tensor::offset_utils::ThreeOffsets_StridedIndexer;
            using ReductionIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;

            InputOutputBatchIndexerT in_out_batch_indexer{
                batch_nd, batch_lhs_offset, batch_rhs_offset, batch_res_offset,
                batch_shape_and_strides};
            ReductionIndexerT reduction_indexer{red_nd, reduction_lhs_offset,
                                                reduction_rhs_offset,
                                                reduction_shape_stride};

            cgh.parallel_for<class dot_product_seq_krn<lhsTy, rhsTy, resTy,
                                                       InputOutputBatchIndexerT,
                                                       ReductionIndexerT>>(
                sycl::range<1>(batches),
                SequentialDotProduct<lhsTy, rhsTy, resTy,
                                     InputOutputBatchIndexerT,
                                     ReductionIndexerT>(
                    lhs_tp, rhs_tp, res_tp, in_out_batch_indexer,
                    reduction_indexer, reduction_nelems));
        });

        return dot_ev;
    }
    else {
        sycl::event res_init_ev = exec_q.submit([&](sycl::handler &cgh) {
            using IndexerT =
                dpctl::tensor::offset_utils::UnpackedStridedIndexer;

            const ssize_t *const &res_shape = batch_shape_and_strides;
            const ssize_t *const &res_strides =
                batch_shape_and_strides + 3 * batch_nd;
            IndexerT res_indexer(batch_nd, batch_res_offset, res_shape,
                                 res_strides);
            using InitKernelName =
                class dot_product_init_krn<lhsTy, rhsTy, resTy>;
            cgh.depends_on(depends);

            cgh.parallel_for<InitKernelName>(
                sycl::range<1>(batches), [=](sycl::id<1> id) {
                    auto res_offset = res_indexer(id[0]);
                    res_tp[res_offset] = 0;
                });
        });

        sycl::event dot_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(res_init_ev);

            using BatchIndexerT =
                dpctl::tensor::offset_utils::ThreeOffsets_StridedIndexer;
            using ReductionIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;

            BatchIndexerT batch_indexer{batch_nd, batch_lhs_offset,
                                        batch_rhs_offset, batch_res_offset,
                                        batch_shape_and_strides};
            ReductionIndexerT reduction_indexer{red_nd, reduction_lhs_offset,
                                                reduction_rhs_offset,
                                                reduction_shape_stride};

            constexpr size_t preferred_reductions_per_wi =
                4; // determined experimentally
            size_t reductions_per_wi =
                (reduction_nelems < preferred_reductions_per_wi * wg)
                    ? std::max<size_t>(1, (reduction_nelems + wg - 1) / wg)
                    : preferred_reductions_per_wi;

            size_t reduction_groups =
                (reduction_nelems + reductions_per_wi * wg - 1) /
                (reductions_per_wi * wg);

            auto globalRange = sycl::range<1>{batches * reduction_groups * wg};
            auto localRange = sycl::range<1>{wg};

            using KernelName =
                class dot_product_krn<lhsTy, rhsTy, resTy, BatchIndexerT,
                                      ReductionIndexerT>;

            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(globalRange, localRange),
                DotProductFunctor<lhsTy, rhsTy, resTy, BatchIndexerT,
                                  ReductionIndexerT>(
                    lhs_tp, rhs_tp, res_tp, batch_indexer, reduction_indexer,
                    reduction_nelems, batches, reductions_per_wi));
        });
        return dot_ev;
    }
}

typedef sycl::event (*dot_product_contig_impl_fn_ptr_t)(
    sycl::queue &,
    size_t,
    size_t,
    const char *,
    const char *,
    char *,
    ssize_t,
    ssize_t,
    ssize_t,
    ssize_t,
    ssize_t,
    const std::vector<sycl::event> &);

template <typename lhsTy, typename rhsTy, typename resTy>
sycl::event
dot_product_contig_impl(sycl::queue &exec_q,
                        size_t batches,
                        size_t reduction_nelems,
                        const char *lhs_cp,
                        const char *rhs_cp,
                        char *res_cp,
                        ssize_t batch_lhs_offset,
                        ssize_t batch_rhs_offset,
                        ssize_t batch_res_offset,
                        ssize_t reduction_lhs_offset,
                        ssize_t reduction_rhs_offset,
                        const std::vector<sycl::event> &depends = {})
{
    const lhsTy *lhs_tp = reinterpret_cast<const lhsTy *>(lhs_cp) +
                          batch_lhs_offset + reduction_lhs_offset;
    const rhsTy *rhs_tp = reinterpret_cast<const rhsTy *>(rhs_cp) +
                          batch_rhs_offset + reduction_rhs_offset;
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp) + batch_res_offset;

    const sycl::device &d = exec_q.get_device();
    const auto &sg_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

    if (reduction_nelems < wg) {
        sycl::event dot_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            using InputBatchIndexerT =
                dpctl::tensor::offset_utils::Strided1DIndexer;
            using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            using InputOutputBatchIndexerT =
                dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer<
                    InputBatchIndexerT, InputBatchIndexerT, NoOpIndexerT>;
            using ReductionIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    NoOpIndexerT, NoOpIndexerT>;

            InputBatchIndexerT inp_batch_indexer{
                0, static_cast<ssize_t>(batches),
                static_cast<ssize_t>(reduction_nelems)};
            InputOutputBatchIndexerT inp_out_batch_indexer{
                inp_batch_indexer, inp_batch_indexer, NoOpIndexerT{}};
            ReductionIndexerT reduction_indexer{NoOpIndexerT{}, NoOpIndexerT{}};

            cgh.parallel_for<class dot_product_seq_krn<lhsTy, rhsTy, resTy,
                                                       InputOutputBatchIndexerT,
                                                       ReductionIndexerT>>(
                sycl::range<1>(batches),
                SequentialDotProduct<lhsTy, rhsTy, resTy,
                                     InputOutputBatchIndexerT,
                                     ReductionIndexerT>(
                    lhs_tp, rhs_tp, res_tp, inp_out_batch_indexer,
                    reduction_indexer, reduction_nelems));
        });

        return dot_ev;
    }
    else {
        sycl::event res_init_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            cgh.fill<resTy>(res_tp, resTy(0), batches);
        });

        sycl::event dot_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(res_init_ev);

            using InputBatchIndexerT =
                dpctl::tensor::offset_utils::Strided1DIndexer;
            using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            using InputOutputBatchIndexerT =
                dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer<
                    InputBatchIndexerT, InputBatchIndexerT, NoOpIndexerT>;
            using ReductionIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    NoOpIndexerT, NoOpIndexerT>;

            InputBatchIndexerT inp_batch_indexer{
                0, static_cast<ssize_t>(batches),
                static_cast<ssize_t>(reduction_nelems)};
            InputOutputBatchIndexerT inp_out_batch_indexer{
                inp_batch_indexer, inp_batch_indexer, NoOpIndexerT{}};
            ReductionIndexerT reduction_indexer{NoOpIndexerT{}, NoOpIndexerT{}};

            constexpr size_t preferred_reductions_per_wi =
                4; // determined experimentally
            size_t reductions_per_wi =
                (reduction_nelems < preferred_reductions_per_wi * wg)
                    ? std::max<size_t>(1, (reduction_nelems + wg - 1) / wg)
                    : preferred_reductions_per_wi;

            size_t reduction_groups =
                (reduction_nelems + reductions_per_wi * wg - 1) /
                (reductions_per_wi * wg);

            auto globalRange = sycl::range<1>{batches * reduction_groups * wg};
            auto localRange = sycl::range<1>{wg};

            using KernelName = class dot_product_krn<lhsTy, rhsTy, resTy,
                                                     InputOutputBatchIndexerT,
                                                     ReductionIndexerT>;

            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(globalRange, localRange),
                DotProductFunctor<lhsTy, rhsTy, resTy, InputOutputBatchIndexerT,
                                  ReductionIndexerT>(
                    lhs_tp, rhs_tp, res_tp, inp_out_batch_indexer,
                    reduction_indexer, reduction_nelems, batches,
                    reductions_per_wi));
        });
        return dot_ev;
    }
}

template <typename lhsT,
          typename rhsT,
          typename outT,
          typename BatchIndexerT,
          typename RedIndexerT>
struct DotProductNoAtomicFunctor
{
private:
    const lhsT *lhs_ = nullptr;
    const rhsT *rhs_ = nullptr;
    outT *out_ = nullptr;
    BatchIndexerT batch_indexer_;
    RedIndexerT reduced_dims_indexer_;
    size_t reduction_max_gid_ = 0;
    size_t batches_ = 1;
    size_t reductions_per_wi = 16;

public:
    DotProductNoAtomicFunctor(const lhsT *lhs,
                              const rhsT *rhs,
                              outT *res,
                              BatchIndexerT batch_indexer,
                              RedIndexerT arg_reduced_dims_indexer,
                              size_t reduction_size,
                              size_t iteration_size,
                              size_t reduction_size_per_wi)
        : lhs_(lhs), rhs_(rhs), out_(res), batch_indexer_(batch_indexer),
          reduced_dims_indexer_(arg_reduced_dims_indexer),
          reduction_max_gid_(reduction_size), batches_(iteration_size),
          reductions_per_wi(reduction_size_per_wi)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        const size_t reduction_lid = it.get_local_id(0);
        const size_t wg = it.get_local_range(0); //   0 <= reduction_lid < wg

        const size_t batch_id = it.get_group(0) % batches_;
        const size_t reduction_batch_id = it.get_group(0) / batches_;
        const size_t n_reduction_groups = it.get_group_range(0) / batches_;

        // work-items operate over input with indices
        //   inp_data_id = reduction_batch_id * wg * reductions_per_wi + m * wg
        //   + reduction_lid
        // for 0 <= m < reductions_per_wi
        // for each input

        auto batch_offsets_ = batch_indexer_(batch_id);
        const auto &lhs_batch_offset = batch_offsets_.get_first_offset();
        const auto &rhs_batch_offset = batch_offsets_.get_second_offset();
        const auto &out_batch_offset = batch_offsets_.get_third_offset();

        outT local_red_val(0);
        size_t arg_reduce_gid0 =
            reduction_lid + reduction_batch_id * wg * reductions_per_wi;
        size_t arg_reduce_gid_max = std::min(
            reduction_max_gid_, arg_reduce_gid0 + reductions_per_wi * wg);

        for (size_t arg_reduce_gid = arg_reduce_gid0;
             arg_reduce_gid < arg_reduce_gid_max; arg_reduce_gid += wg)
        {
            auto reduction_offsets_ = reduced_dims_indexer_(arg_reduce_gid);
            const auto &lhs_reduction_offset =
                reduction_offsets_.get_first_offset();
            const auto &rhs_reduction_offset =
                reduction_offsets_.get_second_offset();

            using dpctl::tensor::type_utils::convert_impl;
            outT val = convert_impl<outT, lhsT>(
                           lhs_[lhs_batch_offset + lhs_reduction_offset]) *
                       convert_impl<outT, rhsT>(
                           rhs_[rhs_batch_offset + rhs_reduction_offset]);

            local_red_val += val;
        }

        auto work_group = it.get_group();
        outT red_val_over_wg = sycl::reduce_over_group(
            work_group, local_red_val, outT(0), sycl::plus<outT>());

        if (work_group.leader()) {
            // each group writes to a different memory location
            out_[out_batch_offset * n_reduction_groups + reduction_batch_id] =
                red_val_over_wg;
        }
    }
};

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class dot_product_tree_krn;

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class dot_product_reduction_over_group_temps_krn;

template <typename lhsTy, typename rhsTy, typename resTy>
sycl::event dot_product_tree_impl(sycl::queue &exec_q,
                                  size_t batches,
                                  size_t reduction_nelems,
                                  const char *lhs_cp,
                                  const char *rhs_cp,
                                  char *res_cp,
                                  int batch_nd,
                                  const ssize_t *batch_shape_and_strides,
                                  ssize_t batch_lhs_offset,
                                  ssize_t batch_rhs_offset,
                                  ssize_t batch_res_offset,
                                  int red_nd,
                                  const ssize_t *reduction_shape_stride,
                                  ssize_t reduction_lhs_offset,
                                  ssize_t reduction_rhs_offset,
                                  const std::vector<sycl::event> &depends = {})
{
    const lhsTy *lhs_tp = reinterpret_cast<const lhsTy *>(lhs_cp);
    const rhsTy *rhs_tp = reinterpret_cast<const rhsTy *>(rhs_cp);
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp);

    const sycl::device &d = exec_q.get_device();
    const auto &sg_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

    if (reduction_nelems < wg) {
        sycl::event dot_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            using InputOutputBatchIndexerT =
                dpctl::tensor::offset_utils::ThreeOffsets_StridedIndexer;
            using ReductionIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;

            InputOutputBatchIndexerT in_out_batch_indexer{
                batch_nd, batch_lhs_offset, batch_rhs_offset, batch_res_offset,
                batch_shape_and_strides};
            ReductionIndexerT reduction_indexer{red_nd, reduction_lhs_offset,
                                                reduction_rhs_offset,
                                                reduction_shape_stride};

            cgh.parallel_for<class dot_product_seq_krn<lhsTy, rhsTy, resTy,
                                                       InputOutputBatchIndexerT,
                                                       ReductionIndexerT>>(
                sycl::range<1>(batches),
                SequentialDotProduct<lhsTy, rhsTy, resTy,
                                     InputOutputBatchIndexerT,
                                     ReductionIndexerT>(
                    lhs_tp, rhs_tp, res_tp, in_out_batch_indexer,
                    reduction_indexer, reduction_nelems));
        });

        return dot_ev;
    }

    constexpr size_t preferred_reductions_per_wi = 8;
    // max_max_wg prevents running out of resources on CPU
    constexpr size_t max_max_wg = 2048;
    size_t max_wg = std::min(
        max_max_wg, d.get_info<sycl::info::device::max_work_group_size>() / 2);

    size_t reductions_per_wi(preferred_reductions_per_wi);
    if (reduction_nelems <= preferred_reductions_per_wi * max_wg) {
        sycl::event dot_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            using BatchIndexerT =
                dpctl::tensor::offset_utils::ThreeOffsets_StridedIndexer;
            using ReductionIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;

            BatchIndexerT batch_indexer{batch_nd, batch_lhs_offset,
                                        batch_rhs_offset, batch_res_offset,
                                        batch_shape_and_strides};
            ReductionIndexerT reduction_indexer{red_nd, reduction_lhs_offset,
                                                reduction_rhs_offset,
                                                reduction_shape_stride};

            if (batches == 1) {
                // increase GPU occupancy
                wg = max_wg;
            }
            reductions_per_wi =
                std::max<size_t>(1, (reduction_nelems + wg - 1) / wg);

            size_t reduction_groups =
                (reduction_nelems + reductions_per_wi * wg - 1) /
                (reductions_per_wi * wg);
            assert(reduction_groups == 1);

            auto globalRange = sycl::range<1>{batches * reduction_groups * wg};
            auto localRange = sycl::range<1>{wg};
            using KernelName =
                dot_product_tree_krn<lhsTy, rhsTy, resTy, BatchIndexerT,
                                     ReductionIndexerT>;
            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(globalRange, localRange),
                DotProductNoAtomicFunctor<lhsTy, rhsTy, resTy, BatchIndexerT,
                                          ReductionIndexerT>(
                    lhs_tp, rhs_tp, res_tp, batch_indexer, reduction_indexer,
                    reduction_nelems, batches, reductions_per_wi));
        });

        return dot_ev;
    }
    else {
        using ReductionOpT = sycl::plus<resTy>;
        constexpr resTy identity_val =
            sycl::known_identity<ReductionOpT, resTy>::value;

        // more than one work-groups is needed, requires a temporary
        size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);
        assert(reduction_groups > 1);

        size_t second_iter_reduction_groups_ =
            (reduction_groups + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);

        resTy *partially_reduced_tmp = sycl::malloc_device<resTy>(
            batches * (reduction_groups + second_iter_reduction_groups_),
            exec_q);
        resTy *partially_reduced_tmp2 = nullptr;

        if (partially_reduced_tmp == nullptr) {
            throw std::runtime_error("Unable to allocate device_memory");
        }
        else {
            partially_reduced_tmp2 =
                partially_reduced_tmp + reduction_groups * batches;
        }

        const sycl::event &first_reduction_ev =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(depends);

                using LhsIndexerT = dpctl::tensor::offset_utils::StridedIndexer;
                using RhsIndexerT =
                    dpctl::tensor::offset_utils::UnpackedStridedIndexer;
                using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
                using InputOutputBatchIndexerT =
                    dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer<
                        LhsIndexerT, RhsIndexerT, ResIndexerT>;
                using ReductionIndexerT =
                    dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;

                LhsIndexerT lhs_indexer(batch_nd, batch_lhs_offset,
                                        batch_shape_and_strides);
                RhsIndexerT rhs_indexer(batch_nd, batch_rhs_offset,
                                        batch_shape_and_strides,
                                        batch_shape_and_strides + 2 * batch_nd);
                ResIndexerT noop_tmp_indexer{};

                InputOutputBatchIndexerT in_out_iter_indexer{
                    lhs_indexer, rhs_indexer, noop_tmp_indexer};
                ReductionIndexerT reduction_indexer{
                    red_nd, reduction_lhs_offset, reduction_rhs_offset,
                    reduction_shape_stride};

                auto globalRange =
                    sycl::range<1>{batches * reduction_groups * wg};
                auto localRange = sycl::range<1>{wg};

                using KernelName =
                    class dot_product_tree_krn<lhsTy, rhsTy, resTy,
                                               InputOutputBatchIndexerT,
                                               ReductionIndexerT>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    DotProductNoAtomicFunctor<lhsTy, rhsTy, resTy,
                                              InputOutputBatchIndexerT,
                                              ReductionIndexerT>(
                        lhs_tp, rhs_tp, partially_reduced_tmp,
                        in_out_iter_indexer, reduction_indexer,
                        reduction_nelems, batches,
                        preferred_reductions_per_wi));
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

            sycl::event partial_reduction_ev =
                exec_q.submit([&](sycl::handler &cgh) {
                    cgh.depends_on(dependent_ev);

                    using InputIndexerT =
                        dpctl::tensor::offset_utils::Strided1DIndexer;
                    using ResIndexerT =
                        dpctl::tensor::offset_utils::NoOpIndexer;
                    using InputOutputIterIndexerT =
                        dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                            InputIndexerT, ResIndexerT>;
                    using ReductionIndexerT =
                        dpctl::tensor::offset_utils::NoOpIndexer;

                    InputIndexerT inp_indexer{
                        0, static_cast<ssize_t>(batches),
                        static_cast<ssize_t>(reduction_groups_)};
                    ResIndexerT res_iter_indexer{};

                    InputOutputIterIndexerT in_out_iter_indexer{
                        inp_indexer, res_iter_indexer};
                    ReductionIndexerT reduction_indexer{};

                    auto globalRange =
                        sycl::range<1>{batches * reduction_groups_ * wg};
                    auto localRange = sycl::range<1>{wg};
                    using KernelName =
                        class dot_product_reduction_over_group_temps_krn<
                            resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                            ReductionIndexerT>;
                    cgh.parallel_for<KernelName>(
                        sycl::nd_range<1>(globalRange, localRange),
                        ReductionOverGroupNoAtomicFunctor<
                            resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                            ReductionIndexerT>(
                            temp_arg, temp2_arg, ReductionOpT(), identity_val,
                            in_out_iter_indexer, reduction_indexer,
                            remaining_reduction_nelems, batches,
                            preferred_reductions_per_wi));
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
                0, static_cast<ssize_t>(batches),
                static_cast<ssize_t>(remaining_reduction_nelems)};
            ResIndexerT res_iter_indexer{batch_nd, batch_res_offset,
                                         /* shape */ batch_shape_and_strides,
                                         /* strides */ batch_shape_and_strides +
                                             2 * batch_nd};

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

            auto globalRange = sycl::range<1>{batches * reduction_groups * wg};
            auto localRange = sycl::range<1>{wg};

            using KernelName = class dot_product_reduction_over_group_temps_krn<
                resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                ReductionIndexerT>;
            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(globalRange, localRange),
                ReductionOverGroupNoAtomicFunctor<resTy, resTy, ReductionOpT,
                                                  InputOutputIterIndexerT,
                                                  ReductionIndexerT>(
                    temp_arg, res_tp, ReductionOpT(), identity_val,
                    in_out_iter_indexer, reduction_indexer,
                    remaining_reduction_nelems, batches, reductions_per_wi));
        });
        sycl::event cleanup_host_task_event =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(final_reduction_ev);
                const sycl::context &ctx = exec_q.get_context();

                cgh.host_task([ctx, partially_reduced_tmp] {
                    sycl::free(partially_reduced_tmp, ctx);
                });
            });

        return cleanup_host_task_event;
    }
}

template <typename lhsTy, typename rhsTy, typename resTy>
sycl::event
dot_product_contig_tree_impl(sycl::queue &exec_q,
                             size_t batches,
                             size_t reduction_nelems,
                             const char *lhs_cp,
                             const char *rhs_cp,
                             char *res_cp,
                             ssize_t batch_lhs_offset,
                             ssize_t batch_rhs_offset,
                             ssize_t batch_res_offset,
                             ssize_t reduction_lhs_offset,
                             ssize_t reduction_rhs_offset,
                             const std::vector<sycl::event> &depends = {})
{
    const lhsTy *lhs_tp = reinterpret_cast<const lhsTy *>(lhs_cp) +
                          batch_lhs_offset + reduction_lhs_offset;
    const rhsTy *rhs_tp = reinterpret_cast<const rhsTy *>(rhs_cp) +
                          batch_rhs_offset + reduction_rhs_offset;
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp) + batch_res_offset;

    const sycl::device &d = exec_q.get_device();
    const auto &sg_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

    if (reduction_nelems < wg) {
        sycl::event dot_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            using InputBatchIndexerT =
                dpctl::tensor::offset_utils::Strided1DIndexer;
            using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            using InputOutputBatchIndexerT =
                dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer<
                    InputBatchIndexerT, InputBatchIndexerT, NoOpIndexerT>;
            using ReductionIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    NoOpIndexerT, NoOpIndexerT>;

            InputBatchIndexerT inp_batch_indexer{
                0, static_cast<ssize_t>(batches),
                static_cast<ssize_t>(reduction_nelems)};
            InputOutputBatchIndexerT inp_out_batch_indexer{
                inp_batch_indexer, inp_batch_indexer, NoOpIndexerT{}};
            ReductionIndexerT reduction_indexer{NoOpIndexerT{}, NoOpIndexerT{}};

            cgh.parallel_for<class dot_product_seq_krn<lhsTy, rhsTy, resTy,
                                                       InputOutputBatchIndexerT,
                                                       ReductionIndexerT>>(
                sycl::range<1>(batches),
                SequentialDotProduct<lhsTy, rhsTy, resTy,
                                     InputOutputBatchIndexerT,
                                     ReductionIndexerT>(
                    lhs_tp, rhs_tp, res_tp, inp_out_batch_indexer,
                    reduction_indexer, reduction_nelems));
        });

        return dot_ev;
    }

    constexpr size_t preferred_reductions_per_wi = 8;
    // max_max_wg prevents running out of resources on CPU
    constexpr size_t max_max_wg = 2048;
    size_t max_wg = std::min(
        max_max_wg, d.get_info<sycl::info::device::max_work_group_size>() / 2);

    size_t reductions_per_wi(preferred_reductions_per_wi);
    if (reduction_nelems <= preferred_reductions_per_wi * max_wg) {
        sycl::event dot_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            using InputBatchIndexerT =
                dpctl::tensor::offset_utils::Strided1DIndexer;
            using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            using InputOutputBatchIndexerT =
                dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer<
                    InputBatchIndexerT, InputBatchIndexerT, NoOpIndexerT>;
            using ReductionIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    NoOpIndexerT, NoOpIndexerT>;

            InputBatchIndexerT inp_batch_indexer{
                0, static_cast<ssize_t>(batches),
                static_cast<ssize_t>(reduction_nelems)};
            InputOutputBatchIndexerT inp_out_batch_indexer{
                inp_batch_indexer, inp_batch_indexer, NoOpIndexerT{}};
            ReductionIndexerT reduction_indexer{NoOpIndexerT{}, NoOpIndexerT{}};

            if (batches == 1) {
                // increase GPU occupancy
                wg = max_wg;
            }
            reductions_per_wi =
                std::max<size_t>(1, (reduction_nelems + wg - 1) / wg);

            size_t reduction_groups =
                (reduction_nelems + reductions_per_wi * wg - 1) /
                (reductions_per_wi * wg);
            assert(reduction_groups == 1);

            auto globalRange = sycl::range<1>{batches * reduction_groups * wg};
            auto localRange = sycl::range<1>{wg};
            using KernelName = dot_product_tree_krn<lhsTy, rhsTy, resTy,
                                                    InputOutputBatchIndexerT,
                                                    ReductionIndexerT>;
            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(globalRange, localRange),
                DotProductNoAtomicFunctor<lhsTy, rhsTy, resTy,
                                          InputOutputBatchIndexerT,
                                          ReductionIndexerT>(
                    lhs_tp, rhs_tp, res_tp, inp_out_batch_indexer,
                    reduction_indexer, reduction_nelems, batches,
                    reductions_per_wi));
        });

        return dot_ev;
    }
    else {
        using ReductionOpT = sycl::plus<resTy>;
        constexpr resTy identity_val =
            sycl::known_identity<ReductionOpT, resTy>::value;

        // more than one work-groups is needed, requires a temporary
        size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);
        assert(reduction_groups > 1);

        size_t second_iter_reduction_groups_ =
            (reduction_groups + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);

        resTy *partially_reduced_tmp = sycl::malloc_device<resTy>(
            batches * (reduction_groups + second_iter_reduction_groups_),
            exec_q);
        resTy *partially_reduced_tmp2 = nullptr;

        if (partially_reduced_tmp == nullptr) {
            throw std::runtime_error("Unable to allocate device_memory");
        }
        else {
            partially_reduced_tmp2 =
                partially_reduced_tmp + reduction_groups * batches;
        }

        const sycl::event &first_reduction_ev = exec_q.submit([&](sycl::handler
                                                                      &cgh) {
            cgh.depends_on(depends);

            using InputBatchIndexerT =
                dpctl::tensor::offset_utils::Strided1DIndexer;
            using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            using InputOutputBatchIndexerT =
                dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer<
                    InputBatchIndexerT, InputBatchIndexerT, NoOpIndexerT>;
            using ReductionIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    NoOpIndexerT, NoOpIndexerT>;

            InputBatchIndexerT inp_batch_indexer{
                0, static_cast<ssize_t>(batches),
                static_cast<ssize_t>(reduction_nelems)};
            InputOutputBatchIndexerT inp_out_batch_indexer{
                inp_batch_indexer, inp_batch_indexer, NoOpIndexerT{}};
            ReductionIndexerT reduction_indexer{NoOpIndexerT{}, NoOpIndexerT{}};

            auto globalRange = sycl::range<1>{batches * reduction_groups * wg};
            auto localRange = sycl::range<1>{wg};

            using KernelName =
                class dot_product_tree_krn<lhsTy, rhsTy, resTy,
                                           InputOutputBatchIndexerT,
                                           ReductionIndexerT>;
            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(globalRange, localRange),
                DotProductNoAtomicFunctor<lhsTy, rhsTy, resTy,
                                          InputOutputBatchIndexerT,
                                          ReductionIndexerT>(
                    lhs_tp, rhs_tp, partially_reduced_tmp,
                    inp_out_batch_indexer, reduction_indexer, reduction_nelems,
                    batches, preferred_reductions_per_wi));
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

            sycl::event partial_reduction_ev =
                exec_q.submit([&](sycl::handler &cgh) {
                    cgh.depends_on(dependent_ev);

                    using InputIndexerT =
                        dpctl::tensor::offset_utils::Strided1DIndexer;
                    using ResIndexerT =
                        dpctl::tensor::offset_utils::NoOpIndexer;
                    using InputOutputIterIndexerT =
                        dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                            InputIndexerT, ResIndexerT>;
                    using ReductionIndexerT =
                        dpctl::tensor::offset_utils::NoOpIndexer;

                    InputIndexerT inp_indexer{
                        0, static_cast<ssize_t>(batches),
                        static_cast<ssize_t>(reduction_groups_)};
                    ResIndexerT res_iter_indexer{};

                    InputOutputIterIndexerT in_out_iter_indexer{
                        inp_indexer, res_iter_indexer};
                    ReductionIndexerT reduction_indexer{};

                    auto globalRange =
                        sycl::range<1>{batches * reduction_groups_ * wg};
                    auto localRange = sycl::range<1>{wg};
                    using KernelName =
                        class dot_product_reduction_over_group_temps_krn<
                            resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                            ReductionIndexerT>;
                    cgh.parallel_for<KernelName>(
                        sycl::nd_range<1>(globalRange, localRange),
                        ReductionOverGroupNoAtomicFunctor<
                            resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                            ReductionIndexerT>(
                            temp_arg, temp2_arg, ReductionOpT(), identity_val,
                            in_out_iter_indexer, reduction_indexer,
                            remaining_reduction_nelems, batches,
                            preferred_reductions_per_wi));
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
                0, static_cast<ssize_t>(batches),
                static_cast<ssize_t>(remaining_reduction_nelems)};
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

            auto globalRange = sycl::range<1>{batches * reduction_groups * wg};
            auto localRange = sycl::range<1>{wg};

            using KernelName = class dot_product_reduction_over_group_temps_krn<
                resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                ReductionIndexerT>;
            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(globalRange, localRange),
                ReductionOverGroupNoAtomicFunctor<resTy, resTy, ReductionOpT,
                                                  InputOutputIterIndexerT,
                                                  ReductionIndexerT>(
                    temp_arg, res_tp, ReductionOpT(), identity_val,
                    in_out_iter_indexer, reduction_indexer,
                    remaining_reduction_nelems, batches, reductions_per_wi));
        });
        sycl::event cleanup_host_task_event =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(final_reduction_ev);
                const sycl::context &ctx = exec_q.get_context();

                cgh.host_task([ctx, partially_reduced_tmp] {
                    sycl::free(partially_reduced_tmp, ctx);
                });
            });

        return cleanup_host_task_event;
    }
}

} // namespace kernels
} // namespace tensor
} // namespace dpctl
