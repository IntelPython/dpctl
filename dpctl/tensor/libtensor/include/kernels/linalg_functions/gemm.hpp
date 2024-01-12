#pragma once

#include <CL/sycl.hpp>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include "kernels/reductions.hpp"
#include "pybind11/pybind11.h"
#include "utils/offset_utils.hpp"
#include "utils/sycl_utils.hpp"
#include "utils/type_utils.hpp"

namespace dpctl
{
namespace tensor
{
namespace kernels
{

namespace gemm_detail
{

template <typename T, size_t m_groups>
void scale_gemm_k_parameters(const size_t &local_mem_size,
                             const size_t &reserved_slm_size,
                             const size_t delta_k,
                             size_t &n_wi,
                             size_t &delta_n)
{
    constexpr size_t slm_elem_size = sizeof(T) * m_groups;

    while (slm_elem_size * (n_wi + delta_n) * delta_k + reserved_slm_size >=
           local_mem_size)
    {
        n_wi = n_wi / 2;
        delta_n = delta_n / 2;
        if (delta_n == 0)
            throw std::runtime_error("Insufficient resources");
    }
}

template <typename T, int wi_delta_m>
void scale_gemm_nm_parameters(const size_t &local_mem_size,
                              const size_t &reserved_slm_size,
                              const size_t &wi_delta_n,
                              size_t &wi_delta_k,
                              size_t &wg_delta_n,
                              size_t &wg_delta_m)
{
    constexpr size_t slm_A_elem_size = sizeof(T);
    constexpr size_t slm_B_elem_size = sizeof(T) * wi_delta_m;

    while ((wi_delta_n * wg_delta_n * wi_delta_k * slm_A_elem_size) +
               (wi_delta_k * wg_delta_m * slm_B_elem_size) +
               reserved_slm_size >=
           local_mem_size)
    {
        wg_delta_n /= 2;
        wg_delta_m /= 2;
        wi_delta_k /= 2;
        if (wg_delta_n == 0)
            throw std::runtime_error("Insufficient resources");
    }
}
} // namespace gemm_detail

using dpctl::tensor::sycl_utils::choose_workgroup_size;

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class gemm_reduction_over_group_temps_strided_krn;

template <typename T, typename ReductionOpT>
sycl::event tree_reduction_for_gemm(sycl::queue &exec_q,
                                    T *partially_reduced_tmp,
                                    T *partially_reduced_tmp2,
                                    T *res_tp,
                                    T identity_val,
                                    size_t iter_nelems,
                                    size_t reduction_nelems,
                                    size_t reduction_groups,
                                    size_t wg,
                                    size_t max_wg,
                                    size_t preferred_reductions_per_wi,
                                    size_t reductions_per_wi,
                                    int res_nd,
                                    py::ssize_t res_offset,
                                    const py::ssize_t *res_shape_strides,
                                    std::vector<sycl::event> depends = {})
{

    const sycl::event &first_reduction_ev = exec_q.submit([&](sycl::handler
                                                                  &cgh) {
        cgh.depends_on(depends);

        using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                NoOpIndexerT, NoOpIndexerT>;
        using ReductionIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;

        // Only 2*iter_nd entries describing shape and strides of
        // iterated dimensions of input array from
        // iter_shape_and_strides are going to be accessed by
        // inp_indexer

        InputOutputIterIndexerT in_out_iter_indexer{NoOpIndexerT{},
                                                    NoOpIndexerT{}};
        ReductionIndexerT reduction_indexer{
            0, /* size */ static_cast<py::ssize_t>(reduction_nelems),
            /* step */ static_cast<py::ssize_t>(iter_nelems)};

        auto globalRange = sycl::range<1>{iter_nelems * reduction_groups * wg};
        auto localRange = sycl::range<1>{wg};

        using KernelName = class gemm_reduction_over_group_temps_strided_krn<
            T, T, ReductionOpT, InputOutputIterIndexerT, ReductionIndexerT>;
        cgh.parallel_for<KernelName>(
            sycl::nd_range<1>(globalRange, localRange),
            ReductionOverGroupNoAtomicFunctor<
                T, T, ReductionOpT, InputOutputIterIndexerT, ReductionIndexerT>(
                partially_reduced_tmp, partially_reduced_tmp2, ReductionOpT(),
                identity_val, in_out_iter_indexer, reduction_indexer,
                reduction_nelems, iter_nelems, reductions_per_wi));
    });

    size_t remaining_reduction_nelems = reduction_groups;

    T *temp_arg = partially_reduced_tmp2;
    T *temp2_arg = partially_reduced_tmp;
    sycl::event dependent_ev = first_reduction_ev;

    while (remaining_reduction_nelems > preferred_reductions_per_wi * max_wg) {
        size_t reduction_groups_ = (remaining_reduction_nelems +
                                    preferred_reductions_per_wi * wg - 1) /
                                   (preferred_reductions_per_wi * wg);
        assert(reduction_groups_ > 1);

        // keep reducing
        sycl::event partial_reduction_ev =
            exec_q.submit([&](sycl::handler &cgh) {
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

                using KernelName =
                    class gemm_reduction_over_group_temps_strided_krn<
                        T, T, ReductionOpT, InputOutputIterIndexerT,
                        ReductionIndexerT>;
                cgh.parallel_for<KernelName>(
                    sycl::nd_range<1>(globalRange, localRange),
                    ReductionOverGroupNoAtomicFunctor<T, T, ReductionOpT,
                                                      InputOutputIterIndexerT,
                                                      ReductionIndexerT>(
                        temp_arg, temp2_arg, ReductionOpT(), identity_val,
                        in_out_iter_indexer, reduction_indexer,
                        remaining_reduction_nelems, iter_nelems,
                        reductions_per_wi));
            });

        remaining_reduction_nelems = reduction_groups_;
        std::swap(temp_arg, temp2_arg);
        dependent_ev = std::move(partial_reduction_ev);
    }

    // final reduction to res
    sycl::event final_reduction_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependent_ev);

        using InputIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
        using ResIndexerT = dpctl::tensor::offset_utils::StridedIndexer;
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                InputIndexerT, ResIndexerT>;
        using ReductionIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;

        InputIndexerT inp_indexer{
            0, static_cast<py::ssize_t>(iter_nelems),
            static_cast<py::ssize_t>(remaining_reduction_nelems)};
        ResIndexerT res_iter_indexer{
            res_nd, static_cast<py::ssize_t>(res_offset), res_shape_strides};

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

        auto globalRange = sycl::range<1>{iter_nelems * reduction_groups * wg};
        auto localRange = sycl::range<1>{wg};

        using KernelName = class gemm_reduction_over_group_temps_strided_krn<
            T, T, ReductionOpT, InputOutputIterIndexerT, ReductionIndexerT>;
        cgh.parallel_for<KernelName>(
            sycl::nd_range<1>(globalRange, localRange),
            ReductionOverGroupNoAtomicFunctor<
                T, T, ReductionOpT, InputOutputIterIndexerT, ReductionIndexerT>(
                temp_arg, res_tp, ReductionOpT(), identity_val,
                in_out_iter_indexer, reduction_indexer,
                remaining_reduction_nelems, iter_nelems, reductions_per_wi));
    });

    return final_reduction_ev;
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class gemm_reduction_over_group_temps_contig_krn;

template <typename T, typename ReductionOpT>
sycl::event
tree_reduction_for_gemm_contig(sycl::queue &exec_q,
                               T *partially_reduced_tmp,
                               T *partially_reduced_tmp2,
                               T *res_tp,
                               T identity_val,
                               size_t iter_nelems,
                               size_t reduction_nelems,
                               size_t reduction_groups,
                               size_t wg,
                               size_t max_wg,
                               size_t preferred_reductions_per_wi,
                               size_t reductions_per_wi,
                               std::vector<sycl::event> depends = {})
{

    const sycl::event &first_reduction_ev = exec_q.submit([&](sycl::handler
                                                                  &cgh) {
        cgh.depends_on(depends);

        using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                NoOpIndexerT, NoOpIndexerT>;
        using ReductionIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;

        // Only 2*iter_nd entries describing shape and strides of
        // iterated dimensions of input array from
        // iter_shape_and_strides are going to be accessed by
        // inp_indexer

        InputOutputIterIndexerT in_out_iter_indexer{NoOpIndexerT{},
                                                    NoOpIndexerT{}};
        ReductionIndexerT reduction_indexer{
            0, /* size */ static_cast<py::ssize_t>(reduction_nelems),
            /* step */ static_cast<py::ssize_t>(iter_nelems)};

        auto globalRange = sycl::range<1>{iter_nelems * reduction_groups * wg};
        auto localRange = sycl::range<1>{wg};

        using KernelName = class gemm_reduction_over_group_temps_contig_krn<
            T, T, ReductionOpT, InputOutputIterIndexerT, ReductionIndexerT>;
        cgh.parallel_for<KernelName>(
            sycl::nd_range<1>(globalRange, localRange),
            ReductionOverGroupNoAtomicFunctor<
                T, T, ReductionOpT, InputOutputIterIndexerT, ReductionIndexerT>(
                partially_reduced_tmp, partially_reduced_tmp2, ReductionOpT(),
                identity_val, in_out_iter_indexer, reduction_indexer,
                reduction_nelems, iter_nelems, reductions_per_wi));
    });

    size_t remaining_reduction_nelems = reduction_groups;

    T *temp_arg = partially_reduced_tmp2;
    T *temp2_arg = partially_reduced_tmp;
    sycl::event dependent_ev = first_reduction_ev;

    while (remaining_reduction_nelems > preferred_reductions_per_wi * max_wg) {
        size_t reduction_groups_ = (remaining_reduction_nelems +
                                    preferred_reductions_per_wi * wg - 1) /
                                   (preferred_reductions_per_wi * wg);
        assert(reduction_groups_ > 1);

        // keep reducing
        sycl::event partial_reduction_ev = exec_q.submit([&](sycl::handler
                                                                 &cgh) {
            cgh.depends_on(dependent_ev);

            using InputIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
            using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            using InputOutputIterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    InputIndexerT, ResIndexerT>;
            using ReductionIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;

            // n * m = iter_nelems because essentially, this process
            // creates a stack of reduction_nelems 2D matrices and we reduce
            // along the stack axis
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

            using KernelName = class gemm_reduction_over_group_temps_contig_krn<
                T, T, ReductionOpT, InputOutputIterIndexerT, ReductionIndexerT>;
            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(globalRange, localRange),
                ReductionOverGroupNoAtomicFunctor<T, T, ReductionOpT,
                                                  InputOutputIterIndexerT,
                                                  ReductionIndexerT>(
                    temp_arg, temp2_arg, ReductionOpT(), identity_val,
                    in_out_iter_indexer, reduction_indexer,
                    remaining_reduction_nelems, iter_nelems,
                    reductions_per_wi));
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

        auto globalRange = sycl::range<1>{iter_nelems * reduction_groups * wg};
        auto localRange = sycl::range<1>{wg};

        using KernelName = class gemm_reduction_over_group_temps_contig_krn<
            T, T, ReductionOpT, InputOutputIterIndexerT, ReductionIndexerT>;
        cgh.parallel_for<KernelName>(
            sycl::nd_range<1>(globalRange, localRange),
            ReductionOverGroupNoAtomicFunctor<
                T, T, ReductionOpT, InputOutputIterIndexerT, ReductionIndexerT>(
                temp_arg, res_tp, ReductionOpT(), identity_val,
                in_out_iter_indexer, reduction_indexer,
                remaining_reduction_nelems, iter_nelems, reductions_per_wi));
    });

    return final_reduction_ev;
}

template <typename lhsT,
          typename rhsT,
          typename resT,
          typename LocAccT1,
          typename LocAccT2,
          typename OuterInnerDimsIndexerT,
          int wi_delta_n,
          int wi_delta_m>
class GemmFunctorThreadNM
{
private:
    const lhsT *lhs = nullptr;
    const rhsT *rhs = nullptr;
    resT *res = nullptr;
    LocAccT1 local_A_block;
    LocAccT2 local_B_block;
    size_t n = 0;
    size_t wg_delta_n = 0;
    size_t k = 0;
    size_t k_blocks = 0;
    size_t wi_delta_k = 0;
    size_t m = 0;
    size_t m_blocks = 0;
    size_t wg_delta_m = 0;
    OuterInnerDimsIndexerT lhs_indexer;
    OuterInnerDimsIndexerT rhs_indexer;
    OuterInnerDimsIndexerT res_indexer;

public:
    GemmFunctorThreadNM(const lhsT *lhs_,
                        const rhsT *rhs_,
                        resT *res_,
                        LocAccT1 local_A_block_,
                        LocAccT2 local_B_block_,
                        size_t n_,
                        size_t wg_delta_n_,
                        size_t k_,
                        size_t k_blocks_,
                        size_t wi_delta_k_,
                        size_t m_,
                        size_t m_blocks_,
                        size_t wg_delta_m_,
                        OuterInnerDimsIndexerT lhs_indexer_,
                        OuterInnerDimsIndexerT rhs_indexer_,
                        OuterInnerDimsIndexerT res_indexer_)
        : lhs(lhs_), rhs(rhs_), res(res_), local_A_block(local_A_block_),
          local_B_block(local_B_block_), n(n_), wg_delta_n(wg_delta_n_), k(k_),
          k_blocks(k_blocks_), wi_delta_k(wi_delta_k_), m(m_),
          m_blocks(m_blocks_), wg_delta_m(wg_delta_m_),
          lhs_indexer(lhs_indexer_), rhs_indexer(rhs_indexer_),
          res_indexer(res_indexer_)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        size_t gr_id = it.get_group_linear_id();
        // lift group_id to (block_i, block_j, block_s),
        //    0 <= block_i < n_blocks, 0 <= block_j < m_blocks, 0 <= block_s <
        //    k_blocks
        size_t block_i = gr_id / (m_blocks * k_blocks);
        size_t block_r = gr_id - block_i * (m_blocks * k_blocks);
        size_t block_j = block_r / k_blocks;
        size_t block_s = block_r - block_j * k_blocks;

        size_t lid = it.get_local_linear_id();
        size_t local_i = lid / wg_delta_m;           // 0<= local_i < wg_delta_n
        size_t local_j = lid - local_i * wg_delta_m; // 0<= local_j < wg_delta_m

        // load A block and B blocks into SLM

        size_t i = block_i * wi_delta_n * wg_delta_n;
        size_t j = block_j * wi_delta_m * wg_delta_m;
        size_t s = block_s * wi_delta_k;

        const std::int64_t a_st0 = k;
        const std::int64_t a_st1 = 1;

        const std::int64_t b_st0 = m;
        const std::int64_t b_st1 = 1;

        const std::int64_t c_st0 = m;
        const std::int64_t c_st1 = 1;

        size_t lws = it.get_local_range(0);

        for (size_t vid = lid; vid < local_A_block.size(); vid += lws) {
            size_t v_i = vid / wi_delta_k; // 0<= v_i < wg_delta_n * wi_delta_n
            size_t v_s = vid - v_i * wi_delta_k; // 0<= v_s < wi_delta_k

            size_t g_i = i + v_i;
            size_t g_s = s + v_s;

            local_A_block[vid] =
                (g_i < n && g_s < k)
                    ? static_cast<resT>(
                          lhs[lhs_indexer(g_i * a_st0 + g_s * a_st1)])
                    : resT(0);
        }

        for (size_t vid = lid; vid < local_B_block.size(); vid += lws) {
            size_t v_j = vid / wi_delta_k;       // 0 <= v_i < wg_delta_m
            size_t v_s = vid - v_j * wi_delta_k; // 0 <= v_s < wi_delta_k

            size_t g_j0 = j + v_j * wi_delta_m;
            size_t g_s = s + v_s;

            sycl::vec<resT, wi_delta_m> vec{};
#pragma unroll
            for (std::uint8_t lane_id = 0; lane_id < wi_delta_m; ++lane_id) {
                size_t g_j = g_j0 + lane_id;
                vec[lane_id] =
                    (g_j < m && g_s < k)
                        ? static_cast<resT>(
                              rhs[rhs_indexer(g_s * b_st0 + g_j * b_st1)])
                        : resT(0);
            }

            local_B_block[vid] = vec;
        }

        it.barrier(sycl::access::fence_space::local_space);

        i += local_i * wi_delta_n;
        j += local_j * wi_delta_m;

        size_t a_offset = local_i * wi_delta_k * wi_delta_n;
        size_t b_offset = local_j * wi_delta_k;

        constexpr resT identity_(0);
        for (std::uint8_t private_i = 0; private_i < wi_delta_n; ++private_i) {
            size_t a_pr_offset = private_i * wi_delta_k;

            sycl::vec<resT, wi_delta_m> local_sum(identity_);
            for (size_t private_s = 0; private_s < wi_delta_k; ++private_s) {
                local_sum = local_sum +
                            (local_A_block[a_offset + a_pr_offset + private_s] *
                             local_B_block[b_offset + private_s]);
            }

            size_t gl_i = i + private_i;

#pragma unroll
            for (std::uint8_t lane_id = 0; lane_id < wi_delta_m; ++lane_id) {
                size_t gl_j = j + lane_id;

                if (gl_i < n && gl_j < m) {
                    sycl::atomic_ref<resT, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        aout(res[res_indexer(gl_i * c_st0 + gl_j * c_st1)]);

                    aout += local_sum[lane_id];
                }
            }
        }
    }
};

// specialization for wi_delta_m == 1
template <typename lhsT,
          typename rhsT,
          typename resT,
          typename LocAccT1,
          typename LocAccT2,
          typename OuterInnerDimsIndexerT,
          int wi_delta_n>
class GemmFunctorThreadNM<lhsT,
                          rhsT,
                          resT,
                          LocAccT1,
                          LocAccT2,
                          OuterInnerDimsIndexerT,
                          wi_delta_n,
                          1>
{
private:
    const lhsT *lhs = nullptr;
    const rhsT *rhs = nullptr;
    resT *res = nullptr;
    LocAccT1 local_A_block;
    LocAccT2 local_B_block;
    size_t n = 0;
    size_t wg_delta_n = 0;
    size_t k = 0;
    size_t k_blocks = 0;
    size_t wi_delta_k = 0;
    size_t m = 0;
    size_t m_blocks = 0;
    size_t wg_delta_m = 0;
    OuterInnerDimsIndexerT lhs_indexer;
    OuterInnerDimsIndexerT rhs_indexer;
    OuterInnerDimsIndexerT res_indexer;

public:
    GemmFunctorThreadNM(const lhsT *lhs_,
                        const rhsT *rhs_,
                        resT *res_,
                        LocAccT1 local_A_block_,
                        LocAccT2 local_B_block_,
                        size_t n_,
                        size_t wg_delta_n_,
                        size_t k_,
                        size_t k_blocks_,
                        size_t wi_delta_k_,
                        size_t m_,
                        size_t m_blocks_,
                        size_t wg_delta_m_,
                        OuterInnerDimsIndexerT lhs_indexer_,
                        OuterInnerDimsIndexerT rhs_indexer_,
                        OuterInnerDimsIndexerT res_indexer_)
        : lhs(lhs_), rhs(rhs_), res(res_), local_A_block(local_A_block_),
          local_B_block(local_B_block_), n(n_), wg_delta_n(wg_delta_n_), k(k_),
          k_blocks(k_blocks_), wi_delta_k(wi_delta_k_), m(m_),
          m_blocks(m_blocks_), wg_delta_m(wg_delta_m_),
          lhs_indexer(lhs_indexer_), rhs_indexer(rhs_indexer_),
          res_indexer(res_indexer_)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        size_t gr_id = it.get_group_linear_id();
        // lift group_id to (block_i, block_j, block_s),
        //    0 <= block_i < n_blocks, 0 <= block_j < m_blocks, 0 <= block_s <
        //    k_blocks
        size_t block_i = gr_id / (m_blocks * k_blocks);
        size_t block_r = gr_id - block_i * (m_blocks * k_blocks);
        size_t block_j = block_r / k_blocks;
        size_t block_s = block_r - block_j * k_blocks;

        size_t lid = it.get_local_linear_id();
        size_t local_i = lid / wg_delta_m;           // 0<= local_i < wg_delta_n
        size_t local_j = lid - local_i * wg_delta_m; // 0<= local_j < wg_delta_m

        // load A block and B blocks into SLM

        size_t i = block_i * wi_delta_n * wg_delta_n;
        size_t j = block_j * wg_delta_m;
        size_t s = block_s * wi_delta_k;

        const std::int64_t a_st0 = k;
        const std::int64_t a_st1 = 1;

        const std::int64_t b_st0 = m;
        const std::int64_t b_st1 = 1;

        const std::int64_t c_st0 = m;
        const std::int64_t c_st1 = 1;

        size_t lws = it.get_local_range(0);

        for (size_t vid = lid; vid < local_A_block.size(); vid += lws) {
            size_t v_i = vid / wi_delta_k; // 0<= v_i < wg_delta_n * wi_delta_n
            size_t v_s = vid - v_i * wi_delta_k; // 0<= v_s < wi_delta_k

            size_t g_i = i + v_i;
            size_t g_s = s + v_s;

            local_A_block[vid] =
                (g_i < n && g_s < k)
                    ? static_cast<resT>(
                          lhs[lhs_indexer(g_i * a_st0 + g_s * a_st1)])
                    : resT(0);
        }

        for (size_t vid = lid; vid < local_B_block.size(); vid += lws) {
            size_t v_j = vid / wi_delta_k;       // 0<= v_i < wg_delta_m
            size_t v_s = vid - v_j * wi_delta_k; // 0<= v_s < wi_delta_k

            size_t g_j0 = j + v_j;
            size_t g_s = s + v_s;

            resT val = (g_j0 < m && g_s < k)
                           ? static_cast<resT>(
                                 rhs[rhs_indexer(g_s * b_st0 + g_j0 * b_st1)])
                           : resT(0);

            local_B_block[vid] = val;
        }

        it.barrier(sycl::access::fence_space::local_space);

        i += local_i * wi_delta_n;
        j += local_j;

        size_t a_offset = local_i * wi_delta_k * wi_delta_n;
        size_t b_offset = local_j * wi_delta_k;

        constexpr resT identity_(0);
        for (std::uint8_t private_i = 0; private_i < wi_delta_n; ++private_i) {
            size_t a_pr_offset = private_i * wi_delta_k;

            resT local_sum(identity_);
            for (size_t private_s = 0; private_s < wi_delta_k; ++private_s) {
                local_sum = local_sum +
                            (local_A_block[a_offset + a_pr_offset + private_s] *
                             local_B_block[b_offset + private_s]);
            }

            size_t gl_i = i + private_i;

            if (gl_i < n && j < m) {
                sycl::atomic_ref<resT, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    aout(res[res_indexer(gl_i * c_st0 + j * c_st1)]);

                aout += local_sum;
            }
        }
    }
};

template <typename lhsT,
          typename rhsT,
          typename resT,
          typename LocAccT,
          typename OuterInnerDimsIndexerT,
          size_t m_groups>
class GemmFunctorThreadK
{
private:
    const lhsT *lhs = nullptr;
    const rhsT *rhs = nullptr;
    resT *res = nullptr;
    LocAccT workspace;
    LocAccT local_B_block;
    size_t n = 0;
    size_t n_blocks = 0;
    size_t delta_n = 0;
    size_t k = 0;
    size_t k_blocks = 0;
    size_t delta_k = 0;
    size_t n_wi = 0;
    size_t m = 0;
    OuterInnerDimsIndexerT lhs_indexer;
    OuterInnerDimsIndexerT rhs_indexer;
    OuterInnerDimsIndexerT res_indexer;

public:
    GemmFunctorThreadK(const lhsT *lhs_,
                       const rhsT *rhs_,
                       resT *res_,
                       LocAccT workspace_,
                       LocAccT local_B_block_,
                       size_t n_,
                       size_t n_blocks_,
                       size_t delta_n_,
                       size_t k_,
                       size_t k_blocks_,
                       size_t delta_k_,
                       size_t n_wi_,
                       size_t m_,
                       OuterInnerDimsIndexerT lhs_indexer_,
                       OuterInnerDimsIndexerT rhs_indexer_,
                       OuterInnerDimsIndexerT res_indexer_)
        : lhs(lhs_), rhs(rhs_), res(res_), workspace(workspace_),
          local_B_block(local_B_block_), n(n_), n_blocks(n_blocks_),
          delta_n(delta_n_), k(k_), k_blocks(k_blocks_), delta_k(delta_k_),
          n_wi(n_wi_), m(m_), lhs_indexer(lhs_indexer_),
          rhs_indexer(rhs_indexer_), res_indexer(res_indexer_)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        size_t gr_id = it.get_group_linear_id();
        size_t lid = it.get_local_linear_id();

        // lift gr_id -> (block_i, block_j, block_s)
        //   block_i moves fastest, then block_s, then block_j

        size_t block_j =
            gr_id / (n_blocks * k_blocks); // 0 <= block_j < m_blocks
        size_t block_r =
            gr_id - block_j * (n_blocks *
                               k_blocks); // 0 <= block_r < n_blocks * k_blocks
        size_t block_s = block_r / n_blocks; // 0 <= block_s < k_blocks
        size_t block_i =
            block_r - block_s * n_blocks; // 0 <= block_i < n_blocks

        size_t local_i = lid / (delta_k);           // 0 <= local_i < delta_n
        size_t local_s = lid - local_i * (delta_k); // 0 <= local_s < delta_k

        size_t i = block_i * delta_n + local_i;
        size_t j = m_groups * block_j;
        size_t s = block_s * delta_k * n_wi + local_s;

        constexpr resT identity_ = resT(0);
        if (local_i == 0) {
            for (size_t q = 0; q < n_wi * delta_k; q += delta_k) {
                size_t sq = s + q;
                size_t sqmj = sq * m + j;
                local_B_block[local_s + q] = sycl::vec<resT, m_groups>(
                    (sq < k && j < m)
                        ? static_cast<resT>(rhs[rhs_indexer(sqmj)])
                        : identity_,
                    (sq < k && j + 1 < m)
                        ? static_cast<resT>(rhs[rhs_indexer(sqmj + 1)])
                        : identity_);
            }
        }

        it.barrier(sycl::access::fence_space::local_space);

        size_t t_shift = block_s * delta_k * n_wi;
        size_t global_s_offset = i * k + t_shift;

        sycl::vec<resT, m_groups> private_sum(identity_);
        for (size_t t = local_s; t < local_B_block.size(); t += delta_k) {
            if (t + t_shift < k) {
                private_sum +=
                    (static_cast<resT>(lhs[lhs_indexer(global_s_offset + t)]) *
                     local_B_block[t]);
            }
        }

        size_t workspace_i_shift = local_i * delta_k;
        workspace[workspace_i_shift + local_s] = private_sum;

        it.barrier(sycl::access::fence_space::local_space);

        if (local_s == 0 && i < n) {
            sycl::vec<resT, m_groups> local_sum(workspace[workspace_i_shift]);
            for (size_t t = 1; t < delta_k; ++t) {
                local_sum += workspace[workspace_i_shift + t];
            }

            sycl::atomic_ref<resT, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                aout0(res[res_indexer(i * m + j)]);

            aout0 += local_sum[0];

            if (j + 1 < m) {
                sycl::atomic_ref<resT, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    aout1(res[res_indexer(i * m + j + 1)]);

                aout1 += local_sum[1];
            }
        }
    }
};

// specialization for m_groups == 1
template <typename lhsT,
          typename rhsT,
          typename resT,
          typename LocAccT,
          typename OuterInnerDimsIndexerT>
class GemmFunctorThreadK<lhsT, rhsT, resT, LocAccT, OuterInnerDimsIndexerT, 1>
{
private:
    const lhsT *lhs = nullptr;
    const rhsT *rhs = nullptr;
    resT *res = nullptr;
    LocAccT workspace;
    LocAccT local_B_block;
    size_t n = 0;
    size_t n_blocks = 0;
    size_t delta_n = 0;
    size_t k = 0;
    size_t k_blocks = 0;
    size_t delta_k = 0;
    size_t n_wi = 0;
    size_t m = 0;
    OuterInnerDimsIndexerT lhs_indexer;
    OuterInnerDimsIndexerT rhs_indexer;
    OuterInnerDimsIndexerT res_indexer;

public:
    GemmFunctorThreadK(const lhsT *lhs_,
                       const rhsT *rhs_,
                       resT *res_,
                       LocAccT workspace_,
                       LocAccT local_B_block_,
                       size_t n_,
                       size_t n_blocks_,
                       size_t delta_n_,
                       size_t k_,
                       size_t k_blocks_,
                       size_t delta_k_,
                       size_t n_wi_,
                       size_t m_,
                       OuterInnerDimsIndexerT lhs_indexer_,
                       OuterInnerDimsIndexerT rhs_indexer_,
                       OuterInnerDimsIndexerT res_indexer_)
        : lhs(lhs_), rhs(rhs_), res(res_), workspace(workspace_),
          local_B_block(local_B_block_), n(n_), n_blocks(n_blocks_),
          delta_n(delta_n_), k(k_), k_blocks(k_blocks_), delta_k(delta_k_),
          n_wi(n_wi_), m(m_), lhs_indexer(lhs_indexer_),
          rhs_indexer(rhs_indexer_), res_indexer(res_indexer_)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        size_t gr_id = it.get_group_linear_id();
        size_t lid = it.get_local_linear_id();

        // lift gr_id -> (block_i, block_j, block_s)
        //   block_i moves fastest, then block_s, then block_j

        size_t block_j =
            gr_id / (n_blocks * k_blocks); // 0 <= block_j < m_blocks
        size_t block_r =
            gr_id - block_j * (n_blocks *
                               k_blocks); // 0 <= block_r < n_blocks * k_blocks
        size_t block_s = block_r / n_blocks; // 0 <= block_s < k_blocks
        size_t block_i =
            block_r - block_s * n_blocks; // 0 <= block_i < n_blocks

        size_t local_i = lid / (delta_k);           // 0 <= local_i < delta_n
        size_t local_s = lid - local_i * (delta_k); // 0 <= local_s < delta_k

        size_t i = block_i * delta_n + local_i;
        size_t j = block_j;
        size_t s = block_s * delta_k * n_wi + local_s;

        constexpr resT identity_ = resT(0);
        if (local_i == 0) {
            for (size_t q = 0; q < n_wi * delta_k; q += delta_k) {
                size_t sq = s + q;
                size_t sqmj = sq * m + j;
                local_B_block[local_s + q] =
                    (sq < k && j < m)
                        ? static_cast<resT>(rhs[rhs_indexer(sqmj)])
                        : identity_;
            }
        }

        it.barrier(sycl::access::fence_space::local_space);

        size_t t_shift = block_s * delta_k * n_wi;
        size_t global_s_offset = i * k + t_shift;

        resT private_sum(identity_);
        for (size_t t = local_s; t < local_B_block.size(); t += delta_k) {
            if (t + t_shift < k) {
                private_sum +=
                    (static_cast<resT>(lhs[lhs_indexer(global_s_offset + t)]) *
                     local_B_block[t]);
            }
        }

        size_t workspace_i_shift = local_i * delta_k;
        workspace[workspace_i_shift + local_s] = private_sum;

        it.barrier(sycl::access::fence_space::local_space);

        if (local_s == 0 && i < n) {
            resT local_sum(workspace[workspace_i_shift]);
            for (size_t t = 1; t < delta_k; ++t) {
                local_sum += workspace[workspace_i_shift + t];
            }

            sycl::atomic_ref<resT, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                aout(res[res_indexer(i * m + j)]);

            aout += local_sum;
        }
    }
};

template <typename T1, typename T2, typename T3> class gemm_init_krn;

template <typename T1, typename T2, typename T3, typename T4, size_t>
class gemm_k_krn;

template <typename T1, typename T2, typename T3, typename T4, size_t>
class gemm_nm_krn;

typedef sycl::event (*gemm_impl_fn_ptr_t)(
    sycl::queue &,
    const char *,        // lhs
    const char *,        // rhs
    char *,              // res
    size_t,              // lhs_outer_nelems (n)
    size_t,              // inner_nelems (k)
    size_t,              // rhs_outer_nelems (m)
    int,                 // inner nd
    int,                 // lhs outer nd
    const py::ssize_t *, // lhs shape and strides
    int,                 // rhs outer nd
    const py::ssize_t *, // rhs shape and strides
    int,                 // res outer nd
    const py::ssize_t *, // res shape and strides
    std::vector<sycl::event> const &);

template <typename lhsTy, typename rhsTy, typename resTy>
sycl::event gemm_impl(sycl::queue &exec_q,
                      const char *lhs_cp,
                      const char *rhs_cp,
                      char *res_cp,
                      size_t n,
                      size_t k,
                      size_t m,
                      int inner_nd,
                      int lhs_outer_nd,
                      const py::ssize_t *lhs_shape_strides,
                      int rhs_outer_nd,
                      const py::ssize_t *rhs_shape_strides,
                      int res_outer_nd,
                      const py::ssize_t *res_shape_strides,
                      std::vector<sycl::event> const &depends = {})
{
    const lhsTy *lhs_tp = reinterpret_cast<const lhsTy *>(lhs_cp);
    const rhsTy *rhs_tp = reinterpret_cast<const rhsTy *>(rhs_cp);
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp);

    sycl::event res_init_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        using IndexerT = dpctl::tensor::offset_utils::StridedIndexer;
        IndexerT res_indexer(res_outer_nd, 0, res_shape_strides);
        using InitKernelName = class gemm_init_krn<lhsTy, rhsTy, resTy>;
        cgh.parallel_for<InitKernelName>(
            sycl::range<1>(n * m), [=](sycl::id<1> id) {
                auto res_offset = res_indexer(id[0]);
                res_tp[res_offset] = resTy(0);
            });
    });

    if (k == 0) {
        return res_init_ev;
    }

    const sycl::device &dev = exec_q.get_device();
    const size_t local_mem_size =
        dev.get_info<sycl::info::device::local_mem_size>();
    const size_t reserved_slm_size = 512;

    sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(res_init_ev);

        using OuterInnerIndexerT = dpctl::tensor::offset_utils::StridedIndexer;
        OuterInnerIndexerT lhs_indexer(inner_nd + lhs_outer_nd, 0,
                                       lhs_shape_strides);
        OuterInnerIndexerT rhs_indexer(inner_nd + rhs_outer_nd, 0,
                                       rhs_shape_strides);
        OuterInnerIndexerT res_indexer(res_outer_nd, 0, res_shape_strides);

        if (m == 1) {
            constexpr size_t m_groups = 1;
            size_t delta_k(4);
            size_t n_wi(4);
            size_t delta_n(4);

            gemm_detail::scale_gemm_k_parameters<resTy, m_groups>(
                local_mem_size, reserved_slm_size, delta_k,
                n_wi,   // modified by reference
                delta_n // modified by reference
            );

            size_t n_blocks = (n + delta_n - 1) / delta_n;
            size_t m_blocks = (m + m_groups - 1) / m_groups;
            size_t k_blocks = (k + n_wi * delta_k - 1) / (n_wi * delta_k);

            size_t lws = delta_n * delta_k;

            auto gRange = sycl::range<1>(n_blocks * m_blocks * k_blocks * lws);
            auto lRange = sycl::range<1>(lws);

            auto ndRange = sycl::nd_range<1>(gRange, lRange);

            using LocAccT = sycl::local_accessor<resTy, 1>;
            LocAccT local_B_block(n_wi * delta_k, cgh);
            LocAccT workspace(delta_n * delta_k, cgh);

            using KernelName = class gemm_k_krn<lhsTy, rhsTy, resTy,
                                                OuterInnerIndexerT, m_groups>;
            cgh.parallel_for<KernelName>(
                ndRange, GemmFunctorThreadK<lhsTy, rhsTy, resTy, LocAccT,
                                            OuterInnerIndexerT, m_groups>(
                             lhs_tp, rhs_tp, res_tp, workspace, local_B_block,
                             n, n_blocks, delta_n, k, k_blocks, delta_k, n_wi,
                             m, lhs_indexer, rhs_indexer, res_indexer));
        }
        else if (k > n && k > m && !exec_q.get_device().is_cpu()) {
            constexpr size_t m_groups = 2;
            size_t delta_k(4);
            size_t n_wi(4);
            size_t delta_n(4);

            gemm_detail::scale_gemm_k_parameters<resTy, m_groups>(
                local_mem_size, reserved_slm_size, delta_k,
                n_wi,   // modified by reference
                delta_n // modified by reference
            );

            size_t n_blocks = (n + delta_n - 1) / delta_n;
            size_t m_blocks = (m + m_groups - 1) / m_groups;
            size_t k_blocks = (k + n_wi * delta_k - 1) / (n_wi * delta_k);

            size_t lws = delta_n * delta_k;

            auto gRange = sycl::range<1>(n_blocks * m_blocks * k_blocks * lws);
            auto lRange = sycl::range<1>(lws);

            auto ndRange = sycl::nd_range<1>(gRange, lRange);

            using LocAccT = sycl::local_accessor<sycl::vec<resTy, m_groups>, 1>;
            LocAccT local_B_block(n_wi * delta_k, cgh);
            LocAccT workspace(delta_n * delta_k, cgh);

            using KernelName = class gemm_k_krn<lhsTy, rhsTy, resTy,
                                                OuterInnerIndexerT, m_groups>;
            cgh.parallel_for<KernelName>(
                ndRange, GemmFunctorThreadK<lhsTy, rhsTy, resTy, LocAccT,
                                            OuterInnerIndexerT, m_groups>(
                             lhs_tp, rhs_tp, res_tp, workspace, local_B_block,
                             n, n_blocks, delta_n, k, k_blocks, delta_k, n_wi,
                             m, lhs_indexer, rhs_indexer, res_indexer));
        }
        else {
            constexpr int wi_delta_n = 2;
            constexpr int wi_delta_m = 4;
            size_t wg_delta_n(16); // rows of A processed in WG
            size_t wg_delta_m(16); // rows of B processed in WG
            size_t wi_delta_k(64); // Elements in K dimension processed by WI

            gemm_detail::scale_gemm_nm_parameters<resTy, wi_delta_m>(
                local_mem_size, reserved_slm_size, wi_delta_n,
                wi_delta_k, // modified by reference
                wg_delta_n, // modified by reference
                wg_delta_m  // modified by reference
            );

            size_t lws = wg_delta_n * wg_delta_m;

            size_t n_blocks =
                ((n + wi_delta_n * wg_delta_n - 1) / (wi_delta_n * wg_delta_n));
            size_t m_blocks =
                ((m + wi_delta_m * wg_delta_m - 1) / (wi_delta_m * wg_delta_m));
            size_t k_blocks = ((k + wi_delta_k - 1) / wi_delta_k);

            auto gwsRange =
                sycl::range<1>(n_blocks * m_blocks * k_blocks * lws);
            auto lwsRange = sycl::range<1>(lws);

            auto ndRange = sycl::nd_range<1>(gwsRange, lwsRange);

            using LocAccT1 = sycl::local_accessor<resTy, 1>;
            LocAccT1 local_A_block(
                sycl::range<1>((wi_delta_n * wg_delta_n) * wi_delta_k), cgh);
            using LocAccT2 =
                sycl::local_accessor<sycl::vec<resTy, wi_delta_m>, 1>;
            LocAccT2 local_B_block(sycl::range<1>(wi_delta_k * wg_delta_m),
                                   cgh);

            using KernelName =
                class gemm_nm_krn<lhsTy, rhsTy, resTy, OuterInnerIndexerT,
                                  wi_delta_m>;
            cgh.parallel_for<KernelName>(
                ndRange,
                GemmFunctorThreadNM<lhsTy, rhsTy, resTy, LocAccT1, LocAccT2,
                                    OuterInnerIndexerT, wi_delta_n, wi_delta_m>(
                    lhs_tp, rhs_tp, res_tp, local_A_block, local_B_block, n,
                    wg_delta_n, k, k_blocks, wi_delta_k, m, m_blocks,
                    wg_delta_m, lhs_indexer, rhs_indexer, res_indexer));
        }
    });

    return gemm_ev;
}

typedef sycl::event (*gemm_contig_impl_fn_ptr_t)(
    sycl::queue &,
    const char *, // lhs
    const char *, // rhs
    char *,       // res
    size_t,       // n
    size_t,       // k
    size_t,       // m
    std::vector<sycl::event> const &);

template <typename lhsTy, typename rhsTy, typename resTy>
sycl::event gemm_contig_impl(sycl::queue &exec_q,
                             const char *lhs_cp,
                             const char *rhs_cp,
                             char *res_cp,
                             size_t n,
                             size_t k,
                             size_t m,
                             std::vector<sycl::event> const &depends = {})
{
    const lhsTy *lhs_tp = reinterpret_cast<const lhsTy *>(lhs_cp);
    const rhsTy *rhs_tp = reinterpret_cast<const rhsTy *>(rhs_cp);
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp);

    sycl::event res_init_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.fill<resTy>(res_tp, resTy(0), n * m);
    });

    if (k == 0) {
        return res_init_ev;
    }

    const sycl::device &dev = exec_q.get_device();
    const size_t local_mem_size =
        dev.get_info<sycl::info::device::local_mem_size>();
    const size_t reserved_slm_size = 512;

    sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(res_init_ev);

        using OuterInnerIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        OuterInnerIndexerT lhs_indexer{};
        OuterInnerIndexerT rhs_indexer{};
        OuterInnerIndexerT res_indexer{};

        if (m == 1) {
            constexpr size_t m_groups = 1;
            size_t delta_k(4);
            size_t n_wi(4);
            size_t delta_n(4);

            gemm_detail::scale_gemm_k_parameters<resTy, m_groups>(
                local_mem_size, reserved_slm_size, delta_k,
                n_wi,   // modified by reference
                delta_n // modified by reference
            );

            size_t n_blocks = (n + delta_n - 1) / delta_n;
            size_t m_blocks = (m + m_groups - 1) / m_groups;
            size_t k_blocks = (k + n_wi * delta_k - 1) / (n_wi * delta_k);

            size_t lws = delta_n * delta_k;

            auto gRange = sycl::range<1>(n_blocks * m_blocks * k_blocks * lws);
            auto lRange = sycl::range<1>(lws);

            auto ndRange = sycl::nd_range<1>(gRange, lRange);

            using LocAccT = sycl::local_accessor<resTy, 1>;
            LocAccT local_B_block(n_wi * delta_k, cgh);
            LocAccT workspace(delta_n * delta_k, cgh);

            using KernelName = class gemm_k_krn<lhsTy, rhsTy, resTy,
                                                OuterInnerIndexerT, m_groups>;
            cgh.parallel_for<KernelName>(
                ndRange, GemmFunctorThreadK<lhsTy, rhsTy, resTy, LocAccT,
                                            OuterInnerIndexerT, m_groups>(
                             lhs_tp, rhs_tp, res_tp, workspace, local_B_block,
                             n, n_blocks, delta_n, k, k_blocks, delta_k, n_wi,
                             m, lhs_indexer, rhs_indexer, res_indexer));
        }
        else if (k > n && k > m && !exec_q.get_device().is_cpu()) {
            constexpr size_t m_groups = 2;
            size_t delta_k(4);
            size_t n_wi(4);
            size_t delta_n(4);

            gemm_detail::scale_gemm_k_parameters<resTy, m_groups>(
                local_mem_size, reserved_slm_size, delta_k,
                n_wi,   // modified by reference
                delta_n // modified by reference
            );

            size_t n_blocks = (n + delta_n - 1) / delta_n;
            size_t m_blocks = (m + m_groups - 1) / m_groups;
            size_t k_blocks = (k + n_wi * delta_k - 1) / (n_wi * delta_k);

            size_t lws = delta_n * delta_k;

            auto gRange = sycl::range<1>(n_blocks * m_blocks * k_blocks * lws);
            auto lRange = sycl::range<1>(lws);

            auto ndRange = sycl::nd_range<1>(gRange, lRange);

            using LocAccT = sycl::local_accessor<sycl::vec<resTy, m_groups>, 1>;
            LocAccT local_B_block(n_wi * delta_k, cgh);
            LocAccT workspace(delta_n * delta_k, cgh);

            using KernelName = class gemm_k_krn<lhsTy, rhsTy, resTy,
                                                OuterInnerIndexerT, m_groups>;
            cgh.parallel_for<KernelName>(
                ndRange, GemmFunctorThreadK<lhsTy, rhsTy, resTy, LocAccT,
                                            OuterInnerIndexerT, m_groups>(
                             lhs_tp, rhs_tp, res_tp, workspace, local_B_block,
                             n, n_blocks, delta_n, k, k_blocks, delta_k, n_wi,
                             m, lhs_indexer, rhs_indexer, res_indexer));
        }
        else {
            constexpr int wi_delta_n = 2;
            constexpr int wi_delta_m = 4;
            size_t wg_delta_n(16); // rows of A processed in WG
            size_t wg_delta_m(16); // rows of B processed in WG
            size_t wi_delta_k(64); // Elements in K dimension processed by WI

            gemm_detail::scale_gemm_nm_parameters<resTy, wi_delta_m>(
                local_mem_size, reserved_slm_size, wi_delta_n,
                wi_delta_k, // modified by reference
                wg_delta_n, // modified by reference
                wg_delta_m  // modified by reference
            );

            size_t lws = wg_delta_n * wg_delta_m;

            size_t n_blocks =
                ((n + wi_delta_n * wg_delta_n - 1) / (wi_delta_n * wg_delta_n));
            size_t m_blocks =
                ((m + wi_delta_m * wg_delta_m - 1) / (wi_delta_m * wg_delta_m));
            size_t k_blocks = ((k + wi_delta_k - 1) / wi_delta_k);

            auto gwsRange =
                sycl::range<1>(n_blocks * m_blocks * k_blocks * lws);
            auto lwsRange = sycl::range<1>(lws);

            auto ndRange = sycl::nd_range<1>(gwsRange, lwsRange);

            using LocAccT1 = sycl::local_accessor<resTy, 1>;
            LocAccT1 local_A_block(
                sycl::range<1>((wi_delta_n * wg_delta_n) * wi_delta_k), cgh);
            using LocAccT2 =
                sycl::local_accessor<sycl::vec<resTy, wi_delta_m>, 1>;
            LocAccT2 local_B_block(sycl::range<1>(wi_delta_k * wg_delta_m),
                                   cgh);

            using KernelName =
                class gemm_nm_krn<lhsTy, rhsTy, resTy, OuterInnerIndexerT,
                                  wi_delta_m>;
            cgh.parallel_for<KernelName>(
                ndRange,
                GemmFunctorThreadNM<lhsTy, rhsTy, resTy, LocAccT1, LocAccT2,
                                    OuterInnerIndexerT, wi_delta_n, wi_delta_m>(
                    lhs_tp, rhs_tp, res_tp, local_A_block, local_B_block, n,
                    wg_delta_n, k, k_blocks, wi_delta_k, m, m_blocks,
                    wg_delta_m, lhs_indexer, rhs_indexer, res_indexer));
        }
    });

    return gemm_ev;
}

template <typename lhsT,
          typename rhsT,
          typename resT,
          typename LocAccT1,
          typename LocAccT2,
          typename OuterInnerDimsIndexerT,
          typename ResIndexerT,
          int wi_delta_n,
          int wi_delta_m>
class GemmNoAtomicFunctorThreadNM
{
private:
    const lhsT *lhs = nullptr;
    const rhsT *rhs = nullptr;
    resT *res = nullptr;
    LocAccT1 local_A_block;
    LocAccT2 local_B_block;
    size_t n = 0;
    size_t wg_delta_n = 0;
    size_t k = 0;
    size_t k_blocks = 0;
    size_t wi_delta_k = 0;
    size_t m = 0;
    size_t m_blocks = 0;
    size_t wg_delta_m = 0;
    OuterInnerDimsIndexerT lhs_indexer;
    OuterInnerDimsIndexerT rhs_indexer;
    ResIndexerT res_indexer;

public:
    GemmNoAtomicFunctorThreadNM(const lhsT *lhs_,
                                const rhsT *rhs_,
                                resT *res_,
                                LocAccT1 local_A_block_,
                                LocAccT2 local_B_block_,
                                size_t n_,
                                size_t wg_delta_n_,
                                size_t k_,
                                size_t k_blocks_,
                                size_t wi_delta_k_,
                                size_t m_,
                                size_t m_blocks_,
                                size_t wg_delta_m_,
                                OuterInnerDimsIndexerT lhs_indexer_,
                                OuterInnerDimsIndexerT rhs_indexer_,
                                ResIndexerT res_indexer_)
        : lhs(lhs_), rhs(rhs_), res(res_), local_A_block(local_A_block_),
          local_B_block(local_B_block_), n(n_), wg_delta_n(wg_delta_n_), k(k_),
          k_blocks(k_blocks_), wi_delta_k(wi_delta_k_), m(m_),
          m_blocks(m_blocks_), wg_delta_m(wg_delta_m_),
          lhs_indexer(lhs_indexer_), rhs_indexer(rhs_indexer_),
          res_indexer(res_indexer_)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        size_t gr_id = it.get_group_linear_id();
        // lift group_id to (block_i, block_j, block_s),
        //    0 <= block_i < n_blocks, 0 <= block_j < m_blocks, 0 <= block_s <
        //    k_blocks
        size_t block_i = gr_id / (m_blocks * k_blocks);
        size_t block_r = gr_id - block_i * (m_blocks * k_blocks);
        size_t block_j = block_r / k_blocks;
        size_t block_s = block_r - block_j * k_blocks;

        size_t lid = it.get_local_linear_id();
        size_t local_i = lid / wg_delta_m;           // 0<= local_i < wg_delta_n
        size_t local_j = lid - local_i * wg_delta_m; // 0<= local_j < wg_delta_m

        // load A block and B blocks into SLM

        size_t i = block_i * wi_delta_n * wg_delta_n;
        size_t j = block_j * wi_delta_m * wg_delta_m;
        size_t s = block_s * wi_delta_k;

        const std::int64_t a_st0 = k;
        const std::int64_t a_st1 = 1;

        const std::int64_t b_st0 = m;
        const std::int64_t b_st1 = 1;

        const std::int64_t c_st0 = m;
        const std::int64_t c_st1 = 1;

        size_t lws = it.get_local_range(0);

        for (size_t vid = lid; vid < local_A_block.size(); vid += lws) {
            size_t v_i = vid / wi_delta_k; // 0<= v_i < wg_delta_n * wi_delta_n
            size_t v_s = vid - v_i * wi_delta_k; // 0<= v_s < wi_delta_k

            size_t g_i = i + v_i;
            size_t g_s = s + v_s;

            local_A_block[vid] =
                (g_i < n && g_s < k)
                    ? static_cast<resT>(
                          lhs[lhs_indexer(g_i * a_st0 + g_s * a_st1)])
                    : resT(0);
        }

        for (size_t vid = lid; vid < local_B_block.size(); vid += lws) {
            size_t v_j = vid / wi_delta_k;       // 0<= v_i < wg_delta_m
            size_t v_s = vid - v_j * wi_delta_k; // 0<= v_s < wi_delta_k

            size_t g_j0 = j + v_j * wi_delta_m;
            size_t g_s = s + v_s;

            sycl::vec<resT, wi_delta_m> vec{};
#pragma unroll
            for (std::uint8_t lane_id = 0; lane_id < wi_delta_m; ++lane_id) {
                size_t g_j = g_j0 + lane_id;
                vec[lane_id] =
                    (g_j < m && g_s < k)
                        ? static_cast<resT>(
                              rhs[rhs_indexer(g_s * b_st0 + g_j * b_st1)])
                        : resT(0);
            }

            local_B_block[vid] = vec;
        }

        it.barrier(sycl::access::fence_space::local_space);

        i += local_i * wi_delta_n;
        j += local_j * wi_delta_m;

        size_t a_offset = local_i * wi_delta_k * wi_delta_n;
        size_t b_offset = local_j * wi_delta_k;

        constexpr resT identity_(0);

        for (std::uint8_t private_i = 0; private_i < wi_delta_n; ++private_i) {
            size_t a_pr_offset = private_i * wi_delta_k;

            sycl::vec<resT, wi_delta_m> local_sum(identity_);
            for (size_t private_s = 0; private_s < wi_delta_k; ++private_s) {
                local_sum = local_sum +
                            (local_A_block[a_offset + a_pr_offset + private_s] *
                             local_B_block[b_offset + private_s]);
            }

            size_t gl_i = i + private_i;

#pragma unroll
            for (std::uint8_t lane_id = 0; lane_id < wi_delta_m; ++lane_id) {
                size_t gl_j = j + lane_id;

                if (gl_i < n && gl_j < m) {
                    res[res_indexer(gl_i * c_st0 + gl_j * c_st1 +
                                    block_s * n * m)] = local_sum[lane_id];
                }
            }
        }
    }
};

template <typename lhsT,
          typename rhsT,
          typename resT,
          typename LocAccT1,
          typename LocAccT2,
          typename OuterInnerDimsIndexerT,
          typename ResIndexerT,
          int wi_delta_n>
class GemmNoAtomicFunctorThreadNM<lhsT,
                                  rhsT,
                                  resT,
                                  LocAccT1,
                                  LocAccT2,
                                  OuterInnerDimsIndexerT,
                                  ResIndexerT,
                                  wi_delta_n,
                                  1>
{
private:
    const lhsT *lhs = nullptr;
    const rhsT *rhs = nullptr;
    resT *res = nullptr;
    LocAccT1 local_A_block;
    LocAccT2 local_B_block;
    size_t n = 0;
    size_t wg_delta_n = 0;
    size_t k = 0;
    size_t k_blocks = 0;
    size_t wi_delta_k = 0;
    size_t m = 0;
    size_t m_blocks = 0;
    size_t wg_delta_m = 0;
    OuterInnerDimsIndexerT lhs_indexer;
    OuterInnerDimsIndexerT rhs_indexer;
    ResIndexerT res_indexer;

public:
    GemmNoAtomicFunctorThreadNM(const lhsT *lhs_,
                                const rhsT *rhs_,
                                resT *res_,
                                LocAccT1 local_A_block_,
                                LocAccT2 local_B_block_,
                                size_t n_,
                                size_t wg_delta_n_,
                                size_t k_,
                                size_t k_blocks_,
                                size_t wi_delta_k_,
                                size_t m_,
                                size_t m_blocks_,
                                size_t wg_delta_m_,
                                OuterInnerDimsIndexerT lhs_indexer_,
                                OuterInnerDimsIndexerT rhs_indexer_,
                                ResIndexerT res_indexer_)
        : lhs(lhs_), rhs(rhs_), res(res_), local_A_block(local_A_block_),
          local_B_block(local_B_block_), n(n_), wg_delta_n(wg_delta_n_), k(k_),
          k_blocks(k_blocks_), wi_delta_k(wi_delta_k_), m(m_),
          m_blocks(m_blocks_), wg_delta_m(wg_delta_m_),
          lhs_indexer(lhs_indexer_), rhs_indexer(rhs_indexer_),
          res_indexer(res_indexer_)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        size_t gr_id = it.get_group_linear_id();
        // lift group_id to (block_i, block_j, block_s),
        //    0 <= block_i < n_blocks, 0 <= block_j < m_blocks, 0 <= block_s <
        //    k_blocks
        size_t block_i = gr_id / (m_blocks * k_blocks);
        size_t block_r = gr_id - block_i * (m_blocks * k_blocks);
        size_t block_j = block_r / k_blocks;
        size_t block_s = block_r - block_j * k_blocks;

        size_t lid = it.get_local_linear_id();
        size_t local_i = lid / wg_delta_m;           // 0<= local_i < wg_delta_n
        size_t local_j = lid - local_i * wg_delta_m; // 0<= local_j < wg_delta_m

        // load A block and B blocks into SLM

        size_t i = block_i * wi_delta_n * wg_delta_n;
        size_t j = block_j * wg_delta_m;
        size_t s = block_s * wi_delta_k;

        const std::int64_t a_st0 = k;
        const std::int64_t a_st1 = 1;

        const std::int64_t b_st0 = m;
        const std::int64_t b_st1 = 1;

        const std::int64_t c_st0 = m;
        const std::int64_t c_st1 = 1;

        size_t lws = it.get_local_range(0);

        for (size_t vid = lid; vid < local_A_block.size(); vid += lws) {
            size_t v_i = vid / wi_delta_k; // 0<= v_i < wg_delta_n * wi_delta_n
            size_t v_s = vid - v_i * wi_delta_k; // 0<= v_s < wi_delta_k

            size_t g_i = i + v_i;
            size_t g_s = s + v_s;

            local_A_block[vid] =
                (g_i < n && g_s < k)
                    ? static_cast<resT>(
                          lhs[lhs_indexer(g_i * a_st0 + g_s * a_st1)])
                    : resT(0);
        }

        for (size_t vid = lid; vid < local_B_block.size(); vid += lws) {
            size_t v_j = vid / wi_delta_k;       // 0<= v_i < wg_delta_m
            size_t v_s = vid - v_j * wi_delta_k; // 0<= v_s < wi_delta_k

            size_t g_j0 = j + v_j;
            size_t g_s = s + v_s;

            resT val = (g_j0 < m && g_s < k)
                           ? static_cast<resT>(
                                 rhs[rhs_indexer(g_s * b_st0 + g_j0 * b_st1)])
                           : resT(0);

            local_B_block[vid] = val;
        }

        it.barrier(sycl::access::fence_space::local_space);

        i += local_i * wi_delta_n;
        j += local_j;

        size_t a_offset = local_i * wi_delta_k * wi_delta_n;
        size_t b_offset = local_j * wi_delta_k;

        constexpr resT identity_(0);

        for (std::uint8_t private_i = 0; private_i < wi_delta_n; ++private_i) {
            size_t a_pr_offset = private_i * wi_delta_k;

            resT local_sum(identity_);
            for (size_t private_s = 0; private_s < wi_delta_k; ++private_s) {
                local_sum = local_sum +
                            (local_A_block[a_offset + a_pr_offset + private_s] *
                             local_B_block[b_offset + private_s]);
            }

            size_t gl_i = i + private_i;

            if (gl_i < n && j < m) {
                res[res_indexer(gl_i * c_st0 + j * c_st1 + block_s * n * m)] =
                    local_sum;
            }
        }
    }
};

template <typename lhsT,
          typename rhsT,
          typename resT,
          typename LocAccT,
          typename OuterInnerDimsIndexerT,
          typename ResIndexerT,
          size_t m_groups>
class GemmNoAtomicFunctorThreadK
{
private:
    const lhsT *lhs = nullptr;
    const rhsT *rhs = nullptr;
    resT *res = nullptr;
    LocAccT workspace;
    LocAccT local_B_block;
    size_t n = 0;
    size_t n_blocks = 0;
    size_t delta_n = 0;
    size_t k = 0;
    size_t k_blocks = 0;
    size_t delta_k = 0;
    size_t n_wi = 0;
    size_t m = 0;
    OuterInnerDimsIndexerT lhs_indexer;
    OuterInnerDimsIndexerT rhs_indexer;
    ResIndexerT res_indexer;

public:
    GemmNoAtomicFunctorThreadK(const lhsT *lhs_,
                               const rhsT *rhs_,
                               resT *res_,
                               LocAccT workspace_,
                               LocAccT local_B_block_,
                               size_t n_,
                               size_t n_blocks_,
                               size_t delta_n_,
                               size_t k_,
                               size_t k_blocks_,
                               size_t delta_k_,
                               size_t n_wi_,
                               size_t m_,
                               OuterInnerDimsIndexerT lhs_indexer_,
                               OuterInnerDimsIndexerT rhs_indexer_,
                               ResIndexerT res_indexer_)
        : lhs(lhs_), rhs(rhs_), res(res_), workspace(workspace_),
          local_B_block(local_B_block_), n(n_), n_blocks(n_blocks_),
          delta_n(delta_n_), k(k_), k_blocks(k_blocks_), delta_k(delta_k_),
          n_wi(n_wi_), m(m_), lhs_indexer(lhs_indexer_),
          rhs_indexer(rhs_indexer_), res_indexer(res_indexer_)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        size_t gr_id = it.get_group_linear_id();
        size_t lid = it.get_local_linear_id();

        // lift gr_id -> (block_i, block_j, block_s)
        //   block_i moves fastest, then block_s, then block_j

        size_t block_j =
            gr_id / (n_blocks * k_blocks); // 0 <= block_j < m_blocks
        size_t block_r =
            gr_id - block_j * (n_blocks *
                               k_blocks); // 0 <= block_r < n_blocks * k_blocks
        size_t block_s = block_r / n_blocks; // 0 <= block_s < k_blocks
        size_t block_i =
            block_r - block_s * n_blocks; // 0 <= block_i < n_blocks

        size_t local_i = lid / (delta_k);           // 0 <= local_i < delta_n
        size_t local_s = lid - local_i * (delta_k); // 0 <= local_s < delta_k

        size_t i = block_i * delta_n + local_i;
        size_t j = m_groups * block_j;
        size_t s = block_s * delta_k * n_wi + local_s;

        constexpr resT identity_ = resT(0);
        if (local_i == 0) {
            for (size_t q = 0; q < n_wi * delta_k; q += delta_k) {
                size_t sq = s + q;
                size_t sqmj = sq * m + j;
                local_B_block[local_s + q] = sycl::vec<resT, m_groups>(
                    (sq < k && j < m)
                        ? static_cast<resT>(rhs[rhs_indexer(sqmj)])
                        : identity_,
                    (sq < k && j + 1 < m)
                        ? static_cast<resT>(rhs[rhs_indexer(sqmj + 1)])
                        : identity_);
            }
        }

        it.barrier(sycl::access::fence_space::local_space);

        size_t t_shift = block_s * delta_k * n_wi;
        size_t global_s_offset = i * k + t_shift;

        sycl::vec<resT, m_groups> private_sum(identity_);
        for (size_t t = local_s; t < local_B_block.size(); t += delta_k) {
            if (t + t_shift < k) {
                private_sum +=
                    (static_cast<resT>(lhs[lhs_indexer(global_s_offset + t)]) *
                     local_B_block[t]);
            }
        }

        size_t workspace_i_shift = local_i * delta_k;
        workspace[workspace_i_shift + local_s] = private_sum;

        it.barrier(sycl::access::fence_space::local_space);

        if (local_s == 0 && i < n) {
            sycl::vec<resT, m_groups> local_sum(workspace[workspace_i_shift]);
            for (size_t t = 1; t < delta_k; ++t) {
                local_sum += workspace[workspace_i_shift + t];
            }

            res[res_indexer(i * m + j) + (block_s * n * m)] = local_sum[0];

            if (j + 1 < m) {
                res[res_indexer(i * m + j + 1) + (block_s * n * m)] =
                    local_sum[1];
            }
        }
    }
};

template <typename lhsT,
          typename rhsT,
          typename resT,
          typename LocAccT,
          typename OuterInnerDimsIndexerT,
          typename ResIndexerT>
class GemmNoAtomicFunctorThreadK<lhsT,
                                 rhsT,
                                 resT,
                                 LocAccT,
                                 OuterInnerDimsIndexerT,
                                 ResIndexerT,
                                 1>
{
private:
    const lhsT *lhs = nullptr;
    const rhsT *rhs = nullptr;
    resT *res = nullptr;
    LocAccT workspace;
    LocAccT local_B_block;
    size_t n = 0;
    size_t n_blocks = 0;
    size_t delta_n = 0;
    size_t k = 0;
    size_t k_blocks = 0;
    size_t delta_k = 0;
    size_t n_wi = 0;
    size_t m = 0;
    OuterInnerDimsIndexerT lhs_indexer;
    OuterInnerDimsIndexerT rhs_indexer;
    ResIndexerT res_indexer;

public:
    GemmNoAtomicFunctorThreadK(const lhsT *lhs_,
                               const rhsT *rhs_,
                               resT *res_,
                               LocAccT workspace_,
                               LocAccT local_B_block_,
                               size_t n_,
                               size_t n_blocks_,
                               size_t delta_n_,
                               size_t k_,
                               size_t k_blocks_,
                               size_t delta_k_,
                               size_t n_wi_,
                               size_t m_,
                               OuterInnerDimsIndexerT lhs_indexer_,
                               OuterInnerDimsIndexerT rhs_indexer_,
                               ResIndexerT res_indexer_)
        : lhs(lhs_), rhs(rhs_), res(res_), workspace(workspace_),
          local_B_block(local_B_block_), n(n_), n_blocks(n_blocks_),
          delta_n(delta_n_), k(k_), k_blocks(k_blocks_), delta_k(delta_k_),
          n_wi(n_wi_), m(m_), lhs_indexer(lhs_indexer_),
          rhs_indexer(rhs_indexer_), res_indexer(res_indexer_)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        size_t gr_id = it.get_group_linear_id();
        size_t lid = it.get_local_linear_id();

        // lift gr_id -> (block_i, block_j, block_s)
        //   block_i moves fastest, then block_s, then block_j

        size_t block_j =
            gr_id / (n_blocks * k_blocks); // 0 <= block_j < m_blocks
        size_t block_r =
            gr_id - block_j * (n_blocks *
                               k_blocks); // 0 <= block_r < n_blocks * k_blocks
        size_t block_s = block_r / n_blocks; // 0 <= block_s < k_blocks
        size_t block_i =
            block_r - block_s * n_blocks; // 0 <= block_i < n_blocks

        size_t local_i = lid / (delta_k);           // 0 <= local_i < delta_n
        size_t local_s = lid - local_i * (delta_k); // 0 <= local_s < delta_k

        size_t i = block_i * delta_n + local_i;
        size_t j = block_j;
        size_t s = block_s * delta_k * n_wi + local_s;

        constexpr resT identity_ = resT(0);
        if (local_i == 0) {
            for (size_t q = 0; q < n_wi * delta_k; q += delta_k) {
                size_t sq = s + q;
                size_t sqmj = sq * m + j;
                local_B_block[local_s + q] =
                    (sq < k && j < m)
                        ? static_cast<resT>(rhs[rhs_indexer(sqmj)])
                        : identity_;
            }
        }

        it.barrier(sycl::access::fence_space::local_space);

        size_t t_shift = block_s * delta_k * n_wi;
        size_t global_s_offset = i * k + t_shift;

        resT private_sum(identity_);
        for (size_t t = local_s; t < local_B_block.size(); t += delta_k) {
            if (t + t_shift < k) {
                private_sum +=
                    (static_cast<resT>(lhs[lhs_indexer(global_s_offset + t)]) *
                     local_B_block[t]);
            }
        }

        size_t workspace_i_shift = local_i * delta_k;
        workspace[workspace_i_shift + local_s] = private_sum;

        it.barrier(sycl::access::fence_space::local_space);

        if (local_s == 0 && i < n) {
            resT local_sum(workspace[workspace_i_shift]);
            for (size_t t = 1; t < delta_k; ++t) {
                local_sum += workspace[workspace_i_shift + t];
            }

            res[res_indexer(i * m + j) + (block_s * n * m)] = local_sum;
        }
    }
};

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class gemm_reduction_seq_strided_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          size_t>
class gemm_tree_nm_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          size_t>
class gemm_tree_k_krn;

template <typename lhsTy, typename rhsTy, typename resTy, size_t m_groups>
sycl::event gemm_tree_k_impl(sycl::queue &exec_q,
                             const lhsTy *lhs_tp,
                             const rhsTy *rhs_tp,
                             resTy *res_tp,
                             size_t n,
                             size_t k,
                             size_t m,
                             int inner_nd,
                             int lhs_outer_nd,
                             const py::ssize_t *lhs_outer_inner_shapes_strides,
                             int rhs_outer_nd,
                             const py::ssize_t *rhs_outer_inner_shapes_strides,
                             int res_nd,
                             const py::ssize_t *res_shapes_strides,
                             const std::vector<sycl::event> &depends)
{
    size_t delta_k(4);
    size_t n_wi(4);
    size_t delta_n(4);

    const sycl::device &dev = exec_q.get_device();
    const size_t local_mem_size =
        dev.get_info<sycl::info::device::local_mem_size>();
    const size_t reserved_slm_size = 512;

    gemm_detail::scale_gemm_k_parameters<resTy, m_groups>(
        local_mem_size, reserved_slm_size, delta_k,
        n_wi,   // modified by reference
        delta_n // modified by reference
    );

    sycl::event gemm_ev;
    if (k <= (delta_k * n_wi)) {
        gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            using OuterInnerDimsIndexerT =
                dpctl::tensor::offset_utils::StridedIndexer;
            OuterInnerDimsIndexerT lhs_indexer(inner_nd + lhs_outer_nd, 0,
                                               lhs_outer_inner_shapes_strides);
            OuterInnerDimsIndexerT rhs_indexer(inner_nd + rhs_outer_nd, 0,
                                               rhs_outer_inner_shapes_strides);
            OuterInnerDimsIndexerT res_indexer(res_nd, 0, res_shapes_strides);

            size_t n_blocks = (n + delta_n - 1) / delta_n;
            size_t k_blocks = (k + n_wi * delta_k - 1) / (n_wi * delta_k);
            size_t m_blocks = (m + m_groups - 1) / m_groups;

            size_t lws = delta_n * delta_k;

            auto gRange = sycl::range<1>(n_blocks * m_blocks * k_blocks * lws);
            auto lRange = sycl::range<1>(lws);

            auto ndRange = sycl::nd_range<1>(gRange, lRange);

            if constexpr (m_groups == 1) {
                using LocAccT = sycl::local_accessor<resTy, 1>;
                LocAccT local_B_block(n_wi * delta_k, cgh);
                LocAccT workspace(delta_n * delta_k, cgh);
                using KernelName =
                    class gemm_tree_k_krn<lhsTy, rhsTy, resTy,
                                          OuterInnerDimsIndexerT,
                                          OuterInnerDimsIndexerT, m_groups>;
                cgh.parallel_for<KernelName>(
                    ndRange,
                    GemmNoAtomicFunctorThreadK<
                        lhsTy, rhsTy, resTy, LocAccT, OuterInnerDimsIndexerT,
                        OuterInnerDimsIndexerT, m_groups>(
                        lhs_tp, rhs_tp, res_tp, workspace, local_B_block, n,
                        n_blocks, delta_n, k, k_blocks, delta_k, n_wi, m,
                        lhs_indexer, rhs_indexer, res_indexer));
            }
            else {
                using LocAccT =
                    sycl::local_accessor<sycl::vec<resTy, m_groups>, 1>;
                LocAccT local_B_block(n_wi * delta_k, cgh);
                LocAccT workspace(delta_n * delta_k, cgh);
                using KernelName =
                    class gemm_tree_k_krn<lhsTy, rhsTy, resTy,
                                          OuterInnerDimsIndexerT,
                                          OuterInnerDimsIndexerT, m_groups>;
                cgh.parallel_for<KernelName>(
                    ndRange,
                    GemmNoAtomicFunctorThreadK<
                        lhsTy, rhsTy, resTy, LocAccT, OuterInnerDimsIndexerT,
                        OuterInnerDimsIndexerT, m_groups>(
                        lhs_tp, rhs_tp, res_tp, workspace, local_B_block, n,
                        n_blocks, delta_n, k, k_blocks, delta_k, n_wi, m,
                        lhs_indexer, rhs_indexer, res_indexer));
            }
        });
        return gemm_ev;
    }
    else {
        using ReductionOpT = sycl::plus<resTy>;
        constexpr resTy identity_val =
            sycl::known_identity<ReductionOpT, resTy>::value;

        size_t iter_nelems = n * m;
        size_t reduction_nelems = (k + delta_k * n_wi - 1) / (delta_k * n_wi);

        // more than one work-groups is needed, requires a temporary
        // delta_k * n_wi elements processed along k, so if more to
        // process use multiple
        const auto &sg_sizes =
            dev.get_info<sycl::info::device::sub_group_sizes>();
        size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

        if (reduction_nelems < wg) {
            resTy *tmp = sycl::malloc_device<resTy>(
                iter_nelems * reduction_nelems, exec_q);
            if (!tmp) {
                throw std::runtime_error("Unable to allocate device memory");
            }
            sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(depends);

                using OuterInnerDimsIndexerT =
                    dpctl::tensor::offset_utils::StridedIndexer;
                OuterInnerDimsIndexerT lhs_indexer(
                    inner_nd + lhs_outer_nd, 0, lhs_outer_inner_shapes_strides);
                OuterInnerDimsIndexerT rhs_indexer(
                    inner_nd + rhs_outer_nd, 0, rhs_outer_inner_shapes_strides);
                using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
                ResIndexerT res_indexer{};

                size_t n_blocks = (n + delta_n - 1) / delta_n;
                size_t k_blocks = (k + n_wi * delta_k - 1) / (n_wi * delta_k);
                size_t m_blocks = (m + m_groups - 1) / m_groups;

                size_t lws = delta_n * delta_k;

                auto gRange =
                    sycl::range<1>(n_blocks * m_blocks * k_blocks * lws);
                auto lRange = sycl::range<1>(lws);

                auto ndRange = sycl::nd_range<1>(gRange, lRange);

                if constexpr (m_groups == 1) {
                    using LocAccT = sycl::local_accessor<resTy, 1>;
                    LocAccT local_B_block(n_wi * delta_k, cgh);
                    LocAccT workspace(delta_n * delta_k, cgh);
                    using KernelName =
                        class gemm_tree_k_krn<lhsTy, rhsTy, resTy,
                                              OuterInnerDimsIndexerT,
                                              ResIndexerT, m_groups>;
                    cgh.parallel_for<KernelName>(
                        ndRange,
                        GemmNoAtomicFunctorThreadK<lhsTy, rhsTy, resTy, LocAccT,
                                                   OuterInnerDimsIndexerT,
                                                   ResIndexerT, m_groups>(
                            lhs_tp, rhs_tp, tmp, workspace, local_B_block, n,
                            n_blocks, delta_n, k, k_blocks, delta_k, n_wi, m,
                            lhs_indexer, rhs_indexer, res_indexer));
                }
                else {
                    using LocAccT =
                        sycl::local_accessor<sycl::vec<resTy, m_groups>, 1>;
                    LocAccT local_B_block(n_wi * delta_k, cgh);
                    LocAccT workspace(delta_n * delta_k, cgh);
                    using KernelName =
                        class gemm_tree_k_krn<lhsTy, rhsTy, resTy,
                                              OuterInnerDimsIndexerT,
                                              ResIndexerT, m_groups>;
                    cgh.parallel_for<KernelName>(
                        ndRange,
                        GemmNoAtomicFunctorThreadK<lhsTy, rhsTy, resTy, LocAccT,
                                                   OuterInnerDimsIndexerT,
                                                   ResIndexerT, m_groups>(
                            lhs_tp, rhs_tp, tmp, workspace, local_B_block, n,
                            n_blocks, delta_n, k, k_blocks, delta_k, n_wi, m,
                            lhs_indexer, rhs_indexer, res_indexer));
                }
            });
            sycl::event red_ev = exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(gemm_ev);

                using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
                using ResIndexerT = dpctl::tensor::offset_utils::StridedIndexer;
                using InputOutputIterIndexerT =
                    dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                        NoOpIndexerT, ResIndexerT>;
                using ReductionIndexerT =
                    dpctl::tensor::offset_utils::Strided1DIndexer;

                ResIndexerT res_iter_indexer{res_nd, 0, res_shapes_strides};
                InputOutputIterIndexerT in_out_iter_indexer{NoOpIndexerT{},
                                                            res_iter_indexer};
                ReductionIndexerT reduction_indexer{
                    0, static_cast<py::ssize_t>(reduction_nelems),
                    static_cast<py::ssize_t>(iter_nelems)};

                sycl::range<1> iter_range{iter_nelems};

                cgh.parallel_for<class gemm_reduction_seq_strided_krn<
                    resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                    ReductionIndexerT>>(
                    iter_range, SequentialReduction<resTy, resTy, ReductionOpT,
                                                    InputOutputIterIndexerT,
                                                    ReductionIndexerT>(
                                    tmp, res_tp, ReductionOpT(), identity_val,
                                    in_out_iter_indexer, reduction_indexer,
                                    reduction_nelems));
            });
            sycl::event cleanup_host_task_event =
                exec_q.submit([&](sycl::handler &cgh) {
                    cgh.depends_on(red_ev);
                    const sycl::context &ctx = exec_q.get_context();

                    cgh.host_task([ctx, tmp] { sycl::free(tmp, ctx); });
                });
            return cleanup_host_task_event;
        }
        // max_max_wg prevents running out of resources on CPU
        constexpr size_t max_max_wg = 2048;
        size_t max_wg = std::min(
            max_max_wg,
            dev.get_info<sycl::info::device::max_work_group_size>() / 2);

        constexpr size_t preferred_reductions_per_wi = 8;
        size_t reductions_per_wi(preferred_reductions_per_wi);

        size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);
        assert(reduction_groups > 1);

        resTy *partially_reduced_tmp = sycl::malloc_device<resTy>(
            iter_nelems * (/* temp */ reduction_nelems +
                           /* first reduction temp */ reduction_groups),
            exec_q);
        resTy *partially_reduced_tmp2 = nullptr;

        if (partially_reduced_tmp == nullptr) {
            throw std::runtime_error("Unable to allocate device memory");
        }
        else {
            partially_reduced_tmp2 =
                partially_reduced_tmp + reduction_nelems * iter_nelems;
        }

        sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            using OuterInnerDimsIndexerT =
                dpctl::tensor::offset_utils::StridedIndexer;
            OuterInnerDimsIndexerT lhs_indexer(inner_nd + lhs_outer_nd, 0,
                                               lhs_outer_inner_shapes_strides);
            OuterInnerDimsIndexerT rhs_indexer(inner_nd + rhs_outer_nd, 0,
                                               rhs_outer_inner_shapes_strides);
            using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            ResIndexerT res_indexer{};

            size_t n_blocks = (n + delta_n - 1) / delta_n;
            size_t k_blocks = (k + n_wi * delta_k - 1) / (n_wi * delta_k);
            size_t m_blocks = (m + m_groups - 1) / m_groups;

            size_t lws = delta_n * delta_k;

            auto gRange = sycl::range<1>(n_blocks * m_blocks * k_blocks * lws);
            auto lRange = sycl::range<1>(lws);

            auto ndRange = sycl::nd_range<1>(gRange, lRange);

            if constexpr (m_groups == 1) {
                using LocAccT = sycl::local_accessor<resTy, 1>;
                LocAccT local_B_block(n_wi * delta_k, cgh);
                LocAccT workspace(delta_n * delta_k, cgh);
                using KernelName = class gemm_tree_k_krn<lhsTy, rhsTy, resTy,
                                                         OuterInnerDimsIndexerT,
                                                         ResIndexerT, m_groups>;
                cgh.parallel_for<KernelName>(
                    ndRange,
                    GemmNoAtomicFunctorThreadK<lhsTy, rhsTy, resTy, LocAccT,
                                               OuterInnerDimsIndexerT,
                                               ResIndexerT, m_groups>(
                        lhs_tp, rhs_tp, partially_reduced_tmp, workspace,
                        local_B_block, n, n_blocks, delta_n, k, k_blocks,
                        delta_k, n_wi, m, lhs_indexer, rhs_indexer,
                        res_indexer));
            }
            else {
                using LocAccT =
                    sycl::local_accessor<sycl::vec<resTy, m_groups>, 1>;
                LocAccT local_B_block(n_wi * delta_k, cgh);
                LocAccT workspace(delta_n * delta_k, cgh);
                using KernelName = class gemm_tree_k_krn<lhsTy, rhsTy, resTy,
                                                         OuterInnerDimsIndexerT,
                                                         ResIndexerT, m_groups>;
                cgh.parallel_for<KernelName>(
                    ndRange,
                    GemmNoAtomicFunctorThreadK<lhsTy, rhsTy, resTy, LocAccT,
                                               OuterInnerDimsIndexerT,
                                               ResIndexerT, m_groups>(
                        lhs_tp, rhs_tp, partially_reduced_tmp, workspace,
                        local_B_block, n, n_blocks, delta_n, k, k_blocks,
                        delta_k, n_wi, m, lhs_indexer, rhs_indexer,
                        res_indexer));
            }
        });
        // tree_reduction_for_gemm returns sycl::event for reduction
        sycl::event red_ev = tree_reduction_for_gemm<resTy, ReductionOpT>(
            exec_q, partially_reduced_tmp, partially_reduced_tmp2, res_tp,
            identity_val, iter_nelems, reduction_nelems, reduction_groups, wg,
            max_wg, preferred_reductions_per_wi, reductions_per_wi, res_nd, 0,
            res_shapes_strides, {gemm_ev});

        sycl::event cleanup_host_task_event =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(red_ev);
                const sycl::context &ctx = exec_q.get_context();

                cgh.host_task([ctx, partially_reduced_tmp] {
                    sycl::free(partially_reduced_tmp, ctx);
                });
            });

        return cleanup_host_task_event;
    }
}

template <typename lhsTy, typename rhsTy, typename resTy, int wi_delta_m>
sycl::event gemm_tree_nm_impl(sycl::queue &exec_q,
                              const lhsTy *lhs_tp,
                              const rhsTy *rhs_tp,
                              resTy *res_tp,
                              size_t n,
                              size_t k,
                              size_t m,
                              int inner_nd,
                              int lhs_outer_nd,
                              const py::ssize_t *lhs_outer_inner_shapes_strides,
                              int rhs_outer_nd,
                              const py::ssize_t *rhs_outer_inner_shapes_strides,
                              int res_nd,
                              const py::ssize_t *res_shapes_strides,
                              const std::vector<sycl::event> &depends)
{
    constexpr int wi_delta_n = 2;
    size_t wg_delta_n(16); // rows of A processed in WG
    size_t wg_delta_m(16); // rows of B processed in WG
    size_t wi_delta_k(64); // Elements in K dimension processed by WI

    const sycl::device &dev = exec_q.get_device();
    const size_t local_mem_size =
        dev.get_info<sycl::info::device::local_mem_size>();
    const size_t reserved_slm_size = 512;

    gemm_detail::scale_gemm_nm_parameters<resTy, wi_delta_m>(
        local_mem_size, reserved_slm_size, wi_delta_n,
        wi_delta_k, // modified by reference
        wg_delta_n, // modified by reference
        wg_delta_m  // modified by reference
    );

    // each group processes delta_k items in a column,
    // so no need to allocate temp memory if one group needed
    if (k <= wi_delta_k) {
        sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            using OuterInnerDimsIndexerT =
                dpctl::tensor::offset_utils::StridedIndexer;
            OuterInnerDimsIndexerT lhs_indexer(inner_nd + lhs_outer_nd, 0,
                                               lhs_outer_inner_shapes_strides);
            OuterInnerDimsIndexerT rhs_indexer(inner_nd + rhs_outer_nd, 0,
                                               rhs_outer_inner_shapes_strides);
            OuterInnerDimsIndexerT res_indexer(res_nd, 0, res_shapes_strides);

            size_t lws = wg_delta_n * wg_delta_m;

            size_t n_blocks =
                ((n + wi_delta_n * wg_delta_n - 1) / (wi_delta_n * wg_delta_n));
            size_t k_blocks = ((k + wi_delta_k - 1) / wi_delta_k);
            size_t m_blocks =
                ((m + wi_delta_m * wg_delta_m - 1) / (wi_delta_m * wg_delta_m));

            auto gwsRange =
                sycl::range<1>(n_blocks * m_blocks * k_blocks * lws);
            auto lwsRange = sycl::range<1>(lws);

            auto ndRange = sycl::nd_range<1>(gwsRange, lwsRange);

            if constexpr (wi_delta_m == 1) {
                using LocAccT1 = sycl::local_accessor<resTy, 1>;
                LocAccT1 local_A_block(
                    sycl::range<1>((wi_delta_n * wg_delta_n) * wi_delta_k),
                    cgh);
                using LocAccT2 = sycl::local_accessor<resTy, 1>;
                LocAccT2 local_B_block(wi_delta_k * wg_delta_m, cgh);

                using KernelName =
                    class gemm_tree_nm_krn<lhsTy, rhsTy, resTy,
                                           OuterInnerDimsIndexerT,
                                           OuterInnerDimsIndexerT, wi_delta_m>;
                cgh.parallel_for<KernelName>(
                    ndRange,
                    GemmNoAtomicFunctorThreadNM<
                        lhsTy, rhsTy, resTy, LocAccT1, LocAccT2,
                        OuterInnerDimsIndexerT, OuterInnerDimsIndexerT,
                        wi_delta_n, wi_delta_m>(
                        lhs_tp, rhs_tp, res_tp, local_A_block, local_B_block, n,
                        wg_delta_n, k, k_blocks, wi_delta_k, m, m_blocks,
                        wg_delta_m, lhs_indexer, rhs_indexer, res_indexer));
            }
            else {
                using LocAccT1 = sycl::local_accessor<resTy, 1>;
                LocAccT1 local_A_block(
                    sycl::range<1>((wi_delta_n * wg_delta_n) * wi_delta_k),
                    cgh);
                using LocAccT2 =
                    sycl::local_accessor<sycl::vec<resTy, wi_delta_m>, 1>;
                LocAccT2 local_B_block(sycl::range<1>(wi_delta_k * wg_delta_m),
                                       cgh);

                using KernelName =
                    class gemm_tree_nm_krn<lhsTy, rhsTy, resTy,
                                           OuterInnerDimsIndexerT,
                                           OuterInnerDimsIndexerT, wi_delta_m>;
                cgh.parallel_for<KernelName>(
                    ndRange,
                    GemmNoAtomicFunctorThreadNM<
                        lhsTy, rhsTy, resTy, LocAccT1, LocAccT2,
                        OuterInnerDimsIndexerT, OuterInnerDimsIndexerT,
                        wi_delta_n, wi_delta_m>(
                        lhs_tp, rhs_tp, res_tp, local_A_block, local_B_block, n,
                        wg_delta_n, k, k_blocks, wi_delta_k, m, m_blocks,
                        wg_delta_m, lhs_indexer, rhs_indexer, res_indexer));
            }
        });
        return gemm_ev;
    }
    else {
        using ReductionOpT = sycl::plus<resTy>;
        constexpr resTy identity_val =
            sycl::known_identity<ReductionOpT, resTy>::value;

        size_t iter_nelems = n * m;
        size_t reduction_nelems = (k + wi_delta_k - 1) / wi_delta_k;

        // more than one work-groups is needed, requires a temporary
        // wi_delta_k elements processed along k, so if more to
        // process use multiple
        const auto &sg_sizes =
            dev.get_info<sycl::info::device::sub_group_sizes>();
        size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

        if (reduction_nelems < wg) {
            resTy *tmp = sycl::malloc_device<resTy>(
                iter_nelems * reduction_nelems, exec_q);
            if (!tmp) {
                throw std::runtime_error("Unable to allocate device memory");
            }
            sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(depends);

                using OuterInnerDimsIndexerT =
                    dpctl::tensor::offset_utils::StridedIndexer;
                OuterInnerDimsIndexerT lhs_indexer(
                    inner_nd + lhs_outer_nd, 0, lhs_outer_inner_shapes_strides);
                OuterInnerDimsIndexerT rhs_indexer(
                    inner_nd + rhs_outer_nd, 0, rhs_outer_inner_shapes_strides);
                using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
                ResIndexerT res_indexer{};

                size_t lws = wg_delta_n * wg_delta_m;

                size_t n_blocks = ((n + wi_delta_n * wg_delta_n - 1) /
                                   (wi_delta_n * wg_delta_n));
                size_t k_blocks = ((k + wi_delta_k - 1) / wi_delta_k);
                size_t m_blocks = ((m + wi_delta_m * wg_delta_m - 1) /
                                   (wi_delta_m * wg_delta_m));

                auto gwsRange =
                    sycl::range<1>(n_blocks * m_blocks * k_blocks * lws);
                auto lwsRange = sycl::range<1>(lws);

                auto ndRange = sycl::nd_range<1>(gwsRange, lwsRange);

                if constexpr (wi_delta_m == 1) {
                    using LocAccT1 = sycl::local_accessor<resTy, 1>;
                    LocAccT1 local_A_block(
                        sycl::range<1>((wi_delta_n * wg_delta_n) * wi_delta_k),
                        cgh);
                    using LocAccT2 = sycl::local_accessor<resTy, 1>;
                    LocAccT2 local_B_block(wi_delta_k * wg_delta_m, cgh);

                    using KernelName =
                        class gemm_tree_nm_krn<lhsTy, rhsTy, resTy,
                                               OuterInnerDimsIndexerT,
                                               ResIndexerT, wi_delta_m>;
                    cgh.parallel_for<KernelName>(
                        ndRange,
                        GemmNoAtomicFunctorThreadNM<
                            lhsTy, rhsTy, resTy, LocAccT1, LocAccT2,
                            OuterInnerDimsIndexerT, ResIndexerT, wi_delta_n,
                            wi_delta_m>(
                            lhs_tp, rhs_tp, tmp, local_A_block, local_B_block,
                            n, wg_delta_n, k, k_blocks, wi_delta_k, m, m_blocks,
                            wg_delta_m, lhs_indexer, rhs_indexer, res_indexer));
                }
                else {
                    using LocAccT1 = sycl::local_accessor<resTy, 1>;
                    LocAccT1 local_A_block(
                        sycl::range<1>((wi_delta_n * wg_delta_n) * wi_delta_k),
                        cgh);
                    using LocAccT2 =
                        sycl::local_accessor<sycl::vec<resTy, wi_delta_m>, 1>;
                    LocAccT2 local_B_block(
                        sycl::range<1>(wi_delta_k * wg_delta_m), cgh);

                    using KernelName =
                        class gemm_tree_nm_krn<lhsTy, rhsTy, resTy,
                                               OuterInnerDimsIndexerT,
                                               ResIndexerT, wi_delta_m>;
                    cgh.parallel_for<KernelName>(
                        ndRange,
                        GemmNoAtomicFunctorThreadNM<
                            lhsTy, rhsTy, resTy, LocAccT1, LocAccT2,
                            OuterInnerDimsIndexerT, ResIndexerT, wi_delta_n,
                            wi_delta_m>(
                            lhs_tp, rhs_tp, tmp, local_A_block, local_B_block,
                            n, wg_delta_n, k, k_blocks, wi_delta_k, m, m_blocks,
                            wg_delta_m, lhs_indexer, rhs_indexer, res_indexer));
                }
            });
            sycl::event red_ev = exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(gemm_ev);

                using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
                using ResIndexerT = dpctl::tensor::offset_utils::StridedIndexer;
                using InputOutputIterIndexerT =
                    dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                        NoOpIndexerT, ResIndexerT>;
                using ReductionIndexerT =
                    dpctl::tensor::offset_utils::Strided1DIndexer;

                ResIndexerT res_iter_indexer{res_nd, 0, res_shapes_strides};
                InputOutputIterIndexerT in_out_iter_indexer{NoOpIndexerT{},
                                                            res_iter_indexer};
                ReductionIndexerT reduction_indexer{
                    0, static_cast<py::ssize_t>(reduction_nelems),
                    static_cast<py::ssize_t>(iter_nelems)};

                sycl::range<1> iter_range{iter_nelems};

                cgh.parallel_for<class gemm_reduction_seq_strided_krn<
                    resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                    ReductionIndexerT>>(
                    iter_range, SequentialReduction<resTy, resTy, ReductionOpT,
                                                    InputOutputIterIndexerT,
                                                    ReductionIndexerT>(
                                    tmp, res_tp, ReductionOpT(), identity_val,
                                    in_out_iter_indexer, reduction_indexer,
                                    reduction_nelems));
            });
            sycl::event cleanup_host_task_event =
                exec_q.submit([&](sycl::handler &cgh) {
                    cgh.depends_on(red_ev);
                    const sycl::context &ctx = exec_q.get_context();

                    cgh.host_task([ctx, tmp] { sycl::free(tmp, ctx); });
                });
            return cleanup_host_task_event;
        }

        // max_max_wg prevents running out of resources on CPU
        constexpr size_t max_max_wg = 2048;
        size_t max_wg = std::min(
            max_max_wg,
            dev.get_info<sycl::info::device::max_work_group_size>() / 2);

        constexpr size_t preferred_reductions_per_wi = 8;
        size_t reductions_per_wi(preferred_reductions_per_wi);

        size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);
        assert(reduction_groups > 1);

        resTy *partially_reduced_tmp = sycl::malloc_device<resTy>(
            iter_nelems * (/* temp */ reduction_nelems +
                           /* first reduction temp */ reduction_groups),
            exec_q);
        resTy *partially_reduced_tmp2 = nullptr;

        if (partially_reduced_tmp == nullptr) {
            throw std::runtime_error("Unable to allocate device_memory");
        }
        else {
            partially_reduced_tmp2 =
                partially_reduced_tmp + reduction_nelems * iter_nelems;
        }

        sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            using OuterInnerDimsIndexerT =
                dpctl::tensor::offset_utils::StridedIndexer;
            OuterInnerDimsIndexerT lhs_indexer(inner_nd + lhs_outer_nd, 0,
                                               lhs_outer_inner_shapes_strides);
            OuterInnerDimsIndexerT rhs_indexer(inner_nd + rhs_outer_nd, 0,
                                               rhs_outer_inner_shapes_strides);
            using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            ResIndexerT res_indexer{};

            size_t lws = wg_delta_n * wg_delta_m;

            size_t n_blocks =
                ((n + wi_delta_n * wg_delta_n - 1) / (wi_delta_n * wg_delta_n));
            size_t k_blocks = ((k + wi_delta_k - 1) / wi_delta_k);
            size_t m_blocks =
                ((m + wi_delta_m * wg_delta_m - 1) / (wi_delta_m * wg_delta_m));

            auto gwsRange =
                sycl::range<1>(n_blocks * m_blocks * k_blocks * lws);
            auto lwsRange = sycl::range<1>(lws);

            auto ndRange = sycl::nd_range<1>(gwsRange, lwsRange);

            if constexpr (wi_delta_m == 1) {
                using LocAccT1 = sycl::local_accessor<resTy, 1>;
                LocAccT1 local_A_block(
                    sycl::range<1>((wi_delta_n * wg_delta_n) * wi_delta_k),
                    cgh);
                using LocAccT2 = sycl::local_accessor<resTy, 1>;
                LocAccT2 local_B_block(wi_delta_k * wg_delta_m, cgh);

                using KernelName =
                    class gemm_tree_nm_krn<lhsTy, rhsTy, resTy,
                                           OuterInnerDimsIndexerT, ResIndexerT,
                                           wi_delta_m>;
                cgh.parallel_for<KernelName>(
                    ndRange, GemmNoAtomicFunctorThreadNM<
                                 lhsTy, rhsTy, resTy, LocAccT1, LocAccT2,
                                 OuterInnerDimsIndexerT, ResIndexerT,
                                 wi_delta_n, wi_delta_m>(
                                 lhs_tp, rhs_tp, partially_reduced_tmp,
                                 local_A_block, local_B_block, n, wg_delta_n, k,
                                 k_blocks, wi_delta_k, m, m_blocks, wg_delta_m,
                                 lhs_indexer, rhs_indexer, res_indexer));
            }
            else {
                using LocAccT1 = sycl::local_accessor<resTy, 1>;
                LocAccT1 local_A_block(
                    sycl::range<1>((wi_delta_n * wg_delta_n) * wi_delta_k),
                    cgh);
                using LocAccT2 =
                    sycl::local_accessor<sycl::vec<resTy, wi_delta_m>, 1>;
                LocAccT2 local_B_block(sycl::range<1>(wi_delta_k * wg_delta_m),
                                       cgh);

                using KernelName =
                    class gemm_tree_nm_krn<lhsTy, rhsTy, resTy,
                                           OuterInnerDimsIndexerT, ResIndexerT,
                                           wi_delta_m>;
                cgh.parallel_for<KernelName>(
                    ndRange, GemmNoAtomicFunctorThreadNM<
                                 lhsTy, rhsTy, resTy, LocAccT1, LocAccT2,
                                 OuterInnerDimsIndexerT, ResIndexerT,
                                 wi_delta_n, wi_delta_m>(
                                 lhs_tp, rhs_tp, partially_reduced_tmp,
                                 local_A_block, local_B_block, n, wg_delta_n, k,
                                 k_blocks, wi_delta_k, m, m_blocks, wg_delta_m,
                                 lhs_indexer, rhs_indexer, res_indexer));
            }
        });

        sycl::event red_ev = tree_reduction_for_gemm<resTy, ReductionOpT>(
            exec_q, partially_reduced_tmp, partially_reduced_tmp2, res_tp,
            identity_val, iter_nelems, reduction_nelems, reduction_groups, wg,
            max_wg, preferred_reductions_per_wi, reductions_per_wi, res_nd, 0,
            res_shapes_strides, {gemm_ev});

        sycl::event cleanup_host_task_event =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(red_ev);
                const sycl::context &ctx = exec_q.get_context();

                cgh.host_task([ctx, partially_reduced_tmp] {
                    sycl::free(partially_reduced_tmp, ctx);
                });
            });

        return cleanup_host_task_event;
    }
}

template <typename T1, typename T2, typename T3> class gemm_tree_empty_krn;

template <typename lhsTy, typename rhsTy, typename resTy>
sycl::event gemm_tree_impl(sycl::queue &exec_q,
                           const char *lhs_cp,
                           const char *rhs_cp,
                           char *res_cp,
                           size_t n,
                           size_t k,
                           size_t m,
                           int inner_nd,
                           int lhs_outer_nd,
                           const py::ssize_t *lhs_outer_inner_shapes_strides,
                           int rhs_outer_nd,
                           const py::ssize_t *rhs_outer_inner_shapes_strides,
                           int res_nd,
                           const py::ssize_t *res_shapes_strides,
                           std::vector<sycl::event> const &depends = {})
{
    const lhsTy *lhs_tp = reinterpret_cast<const lhsTy *>(lhs_cp);
    const rhsTy *rhs_tp = reinterpret_cast<const rhsTy *>(rhs_cp);
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp);

    if (k == 0) {
        sycl::event gemm_no_reduction_ev =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(depends);

                using IndexerT = dpctl::tensor::offset_utils::StridedIndexer;
                IndexerT res_indexer(res_nd, 0, res_shapes_strides);
                using InitKernelName =
                    class gemm_tree_empty_krn<lhsTy, rhsTy, resTy>;
                cgh.parallel_for<InitKernelName>(
                    sycl::range<1>(n * m), [=](sycl::id<1> id) {
                        auto res_offset = res_indexer(id[0]);
                        res_tp[res_offset] = resTy(0);
                    });
            });
        return gemm_no_reduction_ev;
    }

    if (exec_q.get_device().is_cpu()) {
        using dpctl::tensor::type_utils::is_complex;
        if constexpr (!is_complex<resTy>::value) {
            if (m == 1) {
                return gemm_tree_k_impl<lhsTy, rhsTy, resTy, 1>(
                    exec_q, lhs_tp, rhs_tp, res_tp, n, k, m, inner_nd,
                    lhs_outer_nd, lhs_outer_inner_shapes_strides, rhs_outer_nd,
                    rhs_outer_inner_shapes_strides, res_nd, res_shapes_strides,
                    depends);
            }
            else {
                return gemm_tree_nm_impl<lhsTy, rhsTy, resTy, 4>(
                    exec_q, lhs_tp, rhs_tp, res_tp, n, k, m, inner_nd,
                    lhs_outer_nd, lhs_outer_inner_shapes_strides, rhs_outer_nd,
                    rhs_outer_inner_shapes_strides, res_nd, res_shapes_strides,
                    depends);
            }
        }
        else {
            return gemm_tree_nm_impl<lhsTy, rhsTy, resTy, 1>(
                exec_q, lhs_tp, rhs_tp, res_tp, n, k, m, inner_nd, lhs_outer_nd,
                lhs_outer_inner_shapes_strides, rhs_outer_nd,
                rhs_outer_inner_shapes_strides, res_nd, res_shapes_strides,
                depends);
        }
    }
    else {
        if ((k > n && k > m) || m == 1) {
            using dpctl::tensor::type_utils::is_complex;
            if constexpr (!is_complex<resTy>::value) {
                if (m == 1) {
                    return gemm_tree_k_impl<lhsTy, rhsTy, resTy, 1>(
                        exec_q, lhs_tp, rhs_tp, res_tp, n, k, m, inner_nd,
                        lhs_outer_nd, lhs_outer_inner_shapes_strides,
                        rhs_outer_nd, rhs_outer_inner_shapes_strides, res_nd,
                        res_shapes_strides, depends);
                }
                else {
                    return gemm_tree_k_impl<lhsTy, rhsTy, resTy, 2>(
                        exec_q, lhs_tp, rhs_tp, res_tp, n, k, m, inner_nd,
                        lhs_outer_nd, lhs_outer_inner_shapes_strides,
                        rhs_outer_nd, rhs_outer_inner_shapes_strides, res_nd,
                        res_shapes_strides, depends);
                }
            }
            else {
                return gemm_tree_k_impl<lhsTy, rhsTy, resTy, 1>(
                    exec_q, lhs_tp, rhs_tp, res_tp, n, k, m, inner_nd,
                    lhs_outer_nd, lhs_outer_inner_shapes_strides, rhs_outer_nd,
                    rhs_outer_inner_shapes_strides, res_nd, res_shapes_strides,
                    depends);
            }
        }
        else { // m > 1, n > k or m > k
            using dpctl::tensor::type_utils::is_complex;
            if constexpr (!is_complex<resTy>::value) {
                return gemm_tree_nm_impl<lhsTy, rhsTy, resTy, 4>(
                    exec_q, lhs_tp, rhs_tp, res_tp, n, k, m, inner_nd,
                    lhs_outer_nd, lhs_outer_inner_shapes_strides, rhs_outer_nd,
                    rhs_outer_inner_shapes_strides, res_nd, res_shapes_strides,
                    depends);
            }
            else {
                return gemm_tree_nm_impl<lhsTy, rhsTy, resTy, 1>(
                    exec_q, lhs_tp, rhs_tp, res_tp, n, k, m, inner_nd,
                    lhs_outer_nd, lhs_outer_inner_shapes_strides, rhs_outer_nd,
                    rhs_outer_inner_shapes_strides, res_nd, res_shapes_strides,
                    depends);
            }
        }
    }
}

template <typename lhsTy, typename rhsTy, typename resTy, size_t m_groups>
sycl::event gemm_contig_tree_k_impl(sycl::queue &exec_q,
                                    const lhsTy *lhs_tp,
                                    const rhsTy *rhs_tp,
                                    resTy *res_tp,
                                    size_t n,
                                    size_t k,
                                    size_t m,
                                    std::vector<sycl::event> const &depends)
{
    size_t delta_k(4);
    size_t n_wi(4);
    size_t delta_n(4);

    const sycl::device &dev = exec_q.get_device();
    const size_t local_mem_size =
        dev.get_info<sycl::info::device::local_mem_size>();
    const size_t reserved_slm_size = 512;

    gemm_detail::scale_gemm_k_parameters<resTy, m_groups>(
        local_mem_size, reserved_slm_size, delta_k,
        n_wi,   // modified by reference
        delta_n // modified by reference
    );

    sycl::event gemm_ev;
    if (k <= (delta_k * n_wi)) {
        gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            using OuterInnerDimsIndexerT =
                dpctl::tensor::offset_utils::NoOpIndexer;
            OuterInnerDimsIndexerT lhs_indexer{};
            OuterInnerDimsIndexerT rhs_indexer{};
            OuterInnerDimsIndexerT res_indexer{};

            size_t n_blocks = (n + delta_n - 1) / delta_n;
            size_t k_blocks = (k + n_wi * delta_k - 1) / (n_wi * delta_k);
            size_t m_blocks = (m + m_groups - 1) / m_groups;

            size_t lws = delta_n * delta_k;

            auto gRange = sycl::range<1>(n_blocks * m_blocks * k_blocks * lws);
            auto lRange = sycl::range<1>(lws);

            auto ndRange = sycl::nd_range<1>(gRange, lRange);

            if constexpr (m_groups == 1) {
                using LocAccT = sycl::local_accessor<resTy, 1>;
                LocAccT local_B_block(n_wi * delta_k, cgh);
                LocAccT workspace(delta_n * delta_k, cgh);
                using KernelName =
                    class gemm_tree_k_krn<lhsTy, rhsTy, resTy,
                                          OuterInnerDimsIndexerT,
                                          OuterInnerDimsIndexerT, m_groups>;
                cgh.parallel_for<KernelName>(
                    ndRange,
                    GemmNoAtomicFunctorThreadK<
                        lhsTy, rhsTy, resTy, LocAccT, OuterInnerDimsIndexerT,
                        OuterInnerDimsIndexerT, m_groups>(
                        lhs_tp, rhs_tp, res_tp, workspace, local_B_block, n,
                        n_blocks, delta_n, k, k_blocks, delta_k, n_wi, m,
                        lhs_indexer, rhs_indexer, res_indexer));
            }
            else {
                using LocAccT =
                    sycl::local_accessor<sycl::vec<resTy, m_groups>, 1>;
                LocAccT local_B_block(n_wi * delta_k, cgh);
                LocAccT workspace(delta_n * delta_k, cgh);
                using KernelName =
                    class gemm_tree_k_krn<lhsTy, rhsTy, resTy,
                                          OuterInnerDimsIndexerT,
                                          OuterInnerDimsIndexerT, m_groups>;
                cgh.parallel_for<KernelName>(
                    ndRange,
                    GemmNoAtomicFunctorThreadK<
                        lhsTy, rhsTy, resTy, LocAccT, OuterInnerDimsIndexerT,
                        OuterInnerDimsIndexerT, m_groups>(
                        lhs_tp, rhs_tp, res_tp, workspace, local_B_block, n,
                        n_blocks, delta_n, k, k_blocks, delta_k, n_wi, m,
                        lhs_indexer, rhs_indexer, res_indexer));
            }
        });
        return gemm_ev;
    }
    else {
        using ReductionOpT = sycl::plus<resTy>;
        constexpr resTy identity_val =
            sycl::known_identity<ReductionOpT, resTy>::value;

        size_t iter_nelems = n * m;
        size_t reduction_nelems = (k + delta_k * n_wi - 1) / (delta_k * n_wi);

        // more than one work-groups is needed, requires a
        // temporary delta_k * n_wi elements processed along k,
        // so if more to process use multiple
        const auto &sg_sizes =
            dev.get_info<sycl::info::device::sub_group_sizes>();
        size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

        if (reduction_nelems < wg) {
            resTy *tmp = sycl::malloc_device<resTy>(
                iter_nelems * reduction_nelems, exec_q);
            if (!tmp) {
                throw std::runtime_error("Unable to allocate device memory");
            }
            sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(depends);

                using OuterInnerDimsIndexerT =
                    dpctl::tensor::offset_utils::NoOpIndexer;
                OuterInnerDimsIndexerT lhs_indexer{};
                OuterInnerDimsIndexerT rhs_indexer{};
                OuterInnerDimsIndexerT res_indexer{};

                size_t n_blocks = (n + delta_n - 1) / delta_n;
                size_t k_blocks = (k + n_wi * delta_k - 1) / (n_wi * delta_k);
                size_t m_blocks = (m + m_groups - 1) / m_groups;

                size_t lws = delta_n * delta_k;

                auto gRange =
                    sycl::range<1>(n_blocks * m_blocks * k_blocks * lws);
                auto lRange = sycl::range<1>(lws);

                auto ndRange = sycl::nd_range<1>(gRange, lRange);
                if constexpr (m_groups == 1) {
                    using LocAccT = sycl::local_accessor<resTy, 1>;
                    LocAccT local_B_block(n_wi * delta_k, cgh);
                    LocAccT workspace(delta_n * delta_k, cgh);
                    using KernelName =
                        class gemm_tree_k_krn<lhsTy, rhsTy, resTy,
                                              OuterInnerDimsIndexerT,
                                              OuterInnerDimsIndexerT, m_groups>;
                    cgh.parallel_for<KernelName>(
                        ndRange,
                        GemmNoAtomicFunctorThreadK<lhsTy, rhsTy, resTy, LocAccT,
                                                   OuterInnerDimsIndexerT,
                                                   OuterInnerDimsIndexerT,
                                                   m_groups>(
                            lhs_tp, rhs_tp, tmp, workspace, local_B_block, n,
                            n_blocks, delta_n, k, k_blocks, delta_k, n_wi, m,
                            lhs_indexer, rhs_indexer, res_indexer));
                }
                else {
                    using LocAccT =
                        sycl::local_accessor<sycl::vec<resTy, m_groups>, 1>;
                    LocAccT local_B_block(n_wi * delta_k, cgh);
                    LocAccT workspace(delta_n * delta_k, cgh);
                    using KernelName =
                        class gemm_tree_k_krn<lhsTy, rhsTy, resTy,
                                              OuterInnerDimsIndexerT,
                                              OuterInnerDimsIndexerT, m_groups>;
                    cgh.parallel_for<KernelName>(
                        ndRange,
                        GemmNoAtomicFunctorThreadK<lhsTy, rhsTy, resTy, LocAccT,
                                                   OuterInnerDimsIndexerT,
                                                   OuterInnerDimsIndexerT,
                                                   m_groups>(
                            lhs_tp, rhs_tp, tmp, workspace, local_B_block, n,
                            n_blocks, delta_n, k, k_blocks, delta_k, n_wi, m,
                            lhs_indexer, rhs_indexer, res_indexer));
                }
            });
            sycl::event red_ev = exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(gemm_ev);

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

                sycl::range<1> iter_range{iter_nelems};

                cgh.parallel_for<class gemm_reduction_seq_strided_krn<
                    resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                    ReductionIndexerT>>(
                    iter_range, SequentialReduction<resTy, resTy, ReductionOpT,
                                                    InputOutputIterIndexerT,
                                                    ReductionIndexerT>(
                                    tmp, res_tp, ReductionOpT(), identity_val,
                                    in_out_iter_indexer, reduction_indexer,
                                    reduction_nelems));
            });
            sycl::event cleanup_host_task_event =
                exec_q.submit([&](sycl::handler &cgh) {
                    cgh.depends_on(red_ev);
                    const sycl::context &ctx = exec_q.get_context();

                    cgh.host_task([ctx, tmp] { sycl::free(tmp, ctx); });
                });
            return cleanup_host_task_event;
        }
        // max_max_wg prevents running out of resources on CPU
        constexpr size_t max_max_wg = 2048;
        size_t max_wg = std::min(
            max_max_wg,
            dev.get_info<sycl::info::device::max_work_group_size>() / 2);

        constexpr size_t preferred_reductions_per_wi = 8;
        size_t reductions_per_wi(preferred_reductions_per_wi);

        size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);
        assert(reduction_groups > 1);

        resTy *partially_reduced_tmp = sycl::malloc_device<resTy>(
            iter_nelems * (/* temp */ reduction_nelems +
                           /* first reduction temp */ reduction_groups),
            exec_q);
        resTy *partially_reduced_tmp2 = nullptr;

        if (partially_reduced_tmp == nullptr) {
            throw std::runtime_error("Unable to allocate device_memory");
        }
        else {
            partially_reduced_tmp2 =
                partially_reduced_tmp + reduction_nelems * iter_nelems;
        }

        sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            using OuterInnerDimsIndexerT =
                dpctl::tensor::offset_utils::NoOpIndexer;
            OuterInnerDimsIndexerT lhs_indexer{};
            OuterInnerDimsIndexerT rhs_indexer{};
            OuterInnerDimsIndexerT res_indexer{};

            size_t n_blocks = (n + delta_n - 1) / delta_n;
            size_t k_blocks = (k + n_wi * delta_k - 1) / (n_wi * delta_k);
            size_t m_blocks = (m + m_groups - 1) / m_groups;

            size_t lws = delta_n * delta_k;

            auto gRange = sycl::range<1>(n_blocks * m_blocks * k_blocks * lws);
            auto lRange = sycl::range<1>(lws);

            auto ndRange = sycl::nd_range<1>(gRange, lRange);

            if constexpr (m_groups == 1) {
                using LocAccT = sycl::local_accessor<resTy, 1>;
                LocAccT local_B_block(n_wi * delta_k, cgh);
                LocAccT workspace(delta_n * delta_k, cgh);
                using KernelName =
                    class gemm_tree_k_krn<lhsTy, rhsTy, resTy,
                                          OuterInnerDimsIndexerT,
                                          OuterInnerDimsIndexerT, m_groups>;
                cgh.parallel_for<KernelName>(
                    ndRange,
                    GemmNoAtomicFunctorThreadK<
                        lhsTy, rhsTy, resTy, LocAccT, OuterInnerDimsIndexerT,
                        OuterInnerDimsIndexerT, m_groups>(
                        lhs_tp, rhs_tp, partially_reduced_tmp, workspace,
                        local_B_block, n, n_blocks, delta_n, k, k_blocks,
                        delta_k, n_wi, m, lhs_indexer, rhs_indexer,
                        res_indexer));
            }
            else {
                using LocAccT =
                    sycl::local_accessor<sycl::vec<resTy, m_groups>, 1>;
                LocAccT local_B_block(n_wi * delta_k, cgh);
                LocAccT workspace(delta_n * delta_k, cgh);
                using KernelName =
                    class gemm_tree_k_krn<lhsTy, rhsTy, resTy,
                                          OuterInnerDimsIndexerT,
                                          OuterInnerDimsIndexerT, m_groups>;
                cgh.parallel_for<KernelName>(
                    ndRange,
                    GemmNoAtomicFunctorThreadK<
                        lhsTy, rhsTy, resTy, LocAccT, OuterInnerDimsIndexerT,
                        OuterInnerDimsIndexerT, m_groups>(
                        lhs_tp, rhs_tp, partially_reduced_tmp, workspace,
                        local_B_block, n, n_blocks, delta_n, k, k_blocks,
                        delta_k, n_wi, m, lhs_indexer, rhs_indexer,
                        res_indexer));
            }
        });
        // tree_reduction_for_gemm_contig returns sycl::event
        // for reduction
        sycl::event red_ev =
            tree_reduction_for_gemm_contig<resTy, ReductionOpT>(
                exec_q, partially_reduced_tmp, partially_reduced_tmp2, res_tp,
                identity_val, iter_nelems, reduction_nelems, reduction_groups,
                wg, max_wg, preferred_reductions_per_wi, reductions_per_wi,
                {gemm_ev});

        sycl::event cleanup_host_task_event =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(red_ev);
                const sycl::context &ctx = exec_q.get_context();

                cgh.host_task([ctx, partially_reduced_tmp] {
                    sycl::free(partially_reduced_tmp, ctx);
                });
            });

        return cleanup_host_task_event;
    }
}

template <typename lhsTy, typename rhsTy, typename resTy, int wi_delta_m>
sycl::event gemm_contig_tree_nm_impl(sycl::queue &exec_q,
                                     const lhsTy *lhs_tp,
                                     const rhsTy *rhs_tp,
                                     resTy *res_tp,
                                     size_t n,
                                     size_t k,
                                     size_t m,
                                     std::vector<sycl::event> const &depends)
{
    constexpr int wi_delta_n = 2;
    size_t wg_delta_n(16); // rows of A processed in WG
    size_t wg_delta_m(16); // rows of B processed in WG
    size_t wi_delta_k(64); // Elements in K dimension processed by WI

    const sycl::device &dev = exec_q.get_device();
    const size_t local_mem_size =
        dev.get_info<sycl::info::device::local_mem_size>();
    const size_t reserved_slm_size = 512;

    gemm_detail::scale_gemm_nm_parameters<resTy, wi_delta_m>(
        local_mem_size, reserved_slm_size, wi_delta_n,
        wi_delta_k, // modified by reference
        wg_delta_n, // modified by reference
        wg_delta_m  // modified by reference
    );

    // each group processes delta_k items in a column,
    // so no need to allocate temp memory if one group needed
    if (k <= wi_delta_k) {
        sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            using OuterInnerDimsIndexerT =
                dpctl::tensor::offset_utils::NoOpIndexer;
            OuterInnerDimsIndexerT lhs_indexer{};
            OuterInnerDimsIndexerT rhs_indexer{};
            OuterInnerDimsIndexerT res_indexer{};

            size_t lws = wg_delta_n * wg_delta_m;

            size_t n_blocks =
                ((n + wi_delta_n * wg_delta_n - 1) / (wi_delta_n * wg_delta_n));
            size_t k_blocks = ((k + wi_delta_k - 1) / wi_delta_k);
            size_t m_blocks =
                ((m + wi_delta_m * wg_delta_m - 1) / (wi_delta_m * wg_delta_m));

            auto gwsRange =
                sycl::range<1>(n_blocks * m_blocks * k_blocks * lws);
            auto lwsRange = sycl::range<1>(lws);

            auto ndRange = sycl::nd_range<1>(gwsRange, lwsRange);

            if constexpr (wi_delta_m == 1) {
                using LocAccT1 = sycl::local_accessor<resTy, 1>;
                LocAccT1 local_A_block(
                    sycl::range<1>((wi_delta_n * wg_delta_n) * wi_delta_k),
                    cgh);
                using LocAccT2 = sycl::local_accessor<resTy, 1>;
                LocAccT2 local_B_block(wi_delta_k * wg_delta_m, cgh);

                using KernelName =
                    class gemm_tree_nm_krn<lhsTy, rhsTy, resTy,
                                           OuterInnerDimsIndexerT,
                                           OuterInnerDimsIndexerT, wi_delta_m>;
                cgh.parallel_for<KernelName>(
                    ndRange,
                    GemmNoAtomicFunctorThreadNM<
                        lhsTy, rhsTy, resTy, LocAccT1, LocAccT2,
                        OuterInnerDimsIndexerT, OuterInnerDimsIndexerT,
                        wi_delta_n, wi_delta_m>(
                        lhs_tp, rhs_tp, res_tp, local_A_block, local_B_block, n,
                        wg_delta_n, k, k_blocks, wi_delta_k, m, m_blocks,
                        wg_delta_m, lhs_indexer, rhs_indexer, res_indexer));
            }
            else {
                using LocAccT1 = sycl::local_accessor<resTy, 1>;
                LocAccT1 local_A_block(
                    sycl::range<1>((wi_delta_n * wg_delta_n) * wi_delta_k),
                    cgh);
                using LocAccT2 =
                    sycl::local_accessor<sycl::vec<resTy, wi_delta_m>, 1>;
                LocAccT2 local_B_block(sycl::range<1>(wi_delta_k * wg_delta_m),
                                       cgh);

                using KernelName =
                    class gemm_tree_nm_krn<lhsTy, rhsTy, resTy,
                                           OuterInnerDimsIndexerT,
                                           OuterInnerDimsIndexerT, wi_delta_m>;
                cgh.parallel_for<KernelName>(
                    ndRange,
                    GemmNoAtomicFunctorThreadNM<
                        lhsTy, rhsTy, resTy, LocAccT1, LocAccT2,
                        OuterInnerDimsIndexerT, OuterInnerDimsIndexerT,
                        wi_delta_n, wi_delta_m>(
                        lhs_tp, rhs_tp, res_tp, local_A_block, local_B_block, n,
                        wg_delta_n, k, k_blocks, wi_delta_k, m, m_blocks,
                        wg_delta_m, lhs_indexer, rhs_indexer, res_indexer));
            }
        });
        return gemm_ev;
    }
    else {
        using ReductionOpT = sycl::plus<resTy>;
        constexpr resTy identity_val =
            sycl::known_identity<ReductionOpT, resTy>::value;

        size_t iter_nelems = n * m;
        size_t reduction_nelems = (k + wi_delta_k - 1) / wi_delta_k;

        // more than one work-groups is needed, requires a temporary
        // wi_delta_k elements processed along k, so if more to
        // process use multiple
        const auto &sg_sizes =
            dev.get_info<sycl::info::device::sub_group_sizes>();
        size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

        if (reduction_nelems < wg) {
            resTy *tmp = sycl::malloc_device<resTy>(
                iter_nelems * reduction_nelems, exec_q);
            if (!tmp) {
                throw std::runtime_error("Unable to allocate device memory");
            }
            sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(depends);

                using OuterInnerDimsIndexerT =
                    dpctl::tensor::offset_utils::NoOpIndexer;
                OuterInnerDimsIndexerT lhs_indexer{};
                OuterInnerDimsIndexerT rhs_indexer{};
                OuterInnerDimsIndexerT res_indexer{};

                size_t lws = wg_delta_n * wg_delta_m;

                size_t n_blocks = ((n + wi_delta_n * wg_delta_n - 1) /
                                   (wi_delta_n * wg_delta_n));
                size_t k_blocks = ((k + wi_delta_k - 1) / wi_delta_k);
                size_t m_blocks = ((m + wi_delta_m * wg_delta_m - 1) /
                                   (wi_delta_m * wg_delta_m));

                auto gwsRange =
                    sycl::range<1>(n_blocks * m_blocks * k_blocks * lws);
                auto lwsRange = sycl::range<1>(lws);

                auto ndRange = sycl::nd_range<1>(gwsRange, lwsRange);

                if constexpr (wi_delta_m == 1) {
                    using LocAccT1 = sycl::local_accessor<resTy, 1>;
                    LocAccT1 local_A_block(
                        sycl::range<1>((wi_delta_n * wg_delta_n) * wi_delta_k),
                        cgh);
                    using LocAccT2 = sycl::local_accessor<resTy, 1>;
                    LocAccT2 local_B_block(wi_delta_k * wg_delta_m, cgh);

                    using KernelName = class gemm_tree_nm_krn<
                        lhsTy, rhsTy, resTy, OuterInnerDimsIndexerT,
                        OuterInnerDimsIndexerT, wi_delta_m>;
                    cgh.parallel_for<KernelName>(
                        ndRange,
                        GemmNoAtomicFunctorThreadNM<
                            lhsTy, rhsTy, resTy, LocAccT1, LocAccT2,
                            OuterInnerDimsIndexerT, OuterInnerDimsIndexerT,
                            wi_delta_n, wi_delta_m>(
                            lhs_tp, rhs_tp, tmp, local_A_block, local_B_block,
                            n, wg_delta_n, k, k_blocks, wi_delta_k, m, m_blocks,
                            wg_delta_m, lhs_indexer, rhs_indexer, res_indexer));
                }
                else {
                    using LocAccT1 = sycl::local_accessor<resTy, 1>;
                    LocAccT1 local_A_block(
                        sycl::range<1>((wi_delta_n * wg_delta_n) * wi_delta_k),
                        cgh);
                    using LocAccT2 =
                        sycl::local_accessor<sycl::vec<resTy, wi_delta_m>, 1>;
                    LocAccT2 local_B_block(
                        sycl::range<1>(wi_delta_k * wg_delta_m), cgh);

                    using KernelName = class gemm_tree_nm_krn<
                        lhsTy, rhsTy, resTy, OuterInnerDimsIndexerT,
                        OuterInnerDimsIndexerT, wi_delta_m>;
                    cgh.parallel_for<KernelName>(
                        ndRange,
                        GemmNoAtomicFunctorThreadNM<
                            lhsTy, rhsTy, resTy, LocAccT1, LocAccT2,
                            OuterInnerDimsIndexerT, OuterInnerDimsIndexerT,
                            wi_delta_n, wi_delta_m>(
                            lhs_tp, rhs_tp, tmp, local_A_block, local_B_block,
                            n, wg_delta_n, k, k_blocks, wi_delta_k, m, m_blocks,
                            wg_delta_m, lhs_indexer, rhs_indexer, res_indexer));
                }
            });
            sycl::event red_ev = exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(gemm_ev);

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

                sycl::range<1> iter_range{iter_nelems};

                cgh.parallel_for<class gemm_reduction_seq_strided_krn<
                    resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                    ReductionIndexerT>>(
                    iter_range, SequentialReduction<resTy, resTy, ReductionOpT,
                                                    InputOutputIterIndexerT,
                                                    ReductionIndexerT>(
                                    tmp, res_tp, ReductionOpT(), identity_val,
                                    in_out_iter_indexer, reduction_indexer,
                                    reduction_nelems));
            });
            sycl::event cleanup_host_task_event =
                exec_q.submit([&](sycl::handler &cgh) {
                    cgh.depends_on(red_ev);
                    const sycl::context &ctx = exec_q.get_context();

                    cgh.host_task([ctx, tmp] { sycl::free(tmp, ctx); });
                });
            return cleanup_host_task_event;
        }

        // max_max_wg prevents running out of resources on CPU
        constexpr size_t max_max_wg = 2048;
        size_t max_wg = std::min(
            max_max_wg,
            dev.get_info<sycl::info::device::max_work_group_size>() / 2);

        constexpr size_t preferred_reductions_per_wi = 8;
        size_t reductions_per_wi(preferred_reductions_per_wi);

        size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);
        assert(reduction_groups > 1);

        resTy *partially_reduced_tmp = sycl::malloc_device<resTy>(
            iter_nelems * (/* temp */ reduction_nelems +
                           /* first reduction temp */ reduction_groups),
            exec_q);
        resTy *partially_reduced_tmp2 = nullptr;

        if (partially_reduced_tmp == nullptr) {
            throw std::runtime_error("Unable to allocate device_memory");
        }
        else {
            partially_reduced_tmp2 =
                partially_reduced_tmp + reduction_nelems * iter_nelems;
        }

        sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            using OuterInnerDimsIndexerT =
                dpctl::tensor::offset_utils::NoOpIndexer;
            OuterInnerDimsIndexerT lhs_indexer{};
            OuterInnerDimsIndexerT rhs_indexer{};
            OuterInnerDimsIndexerT res_indexer{};

            size_t lws = wg_delta_n * wg_delta_m;

            size_t n_blocks =
                ((n + wi_delta_n * wg_delta_n - 1) / (wi_delta_n * wg_delta_n));
            size_t k_blocks = ((k + wi_delta_k - 1) / wi_delta_k);
            size_t m_blocks =
                ((m + wi_delta_m * wg_delta_m - 1) / (wi_delta_m * wg_delta_m));

            auto gwsRange =
                sycl::range<1>(n_blocks * m_blocks * k_blocks * lws);
            auto lwsRange = sycl::range<1>(lws);

            auto ndRange = sycl::nd_range<1>(gwsRange, lwsRange);

            if constexpr (wi_delta_m == 1) {
                using LocAccT1 = sycl::local_accessor<resTy, 1>;
                LocAccT1 local_A_block(
                    sycl::range<1>((wi_delta_n * wg_delta_n) * wi_delta_k),
                    cgh);
                using LocAccT2 = sycl::local_accessor<resTy, 1>;
                LocAccT2 local_B_block(wi_delta_k * wg_delta_m, cgh);

                using KernelName =
                    class gemm_tree_nm_krn<lhsTy, rhsTy, resTy,
                                           OuterInnerDimsIndexerT,
                                           OuterInnerDimsIndexerT, wi_delta_m>;
                cgh.parallel_for<KernelName>(
                    ndRange, GemmNoAtomicFunctorThreadNM<
                                 lhsTy, rhsTy, resTy, LocAccT1, LocAccT2,
                                 OuterInnerDimsIndexerT, OuterInnerDimsIndexerT,
                                 wi_delta_n, wi_delta_m>(
                                 lhs_tp, rhs_tp, partially_reduced_tmp,
                                 local_A_block, local_B_block, n, wg_delta_n, k,
                                 k_blocks, wi_delta_k, m, m_blocks, wg_delta_m,
                                 lhs_indexer, rhs_indexer, res_indexer));
            }
            else {
                using LocAccT1 = sycl::local_accessor<resTy, 1>;
                LocAccT1 local_A_block(
                    sycl::range<1>((wi_delta_n * wg_delta_n) * wi_delta_k),
                    cgh);
                using LocAccT2 =
                    sycl::local_accessor<sycl::vec<resTy, wi_delta_m>, 1>;
                LocAccT2 local_B_block(sycl::range<1>(wi_delta_k * wg_delta_m),
                                       cgh);

                using KernelName =
                    class gemm_tree_nm_krn<lhsTy, rhsTy, resTy,
                                           OuterInnerDimsIndexerT,
                                           OuterInnerDimsIndexerT, wi_delta_m>;
                cgh.parallel_for<KernelName>(
                    ndRange, GemmNoAtomicFunctorThreadNM<
                                 lhsTy, rhsTy, resTy, LocAccT1, LocAccT2,
                                 OuterInnerDimsIndexerT, OuterInnerDimsIndexerT,
                                 wi_delta_n, wi_delta_m>(
                                 lhs_tp, rhs_tp, partially_reduced_tmp,
                                 local_A_block, local_B_block, n, wg_delta_n, k,
                                 k_blocks, wi_delta_k, m, m_blocks, wg_delta_m,
                                 lhs_indexer, rhs_indexer, res_indexer));
            }
        });

        sycl::event red_ev =
            tree_reduction_for_gemm_contig<resTy, ReductionOpT>(
                exec_q, partially_reduced_tmp, partially_reduced_tmp2, res_tp,
                identity_val, iter_nelems, reduction_nelems, reduction_groups,
                wg, max_wg, preferred_reductions_per_wi, reductions_per_wi,
                {gemm_ev});

        sycl::event cleanup_host_task_event =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(red_ev);
                const sycl::context &ctx = exec_q.get_context();

                cgh.host_task([ctx, partially_reduced_tmp] {
                    sycl::free(partially_reduced_tmp, ctx);
                });
            });

        return cleanup_host_task_event;
    }
}

template <typename lhsTy, typename rhsTy, typename resTy>
sycl::event gemm_contig_tree_impl(sycl::queue &exec_q,
                                  const char *lhs_cp,
                                  const char *rhs_cp,
                                  char *res_cp,
                                  size_t n,
                                  size_t k,
                                  size_t m,
                                  std::vector<sycl::event> const &depends = {})
{
    const lhsTy *lhs_tp = reinterpret_cast<const lhsTy *>(lhs_cp);
    const rhsTy *rhs_tp = reinterpret_cast<const rhsTy *>(rhs_cp);
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp);

    if (k == 0) {
        sycl::event gemm_no_reduction_ev =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(depends);
                cgh.fill<resTy>(res_tp, resTy(0), n * m);
            });
        return gemm_no_reduction_ev;
    }

    if (exec_q.get_device().is_cpu()) {
        using dpctl::tensor::type_utils::is_complex;
        if constexpr (!is_complex<resTy>::value) {
            if (m == 1) {
                return gemm_contig_tree_k_impl<lhsTy, rhsTy, resTy, 1>(
                    exec_q, lhs_tp, rhs_tp, res_tp, n, k, m, depends);
            }
            else {
                return gemm_contig_tree_nm_impl<lhsTy, rhsTy, resTy, 4>(
                    exec_q, lhs_tp, rhs_tp, res_tp, n, k, m, depends);
            }
        }
        else {
            return gemm_contig_tree_nm_impl<lhsTy, rhsTy, resTy, 1>(
                exec_q, lhs_tp, rhs_tp, res_tp, n, k, m, depends);
        }
    }
    else {
        if ((k > n && k > m) || m == 1) {
            using dpctl::tensor::type_utils::is_complex;
            if constexpr (!is_complex<resTy>::value) {
                if (m == 1) {
                    return gemm_contig_tree_k_impl<lhsTy, rhsTy, resTy, 1>(
                        exec_q, lhs_tp, rhs_tp, res_tp, n, k, m, depends);
                }
                else {
                    return gemm_contig_tree_k_impl<lhsTy, rhsTy, resTy, 2>(
                        exec_q, lhs_tp, rhs_tp, res_tp, n, k, m, depends);
                }
            }
            else {
                return gemm_contig_tree_k_impl<lhsTy, rhsTy, resTy, 1>(
                    exec_q, lhs_tp, rhs_tp, res_tp, n, k, m, depends);
            }
        }
        else { // m > 1, n > k or m > k
            using dpctl::tensor::type_utils::is_complex;
            if constexpr (!is_complex<resTy>::value) {
                return gemm_contig_tree_nm_impl<lhsTy, rhsTy, resTy, 4>(
                    exec_q, lhs_tp, rhs_tp, res_tp, n, k, m, depends);
            }
            else {
                return gemm_contig_tree_nm_impl<lhsTy, rhsTy, resTy, 1>(
                    exec_q, lhs_tp, rhs_tp, res_tp, n, k, m, depends);
            }
        }
    }
}

template <typename lhsT,
          typename rhsT,
          typename resT,
          typename LocAccT1,
          typename LocAccT2,
          typename OuterInnerDimsIndexerT,
          typename BatchDimsIndexerT,
          int wi_delta_n,
          int wi_delta_m>
class GemmBatchFunctorThreadNM
{
private:
    const lhsT *lhs = nullptr;
    const rhsT *rhs = nullptr;
    resT *res = nullptr;
    LocAccT1 local_A_block;
    LocAccT2 local_B_block;
    size_t n = 0;
    size_t wg_delta_n = 0;
    size_t k = 0;
    size_t k_blocks = 0;
    size_t wi_delta_k = 0;
    size_t m = 0;
    size_t m_blocks = 0;
    size_t wg_delta_m = 0;
    size_t batch_nelems;
    BatchDimsIndexerT batch_indexer;
    OuterInnerDimsIndexerT lhs_indexer;
    OuterInnerDimsIndexerT rhs_indexer;
    OuterInnerDimsIndexerT res_indexer;

public:
    GemmBatchFunctorThreadNM(const lhsT *lhs_,
                             const rhsT *rhs_,
                             resT *res_,
                             LocAccT1 local_A_block_,
                             LocAccT2 local_B_block_,
                             size_t n_,
                             size_t wg_delta_n_,
                             size_t k_,
                             size_t k_blocks_,
                             size_t wi_delta_k_,
                             size_t m_,
                             size_t m_blocks_,
                             size_t wg_delta_m_,
                             size_t batch_nelems_,
                             BatchDimsIndexerT batch_indexer_,
                             OuterInnerDimsIndexerT lhs_indexer_,
                             OuterInnerDimsIndexerT rhs_indexer_,
                             OuterInnerDimsIndexerT res_indexer_)
        : lhs(lhs_), rhs(rhs_), res(res_), local_A_block(local_A_block_),
          local_B_block(local_B_block_), n(n_), wg_delta_n(wg_delta_n_), k(k_),
          k_blocks(k_blocks_), wi_delta_k(wi_delta_k_), m(m_),
          m_blocks(m_blocks_), wg_delta_m(wg_delta_m_),
          batch_nelems(batch_nelems_), batch_indexer(batch_indexer_),
          lhs_indexer(lhs_indexer_), rhs_indexer(rhs_indexer_),
          res_indexer(res_indexer_)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        size_t m_id =
            it.get_global_linear_id() / (it.get_global_range(0) / batch_nelems);
        size_t gr_id =
            it.get_group_linear_id() % (it.get_group_range(0) / batch_nelems);

        const auto &three_offsets_ =
            batch_indexer(static_cast<py::ssize_t>(m_id));

        // lift group_id to (block_i, block_j, block_s),
        //    0 <= block_i < n_blocks, 0 <= block_j < m_blocks, 0 <= block_s
        //    < k_blocks

        const auto &lhs_offset = three_offsets_.get_first_offset();
        const auto &rhs_offset = three_offsets_.get_second_offset();
        const auto &res_offset = three_offsets_.get_third_offset();

        size_t block_i = gr_id / (m_blocks * k_blocks);
        size_t block_r = gr_id - block_i * (m_blocks * k_blocks);
        size_t block_j = block_r / k_blocks;
        size_t block_s = block_r - block_j * k_blocks;

        size_t lid = it.get_local_linear_id();
        size_t local_i = lid / wg_delta_m;           // 0<= local_i < wg_delta_n
        size_t local_j = lid - local_i * wg_delta_m; // 0<= local_j < wg_delta_m

        // load A block and B blocks into SLM

        size_t i = block_i * wi_delta_n * wg_delta_n;
        size_t j = block_j * wi_delta_m * wg_delta_m;
        size_t s = block_s * wi_delta_k;

        const std::int64_t a_st0 = k;
        const std::int64_t a_st1 = 1;

        const std::int64_t b_st0 = m;
        const std::int64_t b_st1 = 1;

        const std::int64_t c_st0 = m;
        const std::int64_t c_st1 = 1;

        size_t lws = it.get_local_range(0);

        for (size_t vid = lid; vid < local_A_block.size(); vid += lws) {
            size_t v_i = vid / wi_delta_k; // 0<= v_i < wg_delta_n * wi_delta_n
            size_t v_s = vid - v_i * wi_delta_k; // 0<= v_s < wi_delta_k

            size_t g_i = i + v_i;
            size_t g_s = s + v_s;

            local_A_block[vid] =
                (g_i < n && g_s < k)
                    ? static_cast<resT>(
                          lhs[lhs_offset +
                              lhs_indexer(g_i * a_st0 + g_s * a_st1)])
                    : resT(0);
        }

        for (size_t vid = lid; vid < local_B_block.size(); vid += lws) {
            size_t v_j = vid / wi_delta_k;       // 0<= v_i < wg_delta_m
            size_t v_s = vid - v_j * wi_delta_k; // 0<= v_s < wi_delta_k

            size_t g_j0 = j + v_j * wi_delta_m;
            size_t g_s = s + v_s;

            sycl::vec<resT, wi_delta_m> vec{};
#pragma unroll
            for (std::uint8_t lane_id = 0; lane_id < wi_delta_m; ++lane_id) {
                size_t g_j = g_j0 + lane_id;
                vec[lane_id] =
                    (g_j < m && g_s < k)
                        ? static_cast<resT>(
                              rhs[rhs_offset +
                                  rhs_indexer(g_s * b_st0 + g_j * b_st1)])
                        : resT(0);
            }

            local_B_block[vid] = vec;
        }

        it.barrier(sycl::access::fence_space::local_space);

        i += local_i * wi_delta_n;
        j += local_j * wi_delta_m;

        size_t a_offset = local_i * wi_delta_k * wi_delta_n;
        size_t b_offset = local_j * wi_delta_k;

        constexpr resT identity_(0);

        for (std::uint8_t private_i = 0; private_i < wi_delta_n; ++private_i) {
            size_t a_pr_offset = private_i * wi_delta_k;

            sycl::vec<resT, wi_delta_m> local_sum(identity_);
            for (size_t private_s = 0; private_s < wi_delta_k; ++private_s) {
                local_sum = local_sum +
                            (local_A_block[a_offset + a_pr_offset + private_s] *
                             local_B_block[b_offset + private_s]);
            }

            size_t gl_i = i + private_i;

#pragma unroll
            for (std::uint8_t lane_id = 0; lane_id < wi_delta_m; ++lane_id) {
                size_t gl_j = j + lane_id;

                if (gl_i < n && gl_j < m) {
                    sycl::atomic_ref<resT, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        aout(res[res_offset +
                                 res_indexer(gl_i * c_st0 + gl_j * c_st1)]);

                    aout += local_sum[lane_id];
                }
            }
        }
    }
};

template <typename lhsT,
          typename rhsT,
          typename resT,
          typename LocAccT1,
          typename LocAccT2,
          typename OuterInnerDimsIndexerT,
          typename BatchDimsIndexerT,
          int wi_delta_n>
class GemmBatchFunctorThreadNM<lhsT,
                               rhsT,
                               resT,
                               LocAccT1,
                               LocAccT2,
                               OuterInnerDimsIndexerT,
                               BatchDimsIndexerT,
                               wi_delta_n,
                               1>
{
private:
    const lhsT *lhs = nullptr;
    const rhsT *rhs = nullptr;
    resT *res = nullptr;
    LocAccT1 local_A_block;
    LocAccT2 local_B_block;
    size_t n = 0;
    size_t wg_delta_n = 0;
    size_t k = 0;
    size_t k_blocks = 0;
    size_t wi_delta_k = 0;
    size_t m = 0;
    size_t m_blocks = 0;
    size_t wg_delta_m = 0;
    size_t batch_nelems;
    BatchDimsIndexerT batch_indexer;
    OuterInnerDimsIndexerT lhs_indexer;
    OuterInnerDimsIndexerT rhs_indexer;
    OuterInnerDimsIndexerT res_indexer;

public:
    GemmBatchFunctorThreadNM(const lhsT *lhs_,
                             const rhsT *rhs_,
                             resT *res_,
                             LocAccT1 local_A_block_,
                             LocAccT2 local_B_block_,
                             size_t n_,
                             size_t wg_delta_n_,
                             size_t k_,
                             size_t k_blocks_,
                             size_t wi_delta_k_,
                             size_t m_,
                             size_t m_blocks_,
                             size_t wg_delta_m_,
                             size_t batch_nelems_,
                             BatchDimsIndexerT batch_indexer_,
                             OuterInnerDimsIndexerT lhs_indexer_,
                             OuterInnerDimsIndexerT rhs_indexer_,
                             OuterInnerDimsIndexerT res_indexer_)
        : lhs(lhs_), rhs(rhs_), res(res_), local_A_block(local_A_block_),
          local_B_block(local_B_block_), n(n_), wg_delta_n(wg_delta_n_), k(k_),
          k_blocks(k_blocks_), wi_delta_k(wi_delta_k_), m(m_),
          m_blocks(m_blocks_), wg_delta_m(wg_delta_m_),
          batch_nelems(batch_nelems_), batch_indexer(batch_indexer_),
          lhs_indexer(lhs_indexer_), rhs_indexer(rhs_indexer_),
          res_indexer(res_indexer_)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        size_t m_id =
            it.get_global_linear_id() / (it.get_global_range(0) / batch_nelems);
        size_t gr_id =
            it.get_group_linear_id() % (it.get_group_range(0) / batch_nelems);

        const auto &three_offsets_ =
            batch_indexer(static_cast<py::ssize_t>(m_id));

        // lift group_id to (block_i, block_j, block_s),
        //    0 <= block_i < n_blocks, 0 <= block_j < m_blocks, 0 <= block_s
        //    < k_blocks

        const auto &lhs_offset = three_offsets_.get_first_offset();
        const auto &rhs_offset = three_offsets_.get_second_offset();
        const auto &res_offset = three_offsets_.get_third_offset();

        size_t block_i = gr_id / (m_blocks * k_blocks);
        size_t block_r = gr_id - block_i * (m_blocks * k_blocks);
        size_t block_j = block_r / k_blocks;
        size_t block_s = block_r - block_j * k_blocks;

        size_t lid = it.get_local_linear_id();
        size_t local_i = lid / wg_delta_m;           // 0<= local_i < wg_delta_n
        size_t local_j = lid - local_i * wg_delta_m; // 0<= local_j < wg_delta_m

        // load A block and B blocks into SLM

        size_t i = block_i * wi_delta_n * wg_delta_n;
        size_t j = block_j * wg_delta_m;
        size_t s = block_s * wi_delta_k;

        const std::int64_t a_st0 = k;
        const std::int64_t a_st1 = 1;

        const std::int64_t b_st0 = m;
        const std::int64_t b_st1 = 1;

        const std::int64_t c_st0 = m;
        const std::int64_t c_st1 = 1;

        size_t lws = it.get_local_range(0);

        for (size_t vid = lid; vid < local_A_block.size(); vid += lws) {
            size_t v_i = vid / wi_delta_k; // 0<= v_i < wg_delta_n * wi_delta_n
            size_t v_s = vid - v_i * wi_delta_k; // 0<= v_s < wi_delta_k

            size_t g_i = i + v_i;
            size_t g_s = s + v_s;

            local_A_block[vid] =
                (g_i < n && g_s < k)
                    ? static_cast<resT>(
                          lhs[lhs_offset +
                              lhs_indexer(g_i * a_st0 + g_s * a_st1)])
                    : resT(0);
        }

        for (size_t vid = lid; vid < local_B_block.size(); vid += lws) {
            size_t v_j = vid / wi_delta_k;       // 0<= v_i < wg_delta_m
            size_t v_s = vid - v_j * wi_delta_k; // 0<= v_s < wi_delta_k

            size_t g_j0 = j + v_j;
            size_t g_s = s + v_s;

            resT val = (g_j0 < m && g_s < k)
                           ? static_cast<resT>(
                                 rhs[rhs_offset +
                                     rhs_indexer(g_s * b_st0 + g_j0 * b_st1)])
                           : resT(0);

            local_B_block[vid] = val;
        }

        it.barrier(sycl::access::fence_space::local_space);

        i += local_i * wi_delta_n;
        j += local_j;

        size_t a_offset = local_i * wi_delta_k * wi_delta_n;
        size_t b_offset = local_j * wi_delta_k;

        constexpr resT identity_(0);
        for (std::uint8_t private_i = 0; private_i < wi_delta_n; ++private_i) {
            size_t a_pr_offset = private_i * wi_delta_k;

            resT local_sum(identity_);
            for (size_t private_s = 0; private_s < wi_delta_k; ++private_s) {
                local_sum = local_sum +
                            (local_A_block[a_offset + a_pr_offset + private_s] *
                             local_B_block[b_offset + private_s]);
            }

            size_t gl_i = i + private_i;

            if (gl_i < n && j < m) {
                sycl::atomic_ref<resT, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    aout(res[res_offset +
                             res_indexer(gl_i * c_st0 + j * c_st1)]);

                aout += local_sum;
            }
        }
    }
};

template <typename lhsT,
          typename rhsT,
          typename resT,
          typename LocAccT,
          typename OuterInnerDimsIndexerT,
          typename BatchDimsIndexerT,
          size_t m_groups>
class GemmBatchFunctorThreadK
{
private:
    const lhsT *lhs = nullptr;
    const rhsT *rhs = nullptr;
    resT *res = nullptr;
    LocAccT workspace;
    LocAccT local_B_block;
    size_t n = 0;
    size_t n_blocks = 0;
    size_t delta_n = 0;
    size_t k = 0;
    size_t k_blocks = 0;
    size_t delta_k = 0;
    size_t n_wi = 0;
    size_t m = 0;
    size_t batch_nelems = 0;
    BatchDimsIndexerT batch_indexer;
    OuterInnerDimsIndexerT lhs_indexer;
    OuterInnerDimsIndexerT rhs_indexer;
    OuterInnerDimsIndexerT res_indexer;

public:
    GemmBatchFunctorThreadK(const lhsT *lhs_,
                            const rhsT *rhs_,
                            resT *res_,
                            LocAccT workspace_,
                            LocAccT local_B_block_,
                            size_t n_,
                            size_t n_blocks_,
                            size_t delta_n_,
                            size_t k_,
                            size_t k_blocks_,
                            size_t delta_k_,
                            size_t n_wi_,
                            size_t m_,
                            size_t batch_nelems_,
                            BatchDimsIndexerT batch_indexer_,
                            OuterInnerDimsIndexerT lhs_indexer_,
                            OuterInnerDimsIndexerT rhs_indexer_,
                            OuterInnerDimsIndexerT res_indexer_)
        : lhs(lhs_), rhs(rhs_), res(res_), workspace(workspace_),
          local_B_block(local_B_block_), n(n_), n_blocks(n_blocks_),
          delta_n(delta_n_), k(k_), k_blocks(k_blocks_), delta_k(delta_k_),
          n_wi(n_wi_), m(m_), batch_nelems(batch_nelems_),
          batch_indexer(batch_indexer_), lhs_indexer(lhs_indexer_),
          rhs_indexer(rhs_indexer_), res_indexer(res_indexer_)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        // for batching:
        // (current matrix in batch) m_id = global_id / (global_range /
        // batch_nelems) for lhs, offset = m_id * (n * k) for rhs, offset =
        // m_id
        // * (k * m) for res, offset = m_id * (n * m)
        size_t m_id =
            it.get_global_linear_id() / (it.get_global_range(0) / batch_nelems);
        size_t gr_id =
            it.get_group_linear_id() % (it.get_group_range(0) / batch_nelems);
        size_t lid = it.get_local_linear_id();

        const auto &three_offsets_ =
            batch_indexer(static_cast<py::ssize_t>(m_id));

        const auto &lhs_offset = three_offsets_.get_first_offset();
        const auto &rhs_offset = three_offsets_.get_second_offset();
        const auto &res_offset = three_offsets_.get_third_offset();

        // lift gr_id -> (block_i, block_j, block_s)
        //   block_i moves fastest, then block_s, then block_j

        size_t block_j =
            gr_id / (n_blocks * k_blocks); // 0 <= block_j < m_blocks
        size_t block_r =
            gr_id - block_j * (n_blocks *
                               k_blocks); // 0 <= block_r < n_blocks * k_blocks
        size_t block_s = block_r / n_blocks; // 0 <= block_s < k_blocks
        size_t block_i =
            block_r - block_s * n_blocks; // 0 <= block_i < n_blocks

        size_t local_i = lid / (delta_k);           // 0 <= local_i < delta_n
        size_t local_s = lid - local_i * (delta_k); // 0 <= local_s < delta_k

        size_t i = block_i * delta_n + local_i;
        size_t j = m_groups * block_j;
        size_t s = block_s * delta_k * n_wi + local_s;

        constexpr resT identity_ = resT(0);
        if (local_i == 0) {
            for (size_t q = 0; q < n_wi * delta_k; q += delta_k) {
                size_t sq = s + q;
                size_t sqmj = sq * m + j;
                local_B_block[local_s + q] = sycl::vec<resT, m_groups>(
                    (sq < k && j < m)
                        ? static_cast<resT>(rhs[rhs_offset + rhs_indexer(sqmj)])
                        : identity_,
                    (sq < k && j + 1 < m)
                        ? static_cast<resT>(
                              rhs[rhs_offset + rhs_indexer(sqmj + 1)])
                        : identity_);
            }
        }

        it.barrier(sycl::access::fence_space::local_space);

        size_t t_shift = block_s * delta_k * n_wi;
        size_t global_s_offset = i * k + t_shift;

        sycl::vec<resT, m_groups> private_sum(identity_);
        for (size_t t = local_s; t < local_B_block.size(); t += delta_k) {
            if (t + t_shift < k) {
                private_sum +=
                    (static_cast<resT>(
                         lhs[lhs_offset + lhs_indexer(global_s_offset + t)]) *
                     local_B_block[t]);
            }
        }

        size_t workspace_i_shift = local_i * delta_k;
        workspace[workspace_i_shift + local_s] = private_sum;

        it.barrier(sycl::access::fence_space::local_space);

        if (local_s == 0 && i < n) {
            sycl::vec<resT, m_groups> local_sum(workspace[workspace_i_shift]);
            for (size_t t = 1; t < delta_k; ++t) {
                local_sum += workspace[workspace_i_shift + t];
            }

            sycl::atomic_ref<resT, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                aout0(res[res_offset + res_indexer(i * m + j)]);

            aout0 += local_sum[0];

            if (j + 1 < m) {
                sycl::atomic_ref<resT, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    aout1(res[res_offset + res_indexer(i * m + j + 1)]);

                aout1 += local_sum[1];
            }
        }
    }
};

template <typename lhsT,
          typename rhsT,
          typename resT,
          typename LocAccT,
          typename OuterInnerDimsIndexerT,
          typename BatchDimsIndexerT>
class GemmBatchFunctorThreadK<lhsT,
                              rhsT,
                              resT,
                              LocAccT,
                              OuterInnerDimsIndexerT,
                              BatchDimsIndexerT,
                              1>
{
private:
    const lhsT *lhs = nullptr;
    const rhsT *rhs = nullptr;
    resT *res = nullptr;
    LocAccT workspace;
    LocAccT local_B_block;
    size_t n = 0;
    size_t n_blocks = 0;
    size_t delta_n = 0;
    size_t k = 0;
    size_t k_blocks = 0;
    size_t delta_k = 0;
    size_t n_wi = 0;
    size_t m = 0;
    size_t batch_nelems = 0;
    BatchDimsIndexerT batch_indexer;
    OuterInnerDimsIndexerT lhs_indexer;
    OuterInnerDimsIndexerT rhs_indexer;
    OuterInnerDimsIndexerT res_indexer;

public:
    GemmBatchFunctorThreadK(const lhsT *lhs_,
                            const rhsT *rhs_,
                            resT *res_,
                            LocAccT workspace_,
                            LocAccT local_B_block_,
                            size_t n_,
                            size_t n_blocks_,
                            size_t delta_n_,
                            size_t k_,
                            size_t k_blocks_,
                            size_t delta_k_,
                            size_t n_wi_,
                            size_t m_,
                            size_t batch_nelems_,
                            BatchDimsIndexerT batch_indexer_,
                            OuterInnerDimsIndexerT lhs_indexer_,
                            OuterInnerDimsIndexerT rhs_indexer_,
                            OuterInnerDimsIndexerT res_indexer_)
        : lhs(lhs_), rhs(rhs_), res(res_), workspace(workspace_),
          local_B_block(local_B_block_), n(n_), n_blocks(n_blocks_),
          delta_n(delta_n_), k(k_), k_blocks(k_blocks_), delta_k(delta_k_),
          n_wi(n_wi_), m(m_), batch_nelems(batch_nelems_),
          batch_indexer(batch_indexer_), lhs_indexer(lhs_indexer_),
          rhs_indexer(rhs_indexer_), res_indexer(res_indexer_)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        // for batching:
        // (current matrix in batch) m_id = global_id / (global_range /
        // batch_nelems) for lhs, offset = m_id * (n * k) for rhs, offset =
        // m_id
        // * (k * m) for res, offset = m_id * (n * m)
        size_t m_id =
            it.get_global_linear_id() / (it.get_global_range(0) / batch_nelems);
        size_t gr_id =
            it.get_group_linear_id() % (it.get_group_range(0) / batch_nelems);
        size_t lid = it.get_local_linear_id();

        const auto &three_offsets_ =
            batch_indexer(static_cast<py::ssize_t>(m_id));

        const auto &lhs_offset = three_offsets_.get_first_offset();
        const auto &rhs_offset = three_offsets_.get_second_offset();
        const auto &res_offset = three_offsets_.get_third_offset();

        // lift gr_id -> (block_i, block_j, block_s)
        //   block_i moves fastest, then block_s, then block_j

        size_t block_j =
            gr_id / (n_blocks * k_blocks); // 0 <= block_j < m_blocks
        size_t block_r =
            gr_id - block_j * (n_blocks *
                               k_blocks); // 0 <= block_r < n_blocks * k_blocks
        size_t block_s = block_r / n_blocks; // 0 <= block_s < k_blocks
        size_t block_i =
            block_r - block_s * n_blocks; // 0 <= block_i < n_blocks

        size_t local_i = lid / (delta_k);           // 0 <= local_i < delta_n
        size_t local_s = lid - local_i * (delta_k); // 0 <= local_s < delta_k

        size_t i = block_i * delta_n + local_i;
        size_t j = block_j;
        size_t s = block_s * delta_k * n_wi + local_s;

        constexpr resT identity_ = resT(0);
        if (local_i == 0) {
            for (size_t q = 0; q < n_wi * delta_k; q += delta_k) {
                size_t sq = s + q;
                size_t sqmj = sq * m + j;
                local_B_block[local_s + q] =
                    (sq < k && j < m)
                        ? static_cast<resT>(rhs[rhs_offset + rhs_indexer(sqmj)])
                        : identity_;
                ;
            }
        }

        it.barrier(sycl::access::fence_space::local_space);

        size_t t_shift = block_s * delta_k * n_wi;
        size_t global_s_offset = i * k + t_shift;

        resT private_sum(identity_);
        for (size_t t = local_s; t < local_B_block.size(); t += delta_k) {
            if (t + t_shift < k) {
                private_sum +=
                    (static_cast<resT>(
                         lhs[lhs_offset + lhs_indexer(global_s_offset + t)]) *
                     local_B_block[t]);
            }
        }

        size_t workspace_i_shift = local_i * delta_k;
        workspace[workspace_i_shift + local_s] = private_sum;

        it.barrier(sycl::access::fence_space::local_space);

        if (local_s == 0 && i < n) {
            resT local_sum(workspace[workspace_i_shift]);
            for (size_t t = 1; t < delta_k; ++t) {
                local_sum += workspace[workspace_i_shift + t];
            }

            sycl::atomic_ref<resT, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                aout(res[res_offset + res_indexer(i * m + j)]);

            aout += local_sum;
        }
    }
};

template <typename T1, typename T2, typename T3> class gemm_batch_init_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          size_t>
class gemm_batch_k_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          size_t>
class gemm_batch_nm_krn;

typedef sycl::event (*gemm_batch_impl_fn_ptr_t)(
    sycl::queue &,
    const char *,        // lhs
    const char *,        // rhs
    char *,              // res
    size_t,              // batch nelems
    size_t,              // lhs outer nelems (n)
    size_t,              // inner nelems (k)
    size_t,              // rhs outer nelems (m)
    int,                 // batching nd
    const py::ssize_t *, // batch shape strides
    py::ssize_t,         // lhs batch offset
    py::ssize_t,         // rhs batch offset
    py::ssize_t,         // res batch offset
    int,                 // inner dims
    int,                 // lhs outer dims
    const py::ssize_t *, // lhs outer and inner shape and strides
    int,                 // rhs outer dims
    const py::ssize_t *, // rhs outer and inner shape and strides
    int,                 // res outer dims
    const py::ssize_t *, // res outer and inner shape and strides
    const py::ssize_t *, // res full shape and strides
    std::vector<sycl::event> const &);

template <typename lhsTy, typename rhsTy, typename resTy>
sycl::event gemm_batch_impl(sycl::queue &exec_q,
                            const char *lhs_cp,
                            const char *rhs_cp,
                            char *res_cp,
                            size_t batch_nelems,
                            size_t n,
                            size_t k,
                            size_t m,
                            int batch_nd,
                            const py::ssize_t *batch_shape_strides,
                            py::ssize_t lhs_batch_offset,
                            py::ssize_t rhs_batch_offset,
                            py::ssize_t res_batch_offset,
                            int inner_nd,
                            int lhs_outer_nd,
                            const py::ssize_t *lhs_outer_inner_shapes_strides,
                            int rhs_outer_nd,
                            const py::ssize_t *rhs_outer_inner_shapes_strides,
                            int res_outer_nd,
                            const py::ssize_t *res_outer_shapes_strides,
                            const py::ssize_t *res_shape_strides,
                            std::vector<sycl::event> const &depends = {})
{
    const lhsTy *lhs_tp = reinterpret_cast<const lhsTy *>(lhs_cp);
    const rhsTy *rhs_tp = reinterpret_cast<const rhsTy *>(rhs_cp);
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp);

    const sycl::device &dev = exec_q.get_device();
    const size_t local_mem_size =
        dev.get_info<sycl::info::device::local_mem_size>();
    const size_t reserved_slm_size = 512;

    sycl::event res_init_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        using IndexerT = dpctl::tensor::offset_utils::StridedIndexer;
        IndexerT res_indexer(batch_nd + res_outer_nd, res_batch_offset,
                             res_shape_strides);
        using InitKernelName = class gemm_batch_init_krn<lhsTy, rhsTy, resTy>;
        cgh.parallel_for<InitKernelName>(
            sycl::range<1>(n * m * batch_nelems), [=](sycl::id<1> id) {
                auto res_offset = res_indexer(id[0]);
                res_tp[res_offset] = resTy(0);
            });
    });

    if (k == 0) {
        return res_init_ev;
    }

    sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(res_init_ev);

        using OuterInnerDimsIndexerT =
            dpctl::tensor::offset_utils::StridedIndexer;
        OuterInnerDimsIndexerT lhs_indexer(inner_nd + lhs_outer_nd, 0,
                                           lhs_outer_inner_shapes_strides);
        OuterInnerDimsIndexerT rhs_indexer(inner_nd + rhs_outer_nd, 0,
                                           rhs_outer_inner_shapes_strides);
        OuterInnerDimsIndexerT res_indexer(res_outer_nd, 0,
                                           res_outer_shapes_strides);
        using BatchDimsIndexerT =
            dpctl::tensor::offset_utils::ThreeOffsets_StridedIndexer;
        BatchDimsIndexerT batch_indexer(batch_nd, lhs_batch_offset,
                                        rhs_batch_offset, res_batch_offset,
                                        batch_shape_strides);
        if (m == 1) {
            constexpr size_t m_groups = 1;
            size_t delta_k(4);
            size_t n_wi(4);
            size_t delta_n(4);

            gemm_detail::scale_gemm_k_parameters<resTy, m_groups>(
                local_mem_size, reserved_slm_size, delta_k,
                n_wi,   // modified by reference
                delta_n // modified by reference
            );

            size_t n_blocks = (n + delta_n - 1) / delta_n;
            size_t m_blocks = (m + m_groups - 1) / m_groups;
            size_t k_blocks = (k + n_wi * delta_k - 1) / (n_wi * delta_k);

            size_t lws = delta_n * delta_k;

            auto gRange = sycl::range<1>(batch_nelems * n_blocks * m_blocks *
                                         k_blocks * lws);
            auto lRange = sycl::range<1>(lws);

            auto ndRange = sycl::nd_range<1>(gRange, lRange);

            using LocAccT = sycl::local_accessor<resTy, 1>;
            LocAccT local_B_block(n_wi * delta_k, cgh);
            LocAccT workspace(delta_n * delta_k, cgh);

            using KernelName =
                class gemm_batch_k_krn<lhsTy, rhsTy, resTy,
                                       OuterInnerDimsIndexerT,
                                       BatchDimsIndexerT, m_groups>;
            cgh.parallel_for<KernelName>(
                ndRange, GemmBatchFunctorThreadK<lhsTy, rhsTy, resTy, LocAccT,
                                                 OuterInnerDimsIndexerT,
                                                 BatchDimsIndexerT, m_groups>(
                             lhs_tp, rhs_tp, res_tp, workspace, local_B_block,
                             n, n_blocks, delta_n, k, k_blocks, delta_k, n_wi,
                             m, batch_nelems, batch_indexer, lhs_indexer,
                             rhs_indexer, res_indexer));
        }
        else if (k > n && k > m && !exec_q.get_device().is_cpu()) {
            constexpr size_t m_groups = 2;
            size_t delta_k(4);
            size_t n_wi(4);
            size_t delta_n(4);

            gemm_detail::scale_gemm_k_parameters<resTy, m_groups>(
                local_mem_size, reserved_slm_size, delta_k,
                n_wi,   // modified by reference
                delta_n // modified by reference
            );

            size_t n_blocks = (n + delta_n - 1) / delta_n;
            size_t m_blocks = (m + m_groups - 1) / m_groups;
            size_t k_blocks = (k + n_wi * delta_k - 1) / (n_wi * delta_k);

            size_t lws = delta_n * delta_k;

            auto gRange = sycl::range<1>(batch_nelems * n_blocks * m_blocks *
                                         k_blocks * lws);
            auto lRange = sycl::range<1>(lws);

            auto ndRange = sycl::nd_range<1>(gRange, lRange);

            using LocAccT = sycl::local_accessor<sycl::vec<resTy, m_groups>, 1>;
            LocAccT local_B_block(n_wi * delta_k, cgh);
            LocAccT workspace(delta_n * delta_k, cgh);

            using KernelName =
                class gemm_batch_k_krn<lhsTy, rhsTy, resTy,
                                       OuterInnerDimsIndexerT,
                                       BatchDimsIndexerT, m_groups>;
            cgh.parallel_for<KernelName>(
                ndRange, GemmBatchFunctorThreadK<lhsTy, rhsTy, resTy, LocAccT,
                                                 OuterInnerDimsIndexerT,
                                                 BatchDimsIndexerT, m_groups>(
                             lhs_tp, rhs_tp, res_tp, workspace, local_B_block,
                             n, n_blocks, delta_n, k, k_blocks, delta_k, n_wi,
                             m, batch_nelems, batch_indexer, lhs_indexer,
                             rhs_indexer, res_indexer));
        }
        else {
            constexpr int wi_delta_n = 2;
            constexpr int wi_delta_m = 4;
            size_t wg_delta_n(16); // rows of A processed in WG
            size_t wg_delta_m(16); // rows of B processed in WG
            size_t wi_delta_k(64); // Elements in K dimension processed by WI

            gemm_detail::scale_gemm_nm_parameters<resTy, wi_delta_m>(
                local_mem_size, reserved_slm_size, wi_delta_n,
                wi_delta_k, // modified by reference
                wg_delta_n, // modified by reference
                wg_delta_m  // modified by reference
            );

            size_t lws = wg_delta_n * wg_delta_m;

            size_t n_blocks =
                ((n + wi_delta_n * wg_delta_n - 1) / (wi_delta_n * wg_delta_n));
            size_t m_blocks =
                ((m + wi_delta_m * wg_delta_m - 1) / (wi_delta_m * wg_delta_m));
            size_t k_blocks = ((k + wi_delta_k - 1) / wi_delta_k);

            auto gwsRange = sycl::range<1>(batch_nelems * n_blocks * m_blocks *
                                           k_blocks * lws);
            auto lwsRange = sycl::range<1>(lws);

            auto ndRange = sycl::nd_range<1>(gwsRange, lwsRange);

            using LocAccT1 = sycl::local_accessor<resTy, 1>;
            LocAccT1 local_A_block(
                sycl::range<1>((wi_delta_n * wg_delta_n) * wi_delta_k), cgh);
            using LocAccT2 =
                sycl::local_accessor<sycl::vec<resTy, wi_delta_m>, 1>;
            LocAccT2 local_B_block(sycl::range<1>(wi_delta_k * wg_delta_m),
                                   cgh);

            using KernelName =
                class gemm_batch_nm_krn<lhsTy, rhsTy, resTy,
                                        OuterInnerDimsIndexerT,
                                        BatchDimsIndexerT, wi_delta_m>;
            cgh.parallel_for<KernelName>(
                ndRange,
                GemmBatchFunctorThreadNM<lhsTy, rhsTy, resTy, LocAccT1,
                                         LocAccT2, OuterInnerDimsIndexerT,
                                         BatchDimsIndexerT, wi_delta_n,
                                         wi_delta_m>(
                    lhs_tp, rhs_tp, res_tp, local_A_block, local_B_block, n,
                    wg_delta_n, k, k_blocks, wi_delta_k, m, m_blocks,
                    wg_delta_m, batch_nelems, batch_indexer, lhs_indexer,
                    rhs_indexer, res_indexer));
        }
    });

    return gemm_ev;
}

typedef sycl::event (*gemm_batch_contig_impl_fn_ptr_t)(
    sycl::queue &,
    const char *, // lhs
    const char *, // rhs
    char *,       // res
    size_t,       // batch nelems
    size_t,       // n
    size_t,       // k
    size_t,       // m
    py::ssize_t,  // lhs batch offset
    py::ssize_t,  // rhs batch offset
    py::ssize_t,  // res batch offset
    std::vector<sycl::event> const &);

template <typename lhsTy, typename rhsTy, typename resTy>
sycl::event gemm_batch_contig_impl(sycl::queue &exec_q,
                                   const char *lhs_cp,
                                   const char *rhs_cp,
                                   char *res_cp,
                                   size_t batch_nelems,
                                   size_t n,
                                   size_t k,
                                   size_t m,
                                   py::ssize_t lhs_batch_offset,
                                   py::ssize_t rhs_batch_offset,
                                   py::ssize_t res_batch_offset,
                                   std::vector<sycl::event> const &depends = {})
{
    const lhsTy *lhs_tp =
        reinterpret_cast<const lhsTy *>(lhs_cp) + lhs_batch_offset;
    const rhsTy *rhs_tp =
        reinterpret_cast<const rhsTy *>(rhs_cp) + rhs_batch_offset;
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp) + res_batch_offset;

    const sycl::device &dev = exec_q.get_device();
    const size_t local_mem_size =
        dev.get_info<sycl::info::device::local_mem_size>();
    const size_t reserved_slm_size = 512;

    sycl::event res_init_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.fill<resTy>(res_tp, resTy(0), n * m * batch_nelems);
    });

    if (k == 0) {
        return res_init_ev;
    }

    sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(res_init_ev);

        using OuterInnerDimsIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        OuterInnerDimsIndexerT lhs_indexer{};
        OuterInnerDimsIndexerT rhs_indexer{};
        OuterInnerDimsIndexerT res_indexer{};
        using dpctl::tensor::offset_utils::Strided1DIndexer;
        using dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer;
        using BatchDimsIndexerT =
            ThreeOffsets_CombinedIndexer<Strided1DIndexer, Strided1DIndexer,
                                         Strided1DIndexer>;

        BatchDimsIndexerT batch_indexer(
            Strided1DIndexer{0, static_cast<py::ssize_t>(batch_nelems),
                             static_cast<py::ssize_t>(n * k)},
            Strided1DIndexer{0, static_cast<py::ssize_t>(batch_nelems),
                             static_cast<py::ssize_t>(k * m)},
            Strided1DIndexer{0, static_cast<py::ssize_t>(batch_nelems),
                             static_cast<py::ssize_t>(n * m)});
        if (m == 1) {
            constexpr size_t m_groups = 1;
            size_t delta_k(4);
            size_t n_wi(4);
            size_t delta_n(4);

            gemm_detail::scale_gemm_k_parameters<resTy, m_groups>(
                local_mem_size, reserved_slm_size, delta_k,
                n_wi,   // modified by reference
                delta_n // modified by reference
            );

            size_t n_blocks = (n + delta_n - 1) / delta_n;
            size_t m_blocks = (m + m_groups - 1) / m_groups;
            size_t k_blocks = (k + n_wi * delta_k - 1) / (n_wi * delta_k);

            size_t lws = delta_n * delta_k;

            auto gRange = sycl::range<1>(batch_nelems * n_blocks * m_blocks *
                                         k_blocks * lws);
            auto lRange = sycl::range<1>(lws);

            auto ndRange = sycl::nd_range<1>(gRange, lRange);

            using LocAccT = sycl::local_accessor<resTy, 1>;
            LocAccT local_B_block(n_wi * delta_k, cgh);
            LocAccT workspace(delta_n * delta_k, cgh);

            using KernelName =
                class gemm_batch_k_krn<lhsTy, rhsTy, resTy,
                                       OuterInnerDimsIndexerT,
                                       BatchDimsIndexerT, m_groups>;
            cgh.parallel_for<KernelName>(
                ndRange, GemmBatchFunctorThreadK<lhsTy, rhsTy, resTy, LocAccT,
                                                 OuterInnerDimsIndexerT,
                                                 BatchDimsIndexerT, m_groups>(
                             lhs_tp, rhs_tp, res_tp, workspace, local_B_block,
                             n, n_blocks, delta_n, k, k_blocks, delta_k, n_wi,
                             m, batch_nelems, batch_indexer, lhs_indexer,
                             rhs_indexer, res_indexer));
        }
        else if (k > n && k > m && !exec_q.get_device().is_cpu()) {
            constexpr size_t m_groups = 2;
            size_t delta_k(4);
            size_t n_wi(4);
            size_t delta_n(4);

            gemm_detail::scale_gemm_k_parameters<resTy, m_groups>(
                local_mem_size, reserved_slm_size, delta_k,
                n_wi,   // modified by reference
                delta_n // modified by reference
            );

            size_t n_blocks = (n + delta_n - 1) / delta_n;
            size_t m_blocks = (m + m_groups - 1) / m_groups;
            size_t k_blocks = (k + n_wi * delta_k - 1) / (n_wi * delta_k);

            size_t lws = delta_n * delta_k;

            auto gRange = sycl::range<1>(batch_nelems * n_blocks * m_blocks *
                                         k_blocks * lws);
            auto lRange = sycl::range<1>(lws);

            auto ndRange = sycl::nd_range<1>(gRange, lRange);

            using LocAccT = sycl::local_accessor<sycl::vec<resTy, m_groups>, 1>;
            LocAccT local_B_block(n_wi * delta_k, cgh);
            LocAccT workspace(delta_n * delta_k, cgh);

            using KernelName =
                class gemm_batch_k_krn<lhsTy, rhsTy, resTy,
                                       OuterInnerDimsIndexerT,
                                       BatchDimsIndexerT, m_groups>;
            cgh.parallel_for<KernelName>(
                ndRange, GemmBatchFunctorThreadK<lhsTy, rhsTy, resTy, LocAccT,
                                                 OuterInnerDimsIndexerT,
                                                 BatchDimsIndexerT, m_groups>(
                             lhs_tp, rhs_tp, res_tp, workspace, local_B_block,
                             n, n_blocks, delta_n, k, k_blocks, delta_k, n_wi,
                             m, batch_nelems, batch_indexer, lhs_indexer,
                             rhs_indexer, res_indexer));
        }
        else {
            constexpr int wi_delta_n = 2;
            constexpr int wi_delta_m = 4;
            size_t wg_delta_n(16); // rows of A processed in WG
            size_t wg_delta_m(16); // rows of B processed in WG
            size_t wi_delta_k(64); // Elements in K dimension processed by WI

            gemm_detail::scale_gemm_nm_parameters<resTy, wi_delta_m>(
                local_mem_size, reserved_slm_size, wi_delta_n,
                wi_delta_k, // modified by reference
                wg_delta_n, // modified by reference
                wg_delta_m  // modified by reference
            );

            size_t lws = wg_delta_n * wg_delta_m;

            size_t n_blocks =
                ((n + wi_delta_n * wg_delta_n - 1) / (wi_delta_n * wg_delta_n));
            size_t m_blocks =
                ((m + wi_delta_m * wg_delta_m - 1) / (wi_delta_m * wg_delta_m));
            size_t k_blocks = ((k + wi_delta_k - 1) / wi_delta_k);

            auto gwsRange = sycl::range<1>(batch_nelems * n_blocks * m_blocks *
                                           k_blocks * lws);
            auto lwsRange = sycl::range<1>(lws);

            auto ndRange = sycl::nd_range<1>(gwsRange, lwsRange);

            using LocAccT1 = sycl::local_accessor<resTy, 1>;
            LocAccT1 local_A_block(
                sycl::range<1>((wi_delta_n * wg_delta_n) * wi_delta_k), cgh);
            using LocAccT2 =
                sycl::local_accessor<sycl::vec<resTy, wi_delta_m>, 1>;
            LocAccT2 local_B_block(sycl::range<1>(wi_delta_k * wg_delta_m),
                                   cgh);

            using KernelName =
                class gemm_batch_nm_krn<lhsTy, rhsTy, resTy,
                                        OuterInnerDimsIndexerT,
                                        BatchDimsIndexerT, wi_delta_m>;
            cgh.parallel_for<KernelName>(
                ndRange,
                GemmBatchFunctorThreadNM<lhsTy, rhsTy, resTy, LocAccT1,
                                         LocAccT2, OuterInnerDimsIndexerT,
                                         BatchDimsIndexerT, wi_delta_n,
                                         wi_delta_m>(
                    lhs_tp, rhs_tp, res_tp, local_A_block, local_B_block, n,
                    wg_delta_n, k, k_blocks, wi_delta_k, m, m_blocks,
                    wg_delta_m, batch_nelems, batch_indexer, lhs_indexer,
                    rhs_indexer, res_indexer));
        }
    });

    return gemm_ev;
}

template <typename lhsT,
          typename rhsT,
          typename resT,
          typename LocAccT1,
          typename LocAccT2,
          typename OuterInnerDimsIndexerT,
          typename ResIndexerT,
          typename BatchDimsIndexerT,
          int wi_delta_n,
          int wi_delta_m>
class GemmBatchNoAtomicFunctorThreadNM
{
private:
    const lhsT *lhs = nullptr;
    const rhsT *rhs = nullptr;
    resT *res = nullptr;
    LocAccT1 local_A_block;
    LocAccT2 local_B_block;
    size_t n = 0;
    size_t wg_delta_n = 0;
    size_t k = 0;
    size_t k_blocks = 0;
    size_t wi_delta_k = 0;
    size_t m = 0;
    size_t m_blocks = 0;
    size_t wg_delta_m = 0;
    size_t batch_nelems;
    BatchDimsIndexerT batch_indexer;
    OuterInnerDimsIndexerT lhs_indexer;
    OuterInnerDimsIndexerT rhs_indexer;
    ResIndexerT res_indexer;

public:
    GemmBatchNoAtomicFunctorThreadNM(const lhsT *lhs_,
                                     const rhsT *rhs_,
                                     resT *res_,
                                     LocAccT1 local_A_block_,
                                     LocAccT2 local_B_block_,
                                     size_t n_,
                                     size_t wg_delta_n_,
                                     size_t k_,
                                     size_t k_blocks_,
                                     size_t wi_delta_k_,
                                     size_t m_,
                                     size_t m_blocks_,
                                     size_t wg_delta_m_,
                                     size_t batch_nelems_,
                                     BatchDimsIndexerT batch_indexer_,
                                     OuterInnerDimsIndexerT lhs_indexer_,
                                     OuterInnerDimsIndexerT rhs_indexer_,
                                     ResIndexerT res_indexer_)
        : lhs(lhs_), rhs(rhs_), res(res_), local_A_block(local_A_block_),
          local_B_block(local_B_block_), n(n_), wg_delta_n(wg_delta_n_), k(k_),
          k_blocks(k_blocks_), wi_delta_k(wi_delta_k_), m(m_),
          m_blocks(m_blocks_), wg_delta_m(wg_delta_m_),
          batch_nelems(batch_nelems_), batch_indexer(batch_indexer_),
          lhs_indexer(lhs_indexer_), rhs_indexer(rhs_indexer_),
          res_indexer(res_indexer_)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        size_t m_id =
            it.get_global_linear_id() / (it.get_global_range(0) / batch_nelems);
        size_t gr_id =
            it.get_group_linear_id() % (it.get_group_range(0) / batch_nelems);

        const auto &three_offsets_ =
            batch_indexer(static_cast<py::ssize_t>(m_id));

        // lift group_id to (block_i, block_j, block_s),
        //    0 <= block_i < n_blocks, 0 <= block_j < m_blocks, 0 <= block_s
        //    < k_blocks

        const auto &lhs_offset = three_offsets_.get_first_offset();
        const auto &rhs_offset = three_offsets_.get_second_offset();
        const auto &res_offset = three_offsets_.get_third_offset();

        size_t block_i = gr_id / (m_blocks * k_blocks);
        size_t block_r = gr_id - block_i * (m_blocks * k_blocks);
        size_t block_j = block_r / k_blocks;
        size_t block_s = block_r - block_j * k_blocks;

        size_t lid = it.get_local_linear_id();
        size_t local_i = lid / wg_delta_m;           // 0<= local_i < wg_delta_n
        size_t local_j = lid - local_i * wg_delta_m; // 0<= local_j < wg_delta_m

        // load A block and B blocks into SLM

        size_t i = block_i * wi_delta_n * wg_delta_n;
        size_t j = block_j * wi_delta_m * wg_delta_m;
        size_t s = block_s * wi_delta_k;

        const std::int64_t a_st0 = k;
        const std::int64_t a_st1 = 1;

        const std::int64_t b_st0 = m;
        const std::int64_t b_st1 = 1;

        const std::int64_t c_st0 = m;
        const std::int64_t c_st1 = 1;

        size_t lws = it.get_local_range(0);

        for (size_t vid = lid; vid < local_A_block.size(); vid += lws) {
            size_t v_i = vid / wi_delta_k; // 0<= v_i < wg_delta_n * wi_delta_n
            size_t v_s = vid - v_i * wi_delta_k; // 0<= v_s < wi_delta_k

            size_t g_i = i + v_i;
            size_t g_s = s + v_s;

            local_A_block[vid] =
                (g_i < n && g_s < k)
                    ? static_cast<resT>(
                          lhs[lhs_offset +
                              lhs_indexer(g_i * a_st0 + g_s * a_st1)])
                    : resT(0);
        }

        for (size_t vid = lid; vid < local_B_block.size(); vid += lws) {
            size_t v_j = vid / wi_delta_k;       // 0<= v_i < wg_delta_m
            size_t v_s = vid - v_j * wi_delta_k; // 0<= v_s < wi_delta_k

            size_t g_j0 = j + v_j * wi_delta_m;
            size_t g_s = s + v_s;

            sycl::vec<resT, wi_delta_m> vec{};
#pragma unroll
            for (std::uint8_t lane_id = 0; lane_id < wi_delta_m; ++lane_id) {
                size_t g_j = g_j0 + lane_id;
                vec[lane_id] =
                    (g_j < m && g_s < k)
                        ? static_cast<resT>(
                              rhs[rhs_offset +
                                  rhs_indexer(g_s * b_st0 + g_j * b_st1)])
                        : resT(0);
            }

            local_B_block[vid] = vec;
        }

        it.barrier(sycl::access::fence_space::local_space);

        i += local_i * wi_delta_n;
        j += local_j * wi_delta_m;

        size_t a_offset = local_i * wi_delta_k * wi_delta_n;
        size_t b_offset = local_j * wi_delta_k;

        constexpr resT identity_(0);

        for (std::uint8_t private_i = 0; private_i < wi_delta_n; ++private_i) {
            size_t a_pr_offset = private_i * wi_delta_k;

            sycl::vec<resT, wi_delta_m> local_sum(identity_);
            for (size_t private_s = 0; private_s < wi_delta_k; ++private_s) {
                local_sum = local_sum +
                            (local_A_block[a_offset + a_pr_offset + private_s] *
                             local_B_block[b_offset + private_s]);
            }

            size_t gl_i = i + private_i;

#pragma unroll
            for (std::uint8_t lane_id = 0; lane_id < wi_delta_m; ++lane_id) {
                size_t gl_j = j + lane_id;

                if (gl_i < n && gl_j < m) {
                    res[res_offset + res_indexer(gl_i * c_st0 + gl_j * c_st1) +
                        (block_s * n * m * batch_nelems)] = local_sum[lane_id];
                }
            }
        }
    }
};

template <typename lhsT,
          typename rhsT,
          typename resT,
          typename LocAccT1,
          typename LocAccT2,
          typename OuterInnerDimsIndexerT,
          typename ResIndexerT,
          typename BatchDimsIndexerT,
          int wi_delta_n>
class GemmBatchNoAtomicFunctorThreadNM<lhsT,
                                       rhsT,
                                       resT,
                                       LocAccT1,
                                       LocAccT2,
                                       OuterInnerDimsIndexerT,
                                       ResIndexerT,
                                       BatchDimsIndexerT,
                                       wi_delta_n,
                                       1>
{
private:
    const lhsT *lhs = nullptr;
    const rhsT *rhs = nullptr;
    resT *res = nullptr;
    LocAccT1 local_A_block;
    LocAccT2 local_B_block;
    size_t n = 0;
    size_t wg_delta_n = 0;
    size_t k = 0;
    size_t k_blocks = 0;
    size_t wi_delta_k = 0;
    size_t m = 0;
    size_t m_blocks = 0;
    size_t wg_delta_m = 0;
    size_t batch_nelems;
    BatchDimsIndexerT batch_indexer;
    OuterInnerDimsIndexerT lhs_indexer;
    OuterInnerDimsIndexerT rhs_indexer;
    ResIndexerT res_indexer;

public:
    GemmBatchNoAtomicFunctorThreadNM(const lhsT *lhs_,
                                     const rhsT *rhs_,
                                     resT *res_,
                                     LocAccT1 local_A_block_,
                                     LocAccT2 local_B_block_,
                                     size_t n_,
                                     size_t wg_delta_n_,
                                     size_t k_,
                                     size_t k_blocks_,
                                     size_t wi_delta_k_,
                                     size_t m_,
                                     size_t m_blocks_,
                                     size_t wg_delta_m_,
                                     size_t batch_nelems_,
                                     BatchDimsIndexerT batch_indexer_,
                                     OuterInnerDimsIndexerT lhs_indexer_,
                                     OuterInnerDimsIndexerT rhs_indexer_,
                                     ResIndexerT res_indexer_)
        : lhs(lhs_), rhs(rhs_), res(res_), local_A_block(local_A_block_),
          local_B_block(local_B_block_), n(n_), wg_delta_n(wg_delta_n_), k(k_),
          k_blocks(k_blocks_), wi_delta_k(wi_delta_k_), m(m_),
          m_blocks(m_blocks_), wg_delta_m(wg_delta_m_),
          batch_nelems(batch_nelems_), batch_indexer(batch_indexer_),
          lhs_indexer(lhs_indexer_), rhs_indexer(rhs_indexer_),
          res_indexer(res_indexer_)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        size_t m_id =
            it.get_global_linear_id() / (it.get_global_range(0) / batch_nelems);
        size_t gr_id =
            it.get_group_linear_id() % (it.get_group_range(0) / batch_nelems);

        const auto &three_offsets_ =
            batch_indexer(static_cast<py::ssize_t>(m_id));

        // lift group_id to (block_i, block_j, block_s),
        //    0 <= block_i < n_blocks, 0 <= block_j < m_blocks, 0 <= block_s
        //    < k_blocks

        const auto &lhs_offset = three_offsets_.get_first_offset();
        const auto &rhs_offset = three_offsets_.get_second_offset();
        const auto &res_offset = three_offsets_.get_third_offset();

        size_t block_i = gr_id / (m_blocks * k_blocks);
        size_t block_r = gr_id - block_i * (m_blocks * k_blocks);
        size_t block_j = block_r / k_blocks;
        size_t block_s = block_r - block_j * k_blocks;

        size_t lid = it.get_local_linear_id();
        size_t local_i = lid / wg_delta_m;           // 0<= local_i < wg_delta_n
        size_t local_j = lid - local_i * wg_delta_m; // 0<= local_j < wg_delta_m

        // load A block and B blocks into SLM

        size_t i = block_i * wi_delta_n * wg_delta_n;
        size_t j = block_j * wg_delta_m;
        size_t s = block_s * wi_delta_k;

        const std::int64_t a_st0 = k;
        const std::int64_t a_st1 = 1;

        const std::int64_t b_st0 = m;
        const std::int64_t b_st1 = 1;

        const std::int64_t c_st0 = m;
        const std::int64_t c_st1 = 1;

        size_t lws = it.get_local_range(0);

        for (size_t vid = lid; vid < local_A_block.size(); vid += lws) {
            size_t v_i = vid / wi_delta_k; // 0<= v_i < wg_delta_n * wi_delta_n
            size_t v_s = vid - v_i * wi_delta_k; // 0<= v_s < wi_delta_k

            size_t g_i = i + v_i;
            size_t g_s = s + v_s;

            local_A_block[vid] =
                (g_i < n && g_s < k)
                    ? static_cast<resT>(
                          lhs[lhs_offset +
                              lhs_indexer(g_i * a_st0 + g_s * a_st1)])
                    : resT(0);
        }

        for (size_t vid = lid; vid < local_B_block.size(); vid += lws) {
            size_t v_j = vid / wi_delta_k;       // 0<= v_i < wg_delta_m
            size_t v_s = vid - v_j * wi_delta_k; // 0<= v_s < wi_delta_k

            size_t g_j0 = j + v_j;
            size_t g_s = s + v_s;

            resT val = (g_j0 < m && g_s < k)
                           ? static_cast<resT>(
                                 rhs[rhs_offset +
                                     rhs_indexer(g_s * b_st0 + g_j0 * b_st1)])
                           : resT(0);

            local_B_block[vid] = val;
        }

        it.barrier(sycl::access::fence_space::local_space);

        i += local_i * wi_delta_n;
        j += local_j;

        size_t a_offset = local_i * wi_delta_k * wi_delta_n;
        size_t b_offset = local_j * wi_delta_k;

        constexpr resT identity_(0);
        for (std::uint8_t private_i = 0; private_i < wi_delta_n; ++private_i) {
            size_t a_pr_offset = private_i * wi_delta_k;

            resT local_sum(identity_);
            for (size_t private_s = 0; private_s < wi_delta_k; ++private_s) {
                local_sum = local_sum +
                            (local_A_block[a_offset + a_pr_offset + private_s] *
                             local_B_block[b_offset + private_s]);
            }

            size_t gl_i = i + private_i;

            if (gl_i < n && j < m) {
                res[res_offset + res_indexer(gl_i * c_st0 + j * c_st1) +
                    (block_s * n * m * batch_nelems)] = local_sum;
            }
        }
    }
};

template <typename lhsT,
          typename rhsT,
          typename resT,
          typename LocAccT,
          typename OuterInnerDimsIndexerT,
          typename ResIndexerT,
          typename BatchDimsIndexerT,
          size_t m_groups>
class GemmBatchNoAtomicFunctorThreadK
{
private:
    const lhsT *lhs = nullptr;
    const rhsT *rhs = nullptr;
    resT *res = nullptr;
    LocAccT workspace;
    LocAccT local_B_block;
    size_t n = 0;
    size_t n_blocks = 0;
    size_t delta_n = 0;
    size_t k = 0;
    size_t k_blocks = 0;
    size_t delta_k = 0;
    size_t n_wi = 0;
    size_t m = 0;
    size_t batch_nelems = 0;
    BatchDimsIndexerT batch_indexer;
    OuterInnerDimsIndexerT lhs_indexer;
    OuterInnerDimsIndexerT rhs_indexer;
    ResIndexerT res_indexer;

public:
    GemmBatchNoAtomicFunctorThreadK(const lhsT *lhs_,
                                    const rhsT *rhs_,
                                    resT *res_,
                                    LocAccT workspace_,
                                    LocAccT local_B_block_,
                                    size_t n_,
                                    size_t n_blocks_,
                                    size_t delta_n_,
                                    size_t k_,
                                    size_t k_blocks_,
                                    size_t delta_k_,
                                    size_t n_wi_,
                                    size_t m_,
                                    size_t batch_nelems_,
                                    BatchDimsIndexerT batch_indexer_,
                                    OuterInnerDimsIndexerT lhs_indexer_,
                                    OuterInnerDimsIndexerT rhs_indexer_,
                                    ResIndexerT res_indexer_)
        : lhs(lhs_), rhs(rhs_), res(res_), workspace(workspace_),
          local_B_block(local_B_block_), n(n_), n_blocks(n_blocks_),
          delta_n(delta_n_), k(k_), k_blocks(k_blocks_), delta_k(delta_k_),
          n_wi(n_wi_), m(m_), batch_nelems(batch_nelems_),
          batch_indexer(batch_indexer_), lhs_indexer(lhs_indexer_),
          rhs_indexer(rhs_indexer_), res_indexer(res_indexer_)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        size_t m_id =
            it.get_global_linear_id() / (it.get_global_range(0) / batch_nelems);
        size_t gr_id =
            it.get_group_linear_id() % (it.get_group_range(0) / batch_nelems);
        size_t lid = it.get_local_linear_id();

        const auto &three_offsets_ =
            batch_indexer(static_cast<py::ssize_t>(m_id));
        const auto &lhs_offset = three_offsets_.get_first_offset();
        const auto &rhs_offset = three_offsets_.get_second_offset();
        const auto &res_offset = three_offsets_.get_third_offset();

        // lift gr_id -> (block_i, block_j, block_s)
        //   block_i moves fastest, then block_s, then block_j

        size_t block_j =
            gr_id / (n_blocks * k_blocks); // 0 <= block_j < m_blocks
        size_t block_r =
            gr_id - block_j * (n_blocks *
                               k_blocks); // 0 <= block_r < n_blocks * k_blocks
        size_t block_s = block_r / n_blocks; // 0 <= block_s < k_blocks
        size_t block_i =
            block_r - block_s * n_blocks; // 0 <= block_i < n_blocks

        size_t local_i = lid / (delta_k);           // 0 <= local_i < delta_n
        size_t local_s = lid - local_i * (delta_k); // 0 <= local_s < delta_k

        size_t i = block_i * delta_n + local_i;
        size_t j = m_groups * block_j;
        size_t s = block_s * delta_k * n_wi + local_s;

        constexpr resT identity_ = resT(0);
        if (local_i == 0) {
            for (size_t q = 0; q < n_wi * delta_k; q += delta_k) {
                size_t sq = s + q;
                size_t sqmj = sq * m + j;
                local_B_block[local_s + q] =
                    (sq < k && j < m)
                        ? static_cast<resT>(rhs[rhs_offset + rhs_indexer(sqmj)])
                        : identity_;
            }
        }

        it.barrier(sycl::access::fence_space::local_space);

        size_t t_shift = block_s * delta_k * n_wi;
        size_t global_s_offset = i * k + t_shift;

        sycl::vec<resT, m_groups> private_sum(identity_);
        for (size_t t = local_s; t < local_B_block.size(); t += delta_k) {
            if (t + t_shift < k) {
                private_sum +=
                    (static_cast<resT>(
                         lhs[lhs_offset + lhs_indexer(global_s_offset + t)]) *
                     local_B_block[t]);
            }
        }

        size_t workspace_i_shift = local_i * delta_k;
        workspace[workspace_i_shift + local_s] = private_sum;

        it.barrier(sycl::access::fence_space::local_space);

        if (local_s == 0 && i < n) {
            sycl::vec<resT, m_groups> local_sum(workspace[workspace_i_shift]);
            for (size_t t = 1; t < delta_k; ++t) {
                local_sum += workspace[workspace_i_shift + t];
            }

            res[res_offset + res_indexer(i * m + j) +
                (block_s * n * m * batch_nelems)] = local_sum[0];

            if (j + 1 < m) {
                res[res_offset + res_indexer(i * m + j + 1) +
                    (block_s * n * m * batch_nelems)] = local_sum[1];
            }
        }
    }
};

template <typename lhsT,
          typename rhsT,
          typename resT,
          typename LocAccT,
          typename OuterInnerDimsIndexerT,
          typename ResIndexerT,
          typename BatchDimsIndexerT>
class GemmBatchNoAtomicFunctorThreadK<lhsT,
                                      rhsT,
                                      resT,
                                      LocAccT,
                                      OuterInnerDimsIndexerT,
                                      ResIndexerT,
                                      BatchDimsIndexerT,
                                      1>
{
private:
    const lhsT *lhs = nullptr;
    const rhsT *rhs = nullptr;
    resT *res = nullptr;
    LocAccT workspace;
    LocAccT local_B_block;
    size_t n = 0;
    size_t n_blocks = 0;
    size_t delta_n = 0;
    size_t k = 0;
    size_t k_blocks = 0;
    size_t delta_k = 0;
    size_t n_wi = 0;
    size_t m = 0;
    size_t batch_nelems = 0;
    BatchDimsIndexerT batch_indexer;
    OuterInnerDimsIndexerT lhs_indexer;
    OuterInnerDimsIndexerT rhs_indexer;
    ResIndexerT res_indexer;

public:
    GemmBatchNoAtomicFunctorThreadK(const lhsT *lhs_,
                                    const rhsT *rhs_,
                                    resT *res_,
                                    LocAccT workspace_,
                                    LocAccT local_B_block_,
                                    size_t n_,
                                    size_t n_blocks_,
                                    size_t delta_n_,
                                    size_t k_,
                                    size_t k_blocks_,
                                    size_t delta_k_,
                                    size_t n_wi_,
                                    size_t m_,
                                    size_t batch_nelems_,
                                    BatchDimsIndexerT batch_indexer_,
                                    OuterInnerDimsIndexerT lhs_indexer_,
                                    OuterInnerDimsIndexerT rhs_indexer_,
                                    ResIndexerT res_indexer_)
        : lhs(lhs_), rhs(rhs_), res(res_), workspace(workspace_),
          local_B_block(local_B_block_), n(n_), n_blocks(n_blocks_),
          delta_n(delta_n_), k(k_), k_blocks(k_blocks_), delta_k(delta_k_),
          n_wi(n_wi_), m(m_), batch_nelems(batch_nelems_),
          batch_indexer(batch_indexer_), lhs_indexer(lhs_indexer_),
          rhs_indexer(rhs_indexer_), res_indexer(res_indexer_)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        size_t m_id =
            it.get_global_linear_id() / (it.get_global_range(0) / batch_nelems);
        size_t gr_id =
            it.get_group_linear_id() % (it.get_group_range(0) / batch_nelems);
        size_t lid = it.get_local_linear_id();

        const auto &three_offsets_ =
            batch_indexer(static_cast<py::ssize_t>(m_id));
        const auto &lhs_offset = three_offsets_.get_first_offset();
        const auto &rhs_offset = three_offsets_.get_second_offset();
        const auto &res_offset = three_offsets_.get_third_offset();

        // lift gr_id -> (block_i, block_j, block_s)
        //   block_i moves fastest, then block_s, then block_j

        size_t block_j =
            gr_id / (n_blocks * k_blocks); // 0 <= block_j < m_blocks
        size_t block_r =
            gr_id - block_j * (n_blocks *
                               k_blocks); // 0 <= block_r < n_blocks * k_blocks
        size_t block_s = block_r / n_blocks; // 0 <= block_s < k_blocks
        size_t block_i =
            block_r - block_s * n_blocks; // 0 <= block_i < n_blocks

        size_t local_i = lid / (delta_k);           // 0 <= local_i < delta_n
        size_t local_s = lid - local_i * (delta_k); // 0 <= local_s < delta_k

        size_t i = block_i * delta_n + local_i;
        size_t j = block_j;
        size_t s = block_s * delta_k * n_wi + local_s;

        constexpr resT identity_ = resT(0);
        if (local_i == 0) {
            for (size_t q = 0; q < n_wi * delta_k; q += delta_k) {
                size_t sq = s + q;
                size_t sqmj = sq * m + j;
                local_B_block[local_s + q] =
                    (sq < k && j < m)
                        ? static_cast<resT>(rhs[rhs_offset + rhs_indexer(sqmj)])
                        : identity_;
            }
        }

        it.barrier(sycl::access::fence_space::local_space);

        size_t t_shift = block_s * delta_k * n_wi;
        size_t global_s_offset = i * k + t_shift;

        resT private_sum(identity_);
        for (size_t t = local_s; t < local_B_block.size(); t += delta_k) {
            if (t + t_shift < k) {
                private_sum +=
                    (static_cast<resT>(
                         lhs[lhs_offset + lhs_indexer(global_s_offset + t)]) *
                     local_B_block[t]);
            }
        }

        size_t workspace_i_shift = local_i * delta_k;
        workspace[workspace_i_shift + local_s] = private_sum;

        it.barrier(sycl::access::fence_space::local_space);

        if (local_s == 0 && i < n) {
            resT local_sum(workspace[workspace_i_shift]);
            for (size_t t = 1; t < delta_k; ++t) {
                local_sum += workspace[workspace_i_shift + t];
            }

            res[res_offset + res_indexer(i * m + j) +
                (block_s * n * m * batch_nelems)] = local_sum;
        }
    }
};

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6,
          size_t>
class gemm_batch_tree_k_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6,
          size_t>
class gemm_batch_tree_nm_krn;

template <typename lhsTy, typename rhsTy, typename resTy, size_t m_groups>
sycl::event
gemm_batch_tree_k_impl(sycl::queue &exec_q,
                       const lhsTy *lhs_tp,
                       const rhsTy *rhs_tp,
                       resTy *res_tp,
                       size_t batch_nelems,
                       size_t n,
                       size_t k,
                       size_t m,
                       int batch_nd,
                       const py::ssize_t *batch_shape_strides,
                       py::ssize_t lhs_batch_offset,
                       py::ssize_t rhs_batch_offset,
                       py::ssize_t res_batch_offset,
                       int inner_nd,
                       int lhs_outer_nd,
                       const py::ssize_t *lhs_outer_inner_shapes_strides,
                       int rhs_outer_nd,
                       const py::ssize_t *rhs_outer_inner_shapes_strides,
                       int res_outer_nd,
                       const py::ssize_t *res_outer_shapes_strides,
                       const py::ssize_t *res_shape_strides,
                       std::vector<sycl::event> const &depends)
{
    size_t delta_k(4);
    size_t n_wi(4);
    size_t delta_n(4);

    const sycl::device &dev = exec_q.get_device();
    const size_t local_mem_size =
        dev.get_info<sycl::info::device::local_mem_size>();
    const size_t reserved_slm_size = 512;

    gemm_detail::scale_gemm_k_parameters<resTy, m_groups>(
        local_mem_size, reserved_slm_size, delta_k,
        n_wi,   // modified by reference
        delta_n // modified by reference
    );

    if (k <= (delta_k * n_wi)) {
        sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            using OuterInnerDimsIndexerT =
                dpctl::tensor::offset_utils::StridedIndexer;
            OuterInnerDimsIndexerT lhs_indexer(inner_nd + lhs_outer_nd, 0,
                                               lhs_outer_inner_shapes_strides);
            OuterInnerDimsIndexerT rhs_indexer(inner_nd + rhs_outer_nd, 0,
                                               rhs_outer_inner_shapes_strides);
            OuterInnerDimsIndexerT res_indexer(res_outer_nd, 0,
                                               res_outer_shapes_strides);
            using BatchDimsIndexerT =
                dpctl::tensor::offset_utils::ThreeOffsets_StridedIndexer;
            BatchDimsIndexerT batch_indexer(batch_nd, lhs_batch_offset,
                                            rhs_batch_offset, res_batch_offset,
                                            batch_shape_strides);

            size_t n_blocks = (n + delta_n - 1) / delta_n;
            size_t k_blocks = (k + n_wi * delta_k - 1) / (n_wi * delta_k);
            size_t m_blocks = (m + m_groups - 1) / m_groups;

            size_t lws = delta_n * delta_k;

            auto gRange = sycl::range<1>(batch_nelems * n_blocks * m_blocks *
                                         k_blocks * lws);
            auto lRange = sycl::range<1>(lws);

            auto ndRange = sycl::nd_range<1>(gRange, lRange);

            if constexpr (m_groups == 1) {
                using LocAccT = sycl::local_accessor<resTy, 1>;
                LocAccT local_B_block(n_wi * delta_k, cgh);
                LocAccT workspace(delta_n * delta_k, cgh);

                using KernelName = class gemm_batch_tree_k_krn<
                    lhsTy, rhsTy, resTy, OuterInnerDimsIndexerT,
                    OuterInnerDimsIndexerT, BatchDimsIndexerT, m_groups>;

                cgh.parallel_for<KernelName>(
                    ndRange,
                    GemmBatchNoAtomicFunctorThreadK<
                        lhsTy, rhsTy, resTy, LocAccT, OuterInnerDimsIndexerT,
                        OuterInnerDimsIndexerT, BatchDimsIndexerT, m_groups>(
                        lhs_tp, rhs_tp, res_tp, workspace, local_B_block, n,
                        n_blocks, delta_n, k, k_blocks, delta_k, n_wi, m,
                        batch_nelems, batch_indexer, lhs_indexer, rhs_indexer,
                        res_indexer));
            }
            else {
                using LocAccT =
                    sycl::local_accessor<sycl::vec<resTy, m_groups>, 1>;
                LocAccT local_B_block(n_wi * delta_k, cgh);
                LocAccT workspace(delta_n * delta_k, cgh);

                using KernelName = class gemm_batch_tree_k_krn<
                    lhsTy, rhsTy, resTy, OuterInnerDimsIndexerT,
                    OuterInnerDimsIndexerT, BatchDimsIndexerT, m_groups>;

                cgh.parallel_for<KernelName>(
                    ndRange,
                    GemmBatchNoAtomicFunctorThreadK<
                        lhsTy, rhsTy, resTy, LocAccT, OuterInnerDimsIndexerT,
                        OuterInnerDimsIndexerT, BatchDimsIndexerT, m_groups>(
                        lhs_tp, rhs_tp, res_tp, workspace, local_B_block, n,
                        n_blocks, delta_n, k, k_blocks, delta_k, n_wi, m,
                        batch_nelems, batch_indexer, lhs_indexer, rhs_indexer,
                        res_indexer));
            }
        });
        return gemm_ev;
    }
    else {
        using ReductionOpT = sycl::plus<resTy>;
        constexpr resTy identity_val =
            sycl::known_identity<ReductionOpT, resTy>::value;

        size_t iter_nelems = batch_nelems * n * m;
        size_t reduction_nelems = (k + delta_k * n_wi - 1) / (delta_k * n_wi);

        // more than one work-group is needed, requires a
        // temporary delta_k * n_wi elements processed along k,
        // so if more to process use multiple
        const auto &sg_sizes =
            dev.get_info<sycl::info::device::sub_group_sizes>();
        size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

        if (reduction_nelems < wg) {
            resTy *tmp = sycl::malloc_device<resTy>(
                iter_nelems * reduction_nelems, exec_q);
            if (!tmp) {
                throw std::runtime_error("Unable to allocate device memory");
            }
            sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(depends);

                using OuterInnerDimsIndexerT =
                    dpctl::tensor::offset_utils::StridedIndexer;
                using TmpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
                OuterInnerDimsIndexerT lhs_indexer(
                    inner_nd + lhs_outer_nd, 0, lhs_outer_inner_shapes_strides);
                OuterInnerDimsIndexerT rhs_indexer(
                    inner_nd + rhs_outer_nd, 0, rhs_outer_inner_shapes_strides);
                TmpIndexerT res_indexer{};
                using dpctl::tensor::offset_utils::StridedIndexer;
                using dpctl::tensor::offset_utils::UnpackedStridedIndexer;
                using dpctl::tensor::offset_utils::Strided1DIndexer;
                using dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer;
                using BatchDimsIndexerT = ThreeOffsets_CombinedIndexer<
                    StridedIndexer, UnpackedStridedIndexer, Strided1DIndexer>;
                StridedIndexer lhs_batch_indexer(batch_nd, lhs_batch_offset,
                                                 batch_shape_strides);
                UnpackedStridedIndexer rhs_batch_indexer(
                    batch_nd, rhs_batch_offset, batch_shape_strides,
                    batch_shape_strides + 2 * batch_nd);
                Strided1DIndexer tmp_batch_indexer(
                    0, static_cast<py::ssize_t>(batch_nelems), n * m);
                BatchDimsIndexerT batch_indexer(
                    lhs_batch_indexer, rhs_batch_indexer, tmp_batch_indexer);

                size_t n_blocks = (n + delta_n - 1) / delta_n;
                size_t k_blocks = (k + n_wi * delta_k - 1) / (n_wi * delta_k);
                size_t m_blocks = (m + m_groups - 1) / m_groups;

                size_t lws = delta_n * delta_k;

                auto gRange = sycl::range<1>(batch_nelems * n_blocks *
                                             m_blocks * k_blocks * lws);
                auto lRange = sycl::range<1>(lws);

                auto ndRange = sycl::nd_range<1>(gRange, lRange);

                if constexpr (m_groups == 1) {
                    using LocAccT = sycl::local_accessor<resTy, 1>;
                    LocAccT local_B_block(n_wi * delta_k, cgh);
                    LocAccT workspace(delta_n * delta_k, cgh);

                    using KernelName = class gemm_batch_tree_k_krn<
                        lhsTy, rhsTy, resTy, OuterInnerDimsIndexerT,
                        TmpIndexerT, BatchDimsIndexerT, m_groups>;
                    cgh.parallel_for<KernelName>(
                        ndRange,
                        GemmBatchNoAtomicFunctorThreadK<
                            lhsTy, rhsTy, resTy, LocAccT,
                            OuterInnerDimsIndexerT, TmpIndexerT,
                            BatchDimsIndexerT, m_groups>(
                            lhs_tp, rhs_tp, tmp, workspace, local_B_block, n,
                            n_blocks, delta_n, k, k_blocks, delta_k, n_wi, m,
                            batch_nelems, batch_indexer, lhs_indexer,
                            rhs_indexer, res_indexer));
                }
                else {
                    using LocAccT =
                        sycl::local_accessor<sycl::vec<resTy, m_groups>, 1>;
                    LocAccT local_B_block(n_wi * delta_k, cgh);
                    LocAccT workspace(delta_n * delta_k, cgh);

                    using KernelName = class gemm_batch_tree_k_krn<
                        lhsTy, rhsTy, resTy, OuterInnerDimsIndexerT,
                        TmpIndexerT, BatchDimsIndexerT, m_groups>;
                    cgh.parallel_for<KernelName>(
                        ndRange,
                        GemmBatchNoAtomicFunctorThreadK<
                            lhsTy, rhsTy, resTy, LocAccT,
                            OuterInnerDimsIndexerT, TmpIndexerT,
                            BatchDimsIndexerT, m_groups>(
                            lhs_tp, rhs_tp, tmp, workspace, local_B_block, n,
                            n_blocks, delta_n, k, k_blocks, delta_k, n_wi, m,
                            batch_nelems, batch_indexer, lhs_indexer,
                            rhs_indexer, res_indexer));
                }
            });
            sycl::event red_ev = exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(gemm_ev);

                using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
                using ResIndexerT = dpctl::tensor::offset_utils::StridedIndexer;
                using InputOutputIterIndexerT =
                    dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                        NoOpIndexerT, ResIndexerT>;
                using ReductionIndexerT =
                    dpctl::tensor::offset_utils::Strided1DIndexer;

                ResIndexerT res_iter_indexer{
                    batch_nd + res_outer_nd,
                    static_cast<py::ssize_t>(res_batch_offset),
                    res_shape_strides};
                InputOutputIterIndexerT in_out_iter_indexer{NoOpIndexerT{},
                                                            res_iter_indexer};
                ReductionIndexerT reduction_indexer{
                    0, static_cast<py::ssize_t>(reduction_nelems),
                    static_cast<py::ssize_t>(iter_nelems)};

                sycl::range<1> iter_range{iter_nelems};

                cgh.parallel_for<class gemm_reduction_seq_strided_krn<
                    resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                    ReductionIndexerT>>(
                    iter_range, SequentialReduction<resTy, resTy, ReductionOpT,
                                                    InputOutputIterIndexerT,
                                                    ReductionIndexerT>(
                                    tmp, res_tp, ReductionOpT(), identity_val,
                                    in_out_iter_indexer, reduction_indexer,
                                    reduction_nelems));
            });
            sycl::event cleanup_host_task_event =
                exec_q.submit([&](sycl::handler &cgh) {
                    cgh.depends_on(red_ev);
                    const sycl::context &ctx = exec_q.get_context();

                    cgh.host_task([ctx, tmp] { sycl::free(tmp, ctx); });
                });
            return cleanup_host_task_event;
        }

        // max_max_wg prevents running out of resources on CPU
        constexpr size_t max_max_wg = 2048;
        size_t max_wg = std::min(
            max_max_wg,
            dev.get_info<sycl::info::device::max_work_group_size>() / 2);

        constexpr size_t preferred_reductions_per_wi = 4;
        size_t reductions_per_wi(preferred_reductions_per_wi);

        size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);
        assert(reduction_groups > 1);

        resTy *partially_reduced_tmp = sycl::malloc_device<resTy>(
            iter_nelems * (/* temp */ reduction_nelems +
                           /* first reduction temp */ reduction_groups),
            exec_q);
        resTy *partially_reduced_tmp2 = nullptr;

        if (partially_reduced_tmp == nullptr) {
            throw std::runtime_error("Unable to allocate device_memory");
        }
        else {
            partially_reduced_tmp2 =
                partially_reduced_tmp + reduction_nelems * iter_nelems;
        }

        sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            using OuterInnerDimsIndexerT =
                dpctl::tensor::offset_utils::StridedIndexer;
            using TmpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            OuterInnerDimsIndexerT lhs_indexer(inner_nd + lhs_outer_nd, 0,
                                               lhs_outer_inner_shapes_strides);
            OuterInnerDimsIndexerT rhs_indexer(inner_nd + rhs_outer_nd, 0,
                                               rhs_outer_inner_shapes_strides);
            TmpIndexerT res_indexer{};
            using dpctl::tensor::offset_utils::StridedIndexer;
            using dpctl::tensor::offset_utils::Strided1DIndexer;
            using dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer;
            using BatchDimsIndexerT =
                ThreeOffsets_CombinedIndexer<StridedIndexer, StridedIndexer,
                                             Strided1DIndexer>;
            StridedIndexer lhs_batch_indexer(batch_nd, lhs_batch_offset,
                                             batch_shape_strides);
            StridedIndexer rhs_batch_indexer(
                batch_nd, rhs_batch_offset, batch_shape_strides + 2 * batch_nd);
            Strided1DIndexer tmp_batch_indexer(
                0, static_cast<py::ssize_t>(batch_nelems), n * m);
            BatchDimsIndexerT batch_indexer(
                lhs_batch_indexer, rhs_batch_indexer, tmp_batch_indexer);

            size_t n_blocks = (n + delta_n - 1) / delta_n;
            size_t k_blocks = (k + n_wi * delta_k - 1) / (n_wi * delta_k);
            size_t m_blocks = (m + m_groups - 1) / m_groups;

            size_t lws = delta_n * delta_k;

            auto gRange = sycl::range<1>(batch_nelems * n_blocks * m_blocks *
                                         k_blocks * lws);
            auto lRange = sycl::range<1>(lws);

            auto ndRange = sycl::nd_range<1>(gRange, lRange);

            if constexpr (m_groups == 1) {
                using LocAccT = sycl::local_accessor<resTy, 1>;
                LocAccT local_B_block(n_wi * delta_k, cgh);
                LocAccT workspace(delta_n * delta_k, cgh);

                using KernelName = class gemm_batch_tree_k_krn<
                    lhsTy, rhsTy, resTy, OuterInnerDimsIndexerT, TmpIndexerT,
                    BatchDimsIndexerT, m_groups>;

                cgh.parallel_for<KernelName>(
                    ndRange,
                    GemmBatchNoAtomicFunctorThreadK<
                        lhsTy, rhsTy, resTy, LocAccT, OuterInnerDimsIndexerT,
                        TmpIndexerT, BatchDimsIndexerT, m_groups>(
                        lhs_tp, rhs_tp, partially_reduced_tmp, workspace,
                        local_B_block, n, n_blocks, delta_n, k, k_blocks,
                        delta_k, n_wi, m, batch_nelems, batch_indexer,
                        lhs_indexer, rhs_indexer, res_indexer));
            }
            else {
                using LocAccT =
                    sycl::local_accessor<sycl::vec<resTy, m_groups>, 1>;
                LocAccT local_B_block(n_wi * delta_k, cgh);
                LocAccT workspace(delta_n * delta_k, cgh);

                using KernelName = class gemm_batch_tree_k_krn<
                    lhsTy, rhsTy, resTy, OuterInnerDimsIndexerT, TmpIndexerT,
                    BatchDimsIndexerT, m_groups>;
                cgh.parallel_for<KernelName>(
                    ndRange,
                    GemmBatchNoAtomicFunctorThreadK<
                        lhsTy, rhsTy, resTy, LocAccT, OuterInnerDimsIndexerT,
                        TmpIndexerT, BatchDimsIndexerT, m_groups>(
                        lhs_tp, rhs_tp, partially_reduced_tmp, workspace,
                        local_B_block, n, n_blocks, delta_n, k, k_blocks,
                        delta_k, n_wi, m, batch_nelems, batch_indexer,
                        lhs_indexer, rhs_indexer, res_indexer));
            }
        });

        sycl::event red_ev = tree_reduction_for_gemm<resTy, ReductionOpT>(
            exec_q, partially_reduced_tmp, partially_reduced_tmp2, res_tp,
            identity_val, iter_nelems, reduction_nelems, reduction_groups, wg,
            max_wg, preferred_reductions_per_wi, reductions_per_wi,
            batch_nd + res_outer_nd, res_batch_offset, res_shape_strides,
            {gemm_ev});

        sycl::event cleanup_host_task_event =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(red_ev);
                const sycl::context &ctx = exec_q.get_context();

                cgh.host_task([ctx, partially_reduced_tmp] {
                    sycl::free(partially_reduced_tmp, ctx);
                });
            });

        return cleanup_host_task_event;
    }
}

template <typename lhsTy, typename rhsTy, typename resTy, int wi_delta_m>
sycl::event
gemm_batch_tree_nm_impl(sycl::queue &exec_q,
                        const lhsTy *lhs_tp,
                        const rhsTy *rhs_tp,
                        resTy *res_tp,
                        size_t batch_nelems,
                        size_t n,
                        size_t k,
                        size_t m,
                        int batch_nd,
                        const py::ssize_t *batch_shape_strides,
                        py::ssize_t lhs_batch_offset,
                        py::ssize_t rhs_batch_offset,
                        py::ssize_t res_batch_offset,
                        int inner_nd,
                        int lhs_outer_nd,
                        const py::ssize_t *lhs_outer_inner_shapes_strides,
                        int rhs_outer_nd,
                        const py::ssize_t *rhs_outer_inner_shapes_strides,
                        int res_outer_nd,
                        const py::ssize_t *res_outer_shapes_strides,
                        const py::ssize_t *res_shape_strides,
                        std::vector<sycl::event> const &depends)
{
    constexpr int wi_delta_n = 2;
    size_t wg_delta_n(16); // rows of A processed in WG
    size_t wg_delta_m(16); // rows of B processed in WG
    size_t wi_delta_k(64); // Elements in K dimension processed by WI

    const sycl::device &dev = exec_q.get_device();
    const size_t local_mem_size =
        dev.get_info<sycl::info::device::local_mem_size>();
    const size_t reserved_slm_size = 512;

    gemm_detail::scale_gemm_nm_parameters<resTy, wi_delta_m>(
        local_mem_size, reserved_slm_size, wi_delta_n,
        wi_delta_k, // modified by reference
        wg_delta_n, // modified by reference
        wg_delta_m  // modified by reference
    );

    // each group processes delta_k * n_wi
    // items in a column, so no need for allocating
    // temp memory if only one group is needed
    if (k <= wi_delta_k) {
        sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            using OuterInnerDimsIndexerT =
                dpctl::tensor::offset_utils::StridedIndexer;
            OuterInnerDimsIndexerT lhs_indexer(inner_nd + lhs_outer_nd, 0,
                                               lhs_outer_inner_shapes_strides);
            OuterInnerDimsIndexerT rhs_indexer(inner_nd + rhs_outer_nd, 0,
                                               rhs_outer_inner_shapes_strides);
            OuterInnerDimsIndexerT res_indexer(res_outer_nd, 0,
                                               res_outer_shapes_strides);
            using BatchDimsIndexerT =
                dpctl::tensor::offset_utils::ThreeOffsets_StridedIndexer;
            BatchDimsIndexerT batch_indexer(batch_nd, lhs_batch_offset,
                                            rhs_batch_offset, res_batch_offset,
                                            batch_shape_strides);

            size_t lws = wg_delta_n * wg_delta_m;

            size_t n_blocks =
                ((n + wi_delta_n * wg_delta_n - 1) / (wi_delta_n * wg_delta_n));
            size_t k_blocks = ((k + wi_delta_k - 1) / wi_delta_k);
            size_t m_blocks =
                ((m + wi_delta_m * wg_delta_m - 1) / (wi_delta_m * wg_delta_m));

            auto gwsRange = sycl::range<1>(batch_nelems * n_blocks * m_blocks *
                                           k_blocks * lws);
            auto lwsRange = sycl::range<1>(lws);

            auto ndRange = sycl::nd_range<1>(gwsRange, lwsRange);

            if constexpr (wi_delta_m == 1) {
                using LocAccT1 = sycl::local_accessor<resTy, 1>;
                LocAccT1 local_A_block(
                    sycl::range<1>((wi_delta_n * wg_delta_n) * wi_delta_k),
                    cgh);
                using LocAccT2 = sycl::local_accessor<resTy, 1>;
                LocAccT2 local_B_block(sycl::range<1>(wi_delta_k * wg_delta_m),
                                       cgh);

                using KernelName = class gemm_batch_tree_nm_krn<
                    lhsTy, rhsTy, resTy, OuterInnerDimsIndexerT,
                    OuterInnerDimsIndexerT, BatchDimsIndexerT, wi_delta_m>;
                cgh.parallel_for<KernelName>(
                    ndRange,
                    GemmBatchNoAtomicFunctorThreadNM<
                        lhsTy, rhsTy, resTy, LocAccT1, LocAccT2,
                        OuterInnerDimsIndexerT, OuterInnerDimsIndexerT,
                        BatchDimsIndexerT, wi_delta_n, wi_delta_m>(
                        lhs_tp, rhs_tp, res_tp, local_A_block, local_B_block, n,
                        wg_delta_n, k, k_blocks, wi_delta_k, m, m_blocks,
                        wg_delta_m, batch_nelems, batch_indexer, lhs_indexer,
                        rhs_indexer, res_indexer));
            }
            else {
                using LocAccT1 = sycl::local_accessor<resTy, 1>;
                LocAccT1 local_A_block(
                    sycl::range<1>((wi_delta_n * wg_delta_n) * wi_delta_k),
                    cgh);
                using LocAccT2 =
                    sycl::local_accessor<sycl::vec<resTy, wi_delta_m>, 1>;
                LocAccT2 local_B_block(sycl::range<1>(wi_delta_k * wg_delta_m),
                                       cgh);

                using KernelName = class gemm_batch_tree_nm_krn<
                    lhsTy, rhsTy, resTy, OuterInnerDimsIndexerT,
                    OuterInnerDimsIndexerT, BatchDimsIndexerT, wi_delta_m>;
                cgh.parallel_for<KernelName>(
                    ndRange,
                    GemmBatchNoAtomicFunctorThreadNM<
                        lhsTy, rhsTy, resTy, LocAccT1, LocAccT2,
                        OuterInnerDimsIndexerT, OuterInnerDimsIndexerT,
                        BatchDimsIndexerT, wi_delta_n, wi_delta_m>(
                        lhs_tp, rhs_tp, res_tp, local_A_block, local_B_block, n,
                        wg_delta_n, k, k_blocks, wi_delta_k, m, m_blocks,
                        wg_delta_m, batch_nelems, batch_indexer, lhs_indexer,
                        rhs_indexer, res_indexer));
            }
        });
        return gemm_ev;
    }
    else {
        using ReductionOpT = sycl::plus<resTy>;
        constexpr resTy identity_val =
            sycl::known_identity<ReductionOpT, resTy>::value;
        size_t iter_nelems = batch_nelems * n * m;
        size_t reduction_nelems = (k + wi_delta_k - 1) / wi_delta_k;

        // more than one work-group is needed, requires a temporary
        // delta_k * n_wi elements processed along k, so if more to
        // process use multiple
        const auto &sg_sizes =
            dev.get_info<sycl::info::device::sub_group_sizes>();
        size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

        if (reduction_nelems < wg) {
            resTy *tmp = sycl::malloc_device<resTy>(
                iter_nelems * reduction_nelems, exec_q);
            if (!tmp) {
                throw std::runtime_error("Unable to allocate device memory");
            }
            sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(depends);

                using OuterInnerDimsIndexerT =
                    dpctl::tensor::offset_utils::StridedIndexer;
                using TmpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
                OuterInnerDimsIndexerT lhs_indexer(
                    inner_nd + lhs_outer_nd, 0, lhs_outer_inner_shapes_strides);
                OuterInnerDimsIndexerT rhs_indexer(
                    inner_nd + rhs_outer_nd, 0, rhs_outer_inner_shapes_strides);
                TmpIndexerT res_indexer{};
                using dpctl::tensor::offset_utils::StridedIndexer;
                using dpctl::tensor::offset_utils::UnpackedStridedIndexer;
                using dpctl::tensor::offset_utils::Strided1DIndexer;
                using dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer;
                using BatchDimsIndexerT = ThreeOffsets_CombinedIndexer<
                    StridedIndexer, UnpackedStridedIndexer, Strided1DIndexer>;
                StridedIndexer lhs_batch_indexer(batch_nd, lhs_batch_offset,
                                                 batch_shape_strides);
                UnpackedStridedIndexer rhs_batch_indexer(
                    batch_nd, rhs_batch_offset, batch_shape_strides,
                    batch_shape_strides + 2 * batch_nd);
                Strided1DIndexer tmp_batch_indexer(
                    0, static_cast<py::ssize_t>(batch_nelems), n * m);
                BatchDimsIndexerT batch_indexer(
                    lhs_batch_indexer, rhs_batch_indexer, tmp_batch_indexer);

                size_t lws = wg_delta_n * wg_delta_m;

                size_t n_blocks = ((n + wi_delta_n * wg_delta_n - 1) /
                                   (wi_delta_n * wg_delta_n));
                size_t k_blocks = ((k + wi_delta_k - 1) / wi_delta_k);
                size_t m_blocks = ((m + wi_delta_m * wg_delta_m - 1) /
                                   (wi_delta_m * wg_delta_m));

                auto gwsRange = sycl::range<1>(batch_nelems * n_blocks *
                                               m_blocks * k_blocks * lws);
                auto lwsRange = sycl::range<1>(lws);

                auto ndRange = sycl::nd_range<1>(gwsRange, lwsRange);

                if constexpr (wi_delta_m == 1) {
                    using LocAccT1 = sycl::local_accessor<resTy, 1>;
                    LocAccT1 local_A_block(
                        sycl::range<1>((wi_delta_n * wg_delta_n) * wi_delta_k),
                        cgh);
                    using LocAccT2 = sycl::local_accessor<resTy, 1>;
                    LocAccT2 local_B_block(
                        sycl::range<1>(wi_delta_k * wg_delta_m), cgh);

                    using KernelName = class gemm_batch_tree_nm_krn<
                        lhsTy, rhsTy, resTy, OuterInnerDimsIndexerT,
                        TmpIndexerT, BatchDimsIndexerT, wi_delta_m>;
                    cgh.parallel_for<KernelName>(
                        ndRange,
                        GemmBatchNoAtomicFunctorThreadNM<
                            lhsTy, rhsTy, resTy, LocAccT1, LocAccT2,
                            OuterInnerDimsIndexerT, TmpIndexerT,
                            BatchDimsIndexerT, wi_delta_n, wi_delta_m>(
                            lhs_tp, rhs_tp, tmp, local_A_block, local_B_block,
                            n, wg_delta_n, k, k_blocks, wi_delta_k, m, m_blocks,
                            wg_delta_m, batch_nelems, batch_indexer,
                            lhs_indexer, rhs_indexer, res_indexer));
                }
                else {
                    using LocAccT1 = sycl::local_accessor<resTy, 1>;
                    LocAccT1 local_A_block(
                        sycl::range<1>((wi_delta_n * wg_delta_n) * wi_delta_k),
                        cgh);
                    using LocAccT2 =
                        sycl::local_accessor<sycl::vec<resTy, wi_delta_m>, 1>;
                    LocAccT2 local_B_block(
                        sycl::range<1>(wi_delta_k * wg_delta_m), cgh);

                    using KernelName = class gemm_batch_tree_nm_krn<
                        lhsTy, rhsTy, resTy, OuterInnerDimsIndexerT,
                        TmpIndexerT, BatchDimsIndexerT, wi_delta_m>;
                    cgh.parallel_for<KernelName>(
                        ndRange,
                        GemmBatchNoAtomicFunctorThreadNM<
                            lhsTy, rhsTy, resTy, LocAccT1, LocAccT2,
                            OuterInnerDimsIndexerT, TmpIndexerT,
                            BatchDimsIndexerT, wi_delta_n, wi_delta_m>(
                            lhs_tp, rhs_tp, tmp, local_A_block, local_B_block,
                            n, wg_delta_n, k, k_blocks, wi_delta_k, m, m_blocks,
                            wg_delta_m, batch_nelems, batch_indexer,
                            lhs_indexer, rhs_indexer, res_indexer));
                }
            });
            sycl::event red_ev = exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(gemm_ev);

                using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
                using ResIndexerT = dpctl::tensor::offset_utils::StridedIndexer;
                using InputOutputIterIndexerT =
                    dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                        NoOpIndexerT, ResIndexerT>;
                using ReductionIndexerT =
                    dpctl::tensor::offset_utils::Strided1DIndexer;

                ResIndexerT res_iter_indexer{
                    batch_nd + res_outer_nd,
                    static_cast<py::ssize_t>(res_batch_offset),
                    res_shape_strides};
                InputOutputIterIndexerT in_out_iter_indexer{NoOpIndexerT{},
                                                            res_iter_indexer};
                ReductionIndexerT reduction_indexer{
                    0, static_cast<py::ssize_t>(reduction_nelems),
                    static_cast<py::ssize_t>(iter_nelems)};

                sycl::range<1> iter_range{iter_nelems};

                cgh.parallel_for<class gemm_reduction_seq_strided_krn<
                    resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                    ReductionIndexerT>>(
                    iter_range, SequentialReduction<resTy, resTy, ReductionOpT,
                                                    InputOutputIterIndexerT,
                                                    ReductionIndexerT>(
                                    tmp, res_tp, ReductionOpT(), identity_val,
                                    in_out_iter_indexer, reduction_indexer,
                                    reduction_nelems));
            });
            sycl::event cleanup_host_task_event =
                exec_q.submit([&](sycl::handler &cgh) {
                    cgh.depends_on(red_ev);
                    const sycl::context &ctx = exec_q.get_context();

                    cgh.host_task([ctx, tmp] { sycl::free(tmp, ctx); });
                });
            return cleanup_host_task_event;
        }

        // max_max_wg prevents running out of resources on CPU
        constexpr size_t max_max_wg = 2048;
        size_t max_wg = std::min(
            max_max_wg,
            dev.get_info<sycl::info::device::max_work_group_size>() / 2);

        constexpr size_t preferred_reductions_per_wi = 4;
        size_t reductions_per_wi(preferred_reductions_per_wi);

        size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);
        assert(reduction_groups > 1);

        resTy *partially_reduced_tmp = sycl::malloc_device<resTy>(
            iter_nelems * (/* temp */ reduction_nelems +
                           /* first reduction temp */ reduction_groups),
            exec_q);
        resTy *partially_reduced_tmp2 = nullptr;

        if (partially_reduced_tmp == nullptr) {
            throw std::runtime_error("Unable to allocate device_memory");
        }
        else {
            partially_reduced_tmp2 =
                partially_reduced_tmp + reduction_nelems * iter_nelems;
        }

        sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            using OuterInnerDimsIndexerT =
                dpctl::tensor::offset_utils::StridedIndexer;
            using TmpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            OuterInnerDimsIndexerT lhs_indexer(inner_nd + lhs_outer_nd, 0,
                                               lhs_outer_inner_shapes_strides);
            OuterInnerDimsIndexerT rhs_indexer(inner_nd + rhs_outer_nd, 0,
                                               rhs_outer_inner_shapes_strides);
            TmpIndexerT res_indexer{};
            using dpctl::tensor::offset_utils::StridedIndexer;
            using dpctl::tensor::offset_utils::UnpackedStridedIndexer;
            using dpctl::tensor::offset_utils::Strided1DIndexer;
            using dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer;
            using BatchDimsIndexerT = ThreeOffsets_CombinedIndexer<
                StridedIndexer, UnpackedStridedIndexer, Strided1DIndexer>;
            StridedIndexer lhs_batch_indexer(batch_nd, lhs_batch_offset,
                                             batch_shape_strides);
            UnpackedStridedIndexer rhs_batch_indexer(
                batch_nd, rhs_batch_offset, batch_shape_strides,
                batch_shape_strides + 2 * batch_nd);
            Strided1DIndexer tmp_batch_indexer(
                0, static_cast<py::ssize_t>(batch_nelems), n * m);
            BatchDimsIndexerT batch_indexer(
                lhs_batch_indexer, rhs_batch_indexer, tmp_batch_indexer);

            size_t lws = wg_delta_n * wg_delta_m;

            size_t n_blocks =
                ((n + wi_delta_n * wg_delta_n - 1) / (wi_delta_n * wg_delta_n));
            size_t k_blocks = ((k + wi_delta_k - 1) / wi_delta_k);
            size_t m_blocks =
                ((m + wi_delta_m * wg_delta_m - 1) / (wi_delta_m * wg_delta_m));

            auto gwsRange = sycl::range<1>(batch_nelems * n_blocks * m_blocks *
                                           k_blocks * lws);
            auto lwsRange = sycl::range<1>(lws);

            auto ndRange = sycl::nd_range<1>(gwsRange, lwsRange);

            if constexpr (wi_delta_m == 1) {
                using LocAccT1 = sycl::local_accessor<resTy, 1>;
                LocAccT1 local_A_block(
                    sycl::range<1>((wi_delta_n * wg_delta_n) * wi_delta_k),
                    cgh);
                using LocAccT2 = sycl::local_accessor<resTy, 1>;
                LocAccT2 local_B_block(sycl::range<1>(wi_delta_k * wg_delta_m),
                                       cgh);

                using KernelName = class gemm_batch_tree_nm_krn<
                    lhsTy, rhsTy, resTy, OuterInnerDimsIndexerT, TmpIndexerT,
                    BatchDimsIndexerT, wi_delta_m>;
                cgh.parallel_for<KernelName>(
                    ndRange,
                    GemmBatchNoAtomicFunctorThreadNM<
                        lhsTy, rhsTy, resTy, LocAccT1, LocAccT2,
                        OuterInnerDimsIndexerT, TmpIndexerT, BatchDimsIndexerT,
                        wi_delta_n, wi_delta_m>(
                        lhs_tp, rhs_tp, partially_reduced_tmp, local_A_block,
                        local_B_block, n, wg_delta_n, k, k_blocks, wi_delta_k,
                        m, m_blocks, wg_delta_m, batch_nelems, batch_indexer,
                        lhs_indexer, rhs_indexer, res_indexer));
            }
            else {
                using LocAccT1 = sycl::local_accessor<resTy, 1>;
                LocAccT1 local_A_block(
                    sycl::range<1>((wi_delta_n * wg_delta_n) * wi_delta_k),
                    cgh);
                using LocAccT2 =
                    sycl::local_accessor<sycl::vec<resTy, wi_delta_m>, 1>;
                LocAccT2 local_B_block(sycl::range<1>(wi_delta_k * wg_delta_m),
                                       cgh);
                using KernelName = class gemm_batch_tree_nm_krn<
                    lhsTy, rhsTy, resTy, OuterInnerDimsIndexerT, TmpIndexerT,
                    BatchDimsIndexerT, wi_delta_m>;
                cgh.parallel_for<KernelName>(
                    ndRange,
                    GemmBatchNoAtomicFunctorThreadNM<
                        lhsTy, rhsTy, resTy, LocAccT1, LocAccT2,
                        OuterInnerDimsIndexerT, TmpIndexerT, BatchDimsIndexerT,
                        wi_delta_n, wi_delta_m>(
                        lhs_tp, rhs_tp, partially_reduced_tmp, local_A_block,
                        local_B_block, n, wg_delta_n, k, k_blocks, wi_delta_k,
                        m, m_blocks, wg_delta_m, batch_nelems, batch_indexer,
                        lhs_indexer, rhs_indexer, res_indexer));
            }
        });

        sycl::event red_ev = tree_reduction_for_gemm<resTy, ReductionOpT>(
            exec_q, partially_reduced_tmp, partially_reduced_tmp2, res_tp,
            identity_val, iter_nelems, reduction_nelems, reduction_groups, wg,
            max_wg, preferred_reductions_per_wi, reductions_per_wi,
            batch_nd + res_outer_nd, res_batch_offset, res_shape_strides,
            {gemm_ev});

        sycl::event cleanup_host_task_event =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(red_ev);
                const sycl::context &ctx = exec_q.get_context();

                cgh.host_task([ctx, partially_reduced_tmp] {
                    sycl::free(partially_reduced_tmp, ctx);
                });
            });

        return cleanup_host_task_event;
    }
}

template <typename T1, typename T2, typename T3>
class gemm_batch_tree_empty_krn;

template <typename lhsTy, typename rhsTy, typename resTy>
sycl::event
gemm_batch_tree_impl(sycl::queue &exec_q,
                     const char *lhs_cp,
                     const char *rhs_cp,
                     char *res_cp,
                     size_t batch_nelems,
                     size_t n,
                     size_t k,
                     size_t m,
                     int batch_nd,
                     const py::ssize_t *batch_shape_strides,
                     py::ssize_t lhs_batch_offset,
                     py::ssize_t rhs_batch_offset,
                     py::ssize_t res_batch_offset,
                     int inner_nd,
                     int lhs_outer_nd,
                     const py::ssize_t *lhs_outer_inner_shapes_strides,
                     int rhs_outer_nd,
                     const py::ssize_t *rhs_outer_inner_shapes_strides,
                     int res_outer_nd,
                     const py::ssize_t *res_outer_shapes_strides,
                     const py::ssize_t *res_shape_strides,
                     std::vector<sycl::event> const &depends = {})
{
    const lhsTy *lhs_tp = reinterpret_cast<const lhsTy *>(lhs_cp);
    const rhsTy *rhs_tp = reinterpret_cast<const rhsTy *>(rhs_cp);
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp);

    if (k == 0) {
        sycl::event gemm_batch_no_reduction_ev =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(depends);

                using IndexerT = dpctl::tensor::offset_utils::StridedIndexer;
                IndexerT res_indexer(batch_nd + res_outer_nd, res_batch_offset,
                                     res_shape_strides);
                using InitKernelName =
                    class gemm_batch_tree_empty_krn<lhsTy, rhsTy, resTy>;
                cgh.parallel_for<InitKernelName>(
                    sycl::range<1>(n * m * batch_nelems), [=](sycl::id<1> id) {
                        auto res_offset = res_indexer(id[0]);
                        res_tp[res_offset] = resTy(0);
                    });
            });
        return gemm_batch_no_reduction_ev;
    }

    if (exec_q.get_device().is_cpu()) {
        using dpctl::tensor::type_utils::is_complex;
        if constexpr (!is_complex<resTy>::value) {
            if (m == 1) {
                return gemm_batch_tree_k_impl<lhsTy, rhsTy, resTy, 1>(
                    exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m,
                    batch_nd, batch_shape_strides, lhs_batch_offset,
                    rhs_batch_offset, res_batch_offset, inner_nd, lhs_outer_nd,
                    lhs_outer_inner_shapes_strides, rhs_outer_nd,
                    rhs_outer_inner_shapes_strides, res_outer_nd,
                    res_outer_shapes_strides, res_shape_strides, depends);
            }
            else {
                return gemm_batch_tree_nm_impl<lhsTy, rhsTy, resTy, 4>(
                    exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m,
                    batch_nd, batch_shape_strides, lhs_batch_offset,
                    rhs_batch_offset, res_batch_offset, inner_nd, lhs_outer_nd,
                    lhs_outer_inner_shapes_strides, rhs_outer_nd,
                    rhs_outer_inner_shapes_strides, res_outer_nd,
                    res_outer_shapes_strides, res_shape_strides, depends);
            }
        }
        else {
            return gemm_batch_tree_nm_impl<lhsTy, rhsTy, resTy, 1>(
                exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m, batch_nd,
                batch_shape_strides, lhs_batch_offset, rhs_batch_offset,
                res_batch_offset, inner_nd, lhs_outer_nd,
                lhs_outer_inner_shapes_strides, rhs_outer_nd,
                rhs_outer_inner_shapes_strides, res_outer_nd,
                res_outer_shapes_strides, res_shape_strides, depends);
        }
    }
    else {
        if ((k > n && k > m) || m == 1) {
            using dpctl::tensor::type_utils::is_complex;
            if constexpr (!is_complex<resTy>::value) {
                if (m == 1) {
                    return gemm_batch_tree_k_impl<lhsTy, rhsTy, resTy, 1>(
                        exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m,
                        batch_nd, batch_shape_strides, lhs_batch_offset,
                        rhs_batch_offset, res_batch_offset, inner_nd,
                        lhs_outer_nd, lhs_outer_inner_shapes_strides,
                        rhs_outer_nd, rhs_outer_inner_shapes_strides,
                        res_outer_nd, res_outer_shapes_strides,
                        res_shape_strides, depends);
                }
                else {
                    return gemm_batch_tree_k_impl<lhsTy, rhsTy, resTy, 2>(
                        exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m,
                        batch_nd, batch_shape_strides, lhs_batch_offset,
                        rhs_batch_offset, res_batch_offset, inner_nd,
                        lhs_outer_nd, lhs_outer_inner_shapes_strides,
                        rhs_outer_nd, rhs_outer_inner_shapes_strides,
                        res_outer_nd, res_outer_shapes_strides,
                        res_shape_strides, depends);
                }
            }
            else {
                return gemm_batch_tree_k_impl<lhsTy, rhsTy, resTy, 1>(
                    exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m,
                    batch_nd, batch_shape_strides, lhs_batch_offset,
                    rhs_batch_offset, res_batch_offset, inner_nd, lhs_outer_nd,
                    lhs_outer_inner_shapes_strides, rhs_outer_nd,
                    rhs_outer_inner_shapes_strides, res_outer_nd,
                    res_outer_shapes_strides, res_shape_strides, depends);
            }
        }
        else { // m > 1, n > k or m > k
            using dpctl::tensor::type_utils::is_complex;
            if constexpr (!is_complex<resTy>::value) {
                return gemm_batch_tree_nm_impl<lhsTy, rhsTy, resTy, 4>(
                    exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m,
                    batch_nd, batch_shape_strides, lhs_batch_offset,
                    rhs_batch_offset, res_batch_offset, inner_nd, lhs_outer_nd,
                    lhs_outer_inner_shapes_strides, rhs_outer_nd,
                    rhs_outer_inner_shapes_strides, res_outer_nd,
                    res_outer_shapes_strides, res_shape_strides, depends);
            }
            else { // m > 1, n > k or m > k, resTy complex
                return gemm_batch_tree_nm_impl<lhsTy, rhsTy, resTy, 1>(
                    exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m,
                    batch_nd, batch_shape_strides, lhs_batch_offset,
                    rhs_batch_offset, res_batch_offset, inner_nd, lhs_outer_nd,
                    lhs_outer_inner_shapes_strides, rhs_outer_nd,
                    rhs_outer_inner_shapes_strides, res_outer_nd,
                    res_outer_shapes_strides, res_shape_strides, depends);
            }
        }
    }
}

template <typename lhsTy, typename rhsTy, typename resTy, size_t m_groups>
sycl::event
gemm_batch_contig_tree_k_impl(sycl::queue &exec_q,
                              const lhsTy *lhs_tp,
                              const rhsTy *rhs_tp,
                              resTy *res_tp,
                              size_t batch_nelems,
                              size_t n,
                              size_t k,
                              size_t m,
                              std::vector<sycl::event> const &depends)
{
    size_t delta_k(4);
    size_t n_wi(4);
    size_t delta_n(4);

    const sycl::device &dev = exec_q.get_device();
    const size_t local_mem_size =
        dev.get_info<sycl::info::device::local_mem_size>();
    const size_t reserved_slm_size = 512;

    gemm_detail::scale_gemm_k_parameters<resTy, m_groups>(
        local_mem_size, reserved_slm_size, delta_k,
        n_wi,   // modified by reference
        delta_n // modified by reference
    );

    if (k <= (delta_k * n_wi)) {
        sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            using OuterInnerDimsIndexerT =
                dpctl::tensor::offset_utils::NoOpIndexer;
            OuterInnerDimsIndexerT lhs_indexer{};
            OuterInnerDimsIndexerT rhs_indexer{};
            OuterInnerDimsIndexerT res_indexer{};
            using dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer;
            using dpctl::tensor::offset_utils::Strided1DIndexer;
            using BatchDimsIndexerT =
                ThreeOffsets_CombinedIndexer<Strided1DIndexer, Strided1DIndexer,
                                             Strided1DIndexer>;

            using dpctl::tensor::offset_utils::Strided1DIndexer;
            BatchDimsIndexerT batch_indexer(
                Strided1DIndexer{0, static_cast<py::ssize_t>(batch_nelems),
                                 static_cast<py::ssize_t>(n * k)},
                Strided1DIndexer{0, static_cast<py::ssize_t>(batch_nelems),
                                 static_cast<py::ssize_t>(k * m)},
                Strided1DIndexer{0, static_cast<py::ssize_t>(batch_nelems),
                                 static_cast<py::ssize_t>(n * m)});

            size_t n_blocks = (n + delta_n - 1) / delta_n;
            size_t k_blocks = (k + n_wi * delta_k - 1) / (n_wi * delta_k);
            size_t m_blocks = (m + m_groups - 1) / m_groups;

            size_t lws = delta_n * delta_k;

            auto gRange = sycl::range<1>(batch_nelems * n_blocks * m_blocks *
                                         k_blocks * lws);
            auto lRange = sycl::range<1>(lws);

            auto ndRange = sycl::nd_range<1>(gRange, lRange);

            if constexpr (m_groups == 1) {
                using LocAccT = sycl::local_accessor<resTy, 1>;
                LocAccT local_B_block(n_wi * delta_k, cgh);
                LocAccT workspace(delta_n * delta_k, cgh);

                using KernelName = class gemm_batch_tree_k_krn<
                    lhsTy, rhsTy, resTy, OuterInnerDimsIndexerT,
                    OuterInnerDimsIndexerT, BatchDimsIndexerT, m_groups>;

                cgh.parallel_for<KernelName>(
                    ndRange,
                    GemmBatchNoAtomicFunctorThreadK<
                        lhsTy, rhsTy, resTy, LocAccT, OuterInnerDimsIndexerT,
                        OuterInnerDimsIndexerT, BatchDimsIndexerT, m_groups>(
                        lhs_tp, rhs_tp, res_tp, workspace, local_B_block, n,
                        n_blocks, delta_n, k, k_blocks, delta_k, n_wi, m,
                        batch_nelems, batch_indexer, lhs_indexer, rhs_indexer,
                        res_indexer));
            }
            else {
                using LocAccT =
                    sycl::local_accessor<sycl::vec<resTy, m_groups>, 1>;
                LocAccT local_B_block(n_wi * delta_k, cgh);
                LocAccT workspace(delta_n * delta_k, cgh);

                using KernelName = class gemm_batch_tree_k_krn<
                    lhsTy, rhsTy, resTy, OuterInnerDimsIndexerT,
                    OuterInnerDimsIndexerT, BatchDimsIndexerT, m_groups>;

                cgh.parallel_for<KernelName>(
                    ndRange,
                    GemmBatchNoAtomicFunctorThreadK<
                        lhsTy, rhsTy, resTy, LocAccT, OuterInnerDimsIndexerT,
                        OuterInnerDimsIndexerT, BatchDimsIndexerT, m_groups>(
                        lhs_tp, rhs_tp, res_tp, workspace, local_B_block, n,
                        n_blocks, delta_n, k, k_blocks, delta_k, n_wi, m,
                        batch_nelems, batch_indexer, lhs_indexer, rhs_indexer,
                        res_indexer));
            }
        });
        return gemm_ev;
    }
    else {
        using ReductionOpT = sycl::plus<resTy>;
        constexpr resTy identity_val =
            sycl::known_identity<ReductionOpT, resTy>::value;

        size_t iter_nelems = batch_nelems * n * m;
        size_t reduction_nelems = (k + delta_k * n_wi - 1) / (delta_k * n_wi);

        // more than one work-group is needed, requires a
        // temporary delta_k * n_wi elements processed along k,
        // so if more to process use multiple
        const auto &sg_sizes =
            dev.get_info<sycl::info::device::sub_group_sizes>();
        size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

        if (reduction_nelems < wg) {
            resTy *tmp = sycl::malloc_device<resTy>(
                iter_nelems * reduction_nelems, exec_q);
            if (!tmp) {
                throw std::runtime_error("Unable to allocate device memory");
            }
            sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(depends);

                using OuterInnerDimsIndexerT =
                    dpctl::tensor::offset_utils::NoOpIndexer;
                OuterInnerDimsIndexerT lhs_indexer{};
                OuterInnerDimsIndexerT rhs_indexer{};
                OuterInnerDimsIndexerT tmp_indexer{};
                using dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer;
                using dpctl::tensor::offset_utils::Strided1DIndexer;
                using BatchDimsIndexerT = ThreeOffsets_CombinedIndexer<
                    Strided1DIndexer, Strided1DIndexer, Strided1DIndexer>;

                BatchDimsIndexerT batch_indexer(
                    Strided1DIndexer{0, static_cast<py::ssize_t>(batch_nelems),
                                     static_cast<py::ssize_t>(n * k)},
                    Strided1DIndexer{0, static_cast<py::ssize_t>(batch_nelems),
                                     static_cast<py::ssize_t>(k * m)},
                    Strided1DIndexer{0, static_cast<py::ssize_t>(batch_nelems),
                                     static_cast<py::ssize_t>(n * m)});

                size_t n_blocks = (n + delta_n - 1) / delta_n;
                size_t k_blocks = (k + n_wi * delta_k - 1) / (n_wi * delta_k);
                size_t m_blocks = (m + m_groups - 1) / m_groups;

                size_t lws = delta_n * delta_k;

                auto gRange = sycl::range<1>(batch_nelems * n_blocks *
                                             m_blocks * k_blocks * lws);
                auto lRange = sycl::range<1>(lws);

                auto ndRange = sycl::nd_range<1>(gRange, lRange);

                if constexpr (m_groups == 1) {
                    using LocAccT = sycl::local_accessor<resTy, 1>;
                    LocAccT local_B_block(n_wi * delta_k, cgh);
                    LocAccT workspace(delta_n * delta_k, cgh);

                    using KernelName = class gemm_batch_tree_k_krn<
                        lhsTy, rhsTy, resTy, OuterInnerDimsIndexerT,
                        OuterInnerDimsIndexerT, BatchDimsIndexerT, m_groups>;
                    cgh.parallel_for<KernelName>(
                        ndRange,
                        GemmBatchNoAtomicFunctorThreadK<
                            lhsTy, rhsTy, resTy, LocAccT,
                            OuterInnerDimsIndexerT, OuterInnerDimsIndexerT,
                            BatchDimsIndexerT, m_groups>(
                            lhs_tp, rhs_tp, tmp, workspace, local_B_block, n,
                            n_blocks, delta_n, k, k_blocks, delta_k, n_wi, m,
                            batch_nelems, batch_indexer, lhs_indexer,
                            rhs_indexer, tmp_indexer));
                }
                else {
                    using LocAccT =
                        sycl::local_accessor<sycl::vec<resTy, m_groups>, 1>;
                    LocAccT local_B_block(n_wi * delta_k, cgh);
                    LocAccT workspace(delta_n * delta_k, cgh);

                    using KernelName = class gemm_batch_tree_k_krn<
                        lhsTy, rhsTy, resTy, OuterInnerDimsIndexerT,
                        OuterInnerDimsIndexerT, BatchDimsIndexerT, m_groups>;
                    cgh.parallel_for<KernelName>(
                        ndRange,
                        GemmBatchNoAtomicFunctorThreadK<
                            lhsTy, rhsTy, resTy, LocAccT,
                            OuterInnerDimsIndexerT, OuterInnerDimsIndexerT,
                            BatchDimsIndexerT, m_groups>(
                            lhs_tp, rhs_tp, tmp, workspace, local_B_block, n,
                            n_blocks, delta_n, k, k_blocks, delta_k, n_wi, m,
                            batch_nelems, batch_indexer, lhs_indexer,
                            rhs_indexer, tmp_indexer));
                }
            });
            sycl::event red_ev = exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(gemm_ev);

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

                sycl::range<1> iter_range{iter_nelems};

                cgh.parallel_for<class gemm_reduction_seq_strided_krn<
                    resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                    ReductionIndexerT>>(
                    iter_range, SequentialReduction<resTy, resTy, ReductionOpT,
                                                    InputOutputIterIndexerT,
                                                    ReductionIndexerT>(
                                    tmp, res_tp, ReductionOpT(), identity_val,
                                    in_out_iter_indexer, reduction_indexer,
                                    reduction_nelems));
            });
            sycl::event cleanup_host_task_event =
                exec_q.submit([&](sycl::handler &cgh) {
                    cgh.depends_on(red_ev);
                    const sycl::context &ctx = exec_q.get_context();

                    cgh.host_task([ctx, tmp] { sycl::free(tmp, ctx); });
                });
            return cleanup_host_task_event;
        }

        // max_max_wg prevents running out of resources on CPU
        constexpr size_t max_max_wg = 2048;
        size_t max_wg = std::min(
            max_max_wg,
            dev.get_info<sycl::info::device::max_work_group_size>() / 2);

        constexpr size_t preferred_reductions_per_wi = 4;
        size_t reductions_per_wi(preferred_reductions_per_wi);

        size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);
        assert(reduction_groups > 1);

        resTy *partially_reduced_tmp = sycl::malloc_device<resTy>(
            iter_nelems * (/* temp */ reduction_nelems +
                           /* first reduction temp */ reduction_groups),
            exec_q);
        resTy *partially_reduced_tmp2 = nullptr;

        if (partially_reduced_tmp == nullptr) {
            throw std::runtime_error("Unable to allocate device_memory");
        }
        else {
            partially_reduced_tmp2 =
                partially_reduced_tmp + reduction_nelems * iter_nelems;
        }

        sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            using OuterInnerDimsIndexerT =
                dpctl::tensor::offset_utils::NoOpIndexer;
            OuterInnerDimsIndexerT lhs_indexer{};
            OuterInnerDimsIndexerT rhs_indexer{};
            OuterInnerDimsIndexerT tmp_indexer{};
            using dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer;
            using dpctl::tensor::offset_utils::Strided1DIndexer;
            using BatchDimsIndexerT =
                ThreeOffsets_CombinedIndexer<Strided1DIndexer, Strided1DIndexer,
                                             Strided1DIndexer>;

            BatchDimsIndexerT batch_indexer(
                Strided1DIndexer{0, static_cast<py::ssize_t>(batch_nelems),
                                 static_cast<py::ssize_t>(n * k)},
                Strided1DIndexer{0, static_cast<py::ssize_t>(batch_nelems),
                                 static_cast<py::ssize_t>(k * m)},
                Strided1DIndexer{0, static_cast<py::ssize_t>(batch_nelems),
                                 static_cast<py::ssize_t>(n * m)});

            size_t n_blocks = (n + delta_n - 1) / delta_n;
            size_t k_blocks = (k + n_wi * delta_k - 1) / (n_wi * delta_k);
            size_t m_blocks = (m + m_groups - 1) / m_groups;

            size_t lws = delta_n * delta_k;

            auto gRange = sycl::range<1>(batch_nelems * n_blocks * m_blocks *
                                         k_blocks * lws);
            auto lRange = sycl::range<1>(lws);

            auto ndRange = sycl::nd_range<1>(gRange, lRange);

            if constexpr (m_groups == 1) {
                using LocAccT = sycl::local_accessor<resTy, 1>;
                LocAccT local_B_block(n_wi * delta_k, cgh);
                LocAccT workspace(delta_n * delta_k, cgh);

                using KernelName = class gemm_batch_tree_k_krn<
                    lhsTy, rhsTy, resTy, OuterInnerDimsIndexerT,
                    OuterInnerDimsIndexerT, BatchDimsIndexerT, m_groups>;
                cgh.parallel_for<KernelName>(
                    ndRange,
                    GemmBatchNoAtomicFunctorThreadK<
                        lhsTy, rhsTy, resTy, LocAccT, OuterInnerDimsIndexerT,
                        OuterInnerDimsIndexerT, BatchDimsIndexerT, m_groups>(
                        lhs_tp, rhs_tp, partially_reduced_tmp, workspace,
                        local_B_block, n, n_blocks, delta_n, k, k_blocks,
                        delta_k, n_wi, m, batch_nelems, batch_indexer,
                        lhs_indexer, rhs_indexer, tmp_indexer));
            }
            else {
                using LocAccT =
                    sycl::local_accessor<sycl::vec<resTy, m_groups>, 1>;
                LocAccT local_B_block(n_wi * delta_k, cgh);
                LocAccT workspace(delta_n * delta_k, cgh);

                using KernelName = class gemm_batch_tree_k_krn<
                    lhsTy, rhsTy, resTy, OuterInnerDimsIndexerT,
                    OuterInnerDimsIndexerT, BatchDimsIndexerT, m_groups>;
                cgh.parallel_for<KernelName>(
                    ndRange,
                    GemmBatchNoAtomicFunctorThreadK<
                        lhsTy, rhsTy, resTy, LocAccT, OuterInnerDimsIndexerT,
                        OuterInnerDimsIndexerT, BatchDimsIndexerT, m_groups>(
                        lhs_tp, rhs_tp, partially_reduced_tmp, workspace,
                        local_B_block, n, n_blocks, delta_n, k, k_blocks,
                        delta_k, n_wi, m, batch_nelems, batch_indexer,
                        lhs_indexer, rhs_indexer, tmp_indexer));
            }
        });

        sycl::event red_ev =
            tree_reduction_for_gemm_contig<resTy, ReductionOpT>(
                exec_q, partially_reduced_tmp, partially_reduced_tmp2, res_tp,
                identity_val, iter_nelems, reduction_nelems, reduction_groups,
                wg, max_wg, preferred_reductions_per_wi, reductions_per_wi,
                {gemm_ev});

        sycl::event cleanup_host_task_event =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(red_ev);
                const sycl::context &ctx = exec_q.get_context();

                cgh.host_task([ctx, partially_reduced_tmp] {
                    sycl::free(partially_reduced_tmp, ctx);
                });
            });

        return cleanup_host_task_event;
    }
}

template <typename lhsTy, typename rhsTy, typename resTy, int wi_delta_m>
sycl::event
gemm_batch_contig_tree_nm_impl(sycl::queue &exec_q,
                               const lhsTy *lhs_tp,
                               const rhsTy *rhs_tp,
                               resTy *res_tp,
                               size_t batch_nelems,
                               size_t n,
                               size_t k,
                               size_t m,
                               std::vector<sycl::event> const &depends)
{
    constexpr int wi_delta_n = 2;
    size_t wg_delta_n(16); // rows of A processed in WG
    size_t wg_delta_m(16); // rows of B processed in WG
    size_t wi_delta_k(64); // Elements in K dimension processed by WI

    const sycl::device &dev = exec_q.get_device();
    const size_t local_mem_size =
        dev.get_info<sycl::info::device::local_mem_size>();
    const size_t reserved_slm_size = 512;

    gemm_detail::scale_gemm_nm_parameters<resTy, wi_delta_m>(
        local_mem_size, reserved_slm_size, wi_delta_n,
        wi_delta_k, // modified by reference
        wg_delta_n, // modified by reference
        wg_delta_m  // modified by reference
    );

    // each group processes delta_k * n_wi
    // items in a column, so no need for allocating
    // temp memory if only one group is needed
    if (k <= wi_delta_k) {
        sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            using OuterInnerDimsIndexerT =
                dpctl::tensor::offset_utils::NoOpIndexer;
            OuterInnerDimsIndexerT lhs_indexer{};
            OuterInnerDimsIndexerT rhs_indexer{};
            OuterInnerDimsIndexerT res_indexer{};
            using dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer;
            using dpctl::tensor::offset_utils::Strided1DIndexer;
            using BatchDimsIndexerT =
                ThreeOffsets_CombinedIndexer<Strided1DIndexer, Strided1DIndexer,
                                             Strided1DIndexer>;

            BatchDimsIndexerT batch_indexer(
                Strided1DIndexer{0, static_cast<py::ssize_t>(batch_nelems),
                                 static_cast<py::ssize_t>(n * k)},
                Strided1DIndexer{0, static_cast<py::ssize_t>(batch_nelems),
                                 static_cast<py::ssize_t>(k * m)},
                Strided1DIndexer{0, static_cast<py::ssize_t>(batch_nelems),
                                 static_cast<py::ssize_t>(n * m)});

            size_t lws = wg_delta_n * wg_delta_m;

            size_t n_blocks =
                ((n + wi_delta_n * wg_delta_n - 1) / (wi_delta_n * wg_delta_n));
            size_t k_blocks = ((k + wi_delta_k - 1) / wi_delta_k);
            size_t m_blocks =
                ((m + wi_delta_m * wg_delta_m - 1) / (wi_delta_m * wg_delta_m));

            auto gwsRange = sycl::range<1>(batch_nelems * n_blocks * m_blocks *
                                           k_blocks * lws);
            auto lwsRange = sycl::range<1>(lws);

            auto ndRange = sycl::nd_range<1>(gwsRange, lwsRange);

            if constexpr (wi_delta_m == 1) {
                using LocAccT1 = sycl::local_accessor<resTy, 1>;
                LocAccT1 local_A_block(
                    sycl::range<1>((wi_delta_n * wg_delta_n) * wi_delta_k),
                    cgh);
                using LocAccT2 = sycl::local_accessor<resTy, 1>;
                LocAccT2 local_B_block(sycl::range<1>(wi_delta_k * wg_delta_m),
                                       cgh);

                using KernelName = class gemm_batch_tree_nm_krn<
                    lhsTy, rhsTy, resTy, OuterInnerDimsIndexerT,
                    OuterInnerDimsIndexerT, BatchDimsIndexerT, wi_delta_m>;
                cgh.parallel_for<KernelName>(
                    ndRange,
                    GemmBatchNoAtomicFunctorThreadNM<
                        lhsTy, rhsTy, resTy, LocAccT1, LocAccT2,
                        OuterInnerDimsIndexerT, OuterInnerDimsIndexerT,
                        BatchDimsIndexerT, wi_delta_n, wi_delta_m>(
                        lhs_tp, rhs_tp, res_tp, local_A_block, local_B_block, n,
                        wg_delta_n, k, k_blocks, wi_delta_k, m, m_blocks,
                        wg_delta_m, batch_nelems, batch_indexer, lhs_indexer,
                        rhs_indexer, res_indexer));
            }
            else {
                using LocAccT1 = sycl::local_accessor<resTy, 1>;
                LocAccT1 local_A_block(
                    sycl::range<1>((wi_delta_n * wg_delta_n) * wi_delta_k),
                    cgh);
                using LocAccT2 =
                    sycl::local_accessor<sycl::vec<resTy, wi_delta_m>, 1>;
                LocAccT2 local_B_block(sycl::range<1>(wi_delta_k * wg_delta_m),
                                       cgh);

                using KernelName = class gemm_batch_tree_nm_krn<
                    lhsTy, rhsTy, resTy, OuterInnerDimsIndexerT,
                    OuterInnerDimsIndexerT, BatchDimsIndexerT, wi_delta_m>;
                cgh.parallel_for<KernelName>(
                    ndRange,
                    GemmBatchNoAtomicFunctorThreadNM<
                        lhsTy, rhsTy, resTy, LocAccT1, LocAccT2,
                        OuterInnerDimsIndexerT, OuterInnerDimsIndexerT,
                        BatchDimsIndexerT, wi_delta_n, wi_delta_m>(
                        lhs_tp, rhs_tp, res_tp, local_A_block, local_B_block, n,
                        wg_delta_n, k, k_blocks, wi_delta_k, m, m_blocks,
                        wg_delta_m, batch_nelems, batch_indexer, lhs_indexer,
                        rhs_indexer, res_indexer));
            }
        });
        return gemm_ev;
    }
    else {
        using ReductionOpT = sycl::plus<resTy>;
        constexpr resTy identity_val =
            sycl::known_identity<ReductionOpT, resTy>::value;
        size_t iter_nelems = batch_nelems * n * m;
        size_t reduction_nelems = (k + wi_delta_k - 1) / wi_delta_k;

        // more than one work-group is needed, requires a temporary
        // delta_k * n_wi elements processed along k, so if more to
        // process use multiple
        const auto &sg_sizes =
            dev.get_info<sycl::info::device::sub_group_sizes>();
        size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

        if (reduction_nelems < wg) {
            resTy *tmp = sycl::malloc_device<resTy>(
                iter_nelems * reduction_nelems, exec_q);
            if (!tmp) {
                throw std::runtime_error("Unable to allocate device memory");
            }
            sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(depends);

                using OuterInnerDimsIndexerT =
                    dpctl::tensor::offset_utils::NoOpIndexer;
                OuterInnerDimsIndexerT lhs_indexer{};
                OuterInnerDimsIndexerT rhs_indexer{};
                OuterInnerDimsIndexerT tmp_indexer{};
                using dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer;
                using dpctl::tensor::offset_utils::Strided1DIndexer;
                using BatchDimsIndexerT = ThreeOffsets_CombinedIndexer<
                    Strided1DIndexer, Strided1DIndexer, Strided1DIndexer>;

                BatchDimsIndexerT batch_indexer(
                    Strided1DIndexer{0, static_cast<py::ssize_t>(batch_nelems),
                                     static_cast<py::ssize_t>(n * k)},
                    Strided1DIndexer{0, static_cast<py::ssize_t>(batch_nelems),
                                     static_cast<py::ssize_t>(k * m)},
                    Strided1DIndexer{0, static_cast<py::ssize_t>(batch_nelems),
                                     static_cast<py::ssize_t>(n * m)});

                size_t lws = wg_delta_n * wg_delta_m;

                size_t n_blocks = ((n + wi_delta_n * wg_delta_n - 1) /
                                   (wi_delta_n * wg_delta_n));
                size_t k_blocks = ((k + wi_delta_k - 1) / wi_delta_k);
                size_t m_blocks = ((m + wi_delta_m * wg_delta_m - 1) /
                                   (wi_delta_m * wg_delta_m));

                auto gwsRange = sycl::range<1>(batch_nelems * n_blocks *
                                               m_blocks * k_blocks * lws);
                auto lwsRange = sycl::range<1>(lws);

                auto ndRange = sycl::nd_range<1>(gwsRange, lwsRange);

                if constexpr (wi_delta_m == 1) {
                    using LocAccT1 = sycl::local_accessor<resTy, 1>;
                    LocAccT1 local_A_block(
                        sycl::range<1>((wi_delta_n * wg_delta_n) * wi_delta_k),
                        cgh);
                    using LocAccT2 = sycl::local_accessor<resTy, 1>;
                    LocAccT2 local_B_block(
                        sycl::range<1>(wi_delta_k * wg_delta_m), cgh);

                    using KernelName = class gemm_batch_tree_nm_krn<
                        lhsTy, rhsTy, resTy, OuterInnerDimsIndexerT,
                        OuterInnerDimsIndexerT, BatchDimsIndexerT, wi_delta_m>;
                    cgh.parallel_for<KernelName>(
                        ndRange,
                        GemmBatchNoAtomicFunctorThreadNM<
                            lhsTy, rhsTy, resTy, LocAccT1, LocAccT2,
                            OuterInnerDimsIndexerT, OuterInnerDimsIndexerT,
                            BatchDimsIndexerT, wi_delta_n, wi_delta_m>(
                            lhs_tp, rhs_tp, tmp, local_A_block, local_B_block,
                            n, wg_delta_n, k, k_blocks, wi_delta_k, m, m_blocks,
                            wg_delta_m, batch_nelems, batch_indexer,
                            lhs_indexer, rhs_indexer, tmp_indexer));
                }
                else {
                    using LocAccT1 = sycl::local_accessor<resTy, 1>;
                    LocAccT1 local_A_block(
                        sycl::range<1>((wi_delta_n * wg_delta_n) * wi_delta_k),
                        cgh);
                    using LocAccT2 =
                        sycl::local_accessor<sycl::vec<resTy, wi_delta_m>, 1>;
                    LocAccT2 local_B_block(
                        sycl::range<1>(wi_delta_k * wg_delta_m), cgh);

                    using KernelName = class gemm_batch_tree_nm_krn<
                        lhsTy, rhsTy, resTy, OuterInnerDimsIndexerT,
                        OuterInnerDimsIndexerT, BatchDimsIndexerT, wi_delta_m>;
                    cgh.parallel_for<KernelName>(
                        ndRange,
                        GemmBatchNoAtomicFunctorThreadNM<
                            lhsTy, rhsTy, resTy, LocAccT1, LocAccT2,
                            OuterInnerDimsIndexerT, OuterInnerDimsIndexerT,
                            BatchDimsIndexerT, wi_delta_n, wi_delta_m>(
                            lhs_tp, rhs_tp, tmp, local_A_block, local_B_block,
                            n, wg_delta_n, k, k_blocks, wi_delta_k, m, m_blocks,
                            wg_delta_m, batch_nelems, batch_indexer,
                            lhs_indexer, rhs_indexer, tmp_indexer));
                }
            });
            sycl::event red_ev = exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(gemm_ev);

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

                sycl::range<1> iter_range{iter_nelems};

                cgh.parallel_for<class gemm_reduction_seq_strided_krn<
                    resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                    ReductionIndexerT>>(
                    iter_range, SequentialReduction<resTy, resTy, ReductionOpT,
                                                    InputOutputIterIndexerT,
                                                    ReductionIndexerT>(
                                    tmp, res_tp, ReductionOpT(), identity_val,
                                    in_out_iter_indexer, reduction_indexer,
                                    reduction_nelems));
            });
            sycl::event cleanup_host_task_event =
                exec_q.submit([&](sycl::handler &cgh) {
                    cgh.depends_on(red_ev);
                    const sycl::context &ctx = exec_q.get_context();

                    cgh.host_task([ctx, tmp] { sycl::free(tmp, ctx); });
                });
            return cleanup_host_task_event;
        }

        // max_max_wg prevents running out of resources on CPU
        constexpr size_t max_max_wg = 2048;
        size_t max_wg = std::min(
            max_max_wg,
            dev.get_info<sycl::info::device::max_work_group_size>() / 2);

        constexpr size_t preferred_reductions_per_wi = 4;
        size_t reductions_per_wi(preferred_reductions_per_wi);

        size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);
        assert(reduction_groups > 1);

        resTy *partially_reduced_tmp = sycl::malloc_device<resTy>(
            iter_nelems * (/* temp */ reduction_nelems +
                           /* first reduction temp */ reduction_groups),
            exec_q);
        resTy *partially_reduced_tmp2 = nullptr;

        if (partially_reduced_tmp == nullptr) {
            throw std::runtime_error("Unable to allocate device_memory");
        }
        else {
            partially_reduced_tmp2 =
                partially_reduced_tmp + reduction_nelems * iter_nelems;
        }

        sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            using OuterInnerDimsIndexerT =
                dpctl::tensor::offset_utils::NoOpIndexer;
            OuterInnerDimsIndexerT lhs_indexer{};
            OuterInnerDimsIndexerT rhs_indexer{};
            OuterInnerDimsIndexerT tmp_indexer{};
            using dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer;
            using dpctl::tensor::offset_utils::Strided1DIndexer;
            using BatchDimsIndexerT =
                ThreeOffsets_CombinedIndexer<Strided1DIndexer, Strided1DIndexer,
                                             Strided1DIndexer>;

            BatchDimsIndexerT batch_indexer(
                Strided1DIndexer{0, static_cast<py::ssize_t>(batch_nelems),
                                 static_cast<py::ssize_t>(n * k)},
                Strided1DIndexer{0, static_cast<py::ssize_t>(batch_nelems),
                                 static_cast<py::ssize_t>(k * m)},
                Strided1DIndexer{0, static_cast<py::ssize_t>(batch_nelems),
                                 static_cast<py::ssize_t>(n * m)});

            size_t lws = wg_delta_n * wg_delta_m;

            size_t n_blocks =
                ((n + wi_delta_n * wg_delta_n - 1) / (wi_delta_n * wg_delta_n));
            size_t k_blocks = ((k + wi_delta_k - 1) / wi_delta_k);
            size_t m_blocks =
                ((m + wi_delta_m * wg_delta_m - 1) / (wi_delta_m * wg_delta_m));

            auto gwsRange = sycl::range<1>(batch_nelems * n_blocks * m_blocks *
                                           k_blocks * lws);
            auto lwsRange = sycl::range<1>(lws);

            auto ndRange = sycl::nd_range<1>(gwsRange, lwsRange);

            if constexpr (wi_delta_m == 1) {
                using LocAccT1 = sycl::local_accessor<resTy, 1>;
                LocAccT1 local_A_block(
                    sycl::range<1>((wi_delta_n * wg_delta_n) * wi_delta_k),
                    cgh);
                using LocAccT2 = sycl::local_accessor<resTy, 1>;
                LocAccT2 local_B_block(sycl::range<1>(wi_delta_k * wg_delta_m),
                                       cgh);

                using KernelName = class gemm_batch_tree_nm_krn<
                    lhsTy, rhsTy, resTy, OuterInnerDimsIndexerT,
                    OuterInnerDimsIndexerT, BatchDimsIndexerT, wi_delta_m>;
                cgh.parallel_for<KernelName>(
                    ndRange,
                    GemmBatchNoAtomicFunctorThreadNM<
                        lhsTy, rhsTy, resTy, LocAccT1, LocAccT2,
                        OuterInnerDimsIndexerT, OuterInnerDimsIndexerT,
                        BatchDimsIndexerT, wi_delta_n, wi_delta_m>(
                        lhs_tp, rhs_tp, partially_reduced_tmp, local_A_block,
                        local_B_block, n, wg_delta_n, k, k_blocks, wi_delta_k,
                        m, m_blocks, wg_delta_m, batch_nelems, batch_indexer,
                        lhs_indexer, rhs_indexer, tmp_indexer));
            }
            else {
                using LocAccT1 = sycl::local_accessor<resTy, 1>;
                LocAccT1 local_A_block(
                    sycl::range<1>((wi_delta_n * wg_delta_n) * wi_delta_k),
                    cgh);
                using LocAccT2 =
                    sycl::local_accessor<sycl::vec<resTy, wi_delta_m>, 1>;
                LocAccT2 local_B_block(sycl::range<1>(wi_delta_k * wg_delta_m),
                                       cgh);

                using KernelName = class gemm_batch_tree_nm_krn<
                    lhsTy, rhsTy, resTy, OuterInnerDimsIndexerT,
                    OuterInnerDimsIndexerT, BatchDimsIndexerT, wi_delta_m>;
                cgh.parallel_for<KernelName>(
                    ndRange,
                    GemmBatchNoAtomicFunctorThreadNM<
                        lhsTy, rhsTy, resTy, LocAccT1, LocAccT2,
                        OuterInnerDimsIndexerT, OuterInnerDimsIndexerT,
                        BatchDimsIndexerT, wi_delta_n, wi_delta_m>(
                        lhs_tp, rhs_tp, partially_reduced_tmp, local_A_block,
                        local_B_block, n, wg_delta_n, k, k_blocks, wi_delta_k,
                        m, m_blocks, wg_delta_m, batch_nelems, batch_indexer,
                        lhs_indexer, rhs_indexer, tmp_indexer));
            }
        });

        sycl::event red_ev =
            tree_reduction_for_gemm_contig<resTy, ReductionOpT>(
                exec_q, partially_reduced_tmp, partially_reduced_tmp2, res_tp,
                identity_val, iter_nelems, reduction_nelems, reduction_groups,
                wg, max_wg, preferred_reductions_per_wi, reductions_per_wi,
                {gemm_ev});

        sycl::event cleanup_host_task_event =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(red_ev);
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
gemm_batch_contig_tree_impl(sycl::queue &exec_q,
                            const char *lhs_cp,
                            const char *rhs_cp,
                            char *res_cp,
                            size_t batch_nelems,
                            size_t n,
                            size_t k,
                            size_t m,
                            py::ssize_t lhs_batch_offset,
                            py::ssize_t rhs_batch_offset,
                            py::ssize_t res_batch_offset,
                            std::vector<sycl::event> const &depends = {})
{
    const lhsTy *lhs_tp =
        reinterpret_cast<const lhsTy *>(lhs_cp) + lhs_batch_offset;
    const rhsTy *rhs_tp =
        reinterpret_cast<const rhsTy *>(rhs_cp) + rhs_batch_offset;
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp) + res_batch_offset;

    if (k == 0) {
        sycl::event gemm_batch_no_reduction_ev =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(depends);
                cgh.fill<resTy>(res_tp, resTy(0), n * m * batch_nelems);
            });
        return gemm_batch_no_reduction_ev;
    }

    if (exec_q.get_device().is_cpu()) {
        using dpctl::tensor::type_utils::is_complex;
        if constexpr (!is_complex<resTy>::value) {
            if (m == 1) {
                return gemm_batch_contig_tree_k_impl<lhsTy, rhsTy, resTy, 1>(
                    exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m,
                    depends);
            }
            else {
                return gemm_batch_contig_tree_nm_impl<lhsTy, rhsTy, resTy, 4>(
                    exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m,
                    depends);
            }
        }
        else {
            return gemm_batch_contig_tree_nm_impl<lhsTy, rhsTy, resTy, 1>(
                exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m, depends);
        }
    }
    else {
        if ((k > n && k > m) || m == 1) {
            using dpctl::tensor::type_utils::is_complex;
            if constexpr (!is_complex<resTy>::value) {
                if (m == 1) {
                    return gemm_batch_contig_tree_k_impl<lhsTy, rhsTy, resTy,
                                                         1>(
                        exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m,
                        depends);
                }
                else {
                    return gemm_batch_contig_tree_k_impl<lhsTy, rhsTy, resTy,
                                                         2>(
                        exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m,
                        depends);
                }
            }
            else {
                return gemm_batch_contig_tree_k_impl<lhsTy, rhsTy, resTy, 1>(
                    exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m,
                    depends);
            }
        }
        else { // m > 1, n > k or m > k
            using dpctl::tensor::type_utils::is_complex;
            if constexpr (!is_complex<resTy>::value) {
                return gemm_batch_contig_tree_nm_impl<lhsTy, rhsTy, resTy, 4>(
                    exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m,
                    depends);
            }
            else { // m > 1, n > k or m > k, resTy complex
                return gemm_batch_contig_tree_nm_impl<lhsTy, rhsTy, resTy, 1>(
                    exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m,
                    depends);
            }
        }
    }
}

} // namespace kernels
} // namespace tensor
} // namespace dpctl
