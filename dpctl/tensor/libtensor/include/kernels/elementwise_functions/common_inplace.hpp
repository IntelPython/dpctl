//=== common_inplace.hpp -  Common code for in-place elementwise operations
//----- *-C++-*--/===//
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
//===---------------------------------------------------------------------===//
///
/// \file
/// This file defines common code for in-place elementwise tensor operations.
//===---------------------------------------------------------------------===//

#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <sycl/sycl.hpp>

#include "kernels/alignment.hpp"
#include "kernels/dpctl_tensor_types.hpp"
#include "kernels/elementwise_functions/common_detail.hpp"
#include "utils/offset_utils.hpp"
#include "utils/sycl_alloc_utils.hpp"
#include "utils/sycl_utils.hpp"

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace elementwise_common
{

using dpctl::tensor::ssize_t;
using dpctl::tensor::kernels::alignment_utils::
    disabled_sg_loadstore_wrapper_krn;
using dpctl::tensor::kernels::alignment_utils::is_aligned;
using dpctl::tensor::kernels::alignment_utils::required_alignment;

using dpctl::tensor::sycl_utils::sub_group_load;
using dpctl::tensor::sycl_utils::sub_group_store;

template <typename argT,
          typename resT,
          typename BinaryInplaceOperatorT,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
struct BinaryInplaceContigFunctor
{
private:
    const argT *rhs = nullptr;
    resT *lhs = nullptr;
    std::size_t nelems_;

public:
    BinaryInplaceContigFunctor(const argT *rhs_tp,
                               resT *lhs_tp,
                               const std::size_t n_elems)
        : rhs(rhs_tp), lhs(lhs_tp), nelems_(n_elems)
    {
    }

    void operator()(sycl::nd_item<1> ndit) const
    {
        BinaryInplaceOperatorT op{};
        constexpr std::uint8_t elems_per_wi = vec_sz * n_vecs;
        /* Each work-item processes vec_sz elements, contiguous in memory */
        /* NB: Workgroup size must be divisible by sub-group size */

        if constexpr (enable_sg_loadstore &&
                      BinaryInplaceOperatorT::supports_sg_loadstore::value &&
                      BinaryInplaceOperatorT::supports_vec::value &&
                      (vec_sz > 1))
        {
            auto sg = ndit.get_sub_group();
            std::uint16_t sgSize = sg.get_max_local_range()[0];

            std::size_t base =
                elems_per_wi * (ndit.get_group(0) * ndit.get_local_range(0) +
                                sg.get_group_id()[0] * sgSize);

            if (base + elems_per_wi * sgSize < nelems_) {

#pragma unroll
                for (std::uint8_t it = 0; it < elems_per_wi; it += vec_sz) {
                    const std::size_t offset = base + it * sgSize;
                    auto rhs_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&rhs[offset]);
                    auto lhs_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&lhs[offset]);

                    const sycl::vec<argT, vec_sz> &arg_vec =
                        sub_group_load<vec_sz>(sg, rhs_multi_ptr);
                    sycl::vec<resT, vec_sz> res_vec =
                        sub_group_load<vec_sz>(sg, lhs_multi_ptr);
                    op(res_vec, arg_vec);

                    sub_group_store<vec_sz>(sg, res_vec, lhs_multi_ptr);
                }
            }
            else {
                const std::size_t lane_id = sg.get_local_id()[0];
                for (std::size_t k = base + lane_id; k < nelems_; k += sgSize) {
                    op(lhs[k], rhs[k]);
                }
            }
        }
        else if constexpr (enable_sg_loadstore &&
                           BinaryInplaceOperatorT::supports_sg_loadstore::value)
        {
            auto sg = ndit.get_sub_group();
            std::uint16_t sgSize = sg.get_max_local_range()[0];

            std::size_t base =
                elems_per_wi * (ndit.get_group(0) * ndit.get_local_range(0) +
                                sg.get_group_id()[0] * sgSize);

            if (base + elems_per_wi * sgSize < nelems_) {
#pragma unroll
                for (std::uint8_t it = 0; it < elems_per_wi; it += vec_sz) {
                    const std::size_t offset = base + it * sgSize;
                    auto rhs_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&rhs[offset]);
                    auto lhs_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&lhs[offset]);

                    const sycl::vec<argT, vec_sz> arg_vec =
                        sub_group_load<vec_sz>(sg, rhs_multi_ptr);
                    sycl::vec<resT, vec_sz> res_vec =
                        sub_group_load<vec_sz>(sg, lhs_multi_ptr);
#pragma unroll
                    for (std::uint8_t vec_id = 0; vec_id < vec_sz; ++vec_id) {
                        op(res_vec[vec_id], arg_vec[vec_id]);
                    }
                    sub_group_store<vec_sz>(sg, res_vec, lhs_multi_ptr);
                }
            }
            else {
                const std::size_t lane_id = sg.get_local_id()[0];
                for (std::size_t k = base + lane_id; k < nelems_; k += sgSize) {
                    op(lhs[k], rhs[k]);
                }
            }
        }
        else {
            const std::size_t sgSize =
                ndit.get_sub_group().get_local_range()[0];
            const std::size_t gid = ndit.get_global_linear_id();
            const std::size_t elems_per_sg = elems_per_wi * sgSize;

            const std::size_t start =
                (gid / sgSize) * (elems_per_sg - sgSize) + gid;
            const std::size_t end = std::min(nelems_, start + elems_per_sg);
            for (std::size_t offset = start; offset < end; offset += sgSize) {
                op(lhs[offset], rhs[offset]);
            }
        }
    }
};

template <typename argT,
          typename resT,
          typename TwoOffsets_IndexerT,
          typename BinaryInplaceOperatorT>
struct BinaryInplaceStridedFunctor
{
private:
    const argT *rhs = nullptr;
    resT *lhs = nullptr;
    TwoOffsets_IndexerT two_offsets_indexer_;

public:
    BinaryInplaceStridedFunctor(const argT *rhs_tp,
                                resT *lhs_tp,
                                const TwoOffsets_IndexerT &inp_res_indexer)
        : rhs(rhs_tp), lhs(lhs_tp), two_offsets_indexer_(inp_res_indexer)
    {
    }

    void operator()(sycl::id<1> wid) const
    {
        const auto &two_offsets_ =
            two_offsets_indexer_(static_cast<ssize_t>(wid.get(0)));

        const auto &inp_offset = two_offsets_.get_first_offset();
        const auto &lhs_offset = two_offsets_.get_second_offset();

        BinaryInplaceOperatorT op{};
        op(lhs[lhs_offset], rhs[inp_offset]);
    }
};

template <typename argT, typename resT, typename BinaryOperatorT>
struct BinaryInplaceRowMatrixBroadcastingFunctor
{
private:
    const argT *padded_vec;
    resT *mat;
    std::size_t n_elems;
    std::size_t n1;

public:
    BinaryInplaceRowMatrixBroadcastingFunctor(const argT *row_tp,
                                              resT *mat_tp,
                                              std::size_t n_elems_in_mat,
                                              std::size_t n_elems_in_row)
        : padded_vec(row_tp), mat(mat_tp), n_elems(n_elems_in_mat),
          n1(n_elems_in_row)
    {
    }

    void operator()(sycl::nd_item<1> ndit) const
    {
        /* Workgroup size is expected to be a multiple of sub-group size */
        BinaryOperatorT op{};
        static_assert(BinaryOperatorT::supports_sg_loadstore::value);

        auto sg = ndit.get_sub_group();
        const std::size_t gid = ndit.get_global_linear_id();

        std::uint8_t sgSize = sg.get_max_local_range()[0];
        std::size_t base = gid - sg.get_local_id()[0];

        if (base + sgSize < n_elems) {
            auto in_multi_ptr = sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::yes>(&padded_vec[base % n1]);

            auto out_multi_ptr = sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::yes>(&mat[base]);

            const argT vec_el = sub_group_load(sg, in_multi_ptr);
            resT mat_el = sub_group_load(sg, out_multi_ptr);

            op(mat_el, vec_el);

            sub_group_store(sg, mat_el, out_multi_ptr);
        }
        else {
            const std::size_t start = base + sg.get_local_id()[0];
            for (std::size_t k = start; k < n_elems; k += sgSize) {
                op(mat[k], padded_vec[k % n1]);
            }
        }
    }
};

// Typedefs for function pointers

typedef sycl::event (*binary_inplace_contig_impl_fn_ptr_t)(
    sycl::queue &,
    std::size_t,
    const char *,
    ssize_t,
    char *,
    ssize_t,
    const std::vector<sycl::event> &);

typedef sycl::event (*binary_inplace_strided_impl_fn_ptr_t)(
    sycl::queue &,
    std::size_t,
    int,
    const ssize_t *,
    const char *,
    ssize_t,
    char *,
    ssize_t,
    const std::vector<sycl::event> &,
    const std::vector<sycl::event> &);

typedef sycl::event (*binary_inplace_row_matrix_broadcast_impl_fn_ptr_t)(
    sycl::queue &,
    std::vector<sycl::event> &,
    std::size_t,
    std::size_t,
    const char *,
    ssize_t,
    char *,
    ssize_t,
    const std::vector<sycl::event> &);

template <typename argTy,
          typename resTy,
          template <typename T1,
                    typename T2,
                    std::uint8_t vs,
                    std::uint8_t nv,
                    bool enable_sg_loadstore>
          class BinaryInplaceContigFunctorT,
          template <typename T1, typename T2, std::uint8_t vs, std::uint8_t nv>
          class kernel_name,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u>
sycl::event
binary_inplace_contig_impl(sycl::queue &exec_q,
                           std::size_t nelems,
                           const char *rhs_p,
                           ssize_t rhs_offset,
                           char *lhs_p,
                           ssize_t lhs_offset,
                           const std::vector<sycl::event> &depends = {})
{
    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        const std::size_t lws = 128;
        const std::size_t n_groups =
            ((nelems + lws * n_vecs * vec_sz - 1) / (lws * n_vecs * vec_sz));
        const auto gws_range = sycl::range<1>(n_groups * lws);
        const auto lws_range = sycl::range<1>(lws);

        const argTy *arg_tp =
            reinterpret_cast<const argTy *>(rhs_p) + rhs_offset;
        resTy *res_tp = reinterpret_cast<resTy *>(lhs_p) + lhs_offset;

        if (is_aligned<required_alignment>(arg_tp) &&
            is_aligned<required_alignment>(res_tp))
        {
            constexpr bool enable_sg_loadstore = true;
            using KernelName = kernel_name<argTy, resTy, vec_sz, n_vecs>;
            using Impl =
                BinaryInplaceContigFunctorT<argTy, resTy, vec_sz, n_vecs,
                                            enable_sg_loadstore>;

            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(gws_range, lws_range),
                Impl(arg_tp, res_tp, nelems));
        }
        else {
            constexpr bool disable_sg_loadstore = true;
            using InnerKernelName = kernel_name<argTy, resTy, vec_sz, n_vecs>;
            using KernelName =
                disabled_sg_loadstore_wrapper_krn<InnerKernelName>;
            using Impl =
                BinaryInplaceContigFunctorT<argTy, resTy, vec_sz, n_vecs,
                                            disable_sg_loadstore>;

            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(gws_range, lws_range),
                Impl(arg_tp, res_tp, nelems));
        }
    });
    return comp_ev;
}

template <typename argTy,
          typename resTy,
          template <typename T1, typename T2, typename IndT>
          class BinaryInplaceStridedFunctorT,
          template <typename T1, typename T2, typename IndT>
          class kernel_name>
sycl::event
binary_inplace_strided_impl(sycl::queue &exec_q,
                            std::size_t nelems,
                            int nd,
                            const ssize_t *shape_and_strides,
                            const char *rhs_p,
                            ssize_t rhs_offset,
                            char *lhs_p,
                            ssize_t lhs_offset,
                            const std::vector<sycl::event> &depends,
                            const std::vector<sycl::event> &additional_depends)
{
    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.depends_on(additional_depends);

        using IndexerT =
            typename dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;

        const IndexerT indexer{nd, rhs_offset, lhs_offset, shape_and_strides};

        const argTy *arg_tp = reinterpret_cast<const argTy *>(rhs_p);
        resTy *res_tp = reinterpret_cast<resTy *>(lhs_p);

        using Impl = BinaryInplaceStridedFunctorT<argTy, resTy, IndexerT>;

        cgh.parallel_for<kernel_name<argTy, resTy, IndexerT>>(
            {nelems}, Impl(arg_tp, res_tp, indexer));
    });
    return comp_ev;
}

template <typename argT,
          typename resT,
          template <typename T1, typename T3>
          class BinaryInplaceRowMatrixBroadcastFunctorT,
          template <typename T1, typename T3>
          class kernel_name>
sycl::event binary_inplace_row_matrix_broadcast_impl(
    sycl::queue &exec_q,
    std::vector<sycl::event> &host_tasks,
    std::size_t n0,
    std::size_t n1,
    const char *vec_p, // typeless pointer to (n1,) contiguous row
    ssize_t vec_offset,
    char *mat_p, // typeless pointer to (n0, n1) C-contiguous matrix
    ssize_t mat_offset,
    const std::vector<sycl::event> &depends = {})
{
    const argT *vec = reinterpret_cast<const argT *>(vec_p) + vec_offset;
    resT *mat = reinterpret_cast<resT *>(mat_p) + mat_offset;

    const auto &dev = exec_q.get_device();
    const auto &sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
    // Get device-specific kernel info max_sub_group_size
    std::size_t max_sgSize =
        *(std::max_element(std::begin(sg_sizes), std::end(sg_sizes)));

    std::size_t n1_padded = n1 + max_sgSize;
    auto padded_vec_owner =
        dpctl::tensor::alloc_utils::smart_malloc_device<argT>(n1_padded,
                                                              exec_q);
    argT *padded_vec = padded_vec_owner.get();

    sycl::event make_padded_vec_ev =
        dpctl::tensor::kernels::elementwise_detail::populate_padded_vector<
            argT>(exec_q, vec, n1, padded_vec, n1_padded, depends);

    // sub-group spans work-items [I, I + sgSize)
    // base = ndit.get_global_linear_id() - sg.get_local_id()[0]
    // Generically, sub_group_load( &mat[base]) may load arrays from
    // different rows of mat. The start corresponds to row (base / n0)
    // We read sub_group_load(&padded_vec[(base / n0)]). The vector is
    // padded to ensure that reads are accessible

    const std::size_t lws = 128;

    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(make_padded_vec_ev);

        auto lwsRange = sycl::range<1>(lws);
        std::size_t n_elems = n0 * n1;
        std::size_t n_groups = (n_elems + lws - 1) / lws;
        auto gwsRange = sycl::range<1>(n_groups * lws);

        using Impl = BinaryInplaceRowMatrixBroadcastFunctorT<argT, resT>;

        cgh.parallel_for<class kernel_name<argT, resT>>(
            sycl::nd_range<1>(gwsRange, lwsRange),
            Impl(padded_vec, mat, n_elems, n1));
    });

    sycl::event tmp_cleanup_ev = dpctl::tensor::alloc_utils::async_smart_free(
        exec_q, {comp_ev}, padded_vec_owner);
    host_tasks.push_back(tmp_cleanup_ev);

    return comp_ev;
}

} // namespace elementwise_common
} // namespace kernels
} // namespace tensor
} // namespace dpctl
