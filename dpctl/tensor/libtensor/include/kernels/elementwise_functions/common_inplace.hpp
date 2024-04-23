//=== common_inplace.hpp -  Common code for in-place elementwise operations
//----- *-C++-*--/===//
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
/// This file defines common code for in-place elementwise tensor operations.
//===---------------------------------------------------------------------===//

#pragma once
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

#include "kernels/alignment.hpp"
#include "kernels/dpctl_tensor_types.hpp"

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace elementwise_common
{

using dpctl::tensor::kernels::alignment_utils::
    disabled_sg_loadstore_wrapper_krn;
using dpctl::tensor::kernels::alignment_utils::is_aligned;
using dpctl::tensor::kernels::alignment_utils::required_alignment;

template <typename argT,
          typename resT,
          typename BinaryInplaceOperatorT,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
struct BinaryInplaceContigFunctor
{
private:
    const argT *rhs = nullptr;
    resT *lhs = nullptr;
    const size_t nelems_;

public:
    BinaryInplaceContigFunctor(const argT *rhs_tp,
                               resT *lhs_tp,
                               const size_t n_elems)
        : rhs(rhs_tp), lhs(lhs_tp), nelems_(n_elems)
    {
    }

    void operator()(sycl::nd_item<1> ndit) const
    {
        BinaryInplaceOperatorT op{};
        /* Each work-item processes vec_sz elements, contiguous in memory */

        if constexpr (enable_sg_loadstore &&
                      BinaryInplaceOperatorT::supports_sg_loadstore::value &&
                      BinaryInplaceOperatorT::supports_vec::value)
        {
            auto sg = ndit.get_sub_group();
            std::uint8_t sgSize = sg.get_local_range()[0];
            std::uint8_t maxsgSize = sg.get_max_local_range()[0];

            size_t base = n_vecs * vec_sz *
                          (ndit.get_group(0) * ndit.get_local_range(0) +
                           sg.get_group_id()[0] * sgSize);

            if ((base + n_vecs * vec_sz * sgSize < nelems_) &&
                (sgSize == maxsgSize)) {

                sycl::vec<argT, vec_sz> arg_vec;
                sycl::vec<resT, vec_sz> res_vec;

#pragma unroll
                for (std::uint8_t it = 0; it < n_vecs * vec_sz; it += vec_sz) {
                    auto rhs_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&rhs[base + it * sgSize]);
                    auto lhs_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&lhs[base + it * sgSize]);

                    arg_vec = sg.load<vec_sz>(rhs_multi_ptr);
                    res_vec = sg.load<vec_sz>(lhs_multi_ptr);
                    op(res_vec, arg_vec);

                    sg.store<vec_sz>(lhs_multi_ptr, res_vec);
                }
            }
            else {
                for (size_t k = base + sg.get_local_id()[0]; k < nelems_;
                     k += sgSize) {
                    op(lhs[k], rhs[k]);
                }
            }
        }
        else if constexpr (enable_sg_loadstore &&
                           BinaryInplaceOperatorT::supports_sg_loadstore::value)
        {
            auto sg = ndit.get_sub_group();
            std::uint8_t sgSize = sg.get_local_range()[0];
            std::uint8_t maxsgSize = sg.get_max_local_range()[0];

            size_t base = n_vecs * vec_sz *
                          (ndit.get_group(0) * ndit.get_local_range(0) +
                           sg.get_group_id()[0] * sgSize);

            if ((base + n_vecs * vec_sz * sgSize < nelems_) &&
                (sgSize == maxsgSize)) {
                sycl::vec<argT, vec_sz> arg_vec;
                sycl::vec<resT, vec_sz> res_vec;

#pragma unroll
                for (std::uint8_t it = 0; it < n_vecs * vec_sz; it += vec_sz) {
                    auto rhs_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&rhs[base + it * sgSize]);
                    auto lhs_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&lhs[base + it * sgSize]);

                    arg_vec = sg.load<vec_sz>(rhs_multi_ptr);
                    res_vec = sg.load<vec_sz>(lhs_multi_ptr);
#pragma unroll
                    for (std::uint8_t vec_id = 0; vec_id < vec_sz; ++vec_id) {
                        op(res_vec[vec_id], arg_vec[vec_id]);
                    }
                    sg.store<vec_sz>(lhs_multi_ptr, res_vec);
                }
            }
            else {
                for (size_t k = base + sg.get_local_id()[0]; k < nelems_;
                     k += sgSize) {
                    op(lhs[k], rhs[k]);
                }
            }
        }
        else {
            std::uint8_t sgSize = ndit.get_sub_group().get_local_range()[0];
            size_t base = ndit.get_global_linear_id();

            base = (base / sgSize) * sgSize * n_vecs * vec_sz + (base % sgSize);
            for (size_t offset = base;
                 offset < std::min(nelems_, base + sgSize * (n_vecs * vec_sz));
                 offset += sgSize)
            {
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
    const TwoOffsets_IndexerT two_offsets_indexer_;

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
    size_t n_elems;
    size_t n1;

public:
    BinaryInplaceRowMatrixBroadcastingFunctor(const argT *row_tp,
                                              resT *mat_tp,
                                              size_t n_elems_in_mat,
                                              size_t n_elems_in_row)
        : padded_vec(row_tp), mat(mat_tp), n_elems(n_elems_in_mat),
          n1(n_elems_in_row)
    {
    }

    void operator()(sycl::nd_item<1> ndit) const
    {
        BinaryOperatorT op{};
        static_assert(BinaryOperatorT::supports_sg_loadstore::value);

        auto sg = ndit.get_sub_group();
        size_t gid = ndit.get_global_linear_id();

        std::uint8_t sgSize = sg.get_local_range()[0];
        size_t base = gid - sg.get_local_id()[0];

        if (base + sgSize < n_elems) {
            auto in_multi_ptr = sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::yes>(&padded_vec[base % n1]);

            auto out_multi_ptr = sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::yes>(&mat[base]);

            const argT vec_el = sg.load(in_multi_ptr);
            resT mat_el = sg.load(out_multi_ptr);

            op(mat_el, vec_el);

            sg.store(out_multi_ptr, mat_el);
        }
        else {
            for (size_t k = base + sg.get_local_id()[0]; k < n_elems;
                 k += sgSize) {
                op(mat[k], padded_vec[k % n1]);
            }
        }
    }
};

// Typedefs for function pointers

typedef sycl::event (*binary_inplace_contig_impl_fn_ptr_t)(
    sycl::queue &,
    size_t,
    const char *,
    ssize_t,
    char *,
    ssize_t,
    const std::vector<sycl::event> &);

typedef sycl::event (*binary_inplace_strided_impl_fn_ptr_t)(
    sycl::queue &,
    size_t,
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
    size_t,
    size_t,
    const char *,
    ssize_t,
    char *,
    ssize_t,
    const std::vector<sycl::event> &);

template <typename argTy,
          typename resTy,
          template <typename T1,
                    typename T2,
                    unsigned int vs,
                    unsigned int nv,
                    bool enable_sg_loadstore>
          class BinaryInplaceContigFunctorT,
          template <typename T1, typename T2, unsigned int vs, unsigned int nv>
          class kernel_name,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2>
sycl::event
binary_inplace_contig_impl(sycl::queue &exec_q,
                           size_t nelems,
                           const char *rhs_p,
                           ssize_t rhs_offset,
                           char *lhs_p,
                           ssize_t lhs_offset,
                           const std::vector<sycl::event> &depends = {})
{
    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        const size_t lws = 128;
        const size_t n_groups =
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
            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(gws_range, lws_range),
                BinaryInplaceContigFunctorT<argTy, resTy, vec_sz, n_vecs,
                                            enable_sg_loadstore>(arg_tp, res_tp,
                                                                 nelems));
        }
        else {
            constexpr bool disable_sg_loadstore = true;
            using InnerKernelName = kernel_name<argTy, resTy, vec_sz, n_vecs>;
            using KernelName =
                disabled_sg_loadstore_wrapper_krn<InnerKernelName>;

            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(gws_range, lws_range),
                BinaryInplaceContigFunctorT<argTy, resTy, vec_sz, n_vecs,
                                            disable_sg_loadstore>(
                    arg_tp, res_tp, nelems));
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
                            size_t nelems,
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

        cgh.parallel_for<kernel_name<argTy, resTy, IndexerT>>(
            {nelems}, BinaryInplaceStridedFunctorT<argTy, resTy, IndexerT>(
                          arg_tp, res_tp, indexer));
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
    size_t n0,
    size_t n1,
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
    size_t max_sgSize =
        *(std::max_element(std::begin(sg_sizes), std::end(sg_sizes)));

    size_t n1_padded = n1 + max_sgSize;
    argT *padded_vec = sycl::malloc_device<argT>(n1_padded, exec_q);

    if (padded_vec == nullptr) {
        throw std::runtime_error("Could not allocate memory on the device");
    }
    sycl::event make_padded_vec_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends); // ensure vec contains actual data
        cgh.parallel_for({n1_padded}, [=](sycl::id<1> id) {
            auto i = id[0];
            padded_vec[i] = vec[i % n1];
        });
    });

    // sub-group spans work-items [I, I + sgSize)
    // base = ndit.get_global_linear_id() - sg.get_local_id()[0]
    // Generically, sg.load( &mat[base]) may load arrays from
    // different rows of mat. The start corresponds to row (base / n0)
    // We read sg.load(&padded_vec[(base / n0)]). The vector is padded to
    // ensure that reads are accessible

    const size_t lws = 128;

    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(make_padded_vec_ev);

        auto lwsRange = sycl::range<1>(lws);
        size_t n_elems = n0 * n1;
        size_t n_groups = (n_elems + lws - 1) / lws;
        auto gwsRange = sycl::range<1>(n_groups * lws);

        cgh.parallel_for<class kernel_name<argT, resT>>(
            sycl::nd_range<1>(gwsRange, lwsRange),
            BinaryInplaceRowMatrixBroadcastFunctorT<argT, resT>(padded_vec, mat,
                                                                n_elems, n1));
    });

    sycl::event tmp_cleanup_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(comp_ev);
        const sycl::context &ctx = exec_q.get_context();
        cgh.host_task([ctx, padded_vec]() { sycl::free(padded_vec, ctx); });
    });
    host_tasks.push_back(tmp_cleanup_ev);

    return comp_ev;
}

} // namespace elementwise_common
} // namespace kernels
} // namespace tensor
} // namespace dpctl
