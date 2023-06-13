//=== common.hpp -  Common code for elementwise operations ----- *-C++-*--/===//
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
/// This file defines common code for elementwise tensor operations.
//===---------------------------------------------------------------------===//

#pragma once
#include <CL/sycl.hpp>
#include <cstddef>
#include <cstdint>
#include <pybind11/pybind11.h>

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace elementwise_common
{

template <typename argT,
          typename resT,
          typename BinaryInplaceOperatorT,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2>
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

        if constexpr (BinaryInplaceOperatorT::supports_sg_loadstore::value &&
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
                using rhs_ptrT =
                    sycl::multi_ptr<const argT,
                                    sycl::access::address_space::global_space>;
                using lhs_ptrT =
                    sycl::multi_ptr<resT,
                                    sycl::access::address_space::global_space>;
                sycl::vec<argT, vec_sz> arg_vec;
                sycl::vec<resT, vec_sz> res_vec;

#pragma unroll
                for (std::uint8_t it = 0; it < n_vecs * vec_sz; it += vec_sz) {
                    arg_vec =
                        sg.load<vec_sz>(rhs_ptrT(&rhs[base + it * sgSize]));
                    res_vec =
                        sg.load<vec_sz>(lhs_ptrT(&lhs[base + it * sgSize]));
                    op(res_vec, arg_vec);
                    sg.store<vec_sz>(lhs_ptrT(&lhs[base + it * sgSize]),
                                     res_vec);
                }
            }
            else {
                for (size_t k = base + sg.get_local_id()[0]; k < nelems_;
                     k += sgSize) {
                    op(lhs[k], rhs[k]);
                }
            }
        }
        else if constexpr (BinaryInplaceOperatorT::supports_sg_loadstore::value)
        {
            auto sg = ndit.get_sub_group();
            std::uint8_t sgSize = sg.get_local_range()[0];
            std::uint8_t maxsgSize = sg.get_max_local_range()[0];

            size_t base = n_vecs * vec_sz *
                          (ndit.get_group(0) * ndit.get_local_range(0) +
                           sg.get_group_id()[0] * sgSize);

            if ((base + n_vecs * vec_sz * sgSize < nelems_) &&
                (sgSize == maxsgSize)) {
                using rhs_ptrT =
                    sycl::multi_ptr<const argT,
                                    sycl::access::address_space::global_space>;
                using lhs_ptrT =
                    sycl::multi_ptr<resT,
                                    sycl::access::address_space::global_space>;
                sycl::vec<argT, vec_sz> arg_vec;
                sycl::vec<resT, vec_sz> res_vec;

#pragma unroll
                for (std::uint8_t it = 0; it < n_vecs * vec_sz; it += vec_sz) {
                    arg_vec =
                        sg.load<vec_sz>(rhs_ptrT(&rhs[base + it * sgSize]));
                    res_vec =
                        sg.load<vec_sz>(lhs_ptT(&lhs[base + it * sgSize]));
#pragma unroll
                    for (std::uint8_t vec_id = 0; vec_id < vec_sz; ++vec_id) {
                        op(res_vec[vec_id], arg_vec[vec_id]);
                    }
                    sg.store<vec_sz>(lhs_ptrT(&lhs[base + it * sgSize]),
                                     res_vec);
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
    TwoOffsets_IndexerT two_offsets_indexer_;

public:
    BinaryInplaceStridedFunctor(const argT *rhs_tp,
                                resT *lhs_tp,
                                TwoOffsets_IndexerT inp_res_indexer)
        : rhs(rhs_tp), lhs(lhs_tp), two_offsets_indexer_(inp_res_indexer)
    {
    }

    void operator()(sycl::id<1> wid) const
    {
        const auto &two_offsets_ =
            two_offsets_indexer_(static_cast<py::ssize_t>(wid.get(0)));

        const auto &inp_offset = two_offsets_.get_first_offset();
        const auto &lhs_offset = two_offsets_.get_second_offset();

        BinaryInplaceOperatorT op{};
        op(lhs[lhs_offset], rhs[inp_offset]);
    }
};

// Typedefs for function pointers

typedef sycl::event (*binary_inplace_contig_impl_fn_ptr_t)(
    sycl::queue,
    size_t,
    const char *,
    py::ssize_t,
    char *,
    py::ssize_t,
    const std::vector<sycl::event> &);

typedef sycl::event (*binary_inplace_strided_impl_fn_ptr_t)(
    sycl::queue,
    size_t,
    int,
    const py::ssize_t *,
    const char *,
    py::ssize_t,
    char *,
    py::ssize_t,
    const std::vector<sycl::event> &,
    const std::vector<sycl::event> &);

template <typename argTy,
          typename resTy,
          template <typename T1, typename T2, unsigned int vs, unsigned int nv>
          class BinaryInplaceContigFunctorT,
          template <typename T1, typename T2, unsigned int vs, unsigned int nv>
          class kernel_name,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2>
sycl::event
binary_inplace_contig_impl(sycl::queue exec_q,
                           size_t nelems,
                           const char *rhs_p,
                           py::ssize_t rhs_offset,
                           char *lhs_p,
                           py::ssize_t lhs_offset,
                           const std::vector<sycl::event> &depends = {})
{
    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        size_t lws = 64;
        const size_t n_groups =
            ((nelems + lws * n_vecs * vec_sz - 1) / (lws * n_vecs * vec_sz));
        const auto gws_range = sycl::range<1>(n_groups * lws);
        const auto lws_range = sycl::range<1>(lws);

        const argTy *arg_tp =
            reinterpret_cast<const argTy *>(rhs_p) + rhs_offset;
        resTy *res_tp = reinterpret_cast<resTy *>(lhs_p) + lhs_offset;

        cgh.parallel_for<kernel_name<argTy, resTy, vec_sz, n_vecs>>(
            sycl::nd_range<1>(gws_range, lws_range),
            BinaryInplaceContigFunctorT<argTy, resTy, vec_sz, n_vecs>(
                arg_tp, res_tp, nelems));
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
binary_inplace_strided_impl(sycl::queue exec_q,
                            size_t nelems,
                            int nd,
                            const py::ssize_t *shape_and_strides,
                            const char *rhs_p,
                            py::ssize_t rhs_offset,
                            char *lhs_p,
                            py::ssize_t lhs_offset,
                            const std::vector<sycl::event> &depends,
                            const std::vector<sycl::event> &additional_depends)
{
    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.depends_on(additional_depends);

        using IndexerT =
            typename dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;

        IndexerT indexer{nd, rhs_offset, lhs_offset, shape_and_strides};

        const argTy *arg_tp = reinterpret_cast<const argTy *>(rhs_p);
        resTy *res_tp = reinterpret_cast<resTy *>(lhs_p);

        cgh.parallel_for<kernel_name<argTy, resTy, IndexerT>>(
            {nelems}, BinaryInplaceStridedFunctorT<argTy, resTy, IndexerT>(
                          arg_tp, res_tp, indexer));
    });
    return comp_ev;
}

} // namespace elementwise_common
} // namespace kernels
} // namespace tensor
} // namespace dpctl
