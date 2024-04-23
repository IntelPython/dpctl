//=== common.hpp -  Common code for elementwise operations ----- *-C++-*--/===//
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
/// This file defines common code for elementwise tensor operations.
//===---------------------------------------------------------------------===//

#pragma once
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <utility>

#include "kernels/alignment.hpp"
#include "kernels/dpctl_tensor_types.hpp"
#include "utils/offset_utils.hpp"

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

/*! @brief Functor for unary function evaluation on contiguous array */
template <typename argT,
          typename resT,
          typename UnaryOperatorT,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
struct UnaryContigFunctor
{
private:
    const argT *in = nullptr;
    resT *out = nullptr;
    const size_t nelems_;

public:
    UnaryContigFunctor(const argT *inp, resT *res, const size_t n_elems)
        : in(inp), out(res), nelems_(n_elems)
    {
    }

    void operator()(sycl::nd_item<1> ndit) const
    {
        UnaryOperatorT op{};
        /* Each work-item processes vec_sz elements, contiguous in memory */
        /* NOTE: vec_sz must divide sg.max_local_range()[0] */

        if constexpr (enable_sg_loadstore && UnaryOperatorT::is_constant::value)
        {
            // value of operator is known to be a known constant
            constexpr resT const_val = UnaryOperatorT::constant_value;

            auto sg = ndit.get_sub_group();
            std::uint8_t sgSize = sg.get_local_range()[0];
            std::uint8_t max_sgSize = sg.get_max_local_range()[0];
            size_t base = n_vecs * vec_sz *
                          (ndit.get_group(0) * ndit.get_local_range(0) +
                           sg.get_group_id()[0] * sgSize);
            if (base + n_vecs * vec_sz * sgSize < nelems_ &&
                max_sgSize == sgSize) {
                sycl::vec<resT, vec_sz> res_vec(const_val);
#pragma unroll
                for (std::uint8_t it = 0; it < n_vecs * vec_sz; it += vec_sz) {
                    size_t offset = base + static_cast<size_t>(it) *
                                               static_cast<size_t>(sgSize);
                    auto out_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&out[offset]);

                    sg.store<vec_sz>(out_multi_ptr, res_vec);
                }
            }
            else {
                for (size_t k = base + sg.get_local_id()[0]; k < nelems_;
                     k += sgSize) {
                    out[k] = const_val;
                }
            }
        }
        else if constexpr (enable_sg_loadstore &&
                           UnaryOperatorT::supports_sg_loadstore::value &&
                           UnaryOperatorT::supports_vec::value)
        {
            auto sg = ndit.get_sub_group();
            std::uint16_t sgSize = sg.get_local_range()[0];
            std::uint16_t max_sgSize = sg.get_max_local_range()[0];
            size_t base = n_vecs * vec_sz *
                          (ndit.get_group(0) * ndit.get_local_range(0) +
                           sg.get_group_id()[0] * max_sgSize);
            if (base + n_vecs * vec_sz * sgSize < nelems_ &&
                sgSize == max_sgSize) {
                sycl::vec<argT, vec_sz> x;

#pragma unroll
                for (std::uint16_t it = 0; it < n_vecs * vec_sz; it += vec_sz) {
                    size_t offset = base + static_cast<size_t>(it) *
                                               static_cast<size_t>(sgSize);
                    auto in_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&in[offset]);
                    auto out_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&out[offset]);

                    x = sg.load<vec_sz>(in_multi_ptr);
                    sycl::vec<resT, vec_sz> res_vec = op(x);
                    sg.store<vec_sz>(out_multi_ptr, res_vec);
                }
            }
            else {
                for (size_t k = base + sg.get_local_id()[0]; k < nelems_;
                     k += sgSize) {
                    // scalar call
                    out[k] = op(in[k]);
                }
            }
        }
        else if constexpr (enable_sg_loadstore &&
                           UnaryOperatorT::supports_sg_loadstore::value &&
                           std::is_same_v<resT, argT>)
        {
            // default: use scalar-value function

            auto sg = ndit.get_sub_group();
            std::uint8_t sgSize = sg.get_local_range()[0];
            std::uint8_t maxsgSize = sg.get_max_local_range()[0];
            size_t base = n_vecs * vec_sz *
                          (ndit.get_group(0) * ndit.get_local_range(0) +
                           sg.get_group_id()[0] * maxsgSize);

            if ((base + n_vecs * vec_sz * sgSize < nelems_) &&
                (maxsgSize == sgSize)) {
                sycl::vec<argT, vec_sz> arg_vec;

#pragma unroll
                for (std::uint8_t it = 0; it < n_vecs * vec_sz; it += vec_sz) {
                    size_t offset = base + static_cast<size_t>(it) *
                                               static_cast<size_t>(sgSize);
                    auto in_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&in[offset]);
                    auto out_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&out[offset]);

                    arg_vec = sg.load<vec_sz>(in_multi_ptr);
#pragma unroll
                    for (std::uint8_t k = 0; k < vec_sz; ++k) {
                        arg_vec[k] = op(arg_vec[k]);
                    }
                    sg.store<vec_sz>(out_multi_ptr, arg_vec);
                }
            }
            else {
                for (size_t k = base + sg.get_local_id()[0]; k < nelems_;
                     k += sgSize) {
                    out[k] = op(in[k]);
                }
            }
        }
        else if constexpr (enable_sg_loadstore &&
                           UnaryOperatorT::supports_sg_loadstore::value)
        {
            // default: use scalar-value function

            auto sg = ndit.get_sub_group();
            std::uint8_t sgSize = sg.get_local_range()[0];
            std::uint8_t maxsgSize = sg.get_max_local_range()[0];
            size_t base = n_vecs * vec_sz *
                          (ndit.get_group(0) * ndit.get_local_range(0) +
                           sg.get_group_id()[0] * maxsgSize);

            if ((base + n_vecs * vec_sz * sgSize < nelems_) &&
                (maxsgSize == sgSize)) {
                sycl::vec<argT, vec_sz> arg_vec;
                sycl::vec<resT, vec_sz> res_vec;

#pragma unroll
                for (std::uint8_t it = 0; it < n_vecs * vec_sz; it += vec_sz) {
                    size_t offset = base + static_cast<size_t>(it) *
                                               static_cast<size_t>(sgSize);
                    auto in_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&in[offset]);
                    auto out_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&out[offset]);

                    arg_vec = sg.load<vec_sz>(in_multi_ptr);
#pragma unroll
                    for (std::uint8_t k = 0; k < vec_sz; ++k) {
                        res_vec[k] = op(arg_vec[k]);
                    }
                    sg.store<vec_sz>(out_multi_ptr, res_vec);
                }
            }
            else {
                for (size_t k = base + sg.get_local_id()[0]; k < nelems_;
                     k += sgSize) {
                    out[k] = op(in[k]);
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
                out[offset] = op(in[offset]);
            }
        }
    }
};

template <typename argT, typename resT, typename IndexerT, typename UnaryOpT>
struct UnaryStridedFunctor
{
private:
    const argT *inp_ = nullptr;
    resT *res_ = nullptr;
    const IndexerT inp_out_indexer_;

public:
    UnaryStridedFunctor(const argT *inp_p,
                        resT *res_p,
                        const IndexerT &inp_out_indexer)
        : inp_(inp_p), res_(res_p), inp_out_indexer_(inp_out_indexer)
    {
    }

    void operator()(sycl::id<1> wid) const
    {
        const auto &offsets_ = inp_out_indexer_(wid.get(0));
        const ssize_t &inp_offset = offsets_.get_first_offset();
        const ssize_t &res_offset = offsets_.get_second_offset();

        UnaryOpT op{};

        res_[res_offset] = op(inp_[inp_offset]);
    }
};

template <typename argTy,
          template <typename T>
          class UnaryOutputType,
          template <typename A,
                    typename R,
                    unsigned int vs,
                    unsigned int nv,
                    bool enable>
          class ContigFunctorT,
          template <typename A, typename R, unsigned int vs, unsigned int nv>
          class kernel_name,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2>
sycl::event unary_contig_impl(sycl::queue &exec_q,
                              size_t nelems,
                              const char *arg_p,
                              char *res_p,
                              const std::vector<sycl::event> &depends = {})
{
    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        const size_t lws = 128;
        const size_t n_groups =
            ((nelems + lws * n_vecs * vec_sz - 1) / (lws * n_vecs * vec_sz));
        const auto gws_range = sycl::range<1>(n_groups * lws);
        const auto lws_range = sycl::range<1>(lws);

        using resTy = typename UnaryOutputType<argTy>::value_type;
        const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_p);
        resTy *res_tp = reinterpret_cast<resTy *>(res_p);

        if (is_aligned<required_alignment>(arg_p) &&
            is_aligned<required_alignment>(res_p))
        {
            constexpr bool enable_sg_loadstore = true;
            using KernelName = kernel_name<argTy, resTy, vec_sz, n_vecs>;

            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(gws_range, lws_range),
                ContigFunctorT<argTy, resTy, vec_sz, n_vecs,
                               enable_sg_loadstore>(arg_tp, res_tp, nelems));
        }
        else {
            constexpr bool disable_sg_loadstore = false;
            using InnerKernelName = kernel_name<argTy, resTy, vec_sz, n_vecs>;
            using KernelName =
                disabled_sg_loadstore_wrapper_krn<InnerKernelName>;

            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(gws_range, lws_range),
                ContigFunctorT<argTy, resTy, vec_sz, n_vecs,
                               disable_sg_loadstore>(arg_tp, res_tp, nelems));
        }
    });
    return comp_ev;
}

template <typename argTy,
          template <typename T>
          class UnaryOutputType,
          template <typename A, typename R, typename I>
          class StridedFunctorT,
          template <typename A, typename R, typename I>
          class kernel_name>
sycl::event
unary_strided_impl(sycl::queue &exec_q,
                   size_t nelems,
                   int nd,
                   const ssize_t *shape_and_strides,
                   const char *arg_p,
                   ssize_t arg_offset,
                   char *res_p,
                   ssize_t res_offset,
                   const std::vector<sycl::event> &depends,
                   const std::vector<sycl::event> &additional_depends)
{
    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.depends_on(additional_depends);

        using resTy = typename UnaryOutputType<argTy>::value_type;
        using IndexerT =
            typename dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;

        const IndexerT indexer{nd, arg_offset, res_offset, shape_and_strides};

        const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_p);
        resTy *res_tp = reinterpret_cast<resTy *>(res_p);

        cgh.parallel_for<kernel_name<argTy, resTy, IndexerT>>(
            {nelems},
            StridedFunctorT<argTy, resTy, IndexerT>(arg_tp, res_tp, indexer));
    });
    return comp_ev;
}

template <typename argT1,
          typename argT2,
          typename resT,
          typename BinaryOperatorT,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
struct BinaryContigFunctor
{
private:
    const argT1 *in1 = nullptr;
    const argT2 *in2 = nullptr;
    resT *out = nullptr;
    const size_t nelems_;

public:
    BinaryContigFunctor(const argT1 *inp1,
                        const argT2 *inp2,
                        resT *res,
                        const size_t n_elems)
        : in1(inp1), in2(inp2), out(res), nelems_(n_elems)
    {
    }

    void operator()(sycl::nd_item<1> ndit) const
    {
        BinaryOperatorT op{};
        /* Each work-item processes vec_sz elements, contiguous in memory */

        if constexpr (enable_sg_loadstore &&
                      BinaryOperatorT::supports_sg_loadstore::value &&
                      BinaryOperatorT::supports_vec::value)
        {
            auto sg = ndit.get_sub_group();
            std::uint8_t sgSize = sg.get_local_range()[0];
            std::uint8_t maxsgSize = sg.get_max_local_range()[0];

            size_t base = n_vecs * vec_sz *
                          (ndit.get_group(0) * ndit.get_local_range(0) +
                           sg.get_group_id()[0] * sgSize);

            if ((base + n_vecs * vec_sz * sgSize < nelems_) &&
                (sgSize == maxsgSize)) {
                sycl::vec<argT1, vec_sz> arg1_vec;
                sycl::vec<argT2, vec_sz> arg2_vec;
                sycl::vec<resT, vec_sz> res_vec;

#pragma unroll
                for (std::uint8_t it = 0; it < n_vecs * vec_sz; it += vec_sz) {
                    size_t offset = base + static_cast<size_t>(it) *
                                               static_cast<size_t>(sgSize);
                    auto in1_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&in1[offset]);
                    auto in2_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&in2[offset]);
                    auto out_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&out[offset]);

                    arg1_vec = sg.load<vec_sz>(in1_multi_ptr);
                    arg2_vec = sg.load<vec_sz>(in2_multi_ptr);
                    res_vec = op(arg1_vec, arg2_vec);
                    sg.store<vec_sz>(out_multi_ptr, res_vec);
                }
            }
            else {
                for (size_t k = base + sg.get_local_id()[0]; k < nelems_;
                     k += sgSize) {
                    out[k] = op(in1[k], in2[k]);
                }
            }
        }
        else if constexpr (enable_sg_loadstore &&
                           BinaryOperatorT::supports_sg_loadstore::value)
        {
            auto sg = ndit.get_sub_group();
            std::uint8_t sgSize = sg.get_local_range()[0];
            std::uint8_t maxsgSize = sg.get_max_local_range()[0];

            size_t base = n_vecs * vec_sz *
                          (ndit.get_group(0) * ndit.get_local_range(0) +
                           sg.get_group_id()[0] * sgSize);

            if ((base + n_vecs * vec_sz * sgSize < nelems_) &&
                (sgSize == maxsgSize)) {
                sycl::vec<argT1, vec_sz> arg1_vec;
                sycl::vec<argT2, vec_sz> arg2_vec;
                sycl::vec<resT, vec_sz> res_vec;

#pragma unroll
                for (std::uint8_t it = 0; it < n_vecs * vec_sz; it += vec_sz) {
                    size_t offset = base + static_cast<size_t>(it) *
                                               static_cast<size_t>(sgSize);
                    auto in1_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&in1[offset]);
                    auto in2_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&in2[offset]);
                    auto out_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&out[offset]);

                    arg1_vec = sg.load<vec_sz>(in1_multi_ptr);
                    arg2_vec = sg.load<vec_sz>(in2_multi_ptr);
#pragma unroll
                    for (std::uint8_t vec_id = 0; vec_id < vec_sz; ++vec_id) {
                        res_vec[vec_id] =
                            op(arg1_vec[vec_id], arg2_vec[vec_id]);
                    }
                    sg.store<vec_sz>(out_multi_ptr, res_vec);
                }
            }
            else {
                for (size_t k = base + sg.get_local_id()[0]; k < nelems_;
                     k += sgSize) {
                    out[k] = op(in1[k], in2[k]);
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
                out[offset] = op(in1[offset], in2[offset]);
            }
        }
    }
};

template <typename argT1,
          typename argT2,
          typename resT,
          typename ThreeOffsets_IndexerT,
          typename BinaryOperatorT>
struct BinaryStridedFunctor
{
private:
    const argT1 *in1 = nullptr;
    const argT2 *in2 = nullptr;
    resT *out = nullptr;
    const ThreeOffsets_IndexerT three_offsets_indexer_;

public:
    BinaryStridedFunctor(const argT1 *inp1_tp,
                         const argT2 *inp2_tp,
                         resT *res_tp,
                         const ThreeOffsets_IndexerT &inps_res_indexer)
        : in1(inp1_tp), in2(inp2_tp), out(res_tp),
          three_offsets_indexer_(inps_res_indexer)
    {
    }

    void operator()(sycl::id<1> wid) const
    {
        const auto &three_offsets_ =
            three_offsets_indexer_(static_cast<ssize_t>(wid.get(0)));

        const auto &inp1_offset = three_offsets_.get_first_offset();
        const auto &inp2_offset = three_offsets_.get_second_offset();
        const auto &out_offset = three_offsets_.get_third_offset();

        BinaryOperatorT op{};
        out[out_offset] = op(in1[inp1_offset], in2[inp2_offset]);
    }
};

template <typename argT1,
          typename argT2,
          typename resT,
          typename BinaryOperatorT>
struct BinaryContigMatrixContigRowBroadcastingFunctor
{
private:
    const argT1 *mat;
    const argT2 *padded_vec;
    resT *res;
    size_t n_elems;
    size_t n1;

public:
    BinaryContigMatrixContigRowBroadcastingFunctor(const argT1 *mat_tp,
                                                   const argT2 *row_tp,
                                                   resT *res_tp,
                                                   size_t n_elems_in_mat,
                                                   size_t n_elems_in_row)
        : mat(mat_tp), padded_vec(row_tp), res(res_tp), n_elems(n_elems_in_mat),
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
            auto in1_multi_ptr = sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::yes>(&mat[base]);

            auto in2_multi_ptr = sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::yes>(&padded_vec[base % n1]);

            auto out_multi_ptr = sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::yes>(&res[base]);

            const argT1 mat_el = sg.load(in1_multi_ptr);
            const argT2 vec_el = sg.load(in2_multi_ptr);

            resT res_el = op(mat_el, vec_el);

            sg.store(out_multi_ptr, res_el);
        }
        else {
            for (size_t k = base + sg.get_local_id()[0]; k < n_elems;
                 k += sgSize) {
                res[k] = op(mat[k], padded_vec[k % n1]);
            }
        }
    }
};

template <typename argT1,
          typename argT2,
          typename resT,
          typename BinaryOperatorT>
struct BinaryContigRowContigMatrixBroadcastingFunctor
{
private:
    const argT1 *padded_vec;
    const argT2 *mat;
    resT *res;
    size_t n_elems;
    size_t n1;

public:
    BinaryContigRowContigMatrixBroadcastingFunctor(const argT1 *row_tp,
                                                   const argT2 *mat_tp,
                                                   resT *res_tp,
                                                   size_t n_elems_in_mat,
                                                   size_t n_elems_in_row)
        : padded_vec(row_tp), mat(mat_tp), res(res_tp), n_elems(n_elems_in_mat),
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
            auto in1_multi_ptr = sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::yes>(&padded_vec[base % n1]);

            auto in2_multi_ptr = sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::yes>(&mat[base]);

            auto out_multi_ptr = sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::yes>(&res[base]);

            const argT2 mat_el = sg.load(in2_multi_ptr);
            const argT1 vec_el = sg.load(in1_multi_ptr);

            resT res_el = op(vec_el, mat_el);

            sg.store(out_multi_ptr, res_el);
        }
        else {
            for (size_t k = base + sg.get_local_id()[0]; k < n_elems;
                 k += sgSize) {
                res[k] = op(padded_vec[k % n1], mat[k]);
            }
        }
    }
};

// Typedefs for function pointers

typedef sycl::event (*unary_contig_impl_fn_ptr_t)(
    sycl::queue &,
    size_t,
    const char *,
    char *,
    const std::vector<sycl::event> &);

typedef sycl::event (*unary_strided_impl_fn_ptr_t)(
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

typedef sycl::event (*binary_contig_impl_fn_ptr_t)(
    sycl::queue &,
    size_t,
    const char *,
    ssize_t,
    const char *,
    ssize_t,
    char *,
    ssize_t,
    const std::vector<sycl::event> &);

typedef sycl::event (*binary_strided_impl_fn_ptr_t)(
    sycl::queue &,
    size_t,
    int,
    const ssize_t *,
    const char *,
    ssize_t,
    const char *,
    ssize_t,
    char *,
    ssize_t,
    const std::vector<sycl::event> &,
    const std::vector<sycl::event> &);

typedef sycl::event (*binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t)(
    sycl::queue &,
    std::vector<sycl::event> &,
    size_t,
    size_t,
    const char *,
    ssize_t,
    const char *,
    ssize_t,
    char *,
    ssize_t,
    const std::vector<sycl::event> &);

typedef sycl::event (*binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t)(
    sycl::queue &,
    std::vector<sycl::event> &,
    size_t,
    size_t,
    const char *,
    ssize_t,
    const char *,
    ssize_t,
    char *,
    ssize_t,
    const std::vector<sycl::event> &);

template <typename argTy1,
          typename argTy2,
          template <typename T1, typename T2>
          class BinaryOutputType,
          template <typename T1,
                    typename T2,
                    typename T3,
                    unsigned int vs,
                    unsigned int nv,
                    bool enable_sg_loadstore>
          class BinaryContigFunctorT,
          template <typename T1,
                    typename T2,
                    typename T3,
                    unsigned int vs,
                    unsigned int nv>
          class kernel_name,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2>
sycl::event binary_contig_impl(sycl::queue &exec_q,
                               size_t nelems,
                               const char *arg1_p,
                               ssize_t arg1_offset,
                               const char *arg2_p,
                               ssize_t arg2_offset,
                               char *res_p,
                               ssize_t res_offset,
                               const std::vector<sycl::event> &depends = {})
{
    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        const size_t lws = 128;
        const size_t n_groups =
            ((nelems + lws * n_vecs * vec_sz - 1) / (lws * n_vecs * vec_sz));
        const auto gws_range = sycl::range<1>(n_groups * lws);
        const auto lws_range = sycl::range<1>(lws);

        using resTy = typename BinaryOutputType<argTy1, argTy2>::value_type;

        const argTy1 *arg1_tp =
            reinterpret_cast<const argTy1 *>(arg1_p) + arg1_offset;
        const argTy2 *arg2_tp =
            reinterpret_cast<const argTy2 *>(arg2_p) + arg2_offset;
        resTy *res_tp = reinterpret_cast<resTy *>(res_p) + res_offset;

        if (is_aligned<required_alignment>(arg1_tp) &&
            is_aligned<required_alignment>(arg2_tp) &&
            is_aligned<required_alignment>(res_tp))
        {
            constexpr bool enable_sg_loadstore = true;
            using KernelName =
                kernel_name<argTy1, argTy2, resTy, vec_sz, n_vecs>;
            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(gws_range, lws_range),
                BinaryContigFunctorT<argTy1, argTy2, resTy, vec_sz, n_vecs,
                                     enable_sg_loadstore>(arg1_tp, arg2_tp,
                                                          res_tp, nelems));
        }
        else {
            constexpr bool disable_sg_loadstore = false;
            using InnerKernelName =
                kernel_name<argTy1, argTy2, resTy, vec_sz, n_vecs>;
            using KernelName =
                disabled_sg_loadstore_wrapper_krn<InnerKernelName>;
            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(gws_range, lws_range),
                BinaryContigFunctorT<argTy1, argTy2, resTy, vec_sz, n_vecs,
                                     disable_sg_loadstore>(arg1_tp, arg2_tp,
                                                           res_tp, nelems));
        }
    });
    return comp_ev;
}

template <typename argTy1,
          typename argTy2,
          template <typename T1, typename T2>
          class BinaryOutputType,
          template <typename T1, typename T2, typename T3, typename IndT>
          class BinaryStridedFunctorT,
          template <typename T1, typename T2, typename T3, typename IndT>
          class kernel_name>
sycl::event
binary_strided_impl(sycl::queue &exec_q,
                    size_t nelems,
                    int nd,
                    const ssize_t *shape_and_strides,
                    const char *arg1_p,
                    ssize_t arg1_offset,
                    const char *arg2_p,
                    ssize_t arg2_offset,
                    char *res_p,
                    ssize_t res_offset,
                    const std::vector<sycl::event> &depends,
                    const std::vector<sycl::event> &additional_depends)
{
    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.depends_on(additional_depends);

        using resTy = typename BinaryOutputType<argTy1, argTy2>::value_type;

        using IndexerT =
            typename dpctl::tensor::offset_utils::ThreeOffsets_StridedIndexer;

        const IndexerT indexer{nd, arg1_offset, arg2_offset, res_offset,
                               shape_and_strides};

        const argTy1 *arg1_tp = reinterpret_cast<const argTy1 *>(arg1_p);
        const argTy2 *arg2_tp = reinterpret_cast<const argTy2 *>(arg2_p);
        resTy *res_tp = reinterpret_cast<resTy *>(res_p);

        cgh.parallel_for<kernel_name<argTy1, argTy2, resTy, IndexerT>>(
            {nelems}, BinaryStridedFunctorT<argTy1, argTy2, resTy, IndexerT>(
                          arg1_tp, arg2_tp, res_tp, indexer));
    });
    return comp_ev;
}

template <typename argT1,
          typename argT2,
          typename resT,
          template <typename T1, typename T2, typename T3>
          class BinaryContigMatrixContigRowBroadcastFunctorT,
          template <typename T1, typename T2, typename T3>
          class kernel_name>
sycl::event binary_contig_matrix_contig_row_broadcast_impl(
    sycl::queue &exec_q,
    std::vector<sycl::event> &host_tasks,
    size_t n0,
    size_t n1,
    const char *mat_p, // typeless pointer to (n0, n1) C-contiguous matrix
    ssize_t mat_offset,
    const char *vec_p, // typeless pointer to (n1,) contiguous row
    ssize_t vec_offset,
    char *res_p, // typeless pointer to (n0, n1) result C-contig. matrix,
                 //    res[i,j] = op(mat[i,j], vec[j])
    ssize_t res_offset,
    const std::vector<sycl::event> &depends = {})
{
    const argT1 *mat = reinterpret_cast<const argT1 *>(mat_p) + mat_offset;
    const argT2 *vec = reinterpret_cast<const argT2 *>(vec_p) + vec_offset;
    resT *res = reinterpret_cast<resT *>(res_p) + res_offset;

    const auto &dev = exec_q.get_device();
    const auto &sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
    // Get device-specific kernel info max_sub_group_size
    size_t max_sgSize =
        *(std::max_element(std::begin(sg_sizes), std::end(sg_sizes)));

    size_t n1_padded = n1 + max_sgSize;
    argT2 *padded_vec = sycl::malloc_device<argT2>(n1_padded, exec_q);

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

        cgh.parallel_for<class kernel_name<argT1, argT2, resT>>(
            sycl::nd_range<1>(gwsRange, lwsRange),
            BinaryContigMatrixContigRowBroadcastFunctorT<argT1, argT2, resT>(
                mat, padded_vec, res, n_elems, n1));
    });

    sycl::event tmp_cleanup_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(comp_ev);
        const sycl::context &ctx = exec_q.get_context();
        cgh.host_task([ctx, padded_vec]() { sycl::free(padded_vec, ctx); });
    });
    host_tasks.push_back(tmp_cleanup_ev);

    return comp_ev;
}

template <typename argT1,
          typename argT2,
          typename resT,
          template <typename T1, typename T2, typename T3>
          class BinaryContigRowContigMatrixBroadcastFunctorT,
          template <typename T1, typename T2, typename T3>
          class kernel_name>
sycl::event binary_contig_row_contig_matrix_broadcast_impl(
    sycl::queue &exec_q,
    std::vector<sycl::event> &host_tasks,
    size_t n0,
    size_t n1,
    const char *vec_p, // typeless pointer to (n1,) contiguous row
    ssize_t vec_offset,
    const char *mat_p, // typeless pointer to (n0, n1) C-contiguous matrix
    ssize_t mat_offset,
    char *res_p, // typeless pointer to (n0, n1) result C-contig. matrix,
                 //    res[i,j] = op(vec[j], mat[i,j])
    ssize_t res_offset,
    const std::vector<sycl::event> &depends = {})
{
    const argT1 *vec = reinterpret_cast<const argT2 *>(vec_p) + vec_offset;
    const argT2 *mat = reinterpret_cast<const argT1 *>(mat_p) + mat_offset;
    resT *res = reinterpret_cast<resT *>(res_p) + res_offset;

    const auto &dev = exec_q.get_device();
    const auto &sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
    // Get device-specific kernel info max_sub_group_size
    size_t max_sgSize =
        *(std::max_element(std::begin(sg_sizes), std::end(sg_sizes)));

    size_t n1_padded = n1 + max_sgSize;
    argT2 *padded_vec = sycl::malloc_device<argT2>(n1_padded, exec_q);

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

        cgh.parallel_for<class kernel_name<argT1, argT2, resT>>(
            sycl::nd_range<1>(gwsRange, lwsRange),
            BinaryContigRowContigMatrixBroadcastFunctorT<argT1, argT2, resT>(
                padded_vec, mat, res, n_elems, n1));
    });

    sycl::event tmp_cleanup_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(comp_ev);
        const sycl::context &ctx = exec_q.get_context();
        cgh.host_task([ctx, padded_vec]() { sycl::free(padded_vec, ctx); });
    });
    host_tasks.push_back(tmp_cleanup_ev);

    return comp_ev;
};

} // namespace elementwise_common
} // namespace kernels
} // namespace tensor
} // namespace dpctl
