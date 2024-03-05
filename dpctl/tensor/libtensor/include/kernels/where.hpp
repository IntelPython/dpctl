//=== where.hpp -  Implementation of where kernels ---*-C++-*--/===//
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
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines kernels for dpctl.tensor.where.
//===----------------------------------------------------------------------===//

#pragma once
#include <algorithm>
#include <complex>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "dpctl_tensor_types.hpp"
#include "kernels/alignment.hpp"
#include "utils/offset_utils.hpp"
#include "utils/type_utils.hpp"

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace search
{

using namespace dpctl::tensor::offset_utils;

using dpctl::tensor::kernels::alignment_utils::
    disabled_sg_loadstore_wrapper_krn;
using dpctl::tensor::kernels::alignment_utils::is_aligned;
using dpctl::tensor::kernels::alignment_utils::required_alignment;

template <typename T, typename condT, typename IndexerT>
class where_strided_kernel;
template <typename T, typename condT, int vec_sz, int n_vecs>
class where_contig_kernel;

template <typename T,
          typename condT,
          int vec_sz = 4,
          int n_vecs = 2,
          bool enable_sg_loadstore = true>
class WhereContigFunctor
{
private:
    size_t nelems = 0;
    const condT *cond_p = nullptr;
    const T *x1_p = nullptr;
    const T *x2_p = nullptr;
    T *dst_p = nullptr;

public:
    WhereContigFunctor(size_t nelems_,
                       const condT *cond_p_,
                       const T *x1_p_,
                       const T *x2_p_,
                       T *dst_p_)
        : nelems(nelems_), cond_p(cond_p_), x1_p(x1_p_), x2_p(x2_p_),
          dst_p(dst_p_)
    {
    }

    void operator()(sycl::nd_item<1> ndit) const
    {
        using dpctl::tensor::type_utils::is_complex;
        if constexpr (!enable_sg_loadstore || is_complex<condT>::value ||
                      is_complex<T>::value)
        {
            std::uint8_t sgSize = ndit.get_sub_group().get_local_range()[0];
            size_t base = ndit.get_global_linear_id();

            base = (base / sgSize) * sgSize * n_vecs * vec_sz + (base % sgSize);
            for (size_t offset = base;
                 offset < std::min(nelems, base + sgSize * (n_vecs * vec_sz));
                 offset += sgSize)
            {
                using dpctl::tensor::type_utils::convert_impl;
                bool check = convert_impl<bool, condT>(cond_p[offset]);
                dst_p[offset] = check ? x1_p[offset] : x2_p[offset];
            }
        }
        else {
            auto sg = ndit.get_sub_group();
            std::uint8_t sgSize = sg.get_local_range()[0];
            std::uint8_t max_sgSize = sg.get_max_local_range()[0];
            size_t base = n_vecs * vec_sz *
                          (ndit.get_group(0) * ndit.get_local_range(0) +
                           sg.get_group_id()[0] * max_sgSize);

            if (base + n_vecs * vec_sz * sgSize < nelems &&
                sgSize == max_sgSize) {
                sycl::vec<T, vec_sz> dst_vec;
                sycl::vec<T, vec_sz> x1_vec;
                sycl::vec<T, vec_sz> x2_vec;
                sycl::vec<condT, vec_sz> cond_vec;

#pragma unroll
                for (std::uint8_t it = 0; it < n_vecs * vec_sz; it += vec_sz) {
                    auto idx = base + it * sgSize;
                    auto x1_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&x1_p[idx]);
                    auto x2_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&x2_p[idx]);
                    auto cond_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&cond_p[idx]);
                    auto dst_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&dst_p[idx]);

                    x1_vec = sg.load<vec_sz>(x1_multi_ptr);
                    x2_vec = sg.load<vec_sz>(x2_multi_ptr);
                    cond_vec = sg.load<vec_sz>(cond_multi_ptr);
#pragma unroll
                    for (std::uint8_t k = 0; k < vec_sz; ++k) {
                        dst_vec[k] = cond_vec[k] ? x1_vec[k] : x2_vec[k];
                    }
                    sg.store<vec_sz>(dst_multi_ptr, dst_vec);
                }
            }
            else {
                for (size_t k = base + sg.get_local_id()[0]; k < nelems;
                     k += sgSize) {
                    dst_p[k] = cond_p[k] ? x1_p[k] : x2_p[k];
                }
            }
        }
    }
};

typedef sycl::event (*where_contig_impl_fn_ptr_t)(
    sycl::queue &,
    size_t,
    const char *,
    const char *,
    const char *,
    char *,
    const std::vector<sycl::event> &);

template <typename T, typename condT>
sycl::event where_contig_impl(sycl::queue &q,
                              size_t nelems,
                              const char *cond_cp,
                              const char *x1_cp,
                              const char *x2_cp,
                              char *dst_cp,
                              const std::vector<sycl::event> &depends)
{
    const condT *cond_tp = reinterpret_cast<const condT *>(cond_cp);
    const T *x1_tp = reinterpret_cast<const T *>(x1_cp);
    const T *x2_tp = reinterpret_cast<const T *>(x2_cp);
    T *dst_tp = reinterpret_cast<T *>(dst_cp);

    sycl::event where_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        size_t lws = 64;
        constexpr unsigned int vec_sz = 4;
        constexpr unsigned int n_vecs = 2;
        const size_t n_groups =
            ((nelems + lws * n_vecs * vec_sz - 1) / (lws * n_vecs * vec_sz));
        const auto gws_range = sycl::range<1>(n_groups * lws);
        const auto lws_range = sycl::range<1>(lws);

        if (is_aligned<required_alignment>(cond_cp) &&
            is_aligned<required_alignment>(x1_cp) &&
            is_aligned<required_alignment>(x2_cp) &&
            is_aligned<required_alignment>(dst_cp))
        {
            constexpr bool enable_sg_loadstore = true;
            using KernelName = where_contig_kernel<T, condT, vec_sz, n_vecs>;

            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(gws_range, lws_range),
                WhereContigFunctor<T, condT, vec_sz, n_vecs,
                                   enable_sg_loadstore>(nelems, cond_tp, x1_tp,
                                                        x2_tp, dst_tp));
        }
        else {
            constexpr bool disable_sg_loadstore = false;
            using InnerKernelName =
                where_contig_kernel<T, condT, vec_sz, n_vecs>;
            using KernelName =
                disabled_sg_loadstore_wrapper_krn<InnerKernelName>;

            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(gws_range, lws_range),
                WhereContigFunctor<T, condT, vec_sz, n_vecs,
                                   disable_sg_loadstore>(nelems, cond_tp, x1_tp,
                                                         x2_tp, dst_tp));
        }
    });

    return where_ev;
}

template <typename T, typename condT, typename IndexerT>
class WhereStridedFunctor
{
private:
    const T *x1_p = nullptr;
    const T *x2_p = nullptr;
    T *dst_p = nullptr;
    const condT *cond_p = nullptr;
    const IndexerT indexer;

public:
    WhereStridedFunctor(const condT *cond_p_,
                        const T *x1_p_,
                        const T *x2_p_,
                        T *dst_p_,
                        const IndexerT &indexer_)
        : x1_p(x1_p_), x2_p(x2_p_), dst_p(dst_p_), cond_p(cond_p_),
          indexer(indexer_)
    {
    }

    void operator()(sycl::id<1> id) const
    {
        size_t gid = id[0];
        auto offsets = indexer(static_cast<ssize_t>(gid));

        using dpctl::tensor::type_utils::convert_impl;
        bool check =
            convert_impl<bool, condT>(cond_p[offsets.get_first_offset()]);

        dst_p[offsets.get_fourth_offset()] =
            check ? x1_p[offsets.get_second_offset()]
                  : x2_p[offsets.get_third_offset()];
    }
};

typedef sycl::event (*where_strided_impl_fn_ptr_t)(
    sycl::queue &,
    size_t,
    int,
    const char *,
    const char *,
    const char *,
    char *,
    const ssize_t *,
    ssize_t,
    ssize_t,
    ssize_t,
    ssize_t,
    const std::vector<sycl::event> &);

template <typename T, typename condT>
sycl::event where_strided_impl(sycl::queue &q,
                               size_t nelems,
                               int nd,
                               const char *cond_cp,
                               const char *x1_cp,
                               const char *x2_cp,
                               char *dst_cp,
                               const ssize_t *shape_strides,
                               ssize_t x1_offset,
                               ssize_t x2_offset,
                               ssize_t cond_offset,
                               ssize_t dst_offset,
                               const std::vector<sycl::event> &depends)
{
    const condT *cond_tp = reinterpret_cast<const condT *>(cond_cp);
    const T *x1_tp = reinterpret_cast<const T *>(x1_cp);
    const T *x2_tp = reinterpret_cast<const T *>(x2_cp);
    T *dst_tp = reinterpret_cast<T *>(dst_cp);

    sycl::event where_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        const FourOffsets_StridedIndexer indexer{
            nd, cond_offset, x1_offset, x2_offset, dst_offset, shape_strides};

        cgh.parallel_for<
            where_strided_kernel<T, condT, FourOffsets_StridedIndexer>>(
            sycl::range<1>(nelems),
            WhereStridedFunctor<T, condT, FourOffsets_StridedIndexer>(
                cond_tp, x1_tp, x2_tp, dst_tp, indexer));
    });

    return where_ev;
}

template <typename fnT, typename T, typename condT> struct WhereStridedFactory
{
    fnT get()
    {
        fnT fn = where_strided_impl<T, condT>;
        return fn;
    }
};

template <typename fnT, typename T, typename condT> struct WhereContigFactory
{
    fnT get()
    {
        fnT fn = where_contig_impl<T, condT>;
        return fn;
    }
};

} // namespace search
} // namespace kernels
} // namespace tensor
} // namespace dpctl
