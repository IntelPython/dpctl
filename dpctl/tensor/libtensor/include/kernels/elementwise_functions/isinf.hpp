//=== isinf.hpp -   Unary function ISINF                 ------  *-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of ISINF(x)
/// function that tests whether a tensor element is an infinity.
//===---------------------------------------------------------------------===//

#pragma once
#include <CL/sycl.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "utils/offset_utils.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"
#include <pybind11/pybind11.h>

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace isinf
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;
using dpctl::tensor::type_utils::vec_cast;

template <typename argT, typename resT> struct IsInfFunctor
{
    static_assert(std::is_same_v<resT, bool>);

    using is_constant = typename std::disjunction<std::is_same<argT, bool>,
                                                  std::is_integral<argT>>;
    static constexpr resT constant_value = false;
    using supports_vec =
        typename std::disjunction<std::is_same<argT, sycl::half>,
                                  std::is_floating_point<argT>>;
    using supports_sg_loadstore = typename std::negation<
        std::disjunction<is_complex<resT>, is_complex<argT>>>;

    resT operator()(const argT &in) const
    {
        if constexpr (is_complex<argT>::value) {
            const bool real_isinf = std::isinf(std::real(in));
            const bool imag_isinf = std::isinf(std::imag(in));
            return (real_isinf || imag_isinf);
        }
        else if constexpr (std::is_same<argT, bool>::value ||
                           std::is_integral<argT>::value)
        {
            return constant_value;
        }
        else if constexpr (std::is_same_v<argT, sycl::half>) {
            return sycl::isinf(in);
        }
        else {
            return std::isinf(in);
        }
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz> operator()(const sycl::vec<argT, vec_sz> &in)
    {
        auto const &res_vec = sycl::isinf(in);

        using deducedT = typename std::remove_cv_t<
            std::remove_reference_t<decltype(res_vec)>>::element_type;

        return vec_cast<bool, deducedT, vec_sz>(res_vec);
    }
};

template <typename argT,
          typename resT = bool,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2>
using IsInfContigFunctor = elementwise_common::
    UnaryContigFunctor<argT, resT, IsInfFunctor<argT, resT>, vec_sz, n_vecs>;

template <typename argTy, typename resTy, typename IndexerT>
using IsInfStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, IsInfFunctor<argTy, resTy>>;

template <typename argTy> struct IsInfOutputType
{
    using value_type = bool;
};

template <typename T1, typename T2, std::uint8_t vec_sz, std::uint8_t n_vecs>
class isinf_contig_kernel;

template <typename argTy>
sycl::event isinf_contig_impl(sycl::queue exec_q,
                              size_t nelems,
                              const char *arg_p,
                              char *res_p,
                              const std::vector<sycl::event> &depends = {})
{
    sycl::event isinf_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        constexpr size_t lws = 64;
        constexpr std::uint8_t vec_sz = 4;
        constexpr std::uint8_t n_vecs = 2;
        static_assert(lws % vec_sz == 0);
        auto gws_range = sycl::range<1>(
            ((nelems + lws * n_vecs * vec_sz - 1) / (lws * n_vecs * vec_sz)) *
            lws);
        auto lws_range = sycl::range<1>(lws);

        using resTy = typename IsInfOutputType<argTy>::value_type;
        const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_p);
        resTy *res_tp = reinterpret_cast<resTy *>(res_p);

        cgh.parallel_for<
            class isinf_contig_kernel<argTy, resTy, vec_sz, n_vecs>>(
            sycl::nd_range<1>(gws_range, lws_range),
            IsInfContigFunctor<argTy, resTy, vec_sz, n_vecs>(arg_tp, res_tp,
                                                             nelems));
    });
    return isinf_ev;
}

template <typename fnT, typename T> struct IsInfContigFactory
{
    fnT get()
    {
        fnT fn = isinf_contig_impl<T>;
        return fn;
    }
};

template <typename fnT, typename T> struct IsInfTypeMapFactory
{
    /*! @brief get typeid for output type of sycl::isinf(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename IsInfOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3> class isinf_strided_kernel;

template <typename argTy>
sycl::event
isinf_strided_impl(sycl::queue exec_q,
                   size_t nelems,
                   int nd,
                   const py::ssize_t *shape_and_strides,
                   const char *arg_p,
                   py::ssize_t arg_offset,
                   char *res_p,
                   py::ssize_t res_offset,
                   const std::vector<sycl::event> &depends,
                   const std::vector<sycl::event> &additional_depends)
{
    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.depends_on(additional_depends);

        using resTy = typename IsInfOutputType<argTy>::value_type;
        using IndexerT =
            typename dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;

        IndexerT arg_res_indexer{nd, arg_offset, res_offset, shape_and_strides};

        const argTy *arg_tptr = reinterpret_cast<const argTy *>(arg_p);
        resTy *res_tptr = reinterpret_cast<resTy *>(res_p);

        sycl::range<1> gRange{nelems};

        cgh.parallel_for<isinf_strided_kernel<argTy, resTy, IndexerT>>(
            gRange, IsInfStridedFunctor<argTy, resTy, IndexerT>(
                        arg_tptr, res_tptr, arg_res_indexer));
    });
    return comp_ev;
}

template <typename fnT, typename T> struct IsInfStridedFactory
{
    fnT get()
    {
        fnT fn = isinf_strided_impl<T>;
        return fn;
    }
};

} // namespace isinf
} // namespace kernels
} // namespace tensor
} // namespace dpctl
