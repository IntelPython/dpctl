//=== expm1.hpp -   Unary function EXPM1                   ------
//*-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of EXPM1(x) function.
//===---------------------------------------------------------------------===//

#pragma once
#include <CL/sycl.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "kernels/elementwise_functions/common.hpp"

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
namespace expm1
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;

template <typename argT, typename resT> struct Expm1Functor
{

    // is function constant for given argT
    using is_constant = typename std::false_type;
    // constant value, if constant
    // constexpr resT constant_value = resT{};
    // is function defined for sycl::vec
    using supports_vec = typename std::false_type;
    // do both argTy and resTy support sugroup store/load operation
    using supports_sg_loadstore = typename std::negation<
        std::disjunction<is_complex<resT>, is_complex<argT>>>;

    resT operator()(const argT &in)
    {
        if constexpr (is_complex<argT>::value) {
            using realT = typename argT::value_type;
            // expm1(x + I*y) = expm1(x)*cos(y) - 2*sin(y / 2)^2 +
            // I*exp(x)*sin(y)
            const realT x = std::real(in);
            const realT y = std::imag(in);

            realT cosY_val;
            const realT sinY_val = sycl::sincos(y, &cosY_val);
            const realT sinhalfY_val = std::sin(y / 2);

            const realT res_re =
                std::expm1(x) * cosY_val - 2 * sinhalfY_val * sinhalfY_val;
            const realT res_im = std::exp(x) * sinY_val;
            return resT{res_re, res_im};
        }
        else {
            static_assert(std::is_floating_point_v<argT> ||
                          std::is_same_v<argT, sycl::half>);
            return std::expm1(in);
        }
    }
};

template <typename argTy,
          typename resTy = argTy,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2>
using Expm1ContigFunctor =
    elementwise_common::UnaryContigFunctor<argTy,
                                           resTy,
                                           Expm1Functor<argTy, resTy>,
                                           vec_sz,
                                           n_vecs>;

template <typename argTy, typename resTy, typename IndexerT>
using Expm1StridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, Expm1Functor<argTy, resTy>>;

template <typename T> struct Expm1OutputType
{
    using value_type = typename std::disjunction< // disjunction is C++17
                                                  // feature, supported by DPC++
        td_ns::TypeMapResultEntry<T, sycl::half, sycl::half>,
        td_ns::TypeMapResultEntry<T, float, float>,
        td_ns::TypeMapResultEntry<T, double, double>,
        td_ns::TypeMapResultEntry<T, std::complex<float>, std::complex<float>>,
        td_ns::
            TypeMapResultEntry<T, std::complex<double>, std::complex<double>>,
        td_ns::DefaultResultEntry<void>>::result_type;
};

typedef sycl::event (*expm1_contig_impl_fn_ptr_t)(
    sycl::queue,
    size_t,
    const char *,
    char *,
    const std::vector<sycl::event> &);

template <typename T1, typename T2, unsigned int vec_sz, unsigned int n_vecs>
class expm1_contig_kernel;

template <typename argTy>
sycl::event expm1_contig_impl(sycl::queue exec_q,
                              size_t nelems,
                              const char *arg_p,
                              char *res_p,
                              const std::vector<sycl::event> &depends = {})
{
    sycl::event expm1_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        constexpr size_t lws = 64;
        constexpr unsigned int vec_sz = 4;
        constexpr unsigned int n_vecs = 2;
        static_assert(lws % vec_sz == 0);
        auto gws_range = sycl::range<1>(
            ((nelems + n_vecs * lws * vec_sz - 1) / (lws * n_vecs * vec_sz)) *
            lws);
        auto lws_range = sycl::range<1>(lws);

        using resTy = typename Expm1OutputType<argTy>::value_type;
        const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_p);
        resTy *res_tp = reinterpret_cast<resTy *>(res_p);

        cgh.parallel_for<
            class expm1_contig_kernel<argTy, resTy, vec_sz, n_vecs>>(
            sycl::nd_range<1>(gws_range, lws_range),
            Expm1ContigFunctor<argTy, resTy, vec_sz, n_vecs>(arg_tp, res_tp,
                                                             nelems));
    });
    return expm1_ev;
}

template <typename fnT, typename T> struct Expm1ContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename Expm1OutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = expm1_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T> struct Expm1TypeMapFactory
{
    /*! @brief get typeid for output type of std::expm1(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename Expm1OutputType<T>::value_type;
        ;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3> class expm1_strided_kernel;

typedef sycl::event (*expm1_strided_impl_fn_ptr_t)(
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

template <typename argTy>
sycl::event
expm1_strided_impl(sycl::queue exec_q,
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

        using resTy = typename Expm1OutputType<argTy>::value_type;
        using IndexerT =
            typename dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;

        IndexerT arg_res_indexer(nd, arg_offset, res_offset, shape_and_strides);

        const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_p);
        resTy *res_tp = reinterpret_cast<resTy *>(res_p);

        sycl::range<1> gRange{nelems};

        cgh.parallel_for<expm1_strided_kernel<argTy, resTy, IndexerT>>(
            gRange, Expm1StridedFunctor<argTy, resTy, IndexerT>(
                        arg_tp, res_tp, arg_res_indexer));
    });
    return comp_ev;
}

template <typename fnT, typename T> struct Expm1StridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename Expm1OutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = expm1_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace expm1
} // namespace kernels
} // namespace tensor
} // namespace dpctl
