//=== expm1.hpp -   Unary function EXPM1                   ------
//*-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of EXPM1(x) function.
//===---------------------------------------------------------------------===//

#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "kernels/elementwise_functions/common.hpp"

#include "kernels/dpctl_tensor_types.hpp"
#include "utils/offset_utils.hpp"
#include "utils/type_dispatch_building.hpp"
#include "utils/type_utils.hpp"

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace expm1
{

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

    resT operator()(const argT &in) const
    {
        if constexpr (is_complex<argT>::value) {
            using realT = typename argT::value_type;
            // expm1(x + I*y) = expm1(x)*cos(y) - 2*sin(y / 2)^2 +
            // I*exp(x)*sin(y)
            const realT x = std::real(in);
            const realT y = std::imag(in);

            // special cases
            if (std::isinf(x)) {
                if (x > realT(0)) {
                    // positive infinity cases
                    if (!std::isfinite(y)) {
                        return resT{x, std::numeric_limits<realT>::quiet_NaN()};
                    }
                    else if (y == realT(0)) {
                        return in;
                    }
                    else {
                        return (resT{sycl::copysign(x, std::cos(y)),
                                     sycl::copysign(x, std::sin(y))});
                    }
                }
                else {
                    // negative infinity cases
                    if (!std::isfinite(y)) {
                        // copy sign of y to guarantee
                        // conj(expm1(x)) == expm1(conj(x))
                        return resT{realT(-1), sycl::copysign(realT(0), y)};
                    }
                    else {
                        return resT{realT(-1),
                                    sycl::copysign(realT(0), std::sin(y))};
                    }
                }
            }

            if (std::isnan(x)) {
                if (y == realT(0)) {
                    return in;
                }
                else {
                    return resT{std::numeric_limits<realT>::quiet_NaN(),
                                std::numeric_limits<realT>::quiet_NaN()};
                }
            }

            // x, y finite numbers
            const realT cosY_val = std::cos(y);
            const realT sinY_val = (y == 0) ? y : std::sin(y);
            const realT sinhalfY_val = (y == 0) ? y : std::sin(y / 2);

            const realT res_re =
                std::expm1(x) * cosY_val - 2 * sinhalfY_val * sinhalfY_val;
            realT res_im = std::exp(x) * sinY_val;
            return resT{res_re, res_im};
        }
        else {
            static_assert(std::is_floating_point_v<argT> ||
                          std::is_same_v<argT, sycl::half>);
            static_assert(std::is_same_v<argT, resT>);
            if (in == 0) {
                return in;
            }
            return std::expm1(in);
        }
    }
};

template <typename argTy,
          typename resTy = argTy,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using Expm1ContigFunctor =
    elementwise_common::UnaryContigFunctor<argTy,
                                           resTy,
                                           Expm1Functor<argTy, resTy>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

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

template <typename T1, typename T2, unsigned int vec_sz, unsigned int n_vecs>
class expm1_contig_kernel;

template <typename argTy>
sycl::event expm1_contig_impl(sycl::queue &exec_q,
                              size_t nelems,
                              const char *arg_p,
                              char *res_p,
                              const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::unary_contig_impl<
        argTy, Expm1OutputType, Expm1ContigFunctor, expm1_contig_kernel>(
        exec_q, nelems, arg_p, res_p, depends);
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

template <typename argTy>
sycl::event
expm1_strided_impl(sycl::queue &exec_q,
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
    return elementwise_common::unary_strided_impl<
        argTy, Expm1OutputType, Expm1StridedFunctor, expm1_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
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
