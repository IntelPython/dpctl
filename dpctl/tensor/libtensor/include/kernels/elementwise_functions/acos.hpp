//=== acos.hpp -   Unary function ACOS                  ------  *-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of ACOS(x) function.
//===---------------------------------------------------------------------===//

#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "kernels/elementwise_functions/common.hpp"
#include "sycl_complex.hpp"

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
namespace acos
{

namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;

template <typename argT, typename resT> struct AcosFunctor
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

            constexpr realT q_nan = std::numeric_limits<realT>::quiet_NaN();

            const realT x = std::real(in);
            const realT y = std::imag(in);

            if (std::isnan(x)) {
                /* acos(NaN + I*+-Inf) = NaN + I*-+Inf */
                if (std::isinf(y)) {
                    return resT{q_nan, -y};
                }

                /* all other cases involving NaN return NaN + I*NaN. */
                return resT{q_nan, q_nan};
            }
            if (std::isnan(y)) {
                /* acos(+-Inf + I*NaN) = NaN + I*opt(-)Inf */
                if (std::isinf(x)) {
                    return resT{q_nan, -std::numeric_limits<realT>::infinity()};
                }
                /* acos(0 + I*NaN) = PI/2 + I*NaN with inexact */
                if (x == realT(0)) {
                    const realT res_re = std::atan(realT(1)) * 2; // PI/2
                    return resT{res_re, q_nan};
                }

                /* all other cases involving NaN return NaN + I*NaN. */
                return resT{q_nan, q_nan};
            }

            /*
             * For large x or y including acos(+-Inf + I*+-Inf)
             */
            constexpr realT r_eps =
                realT(1) / std::numeric_limits<realT>::epsilon();
            if (sycl::fabs(x) > r_eps || sycl::fabs(y) > r_eps) {
#ifdef USE_SYCL_FOR_COMPLEX_TYPES
                using sycl_complexT = exprm_ns::complex<realT>;
                sycl_complexT log_in =
                    exprm_ns::log(exprm_ns::complex<realT>(in));

                const realT wx = log_in.real();
                const realT wy = log_in.imag();
                const realT rx = sycl::fabs(wy);

                realT ry = wx + std::log(realT(2));
                return resT{rx, (std::signbit(y)) ? ry : -ry};
#else
                resT log_in = std::log(in);
                const realT wx = std::real(log_in);
                const realT wy = std::imag(log_in);
                const realT rx = sycl::fabs(wy);

                realT ry = wx + std::log(realT(2));
                return resT{rx, (std::signbit(y)) ? ry : -ry};
#endif
            }

            /* ordinary cases */
#if USE_SYCL_FOR_COMPLEX_TYPES
            return exprm_ns::acos(
                exprm_ns::complex<realT>(in)); // std::acos(in);
#else
            return std::acos(in);
#endif
        }
        else {
            static_assert(std::is_floating_point_v<argT> ||
                          std::is_same_v<argT, sycl::half>);
            return std::acos(in);
        }
    }
};

template <typename argTy,
          typename resTy = argTy,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using AcosContigFunctor =
    elementwise_common::UnaryContigFunctor<argTy,
                                           resTy,
                                           AcosFunctor<argTy, resTy>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

template <typename argTy, typename resTy, typename IndexerT>
using AcosStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, AcosFunctor<argTy, resTy>>;

template <typename T> struct AcosOutputType
{
    using value_type = typename std::disjunction< // disjunction is C++17
                                                  // feature, supported by DPC++
        td_ns::TypeMapResultEntry<T, sycl::half>,
        td_ns::TypeMapResultEntry<T, float>,
        td_ns::TypeMapResultEntry<T, double>,
        td_ns::TypeMapResultEntry<T, std::complex<float>>,
        td_ns::TypeMapResultEntry<T, std::complex<double>>,
        td_ns::DefaultResultEntry<void>>::result_type;
};

template <typename T1, typename T2, unsigned int vec_sz, unsigned int n_vecs>
class acos_contig_kernel;

template <typename argTy>
sycl::event acos_contig_impl(sycl::queue &exec_q,
                             size_t nelems,
                             const char *arg_p,
                             char *res_p,
                             const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::unary_contig_impl<
        argTy, AcosOutputType, AcosContigFunctor, acos_contig_kernel>(
        exec_q, nelems, arg_p, res_p, depends);
}

template <typename fnT, typename T> struct AcosContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename AcosOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = acos_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T> struct AcosTypeMapFactory
{
    /*! @brief get typeid for output type of std::acos(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename AcosOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3> class acos_strided_kernel;

template <typename argTy>
sycl::event
acos_strided_impl(sycl::queue &exec_q,
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
        argTy, AcosOutputType, AcosStridedFunctor, acos_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T> struct AcosStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename AcosOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = acos_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace acos
} // namespace kernels
} // namespace tensor
} // namespace dpctl
