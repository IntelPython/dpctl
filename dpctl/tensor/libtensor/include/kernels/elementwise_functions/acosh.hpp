//=== acosh.hpp -   Unary function ACOSH                ------  *-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of ACOSH(x) function.
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
namespace acosh
{

namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;

template <typename argT, typename resT> struct AcoshFunctor
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
            /*
             * acosh(in) = I*acos(in) or -I*acos(in)
             * where the sign is chosen so Re(acosh(in)) >= 0.
             * So, we first calculate acos(in) and then acosh(in).
             */
            const realT x = std::real(in);
            const realT y = std::imag(in);

            resT acos_in;
            if (std::isnan(x)) {
                /* acos(NaN + I*+-Inf) = NaN + I*-+Inf */
                if (std::isinf(y)) {
                    acos_in = resT{q_nan, -y};
                }
                else {
                    acos_in = resT{q_nan, q_nan};
                }
            }
            else if (std::isnan(y)) {
                /* acos(+-Inf + I*NaN) = NaN + I*opt(-)Inf */
                constexpr realT inf = std::numeric_limits<realT>::infinity();

                if (std::isinf(x)) {
                    acos_in = resT{q_nan, -inf};
                }
                /* acos(0 + I*NaN) = Pi/2 + I*NaN with inexact */
                else if (x == realT(0)) {
                    const realT pi_half = std::atan(1) * 2;
                    acos_in = resT{pi_half, q_nan};
                }
                else {
                    acos_in = resT{q_nan, q_nan};
                }
            }

            constexpr realT r_eps =
                realT(1) / std::numeric_limits<realT>::epsilon();
            /*
             * For large x or y including acos(+-Inf + I*+-Inf)
             */
            if (sycl::fabs(x) > r_eps || sycl::fabs(y) > r_eps) {
#ifdef USE_SYCL_FOR_COMPLEX_TYPES
                using sycl_complexT = typename exprm_ns::complex<realT>;
                const sycl_complexT log_in = exprm_ns::log(sycl_complexT(in));
                const realT wx = log_in.real();
                const realT wy = log_in.imag();
#else
                const resT log_in = std::log(in);
                const realT wx = std::real(log_in);
                const realT wy = std::imag(log_in);
#endif
                const realT rx = sycl::fabs(wy);
                realT ry = wx + std::log(realT(2));
                acos_in = resT{rx, (std::signbit(y)) ? ry : -ry};
            }
            else {
                /* ordinary cases */
#if USE_SYCL_FOR_COMPLEX_TYPES
                acos_in = exprm_ns::acos(
                    exprm_ns::complex<realT>(in)); // std::acos(in);
#else
                acos_in = std::acos(in);
#endif
            }

            /* Now we calculate acosh(z) */
            const realT rx = std::real(acos_in);
            const realT ry = std::imag(acos_in);

            /* acosh(NaN + I*NaN) = NaN + I*NaN */
            if (std::isnan(rx) && std::isnan(ry)) {
                return resT{ry, rx};
            }
            /* acosh(NaN + I*+-Inf) = +Inf + I*NaN */
            /* acosh(+-Inf + I*NaN) = +Inf + I*NaN */
            if (std::isnan(rx)) {
                return resT{sycl::fabs(ry), rx};
            }
            /* acosh(0 + I*NaN) = NaN + I*NaN */
            if (std::isnan(ry)) {
                return resT{ry, ry};
            }
            /* ordinary cases */
            const realT res_im = sycl::copysign(rx, std::imag(in));
            return resT{sycl::fabs(ry), res_im};
        }
        else {
            static_assert(std::is_floating_point_v<argT> ||
                          std::is_same_v<argT, sycl::half>);
            return std::acosh(in);
        }
    }
};

template <typename argTy,
          typename resTy = argTy,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using AcoshContigFunctor =
    elementwise_common::UnaryContigFunctor<argTy,
                                           resTy,
                                           AcoshFunctor<argTy, resTy>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

template <typename argTy, typename resTy, typename IndexerT>
using AcoshStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, AcoshFunctor<argTy, resTy>>;

template <typename T> struct AcoshOutputType
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
class acosh_contig_kernel;

template <typename argTy>
sycl::event acosh_contig_impl(sycl::queue &exec_q,
                              size_t nelems,
                              const char *arg_p,
                              char *res_p,
                              const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::unary_contig_impl<
        argTy, AcoshOutputType, AcoshContigFunctor, acosh_contig_kernel>(
        exec_q, nelems, arg_p, res_p, depends);
}

template <typename fnT, typename T> struct AcoshContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename AcoshOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = acosh_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T> struct AcoshTypeMapFactory
{
    /*! @brief get typeid for output type of std::acosh(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename AcoshOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3> class acosh_strided_kernel;

template <typename argTy>
sycl::event
acosh_strided_impl(sycl::queue &exec_q,
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
        argTy, AcoshOutputType, AcoshStridedFunctor, acosh_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T> struct AcoshStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename AcoshOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = acosh_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace acosh
} // namespace kernels
} // namespace tensor
} // namespace dpctl
