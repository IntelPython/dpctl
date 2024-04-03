//=== asin.hpp -   Unary function ASIN                  ------  *-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of ASIN(x) function.
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
namespace asin
{

namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;

template <typename argT, typename resT> struct AsinFunctor
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
             * asin(in) = I * conj( asinh(I * conj(in)) )
             * so we first calculate w = asinh(I * conj(in)) with
             * x = real(I * conj(in)) = imag(in)
             * y = imag(I * conj(in)) = real(in)
             * and then return {imag(w), real(w)} which is asin(in)
             */
            const realT x = std::imag(in);
            const realT y = std::real(in);

            if (std::isnan(x)) {
                /* asinh(NaN + I*+-Inf) = opt(+-)Inf + I*NaN */
                if (std::isinf(y)) {
                    const realT asinh_re = y;
                    const realT asinh_im = q_nan;
                    return resT{asinh_im, asinh_re};
                }
                /* asinh(NaN + I*0) = NaN + I*0 */
                if (y == realT(0)) {
                    const realT asinh_re = q_nan;
                    const realT asinh_im = y;
                    return resT{asinh_im, asinh_re};
                }
                /* All other cases involving NaN return NaN + I*NaN. */
                return resT{q_nan, q_nan};
            }
            else if (std::isnan(y)) {
                /* asinh(+-Inf + I*NaN) = +-Inf + I*NaN */
                if (std::isinf(x)) {
                    const realT asinh_re = x;
                    const realT asinh_im = q_nan;
                    return resT{asinh_im, asinh_re};
                }
                /* All other cases involving NaN return NaN + I*NaN. */
                return resT{q_nan, q_nan};
            }

            /*
             * For large x or y including asinh(+-Inf + I*+-Inf)
             * asinh(in) = sign(x)*log(sign(x)*in) + O(1/in^2)   as in ->
             * infinity The above formula works for the imaginary part as well,
             * because Im(asinh(in)) = sign(x)*atan2(sign(x)*y, fabs(x)) +
             * O(y/in^3) as in -> infinity, uniformly in y
             */
            constexpr realT r_eps =
                realT(1) / std::numeric_limits<realT>::epsilon();
            if (sycl::fabs(x) > r_eps || sycl::fabs(y) > r_eps) {
#ifdef USE_SYCL_FOR_COMPLEX_TYPES
                using sycl_complexT = exprm_ns::complex<realT>;
                const sycl_complexT z{x, y};
                realT wx, wy;
                if (!std::signbit(x)) {
                    const auto log_z = exprm_ns::log(z);
                    wx = log_z.real() + std::log(realT(2));
                    wy = log_z.imag();
                }
                else {
                    const auto log_mz = exprm_ns::log(-z);
                    wx = log_mz.real() + std::log(realT(2));
                    wy = log_mz.imag();
                }
#else
                const resT z{x, y};
                realT wx, wy;
                if (!std::signbit(x)) {
                    const auto log_z = std::log(z);
                    wx = std::real(log_z) + std::log(realT(2));
                    wy = std::imag(log_z);
                }
                else {
                    const auto log_mz = std::log(-z);
                    wx = std::real(log_mz) + std::log(realT(2));
                    wy = std::imag(log_mz);
                }
#endif
                const realT asinh_re = sycl::copysign(wx, x);
                const realT asinh_im = sycl::copysign(wy, y);
                return resT{asinh_im, asinh_re};
            }
            /* ordinary cases */
#if USE_SYCL_FOR_COMPLEX_TYPES
            return exprm_ns::asin(
                exprm_ns::complex<realT>(in)); // std::asin(in);
#else
            return std::asin(in);
#endif
        }
        else {
            static_assert(std::is_floating_point_v<argT> ||
                          std::is_same_v<argT, sycl::half>);
            return std::asin(in);
        }
    }
};

template <typename argTy,
          typename resTy = argTy,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using AsinContigFunctor =
    elementwise_common::UnaryContigFunctor<argTy,
                                           resTy,
                                           AsinFunctor<argTy, resTy>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

template <typename argTy, typename resTy, typename IndexerT>
using AsinStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, AsinFunctor<argTy, resTy>>;

template <typename T> struct AsinOutputType
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
class asin_contig_kernel;

template <typename argTy>
sycl::event asin_contig_impl(sycl::queue &exec_q,
                             size_t nelems,
                             const char *arg_p,
                             char *res_p,
                             const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::unary_contig_impl<
        argTy, AsinOutputType, AsinContigFunctor, asin_contig_kernel>(
        exec_q, nelems, arg_p, res_p, depends);
}

template <typename fnT, typename T> struct AsinContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename AsinOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = asin_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T> struct AsinTypeMapFactory
{
    /*! @brief get typeid for output type of std::asin(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename AsinOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3> class asin_strided_kernel;

template <typename argTy>
sycl::event
asin_strided_impl(sycl::queue &exec_q,
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
        argTy, AsinOutputType, AsinStridedFunctor, asin_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T> struct AsinStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename AsinOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = asin_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace asin
} // namespace kernels
} // namespace tensor
} // namespace dpctl
