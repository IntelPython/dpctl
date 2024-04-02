//=== sin.hpp -   Unary function SIN                     ------  *-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of SIN(x) function.
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
namespace sin
{

namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;

template <typename argT, typename resT> struct SinFunctor
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

            realT const &in_re = std::real(in);
            realT const &in_im = std::imag(in);

            const bool in_re_finite = std::isfinite(in_re);
            const bool in_im_finite = std::isfinite(in_im);
            /*
             * Handle the nearly-non-exceptional cases where
             * real and imaginary parts of input are finite.
             */
            if (in_re_finite && in_im_finite) {
#ifdef USE_SYCL_FOR_COMPLEX_TYPES
                resT res = exprm_ns::sin(
                    exprm_ns::complex<realT>(in)); // std::sin(in);
#else
                resT res = std::sin(in);
#endif
                if (in_re == realT(0)) {
                    res.real(sycl::copysign(realT(0), in_re));
                }
                return res;
            }

            /*
             * since sin(in) = -I * sinh(I * in), for special cases,
             * we calculate real and imaginary parts of z = sinh(I * in) and
             * then return { imag(z) , -real(z) } which is sin(in).
             */
            const realT x = -in_im;
            const realT y = in_re;
            const bool xfinite = in_im_finite;
            const bool yfinite = in_re_finite;
            /*
             * sinh(+-0 +- I Inf) = sign(d(+-0, dNaN))0 + I dNaN.
             * The sign of 0 in the result is unspecified.  Choice = normally
             * the same as dNaN.
             *
             * sinh(+-0 +- I NaN) = sign(d(+-0, NaN))0 + I d(NaN).
             * The sign of 0 in the result is unspecified.  Choice = normally
             * the same as d(NaN).
             */
            if (x == realT(0) && !yfinite) {
                const realT sinh_im = q_nan;
                const realT sinh_re = sycl::copysign(realT(0), x * sinh_im);
                return resT{sinh_im, -sinh_re};
            }

            /*
             * sinh(+-Inf +- I 0) = +-Inf + I +-0.
             *
             * sinh(NaN +- I 0)   = d(NaN) + I +-0.
             */
            if (y == realT(0) && !xfinite) {
                if (std::isnan(x)) {
                    const realT sinh_re = x;
                    const realT sinh_im = y;
                    return resT{sinh_im, -sinh_re};
                }
                const realT sinh_re = x;
                const realT sinh_im = sycl::copysign(realT(0), y);
                return resT{sinh_im, -sinh_re};
            }

            /*
             * sinh(x +- I Inf) = dNaN + I dNaN.
             *
             * sinh(x + I NaN) = d(NaN) + I d(NaN).
             */
            if (xfinite && !yfinite) {
                const realT sinh_re = q_nan;
                const realT sinh_im = x * sinh_re;
                return resT{sinh_im, -sinh_re};
            }

            /*
             * sinh(+-Inf + I NaN)  = +-Inf + I d(NaN).
             * The sign of Inf in the result is unspecified.  Choice = normally
             * the same as d(NaN).
             *
             * sinh(+-Inf +- I Inf) = +Inf + I dNaN.
             * The sign of Inf in the result is unspecified.
             * Choice = always - here for sinh to have positive result for
             * imaginary part of sin.
             *
             * sinh(+-Inf + I y)   = +-Inf cos(y) + I Inf sin(y)
             */
            if (std::isinf(x)) {
                if (!yfinite) {
                    const realT sinh_re = -x * x;
                    const realT sinh_im = x * (y - y);
                    return resT{sinh_im, -sinh_re};
                }
                const realT sinh_re = x * std::cos(y);
                const realT sinh_im =
                    std::numeric_limits<realT>::infinity() * std::sin(y);
                return resT{sinh_im, -sinh_re};
            }

            /*
             * sinh(NaN + I NaN)  = d(NaN) + I d(NaN).
             *
             * sinh(NaN +- I Inf) = d(NaN) + I d(NaN).
             *
             * sinh(NaN + I y)    = d(NaN) + I d(NaN).
             */
            const realT y_m_y = (y - y);
            const realT sinh_re = (x * x) * y_m_y;
            const realT sinh_im = (x + x) * y_m_y;
            return resT{sinh_im, -sinh_re};
        }
        else {
            static_assert(std::is_same_v<argT, resT>);
            if (in == 0) {
                return in;
            }
            return std::sin(in);
        }
    }
};

template <typename argTy,
          typename resTy = argTy,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using SinContigFunctor =
    elementwise_common::UnaryContigFunctor<argTy,
                                           resTy,
                                           SinFunctor<argTy, resTy>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

template <typename argTy, typename resTy, typename IndexerT>
using SinStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, SinFunctor<argTy, resTy>>;

template <typename T> struct SinOutputType
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
class sin_contig_kernel;

template <typename argTy>
sycl::event sin_contig_impl(sycl::queue &exec_q,
                            size_t nelems,
                            const char *arg_p,
                            char *res_p,
                            const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::unary_contig_impl<
        argTy, SinOutputType, SinContigFunctor, sin_contig_kernel>(
        exec_q, nelems, arg_p, res_p, depends);
}

template <typename fnT, typename T> struct SinContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename SinOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = sin_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T> struct SinTypeMapFactory
{
    /*! @brief get typeid for output type of std::sin(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename SinOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3> class sin_strided_kernel;

template <typename argTy>
sycl::event sin_strided_impl(sycl::queue &exec_q,
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
        argTy, SinOutputType, SinStridedFunctor, sin_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T> struct SinStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename SinOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = sin_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace sin
} // namespace kernels
} // namespace tensor
} // namespace dpctl
