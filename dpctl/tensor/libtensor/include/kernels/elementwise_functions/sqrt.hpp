//=== sqrt.hpp -   Unary function SQRT                   ------  *-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of SQRT(x)
/// function that compute a square root.
//===---------------------------------------------------------------------===//

#pragma once
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
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
namespace sqrt
{

namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;

template <typename argT, typename resT> struct SqrtFunctor
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
#ifdef USE_SYCL_FOR_COMPLEX_TYPES
            return exprm_ns::sqrt(exprm_ns::complex<realT>(in));
#else
#ifdef _WINDOWS
            // Work around a problem on Windows, where std::sqrt for
            // single precision input uses double type, precluding
            // offloading to devices that do not support double precision
            // i.e. Iris Xe
            if constexpr (std::is_same_v<realT, double>) {
                return std::sqrt(in);
            }
            else {
                return csqrt(in);
            }
#else
            return std::sqrt(in);
#endif
#endif
        }
        else {
            return std::sqrt(in);
        }
    }

private:
    template <typename T> std::complex<T> csqrt(std::complex<T> const &z) const
    {
        // csqrt(x + y*1j)
        //  * csqrt(x - y * 1j) = conj(csqrt(x + y * 1j))
        //  * If x is either +0 or -0 and y is +0, the result is +0 + 0j.
        //  * If x is any value (including NaN) and y is +infinity, the result
        //  is +infinity + infinity j.
        //  * If x is a finite number and y is NaN, the result is NaN + NaN j.

        //  * If x -infinity and y is a positive (i.e., greater than 0) finite
        //  number, the result is NaN + NaN j.
        //  * If x is +infinity and y is a positive (i.e., greater than 0)
        //  finite number, the result is +0 + infinity j.
        //  * If x is -infinity and y is NaN, the result is NaN + infinity j
        //  (sign of the imaginary component is unspecified).
        //  * If x is +infinity and y is NaN, the result is +infinity + NaN j.
        //  * If x is NaN and y is any value, the result is NaN + NaN j.

        using realT = T;
        constexpr realT q_nan = std::numeric_limits<realT>::quiet_NaN();
        constexpr realT p_inf = std::numeric_limits<realT>::infinity();
        constexpr realT zero = realT(0);

        realT x = std::real(z);
        realT y = std::imag(z);

        if (std::isinf(y)) {
            return {p_inf, y};
        }
        else if (std::isnan(x)) {
            return {x, q_nan};
        }
        else if (std::isinf(x)) { // x is an infinity
            // y is either finite, or nan
            if (std::signbit(x)) { // x == -inf
                return {(std::isfinite(y) ? zero : y),
                        sycl::copysign(p_inf, y)};
            }
            else {
                return {p_inf,
                        (std::isfinite(y) ? sycl::copysign(zero, y) : y)};
            }
        }
        else { // x is finite
            if (std::isfinite(y)) {
#ifdef USE_STD_SQRT_FOR_COMPLEX_TYPES
                return std::sqrt(z);
#else
                return csqrt_finite(x, y);
#endif
            }
            else {
                return {q_nan, y};
            }
        }
    }

    int get_normal_scale_float(const float &v) const
    {
        constexpr int float_significant_bits = 23;
        constexpr std::uint32_t exponent_mask = 0xff;
        constexpr int exponent_bias = 127;
        const int scale = static_cast<int>(
            (sycl::bit_cast<std::uint32_t>(v) >> float_significant_bits) &
            exponent_mask);
        return scale - exponent_bias;
    }

    int get_normal_scale_double(const double &v) const
    {
        constexpr int double_significant_bits = 52;
        constexpr std::uint64_t exponent_mask = 0x7ff;
        constexpr int exponent_bias = 1023;
        const int scale = static_cast<int>(
            (sycl::bit_cast<std::uint64_t>(v) >> double_significant_bits) &
            exponent_mask);
        return scale - exponent_bias;
    }

    template <typename T> int get_normal_scale(const T &v) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

        if constexpr (std::is_same_v<T, float>) {
            return get_normal_scale_float(v);
        }
        else {
            return get_normal_scale_double(v);
        }
    }

    template <typename T>
    std::complex<T> csqrt_finite_scaled(T const &x, T const &y) const
    {
        // csqrt(x + y*1j) =
        //     sqrt((cabs(x, y) + x) / 2) +
        //     1j * copysign(sqrt((cabs(x, y) - x) / 2), y)

        using realT = T;
        constexpr realT half = realT(0x1.0p-1f); // 1/2
        constexpr realT zero = realT(0);

        const int exp_x = get_normal_scale<realT>(x);
        const int exp_y = get_normal_scale<realT>(y);

        int sc = std::max<int>(exp_x, exp_y) / 2;
        const realT xx = sycl::ldexp(x, -sc * 2);
        const realT yy = sycl::ldexp(y, -sc * 2);

        if (std::signbit(xx)) {
            const realT m = std::hypot(xx, yy);
            const realT d = std::sqrt((m - xx) * half);
            const realT res_re = (d == zero ? zero : sycl::fabs(yy) / d * half);
            const realT res_im = sycl::copysign(d, yy);
            return {sycl::ldexp(res_re, sc), sycl::ldexp(res_im, sc)};
        }
        else {
            const realT m = std::hypot(xx, yy);
            const realT d = std::sqrt((m + xx) * half);
            const realT res_im =
                (d == zero) ? sycl::copysign(zero, yy) : yy * half / d;
            return {sycl::ldexp(d, sc), sycl::ldexp(res_im, sc)};
        }
    }

    template <typename T>
    std::complex<T> csqrt_finite_unscaled(T const &x, T const &y) const
    {
        // csqrt(x + y*1j) =
        //     sqrt((cabs(x, y) + x) / 2) +
        //     1j * copysign(sqrt((cabs(x, y) - x) / 2), y)

        using realT = T;
        constexpr realT half = realT(0x1.0p-1f); // 1/2
        constexpr realT zero = realT(0);

        if (std::signbit(x)) {
            const realT m = std::hypot(x, y);
            const realT d = std::sqrt((m - x) * half);
            const realT res_re = (d == zero ? zero : sycl::fabs(y) / d * half);
            const realT res_im = sycl::copysign(d, y);
            return {res_re, res_im};
        }
        else {
            const realT m = std::hypot(x, y);
            const realT d = std::sqrt((m + x) * half);
            const realT res_im =
                (d == zero) ? sycl::copysign(zero, y) : y * half / d;
            return {d, res_im};
        }
    }

    template <typename T> T scaling_threshold() const
    {
        if constexpr (std::is_same_v<T, float>) {
            return T(0x1.0p+126f);
        }
        else {
            return T(0x1.0p+1022);
        }
    }

    template <typename T>
    std::complex<T> csqrt_finite(T const &x, T const &y) const
    {
        return (std::max<T>(std::fabs(x), std::fabs(y)) <
                scaling_threshold<T>())
                   ? csqrt_finite_unscaled(x, y)
                   : csqrt_finite_scaled(x, y);
    }
};

template <typename argTy,
          typename resTy = argTy,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using SqrtContigFunctor =
    elementwise_common::UnaryContigFunctor<argTy,
                                           resTy,
                                           SqrtFunctor<argTy, resTy>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

template <typename argTy, typename resTy, typename IndexerT>
using SqrtStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, SqrtFunctor<argTy, resTy>>;

template <typename T> struct SqrtOutputType
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
class sqrt_contig_kernel;

template <typename argTy>
sycl::event sqrt_contig_impl(sycl::queue &exec_q,
                             size_t nelems,
                             const char *arg_p,
                             char *res_p,
                             const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::unary_contig_impl<
        argTy, SqrtOutputType, SqrtContigFunctor, sqrt_contig_kernel>(
        exec_q, nelems, arg_p, res_p, depends);
}

template <typename fnT, typename T> struct SqrtContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename SqrtOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = sqrt_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T> struct SqrtTypeMapFactory
{
    /*! @brief get typeid for output type of std::sqrt(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename SqrtOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3> class sqrt_strided_kernel;

template <typename argTy>
sycl::event
sqrt_strided_impl(sycl::queue &exec_q,
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
        argTy, SqrtOutputType, SqrtStridedFunctor, sqrt_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T> struct SqrtStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename SqrtOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = sqrt_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace sqrt
} // namespace kernels
} // namespace tensor
} // namespace dpctl
