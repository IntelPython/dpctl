//=== sqrt.hpp -   Unary function SQRT                   ------  *-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of SQRT(x)
/// function that compute a square root.
//===---------------------------------------------------------------------===//

#pragma once
#include <CL/sycl.hpp>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
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
namespace sqrt
{

namespace py = pybind11;
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

    resT operator()(const argT &in)
    {
        if constexpr (is_complex<argT>::value) {
            // #ifdef _WINDOWS
            //             return csqrt(in);
            // #else
            //             return std::sqrt(in);
            // #endif
            return csqrt(in);
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
                return {(std::isfinite(y) ? zero : y), std::copysign(p_inf, y)};
            }
            else {
                return {p_inf, (std::isfinite(y) ? std::copysign(zero, y) : y)};
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

    template <typename T>
    std::complex<T> csqrt_finite(T const &x, T const &y) const
    {
        // csqrt(x + y*1j) =
        //     sqrt((cabs(x, y) + x) / 2) +
        //     1j * copysign(sqrt((cabs(x, y) - x) / 2), y)

        using realT = T;
        constexpr realT half = realT(0x1.0p-1f); // 1/2
        constexpr realT zero = realT(0);

        if (std::signbit(x)) {
            realT m = std::hypot(x, y);
            realT d = std::sqrt((m - x) * half);
            return {(d == zero ? zero : std::abs(y) / d * half),
                    std::copysign(d, y)};
        }
        else {
            realT m = std::hypot(x, y);
            realT d = std::sqrt((m + x) * half);
            return {d, (d == zero) ? std::copysign(zero, y) : y * half / d};
        }
    }
};

template <typename argTy,
          typename resTy = argTy,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2>
using SqrtContigFunctor = elementwise_common::
    UnaryContigFunctor<argTy, resTy, SqrtFunctor<argTy, resTy>, vec_sz, n_vecs>;

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
sycl::event sqrt_contig_impl(sycl::queue exec_q,
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
sqrt_strided_impl(sycl::queue exec_q,
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
