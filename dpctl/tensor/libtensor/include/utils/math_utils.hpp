//===------- math_utils.hpp - Implementation of math utils  -------*-C++-*/===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2025 Intel Corporation
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
/// This file defines math utility functions.
//===----------------------------------------------------------------------===//

#pragma once
#include <cmath>
#include <sycl/sycl.hpp>

#ifndef SYCL_EXT_ONEAPI_COMPLEX
#define SYCL_EXT_ONEAPI_COMPLEX 1
#endif
#if __has_include(<sycl/ext/oneapi/experimental/sycl_complex.hpp>)
#include <sycl/ext/oneapi/experimental/sycl_complex.hpp>
#else
#include <sycl/ext/oneapi/experimental/complex/complex.hpp>
#endif

namespace dpctl
{
namespace tensor
{
namespace math_utils
{

namespace exprm_ns = sycl::ext::oneapi::experimental;

template <typename T> bool less_complex(const T &x1, const T &x2)
{
    using realT = typename T::value_type;
    using sycl_complexT = exprm_ns::complex<realT>;
    sycl_complexT z1 = sycl_complexT(x1);
    sycl_complexT z2 = sycl_complexT(x2);
    realT real1 = exprm_ns::real(z1);
    realT real2 = exprm_ns::real(z2);
    realT imag1 = exprm_ns::imag(z1);
    realT imag2 = exprm_ns::imag(z2);

    return (real1 == real2)
               ? (imag1 < imag2)
               : (real1 < real2 && !std::isnan(imag1) && !std::isnan(imag2));
}

template <typename T> bool greater_complex(const T &x1, const T &x2)
{
    using realT = typename T::value_type;
    using sycl_complexT = exprm_ns::complex<realT>;
    sycl_complexT z1 = sycl_complexT(x1);
    sycl_complexT z2 = sycl_complexT(x2);
    realT real1 = exprm_ns::real(z1);
    realT real2 = exprm_ns::real(z2);
    realT imag1 = exprm_ns::imag(z1);
    realT imag2 = exprm_ns::imag(z2);

    return (real1 == real2)
               ? (imag1 > imag2)
               : (real1 > real2 && !std::isnan(imag1) && !std::isnan(imag2));
}

template <typename T> bool less_equal_complex(const T &x1, const T &x2)
{
    using realT = typename T::value_type;
    using sycl_complexT = exprm_ns::complex<realT>;
    sycl_complexT z1 = sycl_complexT(x1);
    sycl_complexT z2 = sycl_complexT(x2);
    realT real1 = exprm_ns::real(z1);
    realT real2 = exprm_ns::real(z2);
    realT imag1 = exprm_ns::imag(z1);
    realT imag2 = exprm_ns::imag(z2);

    return (real1 == real2)
               ? (imag1 <= imag2)
               : (real1 < real2 && !std::isnan(imag1) && !std::isnan(imag2));
}

template <typename T> bool greater_equal_complex(const T &x1, const T &x2)
{
    using realT = typename T::value_type;
    using sycl_complexT = exprm_ns::complex<realT>;
    sycl_complexT z1 = sycl_complexT(x1);
    sycl_complexT z2 = sycl_complexT(x2);
    realT real1 = exprm_ns::real(z1);
    realT real2 = exprm_ns::real(z2);
    realT imag1 = exprm_ns::imag(z1);
    realT imag2 = exprm_ns::imag(z2);

    return (real1 == real2)
               ? (imag1 >= imag2)
               : (real1 > real2 && !std::isnan(imag1) && !std::isnan(imag2));
}

template <typename T> T max_complex(const T &x1, const T &x2)
{
    using realT = typename T::value_type;
    using sycl_complexT = exprm_ns::complex<realT>;
    sycl_complexT z1 = sycl_complexT(x1);
    sycl_complexT z2 = sycl_complexT(x2);
    realT real1 = exprm_ns::real(z1);
    realT real2 = exprm_ns::real(z2);
    realT imag1 = exprm_ns::imag(z1);
    realT imag2 = exprm_ns::imag(z2);

    bool isnan_imag1 = std::isnan(imag1);
    bool gt = (real1 == real2)
                  ? (imag1 > imag2)
                  : (real1 > real2 && !isnan_imag1 && !std::isnan(imag2));
    return (std::isnan(real1) || isnan_imag1 || gt) ? x1 : x2;
}

template <typename T> T min_complex(const T &x1, const T &x2)
{
    using realT = typename T::value_type;
    using sycl_complexT = exprm_ns::complex<realT>;
    sycl_complexT z1 = sycl_complexT(x1);
    sycl_complexT z2 = sycl_complexT(x2);
    realT real1 = exprm_ns::real(z1);
    realT real2 = exprm_ns::real(z2);
    realT imag1 = exprm_ns::imag(z1);
    realT imag2 = exprm_ns::imag(z2);

    bool isnan_imag1 = std::isnan(imag1);
    bool lt = (real1 == real2)
                  ? (imag1 < imag2)
                  : (real1 < real2 && !isnan_imag1 && !std::isnan(imag2));
    return (std::isnan(real1) || isnan_imag1 || lt) ? x1 : x2;
}

template <typename T> T logaddexp(T x, T y)
{
    if (x == y) { // handle signed infinities
        const T log2 = sycl::log(T(2));
        return x + log2;
    }
    else {
        const T tmp = x - y;
        constexpr T zero(0);

        return (tmp > zero)
                   ? (x + sycl::log1p(sycl::exp(-tmp)))
                   : ((tmp <= zero) ? y + sycl::log1p(sycl::exp(tmp))
                                    : std::numeric_limits<T>::quiet_NaN());
    }
}

template <typename T> T plus_complex(const T &x1, const T &x2)
{
    using realT = typename T::value_type;
    using sycl_complexT = exprm_ns::complex<realT>;
    return T(sycl_complexT(x1) + sycl_complexT(x2));
}

template <typename T> T multiplies_complex(const T &x1, const T &x2)
{
    using realT = typename T::value_type;
    using sycl_complexT = exprm_ns::complex<realT>;
    return T(sycl_complexT(x1) * sycl_complexT(x2));
}

} // namespace math_utils
} // namespace tensor
} // namespace dpctl
