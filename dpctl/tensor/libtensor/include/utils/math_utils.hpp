//===------- math_utils.hpp - Implementation of math utils  -------*-C++-*/===//
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
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines math utility functions.
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl.hpp>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>

#include <utils/type_utils.hpp>

namespace dpctl
{
namespace tensor
{
namespace math_utils
{

namespace detail
{
namespace dt_tu = dpctl::tensor::type_utils;

template <typename T> constexpr bool is_complex_v = dt_tu::is_complex<T>::value;

template <typename T,
          typename bitseqT,
          typename scaleT,
          bitseqT significand_bits,
          bitseqT exponent_mask>
bool is_finite(const T &v)
{
    const scaleT scale = static_cast<scaleT>(
        (sycl::bit_cast<bitseqT>(v) >> significand_bits) & exponent_mask);

    return static_cast<bool>(scale ^ static_cast<scaleT>(exponent_mask));
}

template <typename T,
          typename bitseqT,
          typename scaleT,
          bitseqT significand_bits,
          bitseqT exponent_mask>
bool is_inf(const T &v)
{
    constexpr bitseqT significand_mask = ((bitseqT(1) << significand_bits) - 1);

    const bitseqT bits = sycl::bit_cast<bitseqT>(v);
    const scaleT scale =
        static_cast<scaleT>((bits >> significand_bits) & exponent_mask);

    const bitseqT significand = bits & significand_mask;

    return (!static_cast<bool>(scale ^ static_cast<scaleT>(exponent_mask)) &&
            !static_cast<bool>(significand));
}

template <typename T,
          typename bitseqT,
          typename scaleT,
          bitseqT significand_bits,
          bitseqT exponent_mask>
bool is_nan(const T &v)
{
    constexpr bitseqT significand_mask = ((bitseqT(1) << significand_bits) - 1);

    const bitseqT bits = sycl::bit_cast<bitseqT>(v);
    const scaleT scale =
        static_cast<scaleT>((bits >> significand_bits) & exponent_mask);

    const bitseqT significand = bits & significand_mask;

    return (!static_cast<bool>(scale ^ static_cast<scaleT>(exponent_mask)) &&
            static_cast<bool>(significand));
}

} // namespace detail

// Work-arounds for bug in bug in SYCLOS
template <typename T> bool isfinite(const T &x)
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                  std::is_same_v<T, sycl::half>);
    if constexpr (std::is_same_v<T, float>) {
        return detail::is_finite<T, std::uint32_t, std::uint32_t, 23, 0xff>(x);
    }
    else if constexpr (std::is_same_v<T, double>) {
        return detail::is_finite<T, std::uint64_t, std::uint32_t, 52, 0x7ff>(x);
    }
    else if constexpr (std::is_same_v<T, sycl::half>) {
        return detail::is_finite<T, std::uint16_t, std::uint16_t, 10, 0x1f>(x);
    }
}

template <typename T> bool isinf(const T &x)
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                  std::is_same_v<T, sycl::half>);
    if constexpr (std::is_same_v<T, float>) {
        return detail::is_inf<T, std::uint32_t, std::uint32_t, 23, 0xff>(x);
    }
    else if constexpr (std::is_same_v<T, double>) {
        return detail::is_inf<T, std::uint64_t, std::uint32_t, 52, 0x7ff>(x);
    }
    else if constexpr (std::is_same_v<T, sycl::half>) {
        return detail::is_inf<T, std::uint16_t, std::uint16_t, 10, 0x1f>(x);
    }
}

template <typename T> bool isnan(const T &x)
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                  std::is_same_v<T, sycl::half>);
    if constexpr (std::is_same_v<T, float>) {
        return detail::is_nan<T, std::uint32_t, std::uint32_t, 23, 0xff>(x);
    }
    else if constexpr (std::is_same_v<T, double>) {
        return detail::is_nan<T, std::uint64_t, std::uint32_t, 52, 0x7ff>(x);
    }
    else if constexpr (std::is_same_v<T, sycl::half>) {
        return detail::is_nan<T, std::uint16_t, std::uint16_t, 10, 0x1f>(x);
    }
}

template <typename T> bool less_complex(const T &x1, const T &x2)
{
    static_assert(detail::is_complex_v<T>);

    using realT = typename T::value_type;
    realT real1 = std::real(x1);
    realT real2 = std::real(x2);
    realT imag1 = std::imag(x1);
    realT imag2 = std::imag(x2);

    return (real1 == real2) ? (imag1 < imag2)
                            : (real1 < real2 && !isnan(imag1) && !isnan(imag2));
}

template <typename T> bool greater_complex(const T &x1, const T &x2)
{
    static_assert(detail::is_complex_v<T>);

    using realT = typename T::value_type;
    realT real1 = std::real(x1);
    realT real2 = std::real(x2);
    realT imag1 = std::imag(x1);
    realT imag2 = std::imag(x2);

    return (real1 == real2) ? (imag1 > imag2)
                            : (real1 > real2 && !isnan(imag1) && !isnan(imag2));
}

template <typename T> bool less_equal_complex(const T &x1, const T &x2)
{
    static_assert(detail::is_complex_v<T>);

    using realT = typename T::value_type;
    realT real1 = std::real(x1);
    realT real2 = std::real(x2);
    realT imag1 = std::imag(x1);
    realT imag2 = std::imag(x2);

    return (real1 == real2) ? (imag1 <= imag2)
                            : (real1 < real2 && !isnan(imag1) && !isnan(imag2));
}

template <typename T> bool greater_equal_complex(const T &x1, const T &x2)
{
    static_assert(detail::is_complex_v<T>);

    using realT = typename T::value_type;
    realT real1 = std::real(x1);
    realT real2 = std::real(x2);
    realT imag1 = std::imag(x1);
    realT imag2 = std::imag(x2);

    return (real1 == real2) ? (imag1 >= imag2)
                            : (real1 > real2 && !isnan(imag1) && !isnan(imag2));
}

template <typename T> T max_complex(const T &x1, const T &x2)
{
    static_assert(detail::is_complex_v<T>);

    using realT = typename T::value_type;
    realT real1 = std::real(x1);
    realT real2 = std::real(x2);
    realT imag1 = std::imag(x1);
    realT imag2 = std::imag(x2);

    bool isnan_imag1 = isnan(imag1);
    bool gt = (real1 == real2)
                  ? (imag1 > imag2)
                  : (real1 > real2 && !isnan_imag1 && !isnan(imag2));
    return (isnan(real1) || isnan_imag1 || gt) ? x1 : x2;
}

template <typename T> T min_complex(const T &x1, const T &x2)
{
    static_assert(detail::is_complex_v<T>);

    using realT = typename T::value_type;
    realT real1 = std::real(x1);
    realT real2 = std::real(x2);
    realT imag1 = std::imag(x1);
    realT imag2 = std::imag(x2);

    bool isnan_imag1 = isnan(imag1);
    bool lt = (real1 == real2)
                  ? (imag1 < imag2)
                  : (real1 < real2 && !isnan_imag1 && !isnan(imag2));
    return (isnan(real1) || isnan_imag1 || lt) ? x1 : x2;
}

} // namespace math_utils
} // namespace tensor
} // namespace dpctl
