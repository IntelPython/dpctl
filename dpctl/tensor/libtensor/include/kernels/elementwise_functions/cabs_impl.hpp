//===------- cabs_impl.hpp - Implementation of cabs  -------*-C++-*/===//
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
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines an implementation of the complex absolute value.
//===----------------------------------------------------------------------===//

#pragma once
#include <cmath>
#include <complex>
#include <limits>

#include "sycl_complex.hpp"

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace detail
{

template <typename realT> realT cabs(std::complex<realT> const &z)
{
    // Special values for cabs( x + y * 1j):
    //   * If x is either +infinity or -infinity and y is any value
    //   (including NaN), the result is +infinity.
    //   * If x is any value (including NaN) and y is either +infinity or
    //   -infinity, the result is +infinity.
    //   * If x is either +0 or -0, the result is equal to abs(y).
    //   * If y is either +0 or -0, the result is equal to abs(x).
    //   * If x is NaN and y is a finite number, the result is NaN.
    //   * If x is a finite number and y is NaN, the result is NaN.
    //   * If x is NaN and y is NaN, the result is NaN.

    const realT x = std::real(z);
    const realT y = std::imag(z);

    constexpr realT q_nan = std::numeric_limits<realT>::quiet_NaN();
    constexpr realT p_inf = std::numeric_limits<realT>::infinity();

    if (std::isinf(x)) {
        return p_inf;
    }
    else if (std::isinf(y)) {
        return p_inf;
    }
    else if (std::isnan(x)) {
        return q_nan;
    }
    else if (std::isnan(y)) {
        return q_nan;
    }
    else {
#ifdef USE_SYCL_FOR_COMPLEX_TYPES
        return exprm_ns::abs(exprm_ns::complex<realT>(z));
#else
        return std::hypot(std::real(z), std::imag(z));
#endif
    }
}

} // namespace detail
} // namespace kernels
} // namespace tensor
} // namespace dpctl
