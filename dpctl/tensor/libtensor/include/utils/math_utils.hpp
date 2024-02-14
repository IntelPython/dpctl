//===------- math_utils.hpp - Implementation of math utils  -------*-C++-*/===//
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
/// This file defines math utility functions.
//===----------------------------------------------------------------------===//

#pragma once
#include <cmath>
#include <complex>

namespace dpctl
{
namespace tensor
{
namespace math_utils
{

template <typename T> bool less_complex(const T &x1, const T &x2)
{
    using realT = typename T::value_type;
    realT real1 = std::real(x1);
    realT real2 = std::real(x2);
    realT imag1 = std::imag(x1);
    realT imag2 = std::imag(x2);

    return (real1 == real2)
               ? (imag1 < imag2)
               : (real1 < real2 && !std::isnan(imag1) && !std::isnan(imag2));
}

template <typename T> bool greater_complex(const T &x1, const T &x2)
{
    using realT = typename T::value_type;
    realT real1 = std::real(x1);
    realT real2 = std::real(x2);
    realT imag1 = std::imag(x1);
    realT imag2 = std::imag(x2);

    return (real1 == real2)
               ? (imag1 > imag2)
               : (real1 > real2 && !std::isnan(imag1) && !std::isnan(imag2));
}

template <typename T> bool less_equal_complex(const T &x1, const T &x2)
{
    using realT = typename T::value_type;
    realT real1 = std::real(x1);
    realT real2 = std::real(x2);
    realT imag1 = std::imag(x1);
    realT imag2 = std::imag(x2);

    return (real1 == real2)
               ? (imag1 <= imag2)
               : (real1 < real2 && !std::isnan(imag1) && !std::isnan(imag2));
}

template <typename T> bool greater_equal_complex(const T &x1, const T &x2)
{
    using realT = typename T::value_type;
    realT real1 = std::real(x1);
    realT real2 = std::real(x2);
    realT imag1 = std::imag(x1);
    realT imag2 = std::imag(x2);

    return (real1 == real2)
               ? (imag1 >= imag2)
               : (real1 > real2 && !std::isnan(imag1) && !std::isnan(imag2));
}

template <typename T> T max_complex(const T &x1, const T &x2)
{
    using realT = typename T::value_type;
    realT real1 = std::real(x1);
    realT real2 = std::real(x2);
    realT imag1 = std::imag(x1);
    realT imag2 = std::imag(x2);

    bool isnan_imag1 = std::isnan(imag1);
    bool gt = (real1 == real2)
                  ? (imag1 > imag2)
                  : (real1 > real2 && !isnan_imag1 && !std::isnan(imag2));
    return (std::isnan(real1) || isnan_imag1 || gt) ? x1 : x2;
}

template <typename T> T min_complex(const T &x1, const T &x2)
{
    using realT = typename T::value_type;
    realT real1 = std::real(x1);
    realT real2 = std::real(x2);
    realT imag1 = std::imag(x1);
    realT imag2 = std::imag(x2);

    bool isnan_imag1 = std::isnan(imag1);
    bool lt = (real1 == real2)
                  ? (imag1 < imag2)
                  : (real1 < real2 && !isnan_imag1 && !std::isnan(imag2));
    return (std::isnan(real1) || isnan_imag1 || lt) ? x1 : x2;
}

template <typename T> T logaddexp(T x, T y)
{
    if (x == y) { // handle signed infinities
        const T log2 = std::log(T(2));
        return x + log2;
    }
    else {
        const T tmp = x - y;
        if (tmp > 0) {
            return x + std::log1p(std::exp(-tmp));
        }
        else if (tmp <= 0) {
            return y + std::log1p(std::exp(tmp));
        }
        else {
            return std::numeric_limits<T>::quiet_NaN();
        }
    }
}

} // namespace math_utils
} // namespace tensor
} // namespace dpctl
