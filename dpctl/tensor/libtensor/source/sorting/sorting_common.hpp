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
//===--------------------------------------------------------------------===//
///
/// \file
/// This file defines functions of dpctl.tensor._tensor_sorting_impl
/// extension.
//===--------------------------------------------------------------------===//

#pragma once

#include "sycl/sycl.hpp"
#include <type_traits>

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

namespace
{
template <typename fpT> struct ExtendedRealFPLess
{
    /* [R, nan] */
    bool operator()(const fpT v1, const fpT v2) const
    {
        return (!std::isnan(v1) && (std::isnan(v2) || (v1 < v2)));
    }
};

template <typename fpT> struct ExtendedRealFPGreater
{
    bool operator()(const fpT v1, const fpT v2) const
    {
        return (!std::isnan(v2) && (std::isnan(v1) || (v2 < v1)));
    }
};

template <typename cT> struct ExtendedComplexFPLess
{
    /* [(R, R), (R, nan), (nan, R), (nan, nan)] */

    bool operator()(const cT &v1, const cT &v2) const
    {
        using realT = typename cT::value_type;

        const realT real1 = std::real(v1);
        const realT real2 = std::real(v2);

        const bool r1_nan = std::isnan(real1);
        const bool r2_nan = std::isnan(real2);

        const realT imag1 = std::imag(v1);
        const realT imag2 = std::imag(v2);

        const bool i1_nan = std::isnan(imag1);
        const bool i2_nan = std::isnan(imag2);

        const int idx1 = ((r1_nan) ? 2 : 0) + ((i1_nan) ? 1 : 0);
        const int idx2 = ((r2_nan) ? 2 : 0) + ((i2_nan) ? 1 : 0);

        const bool res =
            !(r1_nan && i1_nan) &&
            ((idx1 < idx2) ||
             ((idx1 == idx2) &&
              ((r1_nan && !i1_nan && (imag1 < imag2)) ||
               (!r1_nan && i1_nan && (real1 < real2)) ||
               (!r1_nan && !i1_nan &&
                ((real1 < real2) || (!(real2 < real1) && (imag1 < imag2)))))));

        return res;
    }
};

template <typename cT> struct ExtendedComplexFPGreater
{
    bool operator()(const cT &v1, const cT &v2) const
    {
        auto less_ = ExtendedComplexFPLess<cT>{};
        return less_(v2, v1);
    }
};

template <typename T>
inline constexpr bool is_fp_v = (std::is_same_v<T, sycl::half> ||
                                 std::is_same_v<T, float> ||
                                 std::is_same_v<T, double>);

} // end of anonymous namespace

template <typename argTy> struct AscendingSorter
{
    using type = std::conditional_t<is_fp_v<argTy>,
                                    ExtendedRealFPLess<argTy>,
                                    std::less<argTy>>;
};

template <typename T> struct AscendingSorter<std::complex<T>>
{
    using type = ExtendedComplexFPLess<std::complex<T>>;
};

template <typename argTy> struct DescendingSorter
{
    using type = std::conditional_t<is_fp_v<argTy>,
                                    ExtendedRealFPGreater<argTy>,
                                    std::greater<argTy>>;
};

template <typename T> struct DescendingSorter<std::complex<T>>
{
    using type = ExtendedComplexFPGreater<std::complex<T>>;
};

} // end of namespace py_internal
} // end of namespace tensor
} // end of namespace dpctl
