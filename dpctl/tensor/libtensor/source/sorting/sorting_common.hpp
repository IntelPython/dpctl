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

#include "utils/math_utils.hpp"

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

template <typename cT> struct ComplexLess
{
    bool operator()(const cT &v1, const cT &v2) const
    {
        using dpctl::tensor::math_utils::less_complex;

        return less_complex(v1, v2);
    }
};

template <typename cT> struct ComplexGreater
{
    bool operator()(const cT &v1, const cT &v2) const
    {
        using dpctl::tensor::math_utils::greater_complex;

        return greater_complex(v1, v2);
    }
};

template <typename argTy> struct AscendingSorter
{
    using type = std::less<argTy>;
};

template <typename T> struct AscendingSorter<std::complex<T>>
{
    using type = ComplexLess<std::complex<T>>;
};

template <typename argTy> struct DescendingSorter
{
    using type = std::greater<argTy>;
};

template <typename T> struct DescendingSorter<std::complex<T>>
{
    using type = ComplexGreater<std::complex<T>>;
};

} // end of namespace py_internal
} // end of namespace tensor
} // end of namespace dpctl
