//=== tanh.hpp -   Unary function TANH                     ------
//*-C++-*--/===//
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
/// This file defines utilities for dispatching elementwise dedicated kernels
//  for contiguous inputs.
//===---------------------------------------------------------------------===//

#pragma once

#include <cstdint>

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace vec_size_utils
{

template <typename T, typename... Rest> struct VecSize
{
    static constexpr unsigned int value =
        std::max<unsigned int>(VecSize<T>::value, VecSize<Rest...>::value);
};

template <typename T> struct VecSize<T>
{
    static_assert(sizeof(T) > 0, "Vacuous types are not supported");

    static constexpr unsigned int value =
        1 + ((sizeof(std::uint32_t) - 1) / (sizeof(T)));
};

template <typename T, typename... Rest>
static constexpr unsigned int VecSize_v = VecSize<T, Rest...>::value;

} // end of namespace vec_size_utils
} // end of namespace kernels
} // end of namespace tensor
} // end of namespace dpctl
