//=== vec_size_utils.hpp -                            -------/ /*-C++-*--/===//
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
//===---------------------------------------------------------------------===//
///
/// \file
/// This file defines utilities for selection of hyperparameters for kernels
/// implementing unary and binary elementwise functions for contiguous inputs
//===---------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <type_traits>

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace vec_size_utils
{

template <typename Ty1,
          typename ArgTy1,
          typename Ty2,
          typename ArgTy2,
          std::uint8_t vec_sz_v,
          std::uint8_t n_vecs_v>
struct BinaryContigHyperparameterSetEntry
    : std::conjunction<std::is_same<Ty1, ArgTy1>, std::is_same<Ty2, ArgTy2>>
{
    static constexpr std::uint8_t vec_sz = vec_sz_v;
    static constexpr std::uint8_t n_vecs = n_vecs_v;
};

template <typename Ty,
          typename ArgTy,
          std::uint8_t vec_sz_v,
          std::uint8_t n_vecs_v>
struct UnaryContigHyperparameterSetEntry : std::is_same<Ty, ArgTy>
{
    static constexpr std::uint8_t vec_sz = vec_sz_v;
    static constexpr std::uint8_t n_vecs = n_vecs_v;
};

template <std::uint8_t vec_sz_v, std::uint8_t n_vecs_v>
struct ContigHyperparameterSetDefault : std::true_type
{
    static constexpr std::uint8_t vec_sz = vec_sz_v;
    static constexpr std::uint8_t n_vecs = n_vecs_v;
};

} // end of namespace vec_size_utils
} // end of namespace kernels
} // end of namespace tensor
} // end of namespace dpctl
