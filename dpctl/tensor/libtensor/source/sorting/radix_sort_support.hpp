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
//===--------------------------------------------------------------------===//
///
/// \file
/// This file defines functions of dpctl.tensor._tensor_sorting_impl
/// extension.
//===--------------------------------------------------------------------===//

#pragma once

#include <type_traits>

#include <sycl/sycl.hpp>

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

template <typename Ty, typename ArgTy>
struct TypeDefinedEntry : std::bool_constant<std::is_same_v<Ty, ArgTy>>
{
    static constexpr bool is_defined = true;
};

struct NotDefinedEntry : std::true_type
{
    static constexpr bool is_defined = false;
};

template <typename T> struct RadixSortSupportVector
{
    using resolver_t =
        typename std::disjunction<TypeDefinedEntry<T, bool>,
                                  TypeDefinedEntry<T, std::int8_t>,
                                  TypeDefinedEntry<T, std::uint8_t>,
                                  TypeDefinedEntry<T, std::int16_t>,
                                  TypeDefinedEntry<T, std::uint16_t>,
                                  TypeDefinedEntry<T, std::int32_t>,
                                  TypeDefinedEntry<T, std::uint32_t>,
                                  TypeDefinedEntry<T, std::int64_t>,
                                  TypeDefinedEntry<T, std::uint64_t>,
                                  TypeDefinedEntry<T, sycl::half>,
                                  TypeDefinedEntry<T, float>,
                                  TypeDefinedEntry<T, double>,
                                  NotDefinedEntry>;

    static constexpr bool is_defined = resolver_t::is_defined;
};

} // end of namespace py_internal
} // end of namespace tensor
} // end of namespace dpctl
