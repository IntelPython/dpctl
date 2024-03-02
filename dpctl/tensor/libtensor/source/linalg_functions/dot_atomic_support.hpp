//===-- ------------ Implementation of _tensor_impl module  ----*-C++-*-/===//
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
/// This file defines functions of dpctl.tensor._tensor_impl extensions
//===--------------------------------------------------------------------===//

#pragma once

#include <type_traits>

#include "reductions/reduction_atomic_support.hpp"
#include "utils/type_utils.hpp"

namespace dpctl
{
namespace tensor
{
namespace py_internal
{
namespace atomic_support
{

template <typename fnT, typename T> struct DotAtomicSupportFactory
{
    fnT get()
    {
        using dpctl::tensor::type_utils::is_complex;
        if constexpr (is_complex<T>::value) {
            return atomic_support::fixed_decision<false>;
        }
        else {
            return atomic_support::check_atomic_support<T>;
        }
    }
};

} // namespace atomic_support
} // namespace py_internal
} // namespace tensor
} // namespace dpctl
