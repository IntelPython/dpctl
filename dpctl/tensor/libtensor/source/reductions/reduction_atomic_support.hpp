//===-- ------------ Implementation of _tensor_impl module  ----*-C++-*-/===//
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
/// This file defines functions of dpctl.tensor._tensor_impl extensions
//===--------------------------------------------------------------------===//

#pragma once
#include <CL/sycl.hpp>
#include <complex>
#include <type_traits>

#include "utils/type_utils.hpp"

namespace dpctl
{
namespace tensor
{
namespace py_internal
{
namespace atomic_support
{

typedef bool (*atomic_support_fn_ptr_t)(const sycl::queue &, sycl::usm::alloc);

/*! @brief Function which returns a constant value for atomic support */
template <bool return_value>
bool fixed_decision(const sycl::queue &, sycl::usm::alloc)
{
    return return_value;
}

/*! @brief Template for querying atomic support for a type on a device */
template <typename T>
bool check_atomic_support(const sycl::queue &exec_q,
                          sycl::usm::alloc usm_alloc_type)
{
    constexpr bool atomic32 = (sizeof(T) == 4);
    constexpr bool atomic64 = (sizeof(T) == 8);
    using dpctl::tensor::type_utils::is_complex;
    if constexpr ((!atomic32 && !atomic64) || is_complex<T>::value) {
        return fixed_decision<false>(exec_q, usm_alloc_type);
    }
    else {
        bool supports_atomics = false;
        const sycl::device &dev = exec_q.get_device();
        if constexpr (atomic64) {
            if (!dev.has(sycl::aspect::atomic64)) {
                return false;
            }
        }
        switch (usm_alloc_type) {
        case sycl::usm::alloc::shared:
            supports_atomics =
                dev.has(sycl::aspect::usm_atomic_shared_allocations);
            break;
        case sycl::usm::alloc::host:
            supports_atomics =
                dev.has(sycl::aspect::usm_atomic_host_allocations);
            break;
        case sycl::usm::alloc::device:
            supports_atomics = true;
            break;
        default:
            supports_atomics = false;
        }
        return supports_atomics;
    }
}

template <typename fnT, typename T> struct MaxAtomicSupportFactory
{
    fnT get()
    {
        if constexpr (std::is_floating_point_v<T>) {
            return fixed_decision<false>;
        }
        else {
            return check_atomic_support<T>;
        }
    }
};

template <typename fnT, typename T> struct MinAtomicSupportFactory
{
    fnT get()
    {
        if constexpr (std::is_floating_point_v<T>) {
            return fixed_decision<false>;
        }
        else {
            return check_atomic_support<T>;
        }
    }
};

template <typename fnT, typename T> struct SumAtomicSupportFactory
{
    fnT get()
    {
        if constexpr (std::is_floating_point_v<T>) {
            return fixed_decision<false>;
        }
        else {
            return check_atomic_support<T>;
        }
    }
};

template <typename fnT, typename T> struct ProductAtomicSupportFactory
{
    fnT get()
    {
        if constexpr (std::is_floating_point_v<T>) {
            return fixed_decision<false>;
        }
        else {
            return check_atomic_support<T>;
        }
    }
};

} // namespace atomic_support
} // namespace py_internal
} // namespace tensor
} // namespace dpctl
