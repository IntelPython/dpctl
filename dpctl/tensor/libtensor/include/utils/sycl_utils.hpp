//=== sycl_utils.hpp - Implementation of utilities         ------- *-C++-*/===//
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
/// This file defines utilities used for kernel submission.
//===----------------------------------------------------------------------===//

#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <type_traits>
#include <vector>

#include "math_utils.hpp"

namespace dpctl
{
namespace tensor
{
namespace sycl_utils
{
namespace detail
{

template <typename...> struct TypeList;

template <typename Head, typename... Tail> struct TypeList<Head, Tail...>
{
    using head = Head;
    using tail = TypeList<Tail...>;
};

using NullTypeList = TypeList<>;
template <typename T>
struct IsNullTypeList : std::conditional_t<std::is_same_v<T, NullTypeList>,
                                           std::true_type,
                                           std::false_type>
{
};

// recursively check if type is contained in given TypeList
template <typename T, typename TList>
struct IsContained
    : std::conditional_t<
          std::is_same_v<typename TList::head, std::remove_cv_t<T>>,
          std::true_type,
          IsContained<T, typename TList::tail>>
{
};

template <> struct TypeList<>
{
};

// std::false_type when last case has been checked for membership
template <typename T> struct IsContained<T, NullTypeList> : std::false_type
{
};

template <class T> struct IsComplex : std::false_type
{
};
template <class T> struct IsComplex<std::complex<T>> : std::true_type
{
};

} // namespace detail

template <typename T>
using sycl_ops = detail::TypeList<sycl::plus<T>,
                                  sycl::bit_or<T>,
                                  sycl::bit_xor<T>,
                                  sycl::bit_and<T>,
                                  sycl::maximum<T>,
                                  sycl::minimum<T>,
                                  sycl::multiplies<T>>;

template <typename T, typename Op> struct IsSyclOp
{
    static constexpr bool value =
        detail::IsContained<Op, sycl_ops<std::remove_const_t<T>>>::value ||
        detail::IsContained<Op, sycl_ops<std::add_const_t<T>>>::value;
};

/*! @brief Find the smallest multiple of supported sub-group size larger than
 * nelems */
template <size_t f = 4>
size_t choose_workgroup_size(const size_t nelems,
                             const std::vector<size_t> &sg_sizes)
{
    std::vector<size_t> wg_choices;
    wg_choices.reserve(f * sg_sizes.size());

    for (const auto &sg_size : sg_sizes) {
#pragma unroll
        for (size_t i = 1; i <= f; ++i) {
            wg_choices.push_back(sg_size * i);
        }
    }
    std::sort(std::begin(wg_choices), std::end(wg_choices));

    size_t wg = 1;
    for (size_t i = 0; i < wg_choices.size(); ++i) {
        if (wg_choices[i] == wg) {
            continue;
        }
        wg = wg_choices[i];
        size_t n_groups = ((nelems + wg - 1) / wg);
        if (n_groups == 1)
            break;
    }

    return wg;
}

template <typename T, typename GroupT, typename LocAccT, typename OpT>
T custom_reduce_over_group(const GroupT &wg,
                           LocAccT local_mem_acc,
                           const T &local_val,
                           const OpT &op)
{
    size_t wgs = wg.get_local_linear_range();
    local_mem_acc[wg.get_local_linear_id()] = local_val;

    sycl::group_barrier(wg, sycl::memory_scope::work_group);

    T red_val_over_wg = local_mem_acc[0];
    if (wg.leader()) {
        for (size_t i = 1; i < wgs; ++i) {
            red_val_over_wg = op(red_val_over_wg, local_mem_acc[i]);
        }
    }

    sycl::group_barrier(wg, sycl::memory_scope::work_group);

    return sycl::group_broadcast(wg, red_val_over_wg);
}

template <typename T, typename GroupT, typename LocAccT, typename OpT>
T custom_inclusive_scan_over_group(const GroupT &wg,
                                   LocAccT local_mem_acc,
                                   const T local_val,
                                   const OpT &op)
{
    const std::uint32_t local_id = wg.get_local_id(0);
    const std::uint32_t wgs = wg.get_local_range(0);
    local_mem_acc[local_id] = local_val;

    sycl::group_barrier(wg, sycl::memory_scope::work_group);

    if (wg.leader()) {
        T scan_val = local_mem_acc[0];
        for (std::uint32_t i = 1; i < wgs; ++i) {
            scan_val = op(local_mem_acc[i], scan_val);
            local_mem_acc[i] = scan_val;
        }
    }

    // ensure all work-items see the same SLM that leader updated
    sycl::group_barrier(wg, sycl::memory_scope::work_group);
    return local_mem_acc[local_id];
}

// Reduction functors

// Maximum

template <typename T> struct Maximum
{
    T operator()(const T &x, const T &y) const
    {
        if constexpr (detail::IsComplex<T>::value) {
            using dpctl::tensor::math_utils::max_complex;
            return max_complex<T>(x, y);
        }
        else if constexpr (std::is_floating_point_v<T> ||
                           std::is_same_v<T, sycl::half>) {
            return (std::isnan(x) || x > y) ? x : y;
        }
        else if constexpr (std::is_same_v<T, bool>) {
            return x || y;
        }
        else {
            return (x > y) ? x : y;
        }
    }
};

// Minimum

template <typename T> struct Minimum
{
    T operator()(const T &x, const T &y) const
    {
        if constexpr (detail::IsComplex<T>::value) {
            using dpctl::tensor::math_utils::min_complex;
            return min_complex<T>(x, y);
        }
        else if constexpr (std::is_floating_point_v<T> ||
                           std::is_same_v<T, sycl::half>) {
            return (std::isnan(x) || x < y) ? x : y;
        }
        else if constexpr (std::is_same_v<T, bool>) {
            return x && y;
        }
        else {
            return (x < y) ? x : y;
        }
    }
};

// Define identities and operator checking structs

template <typename Op, typename T, typename = void> struct GetIdentity
{
};

// Maximum

template <typename T, class Op>
using IsMaximum = std::bool_constant<std::is_same_v<Op, sycl::maximum<T>> ||
                                     std::is_same_v<Op, Maximum<T>>>;

template <typename T, class Op>
using IsSyclMaximum = std::bool_constant<std::is_same_v<Op, sycl::maximum<T>>>;

template <typename Op, typename T>
struct GetIdentity<Op, T, std::enable_if_t<IsMaximum<T, Op>::value>>
{
    static constexpr T value =
        static_cast<T>(std::numeric_limits<T>::has_infinity
                           ? static_cast<T>(-std::numeric_limits<T>::infinity())
                           : std::numeric_limits<T>::lowest());
};

template <typename Op>
struct GetIdentity<Op, bool, std::enable_if_t<IsMaximum<bool, Op>::value>>
{
    static constexpr bool value = false;
};

template <typename Op, typename T>
struct GetIdentity<Op,
                   std::complex<T>,
                   std::enable_if_t<IsMaximum<std::complex<T>, Op>::value>>
{
    static constexpr std::complex<T> value{-std::numeric_limits<T>::infinity(),
                                           -std::numeric_limits<T>::infinity()};
};

// Minimum

template <typename T, class Op>
using IsMinimum = std::bool_constant<std::is_same_v<Op, sycl::minimum<T>> ||
                                     std::is_same_v<Op, Minimum<T>>>;

template <typename T, class Op>
using IsSyclMinimum = std::bool_constant<std::is_same_v<Op, sycl::minimum<T>>>;

template <typename Op, typename T>
struct GetIdentity<Op, T, std::enable_if_t<IsMinimum<T, Op>::value>>
{
    static constexpr T value =
        static_cast<T>(std::numeric_limits<T>::has_infinity
                           ? static_cast<T>(std::numeric_limits<T>::infinity())
                           : std::numeric_limits<T>::max());
};

template <typename Op>
struct GetIdentity<Op, bool, std::enable_if_t<IsMinimum<bool, Op>::value>>
{
    static constexpr bool value = true;
};

template <typename Op, typename T>
struct GetIdentity<Op,
                   std::complex<T>,
                   std::enable_if_t<IsMinimum<std::complex<T>, Op>::value>>
{
    static constexpr std::complex<T> value{std::numeric_limits<T>::infinity(),
                                           std::numeric_limits<T>::infinity()};
};

// Plus

template <typename T, class Op>
using IsPlus = std::bool_constant<std::is_same_v<Op, sycl::plus<T>> ||
                                  std::is_same_v<Op, std::plus<T>>>;

template <typename T, class Op>
using IsSyclPlus = std::bool_constant<std::is_same_v<Op, sycl::plus<T>>>;

// Multiplies

template <typename T, class Op>
using IsMultiplies =
    std::bool_constant<std::is_same_v<Op, sycl::multiplies<T>> ||
                       std::is_same_v<Op, std::multiplies<T>>>;

template <typename T, class Op>
using IsSyclMultiplies =
    std::bool_constant<std::is_same_v<Op, sycl::multiplies<T>>>;

template <typename Op, typename T>
struct GetIdentity<Op, T, std::enable_if_t<IsMultiplies<T, Op>::value>>
{
    static constexpr T value = static_cast<T>(1);
};

// LogSumExp

template <typename T> struct LogSumExp
{
    T operator()(const T &x, const T &y) const
    {
        using dpctl::tensor::math_utils::logaddexp;
        return logaddexp<T>(x, y);
    }
};

template <typename T, class Op>
using IsLogSumExp = std::bool_constant<std::is_same_v<Op, LogSumExp<T>>>;

// only defined for types with infinity
template <typename Op, typename T>
struct GetIdentity<Op, T, std::enable_if_t<IsLogSumExp<T, Op>::value>>
{
    static constexpr T value = -std::numeric_limits<T>::infinity();
};

// Hypot

template <typename T> struct Hypot
{
    T operator()(const T &x, const T &y) const
    {
        return sycl::hypot(x, y);
    }
};

template <typename T, class Op>
using IsHypot = std::bool_constant<std::is_same_v<Op, Hypot<T>>>;

template <typename Op, typename T>
struct GetIdentity<Op, T, std::enable_if_t<IsHypot<T, Op>::value>>
{
    static constexpr T value = 0;
};

// Logical_And

template <typename T, class Op>
using IsLogicalAnd =
    std::bool_constant<std::is_same_v<Op, sycl::logical_and<T>> ||
                       std::is_same_v<Op, std::logical_and<T>>>;

template <typename T, class Op>
using IsSyclLogicalAnd =
    std::bool_constant<std::is_same_v<Op, sycl::logical_and<T>>>;

template <typename Op, typename T>
struct GetIdentity<Op, T, std::enable_if_t<IsLogicalAnd<T, Op>::value>>
{
    static constexpr T value = static_cast<T>(1);
};

// Logical_Or

template <typename T, class Op>
using IsLogicalOr =
    std::bool_constant<std::is_same_v<Op, sycl::logical_or<T>> ||
                       std::is_same_v<Op, std::logical_or<T>>>;

template <typename T, class Op>
using IsSyclLogicalOr =
    std::bool_constant<std::is_same_v<Op, sycl::logical_or<T>>>;

template <typename Op, typename T>
struct GetIdentity<Op, T, std::enable_if_t<IsLogicalOr<T, Op>::value>>
{
    static constexpr T value = static_cast<T>(0);
};

// Identity

template <typename Op, typename T, typename = void> struct Identity
{
};

template <typename Op, typename T>
using UseBuiltInIdentity =
    std::conjunction<IsSyclOp<T, Op>, sycl::has_known_identity<Op, T>>;

template <typename Op, typename T>
struct Identity<Op, T, std::enable_if_t<!UseBuiltInIdentity<Op, T>::value>>
{
    static constexpr T value = GetIdentity<Op, T>::value;
};

template <typename Op, typename T>
struct Identity<Op, T, std::enable_if_t<UseBuiltInIdentity<Op, T>::value>>
{
    static constexpr T value = sycl::known_identity<Op, T>::value;
};

} // namespace sycl_utils
} // namespace tensor
} // namespace dpctl
