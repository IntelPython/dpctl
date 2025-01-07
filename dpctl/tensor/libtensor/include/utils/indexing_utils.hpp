//===----- indexing_utils.hpp - Utilities for indexing modes  -----*-C++-*/===//
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
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines utilities for handling out-of-bounds integer indices in
/// kernels that involve indexing operations, such as take, put, or advanced
/// tensor integer indexing.
//===----------------------------------------------------------------------===//

#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "kernels/dpctl_tensor_types.hpp"

namespace dpctl
{
namespace tensor
{
namespace indexing_utils
{

using dpctl::tensor::ssize_t;

/*
 * ssize_t for indices is a design choice, dpctl::tensor::usm_ndarray
 * uses py::ssize_t for shapes and strides internally and Python uses
 * py_ssize_t for sizes of e.g. lists.
 */

template <typename IndT> struct WrapIndex
{
    static_assert(std::is_integral_v<IndT>);

    ssize_t operator()(ssize_t max_item, IndT ind) const
    {
        ssize_t projected;
        constexpr ssize_t unit(1);
        max_item = sycl::max(max_item, unit);

        constexpr std::uintmax_t ind_max = std::numeric_limits<IndT>::max();
        constexpr std::uintmax_t ssize_max =
            std::numeric_limits<ssize_t>::max();

        if constexpr (std::is_signed_v<IndT>) {
            constexpr std::intmax_t ind_min = std::numeric_limits<IndT>::min();
            constexpr std::intmax_t ssize_min =
                std::numeric_limits<ssize_t>::min();

            if constexpr (ind_max <= ssize_max && ind_min >= ssize_min) {
                const ssize_t ind_ = static_cast<ssize_t>(ind);
                const ssize_t lb = -max_item;
                const ssize_t ub = max_item - 1;
                projected = sycl::clamp(ind_, lb, ub);
            }
            else {
                const IndT lb = static_cast<IndT>(-max_item);
                const IndT ub = static_cast<IndT>(max_item - 1);
                projected = static_cast<ssize_t>(sycl::clamp(ind, lb, ub));
            }
            return (projected < 0) ? projected + max_item : projected;
        }
        else {
            if constexpr (ind_max <= ssize_max) {
                const ssize_t ind_ = static_cast<ssize_t>(ind);
                const ssize_t ub = max_item - 1;
                projected = sycl::min(ind_, ub);
            }
            else {
                const IndT ub = static_cast<IndT>(max_item - 1);
                projected = static_cast<ssize_t>(sycl::min(ind, ub));
            }
            return projected;
        }
    }
};

template <typename IndT> struct ClipIndex
{
    static_assert(std::is_integral_v<IndT>);

    ssize_t operator()(ssize_t max_item, IndT ind) const
    {
        ssize_t projected;
        constexpr ssize_t unit(1);
        max_item = sycl::max<ssize_t>(max_item, unit);

        constexpr std::uintmax_t ind_max = std::numeric_limits<IndT>::max();
        constexpr std::uintmax_t ssize_max =
            std::numeric_limits<ssize_t>::max();
        if constexpr (std::is_signed_v<IndT>) {
            constexpr std::intmax_t ind_min = std::numeric_limits<IndT>::min();
            constexpr std::intmax_t ssize_min =
                std::numeric_limits<ssize_t>::min();

            if constexpr (ind_max <= ssize_max && ind_min >= ssize_min) {
                const ssize_t ind_ = static_cast<ssize_t>(ind);
                constexpr ssize_t lb(0);
                const ssize_t ub = max_item - 1;
                projected = sycl::clamp(ind_, lb, ub);
            }
            else {
                constexpr IndT lb(0);
                const IndT ub = static_cast<IndT>(max_item - 1);
                projected = static_cast<std::size_t>(sycl::clamp(ind, lb, ub));
            }
        }
        else {
            if constexpr (ind_max <= ssize_max) {
                const ssize_t ind_ = static_cast<ssize_t>(ind);
                const ssize_t ub = max_item - 1;
                projected = sycl::min(ind_, ub);
            }
            else {
                const IndT ub = static_cast<IndT>(max_item - 1);
                projected = static_cast<ssize_t>(sycl::min(ind, ub));
            }
        }
        return projected;
    }
};

} // namespace indexing_utils
} // namespace tensor
} // namespace dpctl
