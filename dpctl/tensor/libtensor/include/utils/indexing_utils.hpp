//===----- indexing_utils.hpp - Utilities for indexing modes  -----*-C++-*/===//
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
/// This file defines utilities for handling out-of-bounds integer indices in
/// kernels that involve indexing operations, such as take, put, or advanced
/// tensor integer indexing.
//===----------------------------------------------------------------------===//

#pragma once
#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>

#include "kernels/dpctl_tensor_types.hpp"

namespace dpctl
{
namespace tensor
{
namespace indexing_utils
{

/*
 * ssize_t for indices is a design choice, dpctl::tensor::usm_ndarray
 * uses py::ssize_t for shapes and strides internally and Python uses
 * py_ssize_t for sizes of e.g. lists.
 */

template <typename IndT> struct WrapIndex
{
    ssize_t operator()(ssize_t max_item, IndT ind) const
    {
        ssize_t projected;
        max_item = sycl::max<ssize_t>(max_item, 1);

        if constexpr (std::is_signed_v<IndT>) {
            static constexpr std::uintmax_t ind_max =
                std::numeric_limits<IndT>::max();
            static constexpr std::uintmax_t ssize_max =
                std::numeric_limits<ssize_t>::max();
            static constexpr std::intmax_t ind_min =
                std::numeric_limits<IndT>::min();
            static constexpr std::intmax_t ssize_min =
                std::numeric_limits<ssize_t>::min();

            if constexpr (ind_max <= ssize_max && ind_min >= ssize_min) {
                projected = sycl::clamp<ssize_t>(static_cast<ssize_t>(ind),
                                                 -max_item, max_item - 1);
            }
            else {
                projected = sycl::clamp<IndT>(ind, static_cast<IndT>(-max_item),
                                              static_cast<IndT>(max_item - 1));
            }
            return (projected < 0) ? projected + max_item : projected;
        }
        else {
            static constexpr std::uintmax_t ind_max =
                std::numeric_limits<IndT>::max();
            static constexpr std::uintmax_t ssize_max =
                std::numeric_limits<ssize_t>::max();

            if constexpr (ind_max <= ssize_max) {
                projected =
                    sycl::min<ssize_t>(static_cast<ssize_t>(ind), max_item - 1);
            }
            else {
                projected =
                    sycl::min<IndT>(ind, static_cast<IndT>(max_item - 1));
            }
            return projected;
        }
    }
};

template <typename IndT> struct ClipIndex
{
    ssize_t operator()(ssize_t max_item, IndT ind) const
    {
        ssize_t projected;
        max_item = sycl::max<ssize_t>(max_item, 1);

        if constexpr (std::is_signed_v<IndT>) {
            static constexpr std::uintmax_t ind_max =
                std::numeric_limits<IndT>::max();
            static constexpr std::uintmax_t ssize_max =
                std::numeric_limits<ssize_t>::max();
            static constexpr std::intmax_t ind_min =
                std::numeric_limits<IndT>::min();
            static constexpr std::intmax_t ssize_min =
                std::numeric_limits<ssize_t>::min();

            if constexpr (ind_max <= ssize_max && ind_min >= ssize_min) {
                projected = sycl::clamp<ssize_t>(static_cast<ssize_t>(ind),
                                                 ssize_t(0), max_item - 1);
            }
            else {
                projected = sycl::clamp<IndT>(ind, IndT(0),
                                              static_cast<IndT>(max_item - 1));
            }
        }
        else {
            static constexpr std::uintmax_t ind_max =
                std::numeric_limits<IndT>::max();
            static constexpr std::uintmax_t ssize_max =
                std::numeric_limits<ssize_t>::max();

            if constexpr (ind_max <= ssize_max) {
                projected =
                    sycl::min<ssize_t>(static_cast<ssize_t>(ind), max_item - 1);
            }
            else {
                projected =
                    sycl::min<IndT>(ind, static_cast<IndT>(max_item - 1));
            }
        }
        return projected;
    }
};

} // namespace indexing_utils
} // namespace tensor
} // namespace dpctl
