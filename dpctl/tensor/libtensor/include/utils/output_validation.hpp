//===- output_validation.hpp - Utilities for output array validation
//-*-C++-*===//
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
/// This file defines utilities for determining if an array is a valid output
/// array.
//===----------------------------------------------------------------------===//

#pragma once
#include "dpctl4pybind11.hpp"
#include <pybind11/pybind11.h>

namespace dpctl
{

namespace tensor
{

namespace validation
{

/*! @brief Raises a value error if an array is read-only.

    This should be called with an array before writing.*/
struct CheckWritable
{
    static void throw_if_not_writable(const dpctl::tensor::usm_ndarray &arr)
    {
        if (!arr.is_writable()) {
            throw py::value_error("output array is read-only.");
        }
        return;
    }
};

/*! @brief Raises a value error if an array's memory is not sufficiently ample
    to accommodate an input number of elements.

    This should be called with an array before writing.*/
struct AmpleMemory
{
    template <typename T>
    static void throw_if_not_ample(const dpctl::tensor::usm_ndarray &arr,
                                   T nelems)
    {
        auto arr_offsets = arr.get_minmax_offsets();
        T range = static_cast<T>(arr_offsets.second - arr_offsets.first);
        if (range + 1 < nelems) {
            throw py::value_error("Memory addressed by the output array is not "
                                  "sufficiently ample.");
        }
        return;
    }
};

} // namespace validation
} // namespace tensor
} // namespace dpctl
