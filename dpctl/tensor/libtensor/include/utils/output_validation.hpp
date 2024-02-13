//===- output_validation.hpp - Utilities for output array validation
//-*-C++-*===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2022 Intel Corporation
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

/*! @brief Raises a value error if a function would attempt to write
    to an array which is read-only.

    This should always be called on an array before it will be written to.*/
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

} // namespace validation
} // namespace tensor
} // namespace dpctl
