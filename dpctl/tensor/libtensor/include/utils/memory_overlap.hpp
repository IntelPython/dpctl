//===-- memory_overlap.hpp - Array memory overlap determination   ---*-C++-*-//
//                                                                        ===//
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
/// This file defines utility to determine whether two arrays have memory
/// overlap.
//===----------------------------------------------------------------------===//

#pragma once
#include "dpctl4pybind11.hpp"
#include <pybind11/pybind11.h>

/* @brief check for overlap of memory regions behind arrays.

Presently assume that array occupies all bytes between smallest and largest
displaced elements.

TODO: Write proper Frobenius solver to account for holes, e.g.
   overlap( x_contig[::2], x_contig[1::2]) should give False,
   while this implementation gives True.
*/
namespace dpctl
{
namespace tensor
{
namespace overlap
{

struct MemoryOverlap
{

    bool operator()(dpctl::tensor::usm_ndarray ar1,
                    dpctl::tensor::usm_ndarray ar2) const
    {
        const char *ar1_data = ar1.get_data();

        const auto &ar1_offsets = ar1.get_minmax_offsets();
        py::ssize_t ar1_elem_size =
            static_cast<py::ssize_t>(ar1.get_elemsize());

        const char *ar2_data = ar2.get_data();
        const auto &ar2_offsets = ar2.get_minmax_offsets();
        py::ssize_t ar2_elem_size =
            static_cast<py::ssize_t>(ar2.get_elemsize());

        /* Memory of array1 extends from  */
        /*    [ar1_data + ar1_offsets.first * ar1_elem_size, ar1_data +
         * ar1_offsets.second * ar1_elem_size + ar1_elem_size] */
        /* Memory of array2 extends from */
        /*    [ar2_data + ar2_offsets.first * ar2_elem_size, ar2_data +
         * ar2_offsets.second * ar2_elem_size + ar2_elem_size] */

        /* Intervals [x0, x1] and [y0, y1] do not overlap if (x0 <= x1) && (y0
         * <= y1)
         * && (x1 <=y0 || y1 <= x0 ) */
        /* Given that x0 <= x1 and y0 <= y1 are true by construction, the
         * condition for overlap us (x1 > y0) && (y1 > x0) */

        /*  Applying:
            (ar1_data + ar1_offsets.second * ar1_elem_size + ar1_elem_size >
        ar2_data
        + ar2_offsets.first * ar2_elem_size) && (ar2_data + ar2_offsets.second *
        ar2_elem_size + ar2_elem_size > ar1_data + ar1_offsets.first *
        ar1_elem_size)
        */

        auto byte_distance = static_cast<py::ssize_t>(ar2_data - ar1_data);

        py::ssize_t x1_minus_y0 =
            (-byte_distance +
             (ar1_elem_size + (ar1_offsets.second * ar1_elem_size) -
              (ar2_offsets.first * ar2_elem_size)));

        py::ssize_t y1_minus_x0 =
            (byte_distance +
             (ar2_elem_size + (ar2_offsets.second * ar2_elem_size) -
              (ar1_offsets.first * ar1_elem_size)));

        bool memory_overlap = (x1_minus_y0 > 0) && (y1_minus_x0 > 0);

        return memory_overlap;
    }
};

struct SameLogicalTensors
{
    bool operator()(dpctl::tensor::usm_ndarray ar1,
                    dpctl::tensor::usm_ndarray ar2) const
    {
        // Same ndim
        int nd1 = ar1.get_ndim();
        if (nd1 != ar2.get_ndim())
            return false;

        // Same dtype
        int tn1 = ar1.get_typenum();
        if (tn1 != ar2.get_typenum())
            return false;

        // Same pointer
        const char *ar1_data = ar1.get_data();
        const char *ar2_data = ar2.get_data();

        if (ar1_data != ar2_data)
            return false;

        // Same shape and strides
        const py::ssize_t *ar1_shape = ar1.get_shape_raw();
        const py::ssize_t *ar2_shape = ar2.get_shape_raw();

        if (!std::equal(ar1_shape, ar1_shape + nd1, ar2_shape))
            return false;

        // Same shape and strides
        auto const &ar1_strides = ar1.get_strides_vector();
        auto const &ar2_strides = ar2.get_strides_vector();

        auto ar1_beg_it = std::begin(ar1_strides);
        auto ar1_end_it = std::end(ar1_strides);

        auto ar2_beg_it = std::begin(ar2_strides);

        if (!std::equal(ar1_beg_it, ar1_end_it, ar2_beg_it))
            return false;

        // all checks passed: arrays are logical views
        // into the same memory
        return true;
    }
};

} // namespace overlap
} // namespace tensor
} // namespace dpctl
