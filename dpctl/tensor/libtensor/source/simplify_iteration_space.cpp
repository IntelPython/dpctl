//===-- ------------ Implementation of _tensor_impl module  ----*-C++-*-/===//
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
//===--------------------------------------------------------------------===//
///
/// \file
/// This file defines functions of dpctl.tensor._tensor_impl extensions
//===--------------------------------------------------------------------===//

#include "simplify_iteration_space.hpp"
#include "dpctl4pybind11.hpp"
#include "utils/strided_iters.hpp"
#include <pybind11/pybind11.h>
#include <vector>

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

namespace py = pybind11;

using dpctl::tensor::c_contiguous_strides;
using dpctl::tensor::f_contiguous_strides;

void simplify_iteration_space_1(int &nd,
                                const py::ssize_t *const &shape,
                                std::vector<py::ssize_t> const &strides,
                                // output
                                std::vector<py::ssize_t> &simplified_shape,
                                std::vector<py::ssize_t> &simplified_strides,
                                py::ssize_t &offset)
{
    using dpctl::tensor::strides::simplify_iteration_stride;
    if (nd > 1) {
        // Simplify iteration space to reduce dimensionality
        // and improve access pattern
        simplified_shape.reserve(nd);
        simplified_shape.insert(std::end(simplified_shape), shape, shape + nd);

        simplified_strides.reserve(nd);
        simplified_strides.insert(std::end(simplified_strides),
                                  std::begin(strides), std::end(strides));

        assert(simplified_shape.size() == static_cast<size_t>(nd));
        assert(simplified_strides.size() == static_cast<size_t>(nd));
        int contracted_nd = simplify_iteration_stride(
            nd, simplified_shape.data(), simplified_strides.data(),
            offset // modified by reference
        );
        simplified_shape.resize(contracted_nd);
        simplified_strides.resize(contracted_nd);

        nd = contracted_nd;
    }
    else if (nd == 1) {
        offset = 0;
        // Populate vectors
        simplified_shape.reserve(nd);
        simplified_shape.push_back(shape[0]);

        simplified_strides.reserve(nd);
        simplified_strides.push_back((strides[0] >= 0) ? strides[0]
                                                       : -strides[0]);
        if ((strides[0] < 0) && (shape[0] > 1)) {
            offset += (shape[0] - 1) * strides[0];
        }

        assert(simplified_shape.size() == static_cast<size_t>(nd));
        assert(simplified_strides.size() == static_cast<size_t>(nd));
    }
}

void simplify_iteration_space(int &nd,
                              const py::ssize_t *const &shape,
                              std::vector<py::ssize_t> const &src_strides,
                              std::vector<py::ssize_t> const &dst_strides,
                              // output
                              std::vector<py::ssize_t> &simplified_shape,
                              std::vector<py::ssize_t> &simplified_src_strides,
                              std::vector<py::ssize_t> &simplified_dst_strides,
                              py::ssize_t &src_offset,
                              py::ssize_t &dst_offset)
{
    using dpctl::tensor::strides::simplify_iteration_two_strides;
    if (nd > 1) {
        // Simplify iteration space to reduce dimensionality
        // and improve access pattern
        simplified_shape.reserve(nd);
        simplified_shape.insert(std::begin(simplified_shape), shape,
                                shape + nd);
        assert(simplified_shape.size() == static_cast<size_t>(nd));

        simplified_src_strides.reserve(nd);
        simplified_src_strides.insert(std::end(simplified_src_strides),
                                      std::begin(src_strides),
                                      std::end(src_strides));
        assert(simplified_src_strides.size() == static_cast<size_t>(nd));

        simplified_dst_strides.reserve(nd);
        simplified_dst_strides.insert(std::end(simplified_dst_strides),
                                      std::begin(dst_strides),
                                      std::end(dst_strides));
        assert(simplified_dst_strides.size() == static_cast<size_t>(nd));

        int contracted_nd = simplify_iteration_two_strides(
            nd, simplified_shape.data(), simplified_src_strides.data(),
            simplified_dst_strides.data(),
            src_offset, // modified by reference
            dst_offset  // modified by reference
        );
        simplified_shape.resize(contracted_nd);
        simplified_src_strides.resize(contracted_nd);
        simplified_dst_strides.resize(contracted_nd);

        nd = contracted_nd;
    }
    else if (nd == 1) {
        src_offset = 0;
        dst_offset = 0;
        // Populate vectors
        simplified_shape.reserve(nd);
        simplified_shape.push_back(shape[0]);
        assert(simplified_shape.size() == static_cast<size_t>(nd));

        simplified_src_strides.reserve(nd);
        simplified_src_strides.push_back(
            (src_strides[0] >= 0) ? src_strides[0] : -src_strides[0]);
        if ((src_strides[0] < 0) && (shape[0] > 1)) {
            src_offset += (shape[0] - 1) * src_strides[0];
        }
        assert(simplified_src_strides.size() == static_cast<size_t>(nd));

        simplified_dst_strides.reserve(nd);
        simplified_dst_strides.push_back(
            (dst_strides[0] >= 0) ? dst_strides[0] : -dst_strides[0]);
        if ((dst_strides[0] < 0) && (shape[0] > 1)) {
            dst_offset += (shape[0] - 1) * dst_strides[0];
        }
        assert(simplified_dst_strides.size() == static_cast<size_t>(nd));
    }
}

void simplify_iteration_space_3(
    int &nd,
    const py::ssize_t *const &shape,
    // src1
    std::vector<py::ssize_t> const &src1_strides,
    // src2
    std::vector<py::ssize_t> const &src2_strides,
    // dst
    std::vector<py::ssize_t> const &dst_strides,
    // output
    std::vector<py::ssize_t> &simplified_shape,
    std::vector<py::ssize_t> &simplified_src1_strides,
    std::vector<py::ssize_t> &simplified_src2_strides,
    std::vector<py::ssize_t> &simplified_dst_strides,
    py::ssize_t &src1_offset,
    py::ssize_t &src2_offset,
    py::ssize_t &dst_offset)
{
    using dpctl::tensor::strides::simplify_iteration_three_strides;
    if (nd > 1) {
        // Simplify iteration space to reduce dimensionality
        // and improve access pattern
        simplified_shape.reserve(nd);
        simplified_shape.insert(std::end(simplified_shape), shape, shape + nd);
        assert(simplified_shape.size() == static_cast<size_t>(nd));

        simplified_src1_strides.reserve(nd);
        simplified_src1_strides.insert(std::end(simplified_src1_strides),
                                       std::begin(src1_strides),
                                       std::end(src1_strides));
        assert(simplified_src1_strides.size() == static_cast<size_t>(nd));

        simplified_src2_strides.reserve(nd);
        simplified_src2_strides.insert(std::end(simplified_src2_strides),
                                       std::begin(src2_strides),
                                       std::end(src2_strides));
        assert(simplified_src2_strides.size() == static_cast<size_t>(nd));

        simplified_dst_strides.reserve(nd);
        simplified_dst_strides.insert(std::end(simplified_dst_strides),
                                      std::begin(dst_strides),
                                      std::end(dst_strides));
        assert(simplified_dst_strides.size() == static_cast<size_t>(nd));

        int contracted_nd = simplify_iteration_three_strides(
            nd, simplified_shape.data(), simplified_src1_strides.data(),
            simplified_src2_strides.data(), simplified_dst_strides.data(),
            src1_offset, // modified by reference
            src2_offset, // modified by reference
            dst_offset   // modified by reference
        );
        simplified_shape.resize(contracted_nd);
        simplified_src1_strides.resize(contracted_nd);
        simplified_src2_strides.resize(contracted_nd);
        simplified_dst_strides.resize(contracted_nd);

        nd = contracted_nd;
    }
    else if (nd == 1) {
        src1_offset = 0;
        src2_offset = 0;
        dst_offset = 0;
        // Populate vectors
        simplified_shape.reserve(nd);
        simplified_shape.push_back(shape[0]);
        assert(simplified_shape.size() == static_cast<size_t>(nd));

        simplified_src1_strides.reserve(nd);
        simplified_src1_strides.push_back(
            (src1_strides[0] >= 0) ? src1_strides[0] : -src1_strides[0]);
        if ((src1_strides[0] < 0) && (shape[0] > 1)) {
            src1_offset += src1_strides[0] * (shape[0] - 1);
        }
        assert(simplified_src1_strides.size() == static_cast<size_t>(nd));

        simplified_src2_strides.reserve(nd);
        simplified_src2_strides.push_back(
            (src2_strides[0] >= 0) ? src2_strides[0] : -src2_strides[0]);
        if ((src2_strides[0] < 0) && (shape[0] > 1)) {
            src2_offset += src2_strides[0] * (shape[0] - 1);
        }
        assert(simplified_src2_strides.size() == static_cast<size_t>(nd));

        simplified_dst_strides.reserve(nd);
        simplified_dst_strides.push_back(
            (dst_strides[0] >= 0) ? dst_strides[0] : -dst_strides[0]);
        if ((dst_strides[0] < 0) && (shape[0] > 1)) {
            dst_offset += dst_strides[0] * (shape[0] - 1);
        }
        assert(simplified_dst_strides.size() == static_cast<size_t>(nd));
    }
}

void simplify_iteration_space_4(
    int &nd,
    const py::ssize_t *const &shape,
    // src1
    std::vector<py::ssize_t> const &src1_strides,
    // src2
    std::vector<py::ssize_t> const &src2_strides,
    // src3
    std::vector<py::ssize_t> const &src3_strides,
    // dst
    std::vector<py::ssize_t> const &dst_strides,
    // output
    std::vector<py::ssize_t> &simplified_shape,
    std::vector<py::ssize_t> &simplified_src1_strides,
    std::vector<py::ssize_t> &simplified_src2_strides,
    std::vector<py::ssize_t> &simplified_src3_strides,
    std::vector<py::ssize_t> &simplified_dst_strides,
    py::ssize_t &src1_offset,
    py::ssize_t &src2_offset,
    py::ssize_t &src3_offset,
    py::ssize_t &dst_offset)
{
    using dpctl::tensor::strides::simplify_iteration_four_strides;
    if (nd > 1) {
        // Simplify iteration space to reduce dimensionality
        // and improve access pattern
        simplified_shape.reserve(nd);
        simplified_shape.insert(std::end(simplified_shape), shape, shape + nd);
        assert(simplified_shape.size() == static_cast<size_t>(nd));

        simplified_src1_strides.reserve(nd);
        simplified_src1_strides.insert(std::end(simplified_src1_strides),
                                       std::begin(src1_strides),
                                       std::end(src1_strides));
        assert(simplified_src1_strides.size() == static_cast<size_t>(nd));

        simplified_src2_strides.reserve(nd);
        simplified_src2_strides.insert(std::end(simplified_src2_strides),
                                       std::begin(src2_strides),
                                       std::end(src2_strides));
        assert(simplified_src2_strides.size() == static_cast<size_t>(nd));

        simplified_src3_strides.reserve(nd);
        simplified_src3_strides.insert(std::end(simplified_src3_strides),
                                       std::begin(src3_strides),
                                       std::end(src3_strides));
        assert(simplified_src3_strides.size() == static_cast<size_t>(nd));

        simplified_dst_strides.reserve(nd);
        simplified_dst_strides.insert(std::end(simplified_dst_strides),
                                      std::begin(dst_strides),
                                      std::end(dst_strides));
        assert(simplified_dst_strides.size() == static_cast<size_t>(nd));

        int contracted_nd = simplify_iteration_four_strides(
            nd, simplified_shape.data(), simplified_src1_strides.data(),
            simplified_src2_strides.data(), simplified_src3_strides.data(),
            simplified_dst_strides.data(),
            src1_offset, // modified by reference
            src2_offset, // modified by reference
            src3_offset, // modified by reference
            dst_offset   // modified by reference
        );
        simplified_shape.resize(contracted_nd);
        simplified_src1_strides.resize(contracted_nd);
        simplified_src2_strides.resize(contracted_nd);
        simplified_src3_strides.resize(contracted_nd);
        simplified_dst_strides.resize(contracted_nd);

        nd = contracted_nd;
    }
    else if (nd == 1) {
        src1_offset = 0;
        src2_offset = 0;
        src3_offset = 0;
        dst_offset = 0;
        // Populate vectors
        simplified_shape.reserve(nd);
        simplified_shape.push_back(shape[0]);
        assert(simplified_shape.size() == static_cast<size_t>(nd));

        simplified_src1_strides.reserve(nd);
        simplified_src1_strides.push_back(
            (src1_strides[0] >= 0) ? src1_strides[0] : -src1_strides[0]);
        if ((src1_strides[0] < 0) && (shape[0] > 1)) {
            src1_offset += src1_strides[0] * (shape[0] - 1);
        }
        assert(simplified_src1_strides.size() == static_cast<size_t>(nd));

        simplified_src2_strides.reserve(nd);
        simplified_src2_strides.push_back(
            (src2_strides[0] >= 0) ? src2_strides[0] : -src2_strides[0]);
        if ((src2_strides[0] < 0) && (shape[0] > 1)) {
            src2_offset += src2_strides[0] * (shape[0] - 1);
        }
        assert(simplified_src2_strides.size() == static_cast<size_t>(nd));

        simplified_src3_strides.reserve(nd);
        simplified_src3_strides.push_back(
            (src3_strides[0] >= 0) ? src3_strides[0] : -src3_strides[0]);
        if ((src3_strides[0] < 0) && (shape[0] > 1)) {
            src3_offset += src3_strides[0] * (shape[0] - 1);
        }
        assert(simplified_src3_strides.size() == static_cast<size_t>(nd));

        simplified_dst_strides.reserve(nd);
        simplified_dst_strides.push_back(
            (dst_strides[0] >= 0) ? dst_strides[0] : -dst_strides[0]);
        if ((dst_strides[0] < 0) && (shape[0] > 1)) {
            dst_offset += dst_strides[0] * (shape[0] - 1);
        }
        assert(simplified_dst_strides.size() == static_cast<size_t>(nd));
    }
}

py::ssize_t _ravel_multi_index_c(std::vector<py::ssize_t> const &mi,
                                 std::vector<py::ssize_t> const &shape)
{
    size_t nd = shape.size();
    if (nd != mi.size()) {
        throw py::value_error(
            "Multi-index and shape vectors must have the same length.");
    }

    py::ssize_t flat_index = 0;
    py::ssize_t s = 1;
    for (size_t i = 0; i < nd; ++i) {
        flat_index += mi.at(nd - 1 - i) * s;
        s *= shape.at(nd - 1 - i);
    }

    return flat_index;
}

py::ssize_t _ravel_multi_index_f(std::vector<py::ssize_t> const &mi,
                                 std::vector<py::ssize_t> const &shape)
{
    size_t nd = shape.size();
    if (nd != mi.size()) {
        throw py::value_error(
            "Multi-index and shape vectors must have the same length.");
    }

    py::ssize_t flat_index = 0;
    py::ssize_t s = 1;
    for (size_t i = 0; i < nd; ++i) {
        flat_index += mi.at(i) * s;
        s *= shape.at(i);
    }

    return flat_index;
}

std::vector<py::ssize_t> _unravel_index_c(py::ssize_t flat_index,
                                          std::vector<py::ssize_t> const &shape)
{
    size_t nd = shape.size();
    std::vector<py::ssize_t> mi;
    mi.resize(nd);

    py::ssize_t i_ = flat_index;
    for (size_t dim = 0; dim + 1 < nd; ++dim) {
        const py::ssize_t si = shape[nd - 1 - dim];
        const py::ssize_t q = i_ / si;
        const py::ssize_t r = (i_ - q * si);
        mi[nd - 1 - dim] = r;
        i_ = q;
    }
    if (nd) {
        mi[0] = i_;
    }
    return mi;
}

std::vector<py::ssize_t> _unravel_index_f(py::ssize_t flat_index,
                                          std::vector<py::ssize_t> const &shape)
{
    size_t nd = shape.size();
    std::vector<py::ssize_t> mi;
    mi.resize(nd);

    py::ssize_t i_ = flat_index;
    for (size_t dim = 0; dim + 1 < nd; ++dim) {
        const py::ssize_t si = shape[dim];
        const py::ssize_t q = i_ / si;
        const py::ssize_t r = (i_ - q * si);
        mi[dim] = r;
        i_ = q;
    }
    if (nd) {
        mi[nd - 1] = i_;
    }
    return mi;
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
