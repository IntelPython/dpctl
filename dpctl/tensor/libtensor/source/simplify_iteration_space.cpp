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

void simplify_iteration_space(int &nd,
                              const py::ssize_t *&shape,
                              const py::ssize_t *&src_strides,
                              py::ssize_t src_itemsize,
                              bool is_src_c_contig,
                              bool is_src_f_contig,
                              const py::ssize_t *&dst_strides,
                              py::ssize_t dst_itemsize,
                              bool is_dst_c_contig,
                              bool is_dst_f_contig,
                              std::vector<py::ssize_t> &simplified_shape,
                              std::vector<py::ssize_t> &simplified_src_strides,
                              std::vector<py::ssize_t> &simplified_dst_strides,
                              py::ssize_t &src_offset,
                              py::ssize_t &dst_offset)
{
    if (nd > 1) {
        // Simplify iteration space to reduce dimensionality
        // and improve access pattern
        simplified_shape.reserve(nd);
        for (int i = 0; i < nd; ++i) {
            simplified_shape.push_back(shape[i]);
        }

        simplified_src_strides.reserve(nd);
        simplified_dst_strides.reserve(nd);
        if (src_strides == nullptr) {
            if (is_src_c_contig) {
                simplified_src_strides =
                    c_contiguous_strides(nd, shape, src_itemsize);
            }
            else if (is_src_f_contig) {
                simplified_src_strides =
                    f_contiguous_strides(nd, shape, src_itemsize);
            }
            else {
                throw std::runtime_error(
                    "Source array has null strides "
                    "but has neither C- nor F- contiguous flag set");
            }
        }
        else {
            for (int i = 0; i < nd; ++i) {
                simplified_src_strides.push_back(src_strides[i]);
            }
        }
        if (dst_strides == nullptr) {
            if (is_dst_c_contig) {
                simplified_dst_strides =
                    c_contiguous_strides(nd, shape, dst_itemsize);
            }
            else if (is_dst_f_contig) {
                simplified_dst_strides =
                    f_contiguous_strides(nd, shape, dst_itemsize);
            }
            else {
                throw std::runtime_error(
                    "Destination array has null strides "
                    "but has neither C- nor F- contiguous flag set");
            }
        }
        else {
            for (int i = 0; i < nd; ++i) {
                simplified_dst_strides.push_back(dst_strides[i]);
            }
        }

        assert(simplified_shape.size() == static_cast<size_t>(nd));
        assert(simplified_src_strides.size() == static_cast<size_t>(nd));
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
        shape = const_cast<const py::ssize_t *>(simplified_shape.data());
        src_strides =
            const_cast<const py::ssize_t *>(simplified_src_strides.data());
        dst_strides =
            const_cast<const py::ssize_t *>(simplified_dst_strides.data());
    }
    else if (nd == 1) {
        // Populate vectors
        simplified_shape.reserve(nd);
        simplified_shape.push_back(shape[0]);

        simplified_src_strides.reserve(nd);
        simplified_dst_strides.reserve(nd);

        if (src_strides == nullptr) {
            if (is_src_c_contig) {
                simplified_src_strides.push_back(src_itemsize);
            }
            else if (is_src_f_contig) {
                simplified_src_strides.push_back(src_itemsize);
            }
            else {
                throw std::runtime_error(
                    "Source array has null strides "
                    "but has neither C- nor F- contiguous flag set");
            }
        }
        else {
            simplified_src_strides.push_back(src_strides[0]);
        }
        if (dst_strides == nullptr) {
            if (is_dst_c_contig) {
                simplified_dst_strides.push_back(dst_itemsize);
            }
            else if (is_dst_f_contig) {
                simplified_dst_strides.push_back(dst_itemsize);
            }
            else {
                throw std::runtime_error(
                    "Destination array has null strides "
                    "but has neither C- nor F- contiguous flag set");
            }
        }
        else {
            simplified_dst_strides.push_back(dst_strides[0]);
        }

        assert(simplified_shape.size() == static_cast<size_t>(nd));
        assert(simplified_src_strides.size() == static_cast<size_t>(nd));
        assert(simplified_dst_strides.size() == static_cast<size_t>(nd));
    }
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
