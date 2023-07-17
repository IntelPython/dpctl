//===----------- Implementation of _tensor_impl module  ---------*-C++-*-/===//
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
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines functions of dpctl.tensor._tensor_impl extensions,
/// specifically functions for elementwise operations.
//===----------------------------------------------------------------------===//

#pragma once

#include "dpctl4pybind11.hpp"
#include <CL/sycl.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <utility>
#include <vector>

#include "simplify_iteration_space.hpp"
#include "utils/memory_overlap.hpp"
#include "utils/offset_utils.hpp"
#include "utils/type_dispatch.hpp"

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

namespace td_ns = dpctl::tensor::type_dispatch;

extern py::dtype _dtype_from_typenum(td_ns::typenum_t dst_typenum_t);
extern int _result_typeid(int arg_typeid, const int *fn_output_id);

template <typename output_typesT,
          typename contig_dispatchT,
          typename strided_dispatchT>
std::pair<sycl::event, sycl::event>
py_unary_ufunc(dpctl::tensor::usm_ndarray src,
               dpctl::tensor::usm_ndarray dst,
               sycl::queue q,
               const std::vector<sycl::event> &depends,
               //
               const output_typesT &output_type_vec,
               const contig_dispatchT &contig_dispatch_vector,
               const strided_dispatchT &strided_dispatch_vector)
{
    if (!dst.is_writable()) {
        throw py::value_error("Output array is read-only.");
    }

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    const auto &array_types = td_ns::usm_ndarray_types();
    int src_typeid = array_types.typenum_to_lookup_id(src_typenum);
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    int func_output_typeid = output_type_vec[src_typeid];

    // check that types are supported
    if (dst_typeid != func_output_typeid) {
        throw py::value_error(
            "Destination array has unexpected elemental data type.");
    }

    // check that queues are compatible
    if (!dpctl::utils::queues_are_compatible(q, {src, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    // check that dimensions are the same
    int src_nd = src.get_ndim();
    if (src_nd != dst.get_ndim()) {
        throw py::value_error("Array dimensions are not the same.");
    }

    // check that shapes are the same
    const py::ssize_t *src_shape = src.get_shape_raw();
    const py::ssize_t *dst_shape = dst.get_shape_raw();
    bool shapes_equal(true);
    size_t src_nelems(1);

    for (int i = 0; i < src_nd; ++i) {
        src_nelems *= static_cast<size_t>(src_shape[i]);
        shapes_equal = shapes_equal && (src_shape[i] == dst_shape[i]);
    }
    if (!shapes_equal) {
        throw py::value_error("Array shapes are not the same.");
    }

    // if nelems is zero, return
    if (src_nelems == 0) {
        return std::make_pair(sycl::event(), sycl::event());
    }

    // ensure that output is ample enough to accomodate all elements
    auto dst_offsets = dst.get_minmax_offsets();
    // destination must be ample enough to accomodate all elements
    {
        size_t range =
            static_cast<size_t>(dst_offsets.second - dst_offsets.first);
        if (range + 1 < src_nelems) {
            throw py::value_error(
                "Destination array can not accomodate all the "
                "elements of source array.");
        }
    }

    // check memory overlap
    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    auto const &same_logical_tensors =
        dpctl::tensor::overlap::SameLogicalTensors();
    if (overlap(src, dst) && !same_logical_tensors(src, dst)) {
        throw py::value_error("Arrays index overlapping segments of memory");
    }

    const char *src_data = src.get_data();
    char *dst_data = dst.get_data();

    // handle contiguous inputs
    bool is_src_c_contig = src.is_c_contiguous();
    bool is_src_f_contig = src.is_f_contiguous();

    bool is_dst_c_contig = dst.is_c_contiguous();
    bool is_dst_f_contig = dst.is_f_contiguous();

    bool both_c_contig = (is_src_c_contig && is_dst_c_contig);
    bool both_f_contig = (is_src_f_contig && is_dst_f_contig);

    if (both_c_contig || both_f_contig) {
        auto contig_fn = contig_dispatch_vector[src_typeid];

        if (contig_fn == nullptr) {
            throw std::runtime_error(
                "Contiguous implementation is missing for src_typeid=" +
                std::to_string(src_typeid));
        }

        auto comp_ev = contig_fn(q, src_nelems, src_data, dst_data, depends);
        sycl::event ht_ev =
            dpctl::utils::keep_args_alive(q, {src, dst}, {comp_ev});

        return std::make_pair(ht_ev, comp_ev);
    }

    // simplify iteration space
    //     if 1d with strides 1 - input is contig
    //     dispatch to strided

    auto const &src_strides = src.get_strides_vector();
    auto const &dst_strides = dst.get_strides_vector();

    using shT = std::vector<py::ssize_t>;
    shT simplified_shape;
    shT simplified_src_strides;
    shT simplified_dst_strides;
    py::ssize_t src_offset(0);
    py::ssize_t dst_offset(0);

    int nd = src_nd;
    const py::ssize_t *shape = src_shape;

    dpctl::tensor::py_internal::simplify_iteration_space(
        nd, shape, src_strides, dst_strides,
        // output
        simplified_shape, simplified_src_strides, simplified_dst_strides,
        src_offset, dst_offset);

    if (nd == 1 && simplified_src_strides[0] == 1 &&
        simplified_dst_strides[0] == 1) {
        // Special case of contiguous data
        auto contig_fn = contig_dispatch_vector[src_typeid];

        if (contig_fn == nullptr) {
            throw std::runtime_error(
                "Contiguous implementation is missing for src_typeid=" +
                std::to_string(src_typeid));
        }

        int src_elem_size = src.get_elemsize();
        int dst_elem_size = dst.get_elemsize();
        auto comp_ev =
            contig_fn(q, src_nelems, src_data + src_elem_size * src_offset,
                      dst_data + dst_elem_size * dst_offset, depends);

        sycl::event ht_ev =
            dpctl::utils::keep_args_alive(q, {src, dst}, {comp_ev});

        return std::make_pair(ht_ev, comp_ev);
    }

    // Strided implementation
    auto strided_fn = strided_dispatch_vector[src_typeid];

    if (strided_fn == nullptr) {
        throw std::runtime_error(
            "Strided implementation is missing for src_typeid=" +
            std::to_string(src_typeid));
    }

    using dpctl::tensor::offset_utils::device_allocate_and_pack;

    std::vector<sycl::event> host_tasks{};
    host_tasks.reserve(2);

    const auto &ptr_size_event_triple_ = device_allocate_and_pack<py::ssize_t>(
        q, host_tasks, simplified_shape, simplified_src_strides,
        simplified_dst_strides);
    py::ssize_t *shape_strides = std::get<0>(ptr_size_event_triple_);
    sycl::event copy_shape_ev = std::get<2>(ptr_size_event_triple_);

    if (shape_strides == nullptr) {
        throw std::runtime_error("Device memory allocation failed");
    }

    sycl::event strided_fn_ev =
        strided_fn(q, src_nelems, nd, shape_strides, src_data, src_offset,
                   dst_data, dst_offset, depends, {copy_shape_ev});

    // async free of shape_strides temporary
    auto ctx = q.get_context();
    sycl::event tmp_cleanup_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(strided_fn_ev);
        cgh.host_task(
            [ctx, shape_strides]() { sycl::free(shape_strides, ctx); });
    });
    host_tasks.push_back(tmp_cleanup_ev);

    return std::make_pair(
        dpctl::utils::keep_args_alive(q, {src, dst}, host_tasks),
        strided_fn_ev);
}

template <typename output_typesT>
py::object py_unary_ufunc_result_type(py::dtype input_dtype,
                                      const output_typesT &output_types)
{
    int tn = input_dtype.num(); // NumPy type numbers are the same as in dpctl
    int src_typeid = -1;

    auto array_types = td_ns::usm_ndarray_types();

    try {
        src_typeid = array_types.typenum_to_lookup_id(tn);
    } catch (const std::exception &e) {
        throw py::value_error(e.what());
    }

    int dst_typeid = _result_typeid(src_typeid, output_types);

    if (dst_typeid < 0) {
        auto res = py::none();
        return py::cast<py::object>(res);
    }
    else {
        auto dst_typenum_t = static_cast<td_ns::typenum_t>(dst_typeid);

        auto dt = _dtype_from_typenum(dst_typenum_t);

        return py::cast<py::object>(dt);
    }
}

// ======================== Binary functions ===========================

namespace
{
template <class Container, class T>
bool isEqual(Container const &c, std::initializer_list<T> const &l)
{
    return std::equal(std::begin(c), std::end(c), std::begin(l), std::end(l));
}
} // namespace

template <typename output_typesT,
          typename contig_dispatchT,
          typename strided_dispatchT,
          typename contig_matrix_row_dispatchT,
          typename contig_row_matrix_dispatchT>
std::pair<sycl::event, sycl::event> py_binary_ufunc(
    dpctl::tensor::usm_ndarray src1,
    dpctl::tensor::usm_ndarray src2,
    dpctl::tensor::usm_ndarray dst, // dst = op(src1, src2), elementwise
    sycl::queue exec_q,
    const std::vector<sycl::event> depends,
    //
    const output_typesT &output_type_table,
    const contig_dispatchT &contig_dispatch_table,
    const strided_dispatchT &strided_dispatch_table,
    const contig_matrix_row_dispatchT
        &contig_matrix_row_broadcast_dispatch_table,
    const contig_row_matrix_dispatchT
        &contig_row_matrix_broadcast_dispatch_table)
{
    if (!dst.is_writable()) {
        throw py::value_error("Output array is read-only.");
    }
    // check type_nums
    int src1_typenum = src1.get_typenum();
    int src2_typenum = src2.get_typenum();
    int dst_typenum = dst.get_typenum();

    auto array_types = td_ns::usm_ndarray_types();
    int src1_typeid = array_types.typenum_to_lookup_id(src1_typenum);
    int src2_typeid = array_types.typenum_to_lookup_id(src2_typenum);
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    int output_typeid = output_type_table[src1_typeid][src2_typeid];

    if (output_typeid != dst_typeid) {
        throw py::value_error(
            "Destination array has unexpected elemental data type.");
    }

    // check that queues are compatible
    if (!dpctl::utils::queues_are_compatible(exec_q, {src1, src2, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    // check shapes, broadcasting is assumed done by caller
    // check that dimensions are the same
    int dst_nd = dst.get_ndim();
    if (dst_nd != src1.get_ndim() || dst_nd != src2.get_ndim()) {
        throw py::value_error("Array dimensions are not the same.");
    }

    // check that shapes are the same
    const py::ssize_t *src1_shape = src1.get_shape_raw();
    const py::ssize_t *src2_shape = src2.get_shape_raw();
    const py::ssize_t *dst_shape = dst.get_shape_raw();
    bool shapes_equal(true);
    size_t src_nelems(1);

    for (int i = 0; i < dst_nd; ++i) {
        src_nelems *= static_cast<size_t>(src1_shape[i]);
        shapes_equal = shapes_equal && (src1_shape[i] == dst_shape[i] &&
                                        src2_shape[i] == dst_shape[i]);
    }
    if (!shapes_equal) {
        throw py::value_error("Array shapes are not the same.");
    }

    // if nelems is zero, return
    if (src_nelems == 0) {
        return std::make_pair(sycl::event(), sycl::event());
    }

    // ensure that output is ample enough to accomodate all elements
    auto dst_offsets = dst.get_minmax_offsets();
    // destination must be ample enough to accomodate all elements
    {
        size_t range =
            static_cast<size_t>(dst_offsets.second - dst_offsets.first);
        if (range + 1 < src_nelems) {
            throw py::value_error(
                "Destination array can not accomodate all the "
                "elements of source array.");
        }
    }

    // check memory overlap
    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(src1, dst) || overlap(src2, dst)) {
        throw py::value_error("Arrays index overlapping segments of memory");
    }
    // check memory overlap
    const char *src1_data = src1.get_data();
    const char *src2_data = src2.get_data();
    char *dst_data = dst.get_data();

    // handle contiguous inputs
    bool is_src1_c_contig = src1.is_c_contiguous();
    bool is_src1_f_contig = src1.is_f_contiguous();

    bool is_src2_c_contig = src2.is_c_contiguous();
    bool is_src2_f_contig = src2.is_f_contiguous();

    bool is_dst_c_contig = dst.is_c_contiguous();
    bool is_dst_f_contig = dst.is_f_contiguous();

    bool all_c_contig =
        (is_src1_c_contig && is_src2_c_contig && is_dst_c_contig);
    bool all_f_contig =
        (is_src1_f_contig && is_src2_f_contig && is_dst_f_contig);

    // dispatch for contiguous inputs
    if (all_c_contig || all_f_contig) {
        auto contig_fn = contig_dispatch_table[src1_typeid][src2_typeid];

        if (contig_fn != nullptr) {
            auto comp_ev = contig_fn(exec_q, src_nelems, src1_data, 0,
                                     src2_data, 0, dst_data, 0, depends);
            sycl::event ht_ev = dpctl::utils::keep_args_alive(
                exec_q, {src1, src2, dst}, {comp_ev});

            return std::make_pair(ht_ev, comp_ev);
        }
    }

    // simplify strides
    auto const &src1_strides = src1.get_strides_vector();
    auto const &src2_strides = src2.get_strides_vector();
    auto const &dst_strides = dst.get_strides_vector();

    using shT = std::vector<py::ssize_t>;
    shT simplified_shape;
    shT simplified_src1_strides;
    shT simplified_src2_strides;
    shT simplified_dst_strides;
    py::ssize_t src1_offset(0);
    py::ssize_t src2_offset(0);
    py::ssize_t dst_offset(0);

    int nd = dst_nd;
    const py::ssize_t *shape = src1_shape;

    // all args except itemsizes and is_?_contig bools can be modified by
    // reference
    dpctl::tensor::py_internal::simplify_iteration_space_3(
        nd, shape, src1_strides, src2_strides, dst_strides,
        // outputs
        simplified_shape, simplified_src1_strides, simplified_src2_strides,
        simplified_dst_strides, src1_offset, src2_offset, dst_offset);

    std::vector<sycl::event> host_tasks{};
    if (nd < 3) {
        static constexpr auto unit_stride =
            std::initializer_list<py::ssize_t>{1};

        if ((nd == 1) && isEqual(simplified_src1_strides, unit_stride) &&
            isEqual(simplified_src2_strides, unit_stride) &&
            isEqual(simplified_dst_strides, unit_stride))
        {
            auto contig_fn = contig_dispatch_table[src1_typeid][src2_typeid];

            if (contig_fn != nullptr) {
                auto comp_ev = contig_fn(exec_q, src_nelems, src1_data,
                                         src1_offset, src2_data, src2_offset,
                                         dst_data, dst_offset, depends);
                sycl::event ht_ev = dpctl::utils::keep_args_alive(
                    exec_q, {src1, src2, dst}, {comp_ev});

                return std::make_pair(ht_ev, comp_ev);
            }
        }
        if (nd == 2) {
            static constexpr auto zero_one_strides =
                std::initializer_list<py::ssize_t>{0, 1};
            static constexpr auto one_zero_strides =
                std::initializer_list<py::ssize_t>{1, 0};
            constexpr py::ssize_t one{1};
            // special case of C-contiguous matrix and a row
            if (isEqual(simplified_src2_strides, zero_one_strides) &&
                isEqual(simplified_src1_strides, {simplified_shape[1], one}) &&
                isEqual(simplified_dst_strides, {simplified_shape[1], one}))
            {
                auto matrix_row_broadcast_fn =
                    contig_matrix_row_broadcast_dispatch_table[src1_typeid]
                                                              [src2_typeid];
                if (matrix_row_broadcast_fn != nullptr) {
                    size_t n0 = simplified_shape[0];
                    size_t n1 = simplified_shape[1];
                    sycl::event comp_ev = matrix_row_broadcast_fn(
                        exec_q, host_tasks, n0, n1, src1_data, src1_offset,
                        src2_data, src2_offset, dst_data, dst_offset, depends);

                    return std::make_pair(
                        dpctl::utils::keep_args_alive(exec_q, {src1, src2, dst},
                                                      host_tasks),
                        comp_ev);
                }
            }
            if (isEqual(simplified_src1_strides, one_zero_strides) &&
                isEqual(simplified_src2_strides, {one, simplified_shape[0]}) &&
                isEqual(simplified_dst_strides, {one, simplified_shape[0]}))
            {
                auto row_matrix_broadcast_fn =
                    contig_row_matrix_broadcast_dispatch_table[src1_typeid]
                                                              [src2_typeid];
                if (row_matrix_broadcast_fn != nullptr) {
                    size_t n0 = simplified_shape[1];
                    size_t n1 = simplified_shape[0];
                    sycl::event comp_ev = row_matrix_broadcast_fn(
                        exec_q, host_tasks, n0, n1, src1_data, src1_offset,
                        src2_data, src2_offset, dst_data, dst_offset, depends);

                    return std::make_pair(
                        dpctl::utils::keep_args_alive(exec_q, {src1, src2, dst},
                                                      host_tasks),
                        comp_ev);
                }
            }
        }
    }

    // dispatch to strided code
    auto strided_fn = strided_dispatch_table[src1_typeid][src2_typeid];

    if (strided_fn == nullptr) {
        throw std::runtime_error(
            "Strided implementation is missing for src1_typeid=" +
            std::to_string(src1_typeid) +
            " and src2_typeid=" + std::to_string(src2_typeid));
    }

    using dpctl::tensor::offset_utils::device_allocate_and_pack;
    const auto &ptr_sz_event_triple_ = device_allocate_and_pack<py::ssize_t>(
        exec_q, host_tasks, simplified_shape, simplified_src1_strides,
        simplified_src2_strides, simplified_dst_strides);

    py::ssize_t *shape_strides = std::get<0>(ptr_sz_event_triple_);
    sycl::event copy_shape_ev = std::get<2>(ptr_sz_event_triple_);

    if (shape_strides == nullptr) {
        throw std::runtime_error("Unabled to allocate device memory");
    }

    sycl::event strided_fn_ev = strided_fn(
        exec_q, src_nelems, nd, shape_strides, src1_data, src1_offset,
        src2_data, src2_offset, dst_data, dst_offset, depends, {copy_shape_ev});

    // async free of shape_strides temporary
    auto ctx = exec_q.get_context();

    sycl::event tmp_cleanup_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(strided_fn_ev);
        cgh.host_task(
            [ctx, shape_strides]() { sycl::free(shape_strides, ctx); });
    });

    host_tasks.push_back(tmp_cleanup_ev);

    return std::make_pair(
        dpctl::utils::keep_args_alive(exec_q, {src1, src2, dst}, host_tasks),
        strided_fn_ev);
}

template <typename output_typesT>
py::object py_binary_ufunc_result_type(py::dtype input1_dtype,
                                       py::dtype input2_dtype,
                                       const output_typesT &output_types_table)
{
    int tn1 = input1_dtype.num(); // NumPy type numbers are the same as in dpctl
    int tn2 = input2_dtype.num(); // NumPy type numbers are the same as in dpctl
    int src1_typeid = -1;
    int src2_typeid = -1;

    auto array_types = td_ns::usm_ndarray_types();

    try {
        src1_typeid = array_types.typenum_to_lookup_id(tn1);
        src2_typeid = array_types.typenum_to_lookup_id(tn2);
    } catch (const std::exception &e) {
        throw py::value_error(e.what());
    }

    if (src1_typeid < 0 || src1_typeid >= td_ns::num_types || src2_typeid < 0 ||
        src2_typeid >= td_ns::num_types)
    {
        throw std::runtime_error("binary output type lookup failed");
    }
    int dst_typeid = output_types_table[src1_typeid][src2_typeid];

    if (dst_typeid < 0) {
        auto res = py::none();
        return py::cast<py::object>(res);
    }
    else {
        auto dst_typenum_t = static_cast<td_ns::typenum_t>(dst_typeid);

        auto dt = _dtype_from_typenum(dst_typenum_t);

        return py::cast<py::object>(dt);
    }
}

// ==================== Inplace binary functions =======================

template <typename output_typesT,
          typename contig_dispatchT,
          typename strided_dispatchT,
          typename contig_row_matrix_dispatchT>
std::pair<sycl::event, sycl::event>
py_binary_inplace_ufunc(dpctl::tensor::usm_ndarray lhs,
                        dpctl::tensor::usm_ndarray rhs,
                        sycl::queue exec_q,
                        const std::vector<sycl::event> depends,
                        //
                        const output_typesT &output_type_table,
                        const contig_dispatchT &contig_dispatch_table,
                        const strided_dispatchT &strided_dispatch_table,
                        const contig_row_matrix_dispatchT
                            &contig_row_matrix_broadcast_dispatch_table)
{
    if (!lhs.is_writable()) {
        throw py::value_error("Output array is read-only.");
    }

    // check type_nums
    int rhs_typenum = rhs.get_typenum();
    int lhs_typenum = lhs.get_typenum();

    auto array_types = td_ns::usm_ndarray_types();
    int rhs_typeid = array_types.typenum_to_lookup_id(rhs_typenum);
    int lhs_typeid = array_types.typenum_to_lookup_id(lhs_typenum);

    int output_typeid = output_type_table[rhs_typeid][lhs_typeid];

    if (output_typeid != lhs_typeid) {
        throw py::value_error(
            "Left-hand side array has unexpected elemental data type.");
    }

    // check that queues are compatible
    if (!dpctl::utils::queues_are_compatible(exec_q, {rhs, lhs})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    // check shapes, broadcasting is assumed done by caller
    // check that dimensions are the same
    int lhs_nd = lhs.get_ndim();
    if (lhs_nd != rhs.get_ndim()) {
        throw py::value_error("Array dimensions are not the same.");
    }

    // check that shapes are the same
    const py::ssize_t *rhs_shape = rhs.get_shape_raw();
    const py::ssize_t *lhs_shape = lhs.get_shape_raw();
    bool shapes_equal(true);
    size_t rhs_nelems(1);

    for (int i = 0; i < lhs_nd; ++i) {
        rhs_nelems *= static_cast<size_t>(rhs_shape[i]);
        shapes_equal = shapes_equal && (rhs_shape[i] == lhs_shape[i]);
    }
    if (!shapes_equal) {
        throw py::value_error("Array shapes are not the same.");
    }

    // if nelems is zero, return
    if (rhs_nelems == 0) {
        return std::make_pair(sycl::event(), sycl::event());
    }

    // ensure that output is ample enough to accomodate all elements
    auto lhs_offsets = lhs.get_minmax_offsets();
    // destination must be ample enough to accomodate all elements
    {
        size_t range =
            static_cast<size_t>(lhs_offsets.second - lhs_offsets.first);
        if (range + 1 < rhs_nelems) {
            throw py::value_error(
                "Destination array can not accomodate all the "
                "elements of source array.");
        }
    }

    // check memory overlap
    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(rhs, lhs)) {
        throw py::value_error("Arrays index overlapping segments of memory");
    }
    // check memory overlap
    const char *rhs_data = rhs.get_data();
    char *lhs_data = lhs.get_data();

    // handle contiguous inputs
    bool is_rhs_c_contig = rhs.is_c_contiguous();
    bool is_rhs_f_contig = rhs.is_f_contiguous();

    bool is_lhs_c_contig = lhs.is_c_contiguous();
    bool is_lhs_f_contig = lhs.is_f_contiguous();

    bool both_c_contig = (is_rhs_c_contig && is_lhs_c_contig);
    bool both_f_contig = (is_rhs_f_contig && is_lhs_f_contig);

    // dispatch for contiguous inputs
    if (both_c_contig || both_f_contig) {
        auto contig_fn = contig_dispatch_table[rhs_typeid][lhs_typeid];

        if (contig_fn != nullptr) {
            auto comp_ev = contig_fn(exec_q, rhs_nelems, rhs_data, 0, lhs_data,
                                     0, depends);
            sycl::event ht_ev =
                dpctl::utils::keep_args_alive(exec_q, {rhs, lhs}, {comp_ev});

            return std::make_pair(ht_ev, comp_ev);
        }
    }

    // simplify strides
    auto const &rhs_strides = rhs.get_strides_vector();
    auto const &lhs_strides = lhs.get_strides_vector();

    using shT = std::vector<py::ssize_t>;
    shT simplified_shape;
    shT simplified_rhs_strides;
    shT simplified_lhs_strides;
    py::ssize_t rhs_offset(0);
    py::ssize_t lhs_offset(0);

    int nd = lhs_nd;
    const py::ssize_t *shape = rhs_shape;

    // all args except itemsizes and is_?_contig bools can be modified by
    // reference
    dpctl::tensor::py_internal::simplify_iteration_space(
        nd, shape, rhs_strides, lhs_strides,
        // outputs
        simplified_shape, simplified_rhs_strides, simplified_lhs_strides,
        rhs_offset, lhs_offset);

    std::vector<sycl::event> host_tasks{};
    if (nd < 3) {
        static constexpr auto unit_stride =
            std::initializer_list<py::ssize_t>{1};

        if ((nd == 1) && isEqual(simplified_rhs_strides, unit_stride) &&
            isEqual(simplified_lhs_strides, unit_stride))
        {
            auto contig_fn = contig_dispatch_table[rhs_typeid][lhs_typeid];

            if (contig_fn != nullptr) {
                auto comp_ev =
                    contig_fn(exec_q, rhs_nelems, rhs_data, rhs_offset,
                              lhs_data, lhs_offset, depends);
                sycl::event ht_ev = dpctl::utils::keep_args_alive(
                    exec_q, {rhs, lhs}, {comp_ev});

                return std::make_pair(ht_ev, comp_ev);
            }
        }
        if (nd == 2) {
            static constexpr auto one_zero_strides =
                std::initializer_list<py::ssize_t>{1, 0};
            constexpr py::ssize_t one{1};
            // special case of C-contiguous matrix and a row
            if (isEqual(simplified_rhs_strides, one_zero_strides) &&
                isEqual(simplified_lhs_strides, {one, simplified_shape[0]}))
            {
                auto row_matrix_broadcast_fn =
                    contig_row_matrix_broadcast_dispatch_table[rhs_typeid]
                                                              [lhs_typeid];
                if (row_matrix_broadcast_fn != nullptr) {
                    size_t n0 = simplified_shape[1];
                    size_t n1 = simplified_shape[0];
                    sycl::event comp_ev = row_matrix_broadcast_fn(
                        exec_q, host_tasks, n0, n1, rhs_data, rhs_offset,
                        lhs_data, lhs_offset, depends);

                    return std::make_pair(dpctl::utils::keep_args_alive(
                                              exec_q, {lhs, rhs}, host_tasks),
                                          comp_ev);
                }
            }
        }
    }

    // dispatch to strided code
    auto strided_fn = strided_dispatch_table[rhs_typeid][lhs_typeid];

    if (strided_fn == nullptr) {
        throw std::runtime_error(
            "Strided implementation is missing for rhs_typeid=" +
            std::to_string(rhs_typeid) +
            " and lhs_typeid=" + std::to_string(lhs_typeid));
    }

    using dpctl::tensor::offset_utils::device_allocate_and_pack;
    const auto &ptr_sz_event_triple_ = device_allocate_and_pack<py::ssize_t>(
        exec_q, host_tasks, simplified_shape, simplified_rhs_strides,
        simplified_lhs_strides);

    py::ssize_t *shape_strides = std::get<0>(ptr_sz_event_triple_);
    sycl::event copy_shape_ev = std::get<2>(ptr_sz_event_triple_);

    if (shape_strides == nullptr) {
        throw std::runtime_error("Unabled to allocate device memory");
    }

    sycl::event strided_fn_ev =
        strided_fn(exec_q, rhs_nelems, nd, shape_strides, rhs_data, rhs_offset,
                   lhs_data, lhs_offset, depends, {copy_shape_ev});

    // async free of shape_strides temporary
    auto ctx = exec_q.get_context();

    sycl::event tmp_cleanup_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(strided_fn_ev);
        cgh.host_task(
            [ctx, shape_strides]() { sycl::free(shape_strides, ctx); });
    });

    host_tasks.push_back(tmp_cleanup_ev);

    return std::make_pair(
        dpctl::utils::keep_args_alive(exec_q, {rhs, lhs}, host_tasks),
        strided_fn_ev);
}

extern void init_elementwise_functions(py::module_ m);

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
