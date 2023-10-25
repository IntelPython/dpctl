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
/// specifically functions for reductions.
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <sycl/sycl.hpp>
#include <type_traits>
#include <utility>
#include <vector>

#include "dpctl4pybind11.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kernels/reductions.hpp"
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

template <bool require_atomic64 = false>
bool check_atomic_support(const sycl::queue &exec_q,
                          sycl::usm::alloc usm_alloc_type)
{
    bool supports_atomics = false;

    const sycl::device &dev = exec_q.get_device();

    if constexpr (require_atomic64) {
        if (!dev.has(sycl::aspect::atomic64))
            return false;
    }

    switch (usm_alloc_type) {
    case sycl::usm::alloc::shared:
        supports_atomics = dev.has(sycl::aspect::usm_atomic_shared_allocations);
        break;
    case sycl::usm::alloc::host:
        supports_atomics = dev.has(sycl::aspect::usm_atomic_host_allocations);
        break;
    case sycl::usm::alloc::device:
        supports_atomics = true;
        break;
    default:
        supports_atomics = false;
    }

    return supports_atomics;
}

template <bool return_value>
bool fixed_decision(const sycl::queue &, sycl::usm::alloc)
{
    return return_value;
}

/* ====================== dtype supported ======================== */

template <typename fnT, typename CheckAtomicSupportFnT>
bool py_reduction_dtype_supported(
    const py::dtype &input_dtype,
    const py::dtype &output_dtype,
    const std::string &dst_usm_type,
    sycl::queue &q,
    const fnT &atomic_dispatch_table,
    const fnT &temps_dispatch_table,
    const CheckAtomicSupportFnT &check_atomic_support_size4,
    const CheckAtomicSupportFnT &check_atomic_support_size8)
{
    int arg_tn =
        input_dtype.num(); // NumPy type numbers are the same as in dpctl
    int out_tn =
        output_dtype.num(); // NumPy type numbers are the same as in dpctl
    int arg_typeid = -1;
    int out_typeid = -1;

    auto array_types = td_ns::usm_ndarray_types();

    try {
        arg_typeid = array_types.typenum_to_lookup_id(arg_tn);
        out_typeid = array_types.typenum_to_lookup_id(out_tn);
    } catch (const std::exception &e) {
        throw py::value_error(e.what());
    }

    if (arg_typeid < 0 || arg_typeid >= td_ns::num_types || out_typeid < 0 ||
        out_typeid >= td_ns::num_types)
    {
        throw std::runtime_error("Reduction type support check: lookup failed");
    }

    // remove_all_extents gets underlying type of table
    using fn_ptrT = typename std::remove_all_extents<fnT>::type;
    fn_ptrT fn = nullptr;

    sycl::usm::alloc kind = sycl::usm::alloc::unknown;

    if (dst_usm_type == "device") {
        kind = sycl::usm::alloc::device;
    }
    else if (dst_usm_type == "shared") {
        kind = sycl::usm::alloc::shared;
    }
    else if (dst_usm_type == "host") {
        kind = sycl::usm::alloc::host;
    }
    else {
        throw py::value_error("Unrecognized `dst_usm_type` argument.");
    }

    bool supports_atomics = false;

    switch (output_dtype.itemsize()) {
    case sizeof(float):
    {
        supports_atomics = check_atomic_support_size4(q, kind);
    } break;
    case sizeof(double):
    {
        supports_atomics = check_atomic_support_size8(q, kind);
    } break;
    }

    if (supports_atomics) {
        fn = atomic_dispatch_table[arg_typeid][out_typeid];
    }

    if (fn == nullptr) {
        // use slower reduction implementation using temporaries
        fn = temps_dispatch_table[arg_typeid][out_typeid];
    }

    return (fn != nullptr);
}

/* ==================== Generic reductions ====================== */

template <typename strided_fnT, typename contig_fnT, typename SupportAtomicFnT>
std::pair<sycl::event, sycl::event> py_reduction_over_axis(
    const dpctl::tensor::usm_ndarray &src,
    int trailing_dims_to_reduce, // comp over this many trailing indexes
    const dpctl::tensor::usm_ndarray &dst,
    sycl::queue &exec_q,
    const std::vector<sycl::event> &depends,
    const strided_fnT &atomic_dispatch_table,
    const strided_fnT &temps_dispatch_table,
    const contig_fnT &axis0_dispatch_table,
    const contig_fnT &axis1_dispatch_table,
    const SupportAtomicFnT &check_atomic_support_size4,
    const SupportAtomicFnT &check_atomic_support_size8)
{
    int src_nd = src.get_ndim();
    int iteration_nd = src_nd - trailing_dims_to_reduce;
    if (trailing_dims_to_reduce <= 0 || iteration_nd < 0) {
        throw py::value_error("Trailing_dim_to_reduce must be positive, but no "
                              "greater than rank of the array being reduced");
    }

    int dst_nd = dst.get_ndim();
    if (dst_nd != iteration_nd) {
        throw py::value_error("Destination array rank does not match input "
                              "array rank and number of reduced dimensions");
    }

    const py::ssize_t *src_shape_ptr = src.get_shape_raw();
    const py::ssize_t *dst_shape_ptr = dst.get_shape_raw();

    bool same_shapes = true;
    for (int i = 0; same_shapes && (i < dst_nd); ++i) {
        same_shapes = same_shapes && (src_shape_ptr[i] == dst_shape_ptr[i]);
    }

    if (!same_shapes) {
        throw py::value_error("Destination shape does not match unreduced "
                              "dimensions of the input shape");
    }

    if (!dpctl::utils::queues_are_compatible(exec_q, {src, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    size_t dst_nelems = dst.get_size();

    size_t reduction_nelems(1);
    for (int i = dst_nd; i < src_nd; ++i) {
        reduction_nelems *= static_cast<size_t>(src_shape_ptr[i]);
    }

    // check that dst and src do not overlap
    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(src, dst)) {
        throw py::value_error("Arrays index overlapping segments of memory");
    }

    // destination must be ample enough to accommodate all elements
    {
        auto dst_offsets = dst.get_minmax_offsets();
        size_t range =
            static_cast<size_t>(dst_offsets.second - dst_offsets.first);
        if (range + 1 < dst_nelems) {
            throw py::value_error(
                "Destination array can not accommodate all the "
                "elements of source array.");
        }
    }

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    namespace td_ns = dpctl::tensor::type_dispatch;
    const auto &array_types = td_ns::usm_ndarray_types();
    int src_typeid = array_types.typenum_to_lookup_id(src_typenum);
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    int dst_itemsize = dst.get_elemsize();
    bool supports_atomics = false;

    switch (dst_itemsize) {
    case sizeof(float):
    {
        void *data_ptr = dst.get_data();
        const auto &ctx = exec_q.get_context();
        auto usm_type = sycl::get_pointer_type(data_ptr, ctx);
        supports_atomics = check_atomic_support_size4(exec_q, usm_type);
    } break;
    case sizeof(double):
    {
        void *data_ptr = dst.get_data();
        const auto &ctx = exec_q.get_context();
        auto usm_type = sycl::get_pointer_type(data_ptr, ctx);

        supports_atomics = check_atomic_support_size8(exec_q, usm_type);
    } break;
    }

    // handle special case when both reduction and iteration are 1D contiguous
    // and can be done with atomics
    if (supports_atomics) {
        bool is_src_c_contig = src.is_c_contiguous();
        bool is_dst_c_contig = dst.is_c_contiguous();
        bool is_src_f_contig = src.is_f_contiguous();

        if ((is_src_c_contig && is_dst_c_contig) ||
            (is_src_f_contig && dst_nelems == 1))
        {
            auto fn = axis1_dispatch_table[src_typeid][dst_typeid];

            if (fn != nullptr) {
                size_t iter_nelems = dst_nelems;

                constexpr py::ssize_t zero_offset = 0;

                sycl::event reduction_over_axis_contig_ev =
                    fn(exec_q, iter_nelems, reduction_nelems, src.get_data(),
                       dst.get_data(),
                       zero_offset, // iteration_src_offset
                       zero_offset, // iteration_dst_offset
                       zero_offset, // reduction_src_offset
                       depends);

                sycl::event keep_args_event = dpctl::utils::keep_args_alive(
                    exec_q, {src, dst}, {reduction_over_axis_contig_ev});

                return std::make_pair(keep_args_event,
                                      reduction_over_axis_contig_ev);
            }
        }
        else if (is_src_f_contig &&
                 ((is_dst_c_contig && dst_nd == 1) || dst.is_f_contiguous()))
        {
            auto fn = axis0_dispatch_table[src_typeid][dst_typeid];
            if (fn != nullptr) {
                size_t iter_nelems = dst_nelems;

                constexpr py::ssize_t zero_offset = 0;

                sycl::event reduction_over_axis_contig_ev =
                    fn(exec_q, iter_nelems, reduction_nelems, src.get_data(),
                       dst.get_data(),
                       zero_offset, // iteration_src_offset
                       zero_offset, // iteration_dst_offset
                       zero_offset, // reduction_src_offset
                       depends);

                sycl::event keep_args_event = dpctl::utils::keep_args_alive(
                    exec_q, {src, dst}, {reduction_over_axis_contig_ev});

                return std::make_pair(keep_args_event,
                                      reduction_over_axis_contig_ev);
            }
        }
    }

    using dpctl::tensor::py_internal::simplify_iteration_space;
    using dpctl::tensor::py_internal::simplify_iteration_space_1;

    auto const &src_shape_vecs = src.get_shape_vector();
    auto const &src_strides_vecs = src.get_strides_vector();
    auto const &dst_strides_vecs = dst.get_strides_vector();

    int reduction_nd = trailing_dims_to_reduce;
    const py::ssize_t *reduction_shape_ptr = src_shape_ptr + dst_nd;
    using shT = std::vector<py::ssize_t>;
    shT reduction_src_strides(std::begin(src_strides_vecs) + dst_nd,
                              std::end(src_strides_vecs));

    shT simplified_reduction_shape;
    shT simplified_reduction_src_strides;
    py::ssize_t reduction_src_offset(0);

    simplify_iteration_space_1(
        reduction_nd, reduction_shape_ptr, reduction_src_strides,
        // output
        simplified_reduction_shape, simplified_reduction_src_strides,
        reduction_src_offset);

    const py::ssize_t *iteration_shape_ptr = src_shape_ptr;

    shT iteration_src_strides(std::begin(src_strides_vecs),
                              std::begin(src_strides_vecs) + iteration_nd);
    shT const &iteration_dst_strides = dst_strides_vecs;

    shT simplified_iteration_shape;
    shT simplified_iteration_src_strides;
    shT simplified_iteration_dst_strides;
    py::ssize_t iteration_src_offset(0);
    py::ssize_t iteration_dst_offset(0);

    if (iteration_nd == 0) {
        if (dst_nelems != 1) {
            throw std::runtime_error("iteration_nd == 0, but dst_nelems != 1");
        }
        iteration_nd = 1;
        simplified_iteration_shape.push_back(1);
        simplified_iteration_src_strides.push_back(0);
        simplified_iteration_dst_strides.push_back(0);
    }
    else {
        simplify_iteration_space(iteration_nd, iteration_shape_ptr,
                                 iteration_src_strides, iteration_dst_strides,
                                 // output
                                 simplified_iteration_shape,
                                 simplified_iteration_src_strides,
                                 simplified_iteration_dst_strides,
                                 iteration_src_offset, iteration_dst_offset);
    }

    if (supports_atomics && (reduction_nd == 1) && (iteration_nd == 1)) {
        bool mat_reduce_over_axis1 = false;
        bool mat_reduce_over_axis0 = false;
        bool array_reduce_all_elems = false;
        size_t iter_nelems = dst_nelems;

        if (simplified_reduction_src_strides[0] == 1) {
            array_reduce_all_elems = (simplified_iteration_shape[0] == 1);
            mat_reduce_over_axis1 =
                (simplified_iteration_dst_strides[0] == 1) &&
                (static_cast<size_t>(simplified_iteration_src_strides[0]) ==
                 reduction_nelems);
        }
        else if (static_cast<size_t>(simplified_reduction_src_strides[0]) ==
                 iter_nelems)
        {
            mat_reduce_over_axis0 =
                (simplified_iteration_dst_strides[0] == 1) &&
                (simplified_iteration_src_strides[0] == 1);
        }

        if (mat_reduce_over_axis1 || array_reduce_all_elems) {
            auto fn = axis1_dispatch_table[src_typeid][dst_typeid];
            if (fn != nullptr) {
                sycl::event reduction_over_axis1_contig_ev =
                    fn(exec_q, iter_nelems, reduction_nelems, src.get_data(),
                       dst.get_data(), iteration_src_offset,
                       iteration_dst_offset, reduction_src_offset, depends);

                sycl::event keep_args_event = dpctl::utils::keep_args_alive(
                    exec_q, {src, dst}, {reduction_over_axis1_contig_ev});

                return std::make_pair(keep_args_event,
                                      reduction_over_axis1_contig_ev);
            }
        }
        else if (mat_reduce_over_axis0) {
            auto fn = axis0_dispatch_table[src_typeid][dst_typeid];
            if (fn != nullptr) {
                sycl::event reduction_over_axis0_contig_ev =
                    fn(exec_q, iter_nelems, reduction_nelems, src.get_data(),
                       dst.get_data(), iteration_src_offset,
                       iteration_dst_offset, reduction_src_offset, depends);

                sycl::event keep_args_event = dpctl::utils::keep_args_alive(
                    exec_q, {src, dst}, {reduction_over_axis0_contig_ev});

                return std::make_pair(keep_args_event,
                                      reduction_over_axis0_contig_ev);
            }
        }
    }

    // remove_all_extents gets underlying type of table
    using strided_fn_ptr_T =
        typename std::remove_all_extents<strided_fnT>::type;
    strided_fn_ptr_T fn = nullptr;

    if (supports_atomics) {
        fn = atomic_dispatch_table[src_typeid][dst_typeid];
    }

    if (fn == nullptr) {
        // use slower reduction implementation using temporaries
        fn = temps_dispatch_table[src_typeid][dst_typeid];
        if (fn == nullptr) {
            throw std::runtime_error("Datatypes are not supported");
        }
    }

    std::vector<sycl::event> host_task_events{};

    using dpctl::tensor::offset_utils::device_allocate_and_pack;

    const auto &arrays_metainfo_packing_triple_ =
        device_allocate_and_pack<py::ssize_t>(
            exec_q, host_task_events,
            // iteration metadata
            simplified_iteration_shape, simplified_iteration_src_strides,
            simplified_iteration_dst_strides,
            // reduction metadata
            simplified_reduction_shape, simplified_reduction_src_strides);
    py::ssize_t *temp_allocation_ptr =
        std::get<0>(arrays_metainfo_packing_triple_);
    if (temp_allocation_ptr == nullptr) {
        throw std::runtime_error("Unable to allocate memory on device");
    }
    const auto &copy_metadata_ev = std::get<2>(arrays_metainfo_packing_triple_);

    py::ssize_t *iter_shape_and_strides = temp_allocation_ptr;
    py::ssize_t *reduction_shape_stride =
        temp_allocation_ptr + 3 * simplified_iteration_shape.size();

    std::vector<sycl::event> all_deps;
    all_deps.reserve(depends.size() + 1);
    all_deps.resize(depends.size());
    std::copy(depends.begin(), depends.end(), all_deps.begin());
    all_deps.push_back(copy_metadata_ev);

    auto reduction_ev =
        fn(exec_q, dst_nelems, reduction_nelems, src.get_data(), dst.get_data(),
           iteration_nd, iter_shape_and_strides, iteration_src_offset,
           iteration_dst_offset,
           reduction_nd, // number dimensions being reduced
           reduction_shape_stride, reduction_src_offset, all_deps);

    sycl::event temp_cleanup_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(reduction_ev);
        const auto &ctx = exec_q.get_context();
        cgh.host_task([ctx, temp_allocation_ptr] {
            sycl::free(temp_allocation_ptr, ctx);
        });
    });
    host_task_events.push_back(temp_cleanup_ev);

    sycl::event keep_args_event =
        dpctl::utils::keep_args_alive(exec_q, {src, dst}, host_task_events);

    return std::make_pair(keep_args_event, reduction_ev);
}

/* ==================== Search reductions ====================== */

template <typename fn_tableT>
std::pair<sycl::event, sycl::event> py_search_over_axis(
    const dpctl::tensor::usm_ndarray &src,
    int trailing_dims_to_reduce, // comp over this many trailing indexes
    const dpctl::tensor::usm_ndarray &dst,
    sycl::queue &exec_q,
    const std::vector<sycl::event> &depends,
    const fn_tableT &dispatch_table)
{
    int src_nd = src.get_ndim();
    int iteration_nd = src_nd - trailing_dims_to_reduce;
    if (trailing_dims_to_reduce <= 0 || iteration_nd < 0) {
        throw py::value_error("Trailing_dim_to_reduce must be positive, but no "
                              "greater than rank of the array being reduced");
    }

    int dst_nd = dst.get_ndim();
    if (dst_nd != iteration_nd) {
        throw py::value_error("Destination array rank does not match input "
                              "array rank and number of reduced dimensions");
    }

    const py::ssize_t *src_shape_ptr = src.get_shape_raw();
    const py::ssize_t *dst_shape_ptr = dst.get_shape_raw();

    bool same_shapes = true;
    for (int i = 0; same_shapes && (i < dst_nd); ++i) {
        same_shapes = same_shapes && (src_shape_ptr[i] == dst_shape_ptr[i]);
    }

    if (!same_shapes) {
        throw py::value_error("Destination shape does not match unreduced "
                              "dimensions of the input shape");
    }

    if (!dpctl::utils::queues_are_compatible(exec_q, {src, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    size_t dst_nelems = dst.get_size();

    size_t reduction_nelems(1);
    for (int i = dst_nd; i < src_nd; ++i) {
        reduction_nelems *= static_cast<size_t>(src_shape_ptr[i]);
    }

    // check that dst and src do not overlap
    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(src, dst)) {
        throw py::value_error("Arrays index overlapping segments of memory");
    }

    // destination must be ample enough to accommodate all elements
    {
        auto dst_offsets = dst.get_minmax_offsets();
        size_t range =
            static_cast<size_t>(dst_offsets.second - dst_offsets.first);
        if (range + 1 < dst_nelems) {
            throw py::value_error(
                "Destination array can not accommodate all the "
                "elements of source array.");
        }
    }

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    namespace td_ns = dpctl::tensor::type_dispatch;
    const auto &array_types = td_ns::usm_ndarray_types();
    int src_typeid = array_types.typenum_to_lookup_id(src_typenum);
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    using dpctl::tensor::py_internal::simplify_iteration_space;
    using dpctl::tensor::py_internal::simplify_iteration_space_1;

    auto const &src_shape_vecs = src.get_shape_vector();
    auto const &src_strides_vecs = src.get_strides_vector();
    auto const &dst_strides_vecs = dst.get_strides_vector();

    int reduction_nd = trailing_dims_to_reduce;
    const py::ssize_t *reduction_shape_ptr = src_shape_ptr + dst_nd;
    using shT = std::vector<py::ssize_t>;
    shT reduction_src_strides(std::begin(src_strides_vecs) + dst_nd,
                              std::end(src_strides_vecs));

    shT compact_reduction_shape;
    shT compact_reduction_src_strides;
    py::ssize_t reduction_src_offset(0);

    compact_iteration_space(
        reduction_nd, reduction_shape_ptr, reduction_src_strides,
        // output
        compact_reduction_shape, compact_reduction_src_strides);

    const py::ssize_t *iteration_shape_ptr = src_shape_ptr;

    shT iteration_src_strides(std::begin(src_strides_vecs),
                              std::begin(src_strides_vecs) + iteration_nd);
    shT const &iteration_dst_strides = dst_strides_vecs;

    shT simplified_iteration_shape;
    shT simplified_iteration_src_strides;
    shT simplified_iteration_dst_strides;
    py::ssize_t iteration_src_offset(0);
    py::ssize_t iteration_dst_offset(0);

    if (iteration_nd == 0) {
        if (dst_nelems != 1) {
            throw std::runtime_error("iteration_nd == 0, but dst_nelems != 1");
        }
        iteration_nd = 1;
        simplified_iteration_shape.push_back(1);
        simplified_iteration_src_strides.push_back(0);
        simplified_iteration_dst_strides.push_back(0);
    }
    else {
        simplify_iteration_space(iteration_nd, iteration_shape_ptr,
                                 iteration_src_strides, iteration_dst_strides,
                                 // output
                                 simplified_iteration_shape,
                                 simplified_iteration_src_strides,
                                 simplified_iteration_dst_strides,
                                 iteration_src_offset, iteration_dst_offset);
    }

    auto fn = dispatch_table[src_typeid][dst_typeid];
    if (fn == nullptr) {
        throw std::runtime_error("Datatypes are not supported");
    }

    std::vector<sycl::event> host_task_events{};

    using dpctl::tensor::offset_utils::device_allocate_and_pack;

    const auto &arrays_metainfo_packing_triple_ =
        device_allocate_and_pack<py::ssize_t>(
            exec_q, host_task_events,
            // iteration metadata
            simplified_iteration_shape, simplified_iteration_src_strides,
            simplified_iteration_dst_strides,
            // reduction metadata
            compact_reduction_shape, compact_reduction_src_strides);
    py::ssize_t *temp_allocation_ptr =
        std::get<0>(arrays_metainfo_packing_triple_);
    if (temp_allocation_ptr == nullptr) {
        throw std::runtime_error("Unable to allocate memory on device");
    }
    const auto &copy_metadata_ev = std::get<2>(arrays_metainfo_packing_triple_);

    py::ssize_t *iter_shape_and_strides = temp_allocation_ptr;
    py::ssize_t *reduction_shape_stride =
        temp_allocation_ptr + 3 * simplified_iteration_shape.size();

    std::vector<sycl::event> all_deps;
    all_deps.reserve(depends.size() + 1);
    all_deps.resize(depends.size());
    std::copy(depends.begin(), depends.end(), all_deps.begin());
    all_deps.push_back(copy_metadata_ev);

    auto comp_ev = fn(exec_q, dst_nelems, reduction_nelems, src.get_data(),
                      dst.get_data(), iteration_nd, iter_shape_and_strides,
                      iteration_src_offset, iteration_dst_offset,
                      reduction_nd, // number dimensions being reduced
                      reduction_shape_stride, reduction_src_offset, all_deps);

    sycl::event temp_cleanup_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(comp_ev);
        const auto &ctx = exec_q.get_context();
        cgh.host_task([ctx, temp_allocation_ptr] {
            sycl::free(temp_allocation_ptr, ctx);
        });
    });
    host_task_events.push_back(temp_cleanup_ev);

    sycl::event keep_args_event =
        dpctl::utils::keep_args_alive(exec_q, {src, dst}, host_task_events);

    return std::make_pair(keep_args_event, comp_ev);
}

extern void init_reduction_functions(py::module_ m);

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
