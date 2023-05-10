//===-- boolean_reductions.cpp - Implementation of boolean reductions
//--*-C++-*-/===//
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
/// This file defines Python API for implementation functions of
/// dpctl.tensor.all and dpctl.tensor.any
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <utility>
#include <vector>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "boolean_reductions.hpp"
#include "dpctl4pybind11.hpp"
#include "kernels/boolean_reductions.hpp"
#include "simplify_iteration_space.hpp"
#include "utils/memory_overlap.hpp"
#include "utils/offset_utils.hpp"
#include "utils/type_utils.hpp"

namespace py = pybind11;

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

namespace td_ns = dpctl::tensor::type_dispatch;
constexpr int _ns = td_ns::num_types;
using dpctl::tensor::kernels::boolean_reduction_strided_impl_fn_ptr;
static boolean_reduction_strided_impl_fn_ptr
    all_reduction_strided_dispatch_vector[_ns];
static boolean_reduction_strided_impl_fn_ptr
    any_reduction_strided_dispatch_vector[_ns];

using dpctl::tensor::kernels::boolean_reduction_contig_impl_fn_ptr;
static boolean_reduction_contig_impl_fn_ptr
    all_reduction_contig_dispatch_vector[_ns];
static boolean_reduction_contig_impl_fn_ptr
    any_reduction_contig_dispatch_vector[_ns];

template <typename contig_dispatchT, typename strided_dispatchT>
std::pair<sycl::event, sycl::event>
py_boolean_reduction(dpctl::tensor::usm_ndarray src,
                     int trailing_dims_to_reduce,
                     dpctl::tensor::usm_ndarray dst,
                     sycl::queue exec_q,
                     const std::vector<sycl::event> &depends,
                     const contig_dispatchT &contig_dispatch_vector,
                     const strided_dispatchT &strided_dispatch_vector)
{
    int src_nd = src.get_ndim();
    int iter_nd = src_nd - trailing_dims_to_reduce;
    if (trailing_dims_to_reduce <= 0 || iter_nd < 0) {
        throw py::value_error("Trailing_dim_to_reduce must be positive, but no "
                              "greater than rank of the array being reduced");
    }

    int dst_nd = dst.get_ndim();
    if (dst_nd != iter_nd) {
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

    size_t red_nelems(1);
    for (int i = dst_nd; i < src_nd; ++i) {
        red_nelems *= static_cast<size_t>(src_shape_ptr[i]);
    }

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(dst, src)) {
        throw py::value_error("Arrays are expected to have no memory overlap");
    }

    // ensure that dst is sufficiently ample
    auto dst_offsets = dst.get_minmax_offsets();
    // destination must be ample enough to accomodate all elements
    {
        size_t range =
            static_cast<size_t>(dst_offsets.second - dst_offsets.first);
        if (range + 1 < static_cast<size_t>(dst_nelems)) {
            throw py::value_error(
                "Memory addressed by the destination array can not "
                "accomodate all the array elements.");
        }
    }

    const char *src_data = src.get_data();
    char *dst_data = dst.get_data();

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    const auto &array_types = td_ns::usm_ndarray_types();
    int src_typeid = array_types.typenum_to_lookup_id(src_typenum);
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    constexpr int int32_typeid = static_cast<int>(td_ns::typenum_t::INT32);
    if (dst_typeid != int32_typeid) {
        throw py::value_error(
            "Unexpected data type of destination array, expecting 'int32'");
    }

    bool is_src_c_contig = src.is_c_contiguous();
    bool is_src_f_contig = src.is_f_contiguous();

    bool is_dst_c_contig = dst.is_c_contiguous();

    if ((is_src_c_contig && is_dst_c_contig) ||
        (is_src_f_contig && dst_nd == 0)) {
        auto fn = contig_dispatch_vector[src_typeid];
        constexpr py::ssize_t zero_offset = 0;

        auto red_ev = fn(exec_q, dst_nelems, red_nelems, src_data, dst_data,
                         zero_offset, zero_offset, zero_offset, depends);

        sycl::event keep_args_event =
            dpctl::utils::keep_args_alive(exec_q, {src, dst}, {red_ev});

        return std::make_pair(keep_args_event, red_ev);
    }

    auto src_shape_vecs = src.get_shape_vector();
    auto src_strides_vecs = src.get_strides_vector();
    auto dst_strides_vecs = dst.get_strides_vector();

    int simplified_red_nd = trailing_dims_to_reduce;

    using shT = std::vector<py::ssize_t>;
    shT red_src_strides(std::begin(src_strides_vecs) + dst_nd,
                        std::end(src_strides_vecs));

    shT simplified_red_shape;
    shT simplified_red_src_strides;
    py::ssize_t red_src_offset(0);

    using dpctl::tensor::py_internal::simplify_iteration_space_1;
    simplify_iteration_space_1(
        simplified_red_nd, src_shape_ptr + dst_nd, red_src_strides,
        // output
        simplified_red_shape, simplified_red_src_strides, red_src_offset);

    shT iter_src_strides(std::begin(src_strides_vecs),
                         std::begin(src_strides_vecs) + iter_nd);
    shT const &iter_dst_strides = dst_strides_vecs;

    shT simplified_iter_shape;
    shT simplified_iter_src_strides;
    shT simplified_iter_dst_strides;
    py::ssize_t iter_src_offset(0);
    py::ssize_t iter_dst_offset(0);

    if (iter_nd == 0) {
        if (dst_nelems != 1) {
            throw std::runtime_error("iteration_nd == 0, but dst_nelems != 1");
        }
        iter_nd = 1;
        simplified_iter_shape.push_back(1);
        simplified_iter_src_strides.push_back(0);
        simplified_iter_dst_strides.push_back(0);
    }
    else {
        using dpctl::tensor::py_internal::simplify_iteration_space;
        simplify_iteration_space(
            iter_nd, src_shape_ptr, iter_src_strides, iter_dst_strides,
            // output
            simplified_iter_shape, simplified_iter_src_strides,
            simplified_iter_dst_strides, iter_src_offset, iter_dst_offset);
    }

    if ((simplified_red_nd == 1) && (simplified_red_src_strides[0] == 1) &&
        (iter_nd == 1) &&
        ((simplified_iter_shape[0] == 1) ||
         ((simplified_iter_dst_strides[0] == 1) &&
          (simplified_iter_src_strides[0] ==
           static_cast<py::ssize_t>(red_nelems)))))
    {
        auto fn = contig_dispatch_vector[src_typeid];
        size_t iter_nelems = dst_nelems;

        sycl::event red_ev =
            fn(exec_q, iter_nelems, red_nelems, src.get_data(), dst.get_data(),
               iter_src_offset, iter_dst_offset, red_src_offset, depends);

        sycl::event keep_args_event =
            dpctl::utils::keep_args_alive(exec_q, {src, dst}, {red_ev});

        return std::make_pair(keep_args_event, red_ev);
    }

    auto fn = strided_dispatch_vector[src_typeid];

    std::vector<sycl::event> host_task_events{};
    const auto &iter_src_dst_metadata_packing_triple_ =
        dpctl::tensor::offset_utils::device_allocate_and_pack<py::ssize_t>(
            exec_q, host_task_events, simplified_iter_shape,
            simplified_iter_src_strides, simplified_iter_dst_strides);
    py::ssize_t *iter_shape_and_strides =
        std::get<0>(iter_src_dst_metadata_packing_triple_);
    if (iter_shape_and_strides == nullptr) {
        throw std::runtime_error("Unable to allocate memory on device");
    }
    const auto &copy_iter_metadata_ev =
        std::get<2>(iter_src_dst_metadata_packing_triple_);

    const auto &red_metadata_packing_triple_ =
        dpctl::tensor::offset_utils::device_allocate_and_pack<py::ssize_t>(
            exec_q, host_task_events, simplified_red_shape,
            simplified_red_src_strides);
    py::ssize_t *red_shape_stride = std::get<0>(red_metadata_packing_triple_);
    if (red_shape_stride == nullptr) {
        sycl::event::wait(host_task_events);
        sycl::free(iter_shape_and_strides, exec_q);
        throw std::runtime_error("Unable to allocate memory on device");
    }
    const auto &copy_red_metadata_ev =
        std::get<2>(red_metadata_packing_triple_);

    std::vector<sycl::event> all_deps;
    all_deps.reserve(depends.size() + 2);
    all_deps.resize(depends.size());
    std::copy(depends.begin(), depends.end(), all_deps.begin());
    all_deps.push_back(copy_iter_metadata_ev);
    all_deps.push_back(copy_red_metadata_ev);

    auto red_ev =
        fn(exec_q, dst_nelems, red_nelems, src_data, dst_data, dst_nd,
           iter_shape_and_strides, iter_src_offset, iter_dst_offset,
           simplified_red_nd, red_shape_stride, red_src_offset, all_deps);

    sycl::event temp_cleanup_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(red_ev);
        auto ctx = exec_q.get_context();
        cgh.host_task([ctx, iter_shape_and_strides, red_shape_stride] {
            sycl::free(iter_shape_and_strides, ctx);
            sycl::free(red_shape_stride, ctx);
        });
    });
    host_task_events.push_back(temp_cleanup_ev);

    sycl::event keep_args_event =
        dpctl::utils::keep_args_alive(exec_q, {src, dst}, host_task_events);

    return std::make_pair(keep_args_event, red_ev);
}

std::pair<sycl::event, sycl::event>
py_all(dpctl::tensor::usm_ndarray src,
       int trailing_dims_to_reduce,
       dpctl::tensor::usm_ndarray dst,
       sycl::queue exec_q,
       const std::vector<sycl::event> &depends = {})
{
    return py_boolean_reduction(src, trailing_dims_to_reduce, dst, exec_q,
                                depends, all_reduction_contig_dispatch_vector,
                                all_reduction_strided_dispatch_vector);
}

std::pair<sycl::event, sycl::event>
py_any(dpctl::tensor::usm_ndarray src,
       int trailing_dims_to_reduce,
       dpctl::tensor::usm_ndarray dst,
       sycl::queue exec_q,
       const std::vector<sycl::event> &depends = {})
{
    return py_boolean_reduction(src, trailing_dims_to_reduce, dst, exec_q,
                                depends, any_reduction_contig_dispatch_vector,
                                any_reduction_strided_dispatch_vector);
}

void populate_boolean_reduction_dispatch_vectors(void)
{
    using td_ns::DispatchVectorBuilder;

    using dpctl::tensor::kernels::boolean_reduction_strided_impl_fn_ptr;

    using dpctl::tensor::kernels::AllStridedFactory;
    DispatchVectorBuilder<boolean_reduction_strided_impl_fn_ptr,
                          AllStridedFactory, _ns>
        all_dvb1;
    all_dvb1.populate_dispatch_vector(all_reduction_strided_dispatch_vector);

    using dpctl::tensor::kernels::AnyStridedFactory;
    DispatchVectorBuilder<boolean_reduction_strided_impl_fn_ptr,
                          AnyStridedFactory, _ns>
        any_dvb1;
    any_dvb1.populate_dispatch_vector(any_reduction_strided_dispatch_vector);

    using dpctl::tensor::kernels::boolean_reduction_contig_impl_fn_ptr;

    using dpctl::tensor::kernels::AllContigFactory;
    DispatchVectorBuilder<boolean_reduction_contig_impl_fn_ptr,
                          AllContigFactory, _ns>
        all_dvb2;
    all_dvb2.populate_dispatch_vector(all_reduction_contig_dispatch_vector);

    using dpctl::tensor::kernels::AnyContigFactory;
    DispatchVectorBuilder<boolean_reduction_contig_impl_fn_ptr,
                          AnyContigFactory, _ns>
        any_dvb2;
    any_dvb2.populate_dispatch_vector(any_reduction_contig_dispatch_vector);
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
