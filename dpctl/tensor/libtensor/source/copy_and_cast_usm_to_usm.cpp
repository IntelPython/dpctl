//===-- tensor_py.cpp - Implementation of _tensor_impl module  --*-C++-*-/===//
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
/// This file defines functions of dpctl.tensor._tensor_impl extensions
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <algorithm>
#include <complex>
#include <cstdint>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <thread>
#include <type_traits>
#include <utility>

#include "dpctl4pybind11.hpp"
#include "kernels/copy_and_cast.hpp"
#include "utils/memory_overlap.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

#include "simplify_iteration_space.hpp"

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

namespace _ns = dpctl::tensor::detail;

using dpctl::tensor::kernels::copy_and_cast::copy_and_cast_1d_fn_ptr_t;
using dpctl::tensor::kernels::copy_and_cast::copy_and_cast_contig_fn_ptr_t;
using dpctl::tensor::kernels::copy_and_cast::copy_and_cast_generic_fn_ptr_t;

static copy_and_cast_generic_fn_ptr_t
    copy_and_cast_generic_dispatch_table[_ns::num_types][_ns::num_types];
static copy_and_cast_1d_fn_ptr_t
    copy_and_cast_1d_dispatch_table[_ns::num_types][_ns::num_types];
static copy_and_cast_contig_fn_ptr_t
    copy_and_cast_contig_dispatch_table[_ns::num_types][_ns::num_types];

namespace py = pybind11;

using dpctl::utils::keep_args_alive;

std::pair<sycl::event, sycl::event>
copy_usm_ndarray_into_usm_ndarray(dpctl::tensor::usm_ndarray src,
                                  dpctl::tensor::usm_ndarray dst,
                                  sycl::queue exec_q,
                                  const std::vector<sycl::event> &depends = {})
{
    // array dimensions must be the same
    int src_nd = src.get_ndim();
    int dst_nd = dst.get_ndim();

    if (src_nd != dst_nd) {
        throw py::value_error("Array dimensions are not the same.");
    }

    // shapes must be the same
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

    if (src_nelems == 0) {
        // nothing to do
        return std::make_pair(sycl::event(), sycl::event());
    }

    // destination must be ample enough to accomodate all elements
    {
        auto dst_offsets = dst.get_minmax_offsets();
        size_t range =
            static_cast<size_t>(dst_offsets.second - dst_offsets.first);
        if (range + 1 < src_nelems) {
            throw py::value_error(
                "Destination array can not accomodate all the "
                "elements of source array.");
        }
    }

    // check compatibility of execution queue and allocation queue
    if (!dpctl::utils::queues_are_compatible(exec_q, {src, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    auto array_types = dpctl::tensor::detail::usm_ndarray_types();
    int src_type_id = array_types.typenum_to_lookup_id(src_typenum);
    int dst_type_id = array_types.typenum_to_lookup_id(dst_typenum);

    char *src_data = src.get_data();
    char *dst_data = dst.get_data();

    // check that arrays do not overlap, and concurrent copying is safe.
    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(src, dst)) {
        // TODO: could use a temporary, but this is done by the caller
        throw py::value_error("Arrays index overlapping segments of memory");
    }

    bool is_src_c_contig = src.is_c_contiguous();
    bool is_src_f_contig = src.is_f_contiguous();

    bool is_dst_c_contig = dst.is_c_contiguous();
    bool is_dst_f_contig = dst.is_f_contiguous();

    // check for applicability of special cases:
    //      (both C-contiguous || both F-contiguous)
    bool both_c_contig = (is_src_c_contig && is_dst_c_contig);
    bool both_f_contig = (is_src_f_contig && is_dst_f_contig);
    if (both_c_contig || both_f_contig) {

        sycl::event copy_ev;
        if (src_type_id == dst_type_id) {

            int src_elem_size = src.get_elemsize();

            copy_ev = exec_q.memcpy(static_cast<void *>(dst_data),
                                    static_cast<const void *>(src_data),
                                    src_nelems * src_elem_size, depends);
        }
        else {
            auto contig_fn =
                copy_and_cast_contig_dispatch_table[dst_type_id][src_type_id];
            copy_ev =
                contig_fn(exec_q, src_nelems, src_data, dst_data, depends);
        }
        // make sure src and dst are not GC-ed before copy_ev is complete
        return std::make_pair(keep_args_alive(exec_q, {src, dst}, {copy_ev}),
                              copy_ev);
    }

    const py::ssize_t *src_strides = src.get_strides_raw();
    const py::ssize_t *dst_strides = dst.get_strides_raw();

    using shT = std::vector<py::ssize_t>;
    shT simplified_shape;
    shT simplified_src_strides;
    shT simplified_dst_strides;
    py::ssize_t src_offset(0);
    py::ssize_t dst_offset(0);

    int nd = src_nd;
    const py::ssize_t *shape = src_shape;

    constexpr py::ssize_t src_itemsize = 1; // in elements
    constexpr py::ssize_t dst_itemsize = 1; // in elements

    // all args except itemsizes and is_?_contig bools can be modified by
    // reference
    dpctl::tensor::py_internal::simplify_iteration_space(
        nd, shape, src_strides, src_itemsize, is_src_c_contig, is_src_f_contig,
        dst_strides, dst_itemsize, is_dst_c_contig, is_dst_f_contig,
        simplified_shape, simplified_src_strides, simplified_dst_strides,
        src_offset, dst_offset);

    if (nd < 2) {
        if (nd == 1) {
            std::array<py::ssize_t, 1> shape_arr = {shape[0]};
            // strides may be null
            std::array<py::ssize_t, 1> src_strides_arr = {
                (src_strides ? src_strides[0] : 1)};
            std::array<py::ssize_t, 1> dst_strides_arr = {
                (dst_strides ? dst_strides[0] : 1)};

            sycl::event copy_and_cast_1d_event;
            if ((src_strides_arr[0] == 1) && (dst_strides_arr[0] == 1) &&
                (src_offset == 0) && (dst_offset == 0))
            {
                auto contig_fn =
                    copy_and_cast_contig_dispatch_table[dst_type_id]
                                                       [src_type_id];
                sycl::event copy_and_cast_1d_event =
                    contig_fn(exec_q, src_nelems, src_data, dst_data, depends);
            }
            else {
                auto fn =
                    copy_and_cast_1d_dispatch_table[dst_type_id][src_type_id];
                copy_and_cast_1d_event =
                    fn(exec_q, src_nelems, shape_arr, src_strides_arr,
                       dst_strides_arr, src_data, src_offset, dst_data,
                       dst_offset, depends);
            }
            return std::make_pair(
                keep_args_alive(exec_q, {src, dst}, {copy_and_cast_1d_event}),
                copy_and_cast_1d_event);
        }
        else if (nd == 0) { // case of a scalar
            assert(src_nelems == 1);
            std::array<py::ssize_t, 1> shape_arr = {1};
            std::array<py::ssize_t, 1> src_strides_arr = {1};
            std::array<py::ssize_t, 1> dst_strides_arr = {1};

            auto fn = copy_and_cast_1d_dispatch_table[dst_type_id][src_type_id];

            sycl::event copy_and_cast_0d_event = fn(
                exec_q, src_nelems, shape_arr, src_strides_arr, dst_strides_arr,
                src_data, src_offset, dst_data, dst_offset, depends);

            return std::make_pair(
                keep_args_alive(exec_q, {src, dst}, {copy_and_cast_0d_event}),
                copy_and_cast_0d_event);
        }
    }

    // Generic implementation
    auto copy_and_cast_fn =
        copy_and_cast_generic_dispatch_table[dst_type_id][src_type_id];

    std::vector<sycl::event> host_task_events;
    host_task_events.reserve(2);

    using dpctl::tensor::offset_utils::device_allocate_and_pack;
    const auto &ptr_size_event_tuple = device_allocate_and_pack<py::ssize_t>(
        exec_q, host_task_events, simplified_shape, simplified_src_strides,
        simplified_dst_strides);
    py::ssize_t *shape_strides = std::get<0>(ptr_size_event_tuple);
    if (shape_strides == nullptr) {
        throw std::runtime_error("Unable to allocate device memory");
    }
    sycl::event copy_shape_ev = std::get<2>(ptr_size_event_tuple);

    sycl::event copy_and_cast_generic_ev = copy_and_cast_fn(
        exec_q, src_nelems, nd, shape_strides, src_data, src_offset, dst_data,
        dst_offset, depends, {copy_shape_ev});

    // async free of shape_strides temporary
    auto ctx = exec_q.get_context();
    auto temporaries_cleanup_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(copy_and_cast_generic_ev);
        cgh.host_task(
            [ctx, shape_strides]() { sycl::free(shape_strides, ctx); });
    });

    host_task_events.push_back(temporaries_cleanup_ev);

    return std::make_pair(keep_args_alive(exec_q, {src, dst}, host_task_events),
                          copy_and_cast_generic_ev);
}

void init_copy_and_cast_usm_to_usm_dispatch_tables(void)
{
    using namespace dpctl::tensor::detail;

    using dpctl::tensor::kernels::copy_and_cast::CopyAndCastContigFactory;
    DispatchTableBuilder<copy_and_cast_contig_fn_ptr_t,
                         CopyAndCastContigFactory, num_types>
        dtb_contig;
    dtb_contig.populate_dispatch_table(copy_and_cast_contig_dispatch_table);

    using dpctl::tensor::kernels::copy_and_cast::CopyAndCastGenericFactory;
    DispatchTableBuilder<copy_and_cast_generic_fn_ptr_t,
                         CopyAndCastGenericFactory, num_types>
        dtb_generic;
    dtb_generic.populate_dispatch_table(copy_and_cast_generic_dispatch_table);

    using dpctl::tensor::kernels::copy_and_cast::CopyAndCast1DFactory;
    DispatchTableBuilder<copy_and_cast_1d_fn_ptr_t, CopyAndCast1DFactory,
                         num_types>
        dtb_1d;
    dtb_1d.populate_dispatch_table(copy_and_cast_1d_dispatch_table);
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
