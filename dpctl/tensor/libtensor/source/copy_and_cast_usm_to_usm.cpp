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
using dpctl::tensor::kernels::copy_and_cast::copy_and_cast_2d_fn_ptr_t;
using dpctl::tensor::kernels::copy_and_cast::copy_and_cast_generic_fn_ptr_t;

static copy_and_cast_generic_fn_ptr_t
    copy_and_cast_generic_dispatch_table[_ns::num_types][_ns::num_types];
static copy_and_cast_1d_fn_ptr_t
    copy_and_cast_1d_dispatch_table[_ns::num_types][_ns::num_types];
static copy_and_cast_2d_fn_ptr_t
    copy_and_cast_2d_dispatch_table[_ns::num_types][_ns::num_types];

namespace py = pybind11;

using dpctl::tensor::c_contiguous_strides;
using dpctl::tensor::f_contiguous_strides;

using dpctl::utils::keep_args_alive;

sycl::event _populate_packed_shape_strides_for_copycast_kernel(
    sycl::queue exec_q,
    py::ssize_t *device_shape_strides, // to be populated
    const std::vector<py::ssize_t> &common_shape,
    const std::vector<py::ssize_t> &src_strides,
    const std::vector<py::ssize_t> &dst_strides)
{
    // memory transfer optimization, use USM-host for temporary speeds up
    // tranfer to device, especially on dGPUs
    using usm_host_allocatorT =
        sycl::usm_allocator<py::ssize_t, sycl::usm::alloc::host>;
    using shT = std::vector<py::ssize_t, usm_host_allocatorT>;
    size_t nd = common_shape.size();

    usm_host_allocatorT allocator(exec_q);

    // create host temporary for packed shape and strides managed by shared
    // pointer. Packed vector is concatenation of common_shape, src_stride and
    // std_strides
    std::shared_ptr<shT> shp_host_shape_strides =
        std::make_shared<shT>(3 * nd, allocator);
    std::copy(common_shape.begin(), common_shape.end(),
              shp_host_shape_strides->begin());

    std::copy(src_strides.begin(), src_strides.end(),
              shp_host_shape_strides->begin() + nd);

    std::copy(dst_strides.begin(), dst_strides.end(),
              shp_host_shape_strides->begin() + 2 * nd);

    sycl::event copy_shape_ev = exec_q.copy<py::ssize_t>(
        shp_host_shape_strides->data(), device_shape_strides,
        shp_host_shape_strides->size());

    exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(copy_shape_ev);
        cgh.host_task([shp_host_shape_strides]() {
            // increment shared pointer ref-count to keep it alive
            // till copy operation completes;
        });
    });

    return copy_shape_ev;
}

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

    // check compatibility of execution queue and allocation queue
    sycl::queue src_q = src.get_queue();
    sycl::queue dst_q = dst.get_queue();

    if (!dpctl::utils::queues_are_compatible(exec_q, {src_q, dst_q})) {
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
    auto src_offsets = src.get_minmax_offsets();
    int src_elem_size = src.get_elemsize();
    int dst_elem_size = dst.get_elemsize();

    bool memory_overlap =
        ((dst_data - src_data > src_offsets.second * src_elem_size -
                                    dst_offsets.first * dst_elem_size) &&
         (src_data - dst_data > dst_offsets.second * dst_elem_size -
                                    src_offsets.first * src_elem_size));
    if (memory_overlap) {
        // TODO: could use a temporary, but this is done by the caller
        throw py::value_error("Arrays index overlapping segments of memory");
    }

    bool is_src_c_contig = src.is_c_contiguous();
    bool is_src_f_contig = src.is_f_contiguous();

    bool is_dst_c_contig = dst.is_c_contiguous();
    bool is_dst_f_contig = dst.is_f_contiguous();

    // check for applicability of special cases:
    //      (same type && (both C-contiguous || both F-contiguous)
    bool both_c_contig = (is_src_c_contig && is_dst_c_contig);
    bool both_f_contig = (is_src_f_contig && is_dst_f_contig);
    if (both_c_contig || both_f_contig) {
        if (src_type_id == dst_type_id) {

            sycl::event copy_ev =
                exec_q.memcpy(static_cast<void *>(dst_data),
                              static_cast<const void *>(src_data),
                              src_nelems * src_elem_size, depends);

            // make sure src and dst are not GC-ed before copy_ev is complete
            return std::make_pair(
                keep_args_alive(exec_q, {src, dst}, {copy_ev}), copy_ev);
        }
        // With contract_iter2 in place, there is no need to write
        // dedicated kernels for casting between contiguous arrays
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

    if (nd < 3) {
        if (nd == 1) {
            std::array<py::ssize_t, 1> shape_arr = {shape[0]};
            // strides may be null
            std::array<py::ssize_t, 1> src_strides_arr = {
                (src_strides ? src_strides[0] : 1)};
            std::array<py::ssize_t, 1> dst_strides_arr = {
                (dst_strides ? dst_strides[0] : 1)};

            auto fn = copy_and_cast_1d_dispatch_table[dst_type_id][src_type_id];
            sycl::event copy_and_cast_1d_event = fn(
                exec_q, src_nelems, shape_arr, src_strides_arr, dst_strides_arr,
                src_data, src_offset, dst_data, dst_offset, depends);

            return std::make_pair(
                keep_args_alive(exec_q, {src, dst}, {copy_and_cast_1d_event}),
                copy_and_cast_1d_event);
        }
        else if (nd == 2) {
            std::array<py::ssize_t, 2> shape_arr = {shape[0], shape[1]};
            std::array<py::ssize_t, 2> src_strides_arr = {src_strides[0],
                                                          src_strides[1]};
            std::array<py::ssize_t, 2> dst_strides_arr = {dst_strides[0],
                                                          dst_strides[1]};

            auto fn = copy_and_cast_2d_dispatch_table[dst_type_id][src_type_id];

            sycl::event copy_and_cast_2d_event = fn(
                exec_q, src_nelems, shape_arr, src_strides_arr, dst_strides_arr,
                src_data, src_offset, dst_data, dst_offset, depends);

            return std::make_pair(
                keep_args_alive(exec_q, {src, dst}, {copy_and_cast_2d_event}),
                copy_and_cast_2d_event);
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

    //   If shape/strides are accessed with accessors, buffer destructor
    //   will force syncronization.
    py::ssize_t *shape_strides =
        sycl::malloc_device<py::ssize_t>(3 * nd, exec_q);

    if (shape_strides == nullptr) {
        throw std::runtime_error("Unabled to allocate device memory");
    }

    sycl::event copy_shape_ev =
        _populate_packed_shape_strides_for_copycast_kernel(
            exec_q, shape_strides, simplified_shape, simplified_src_strides,
            simplified_dst_strides);

    sycl::event copy_and_cast_generic_ev = copy_and_cast_fn(
        exec_q, src_nelems, nd, shape_strides, src_data, src_offset, dst_data,
        dst_offset, depends, {copy_shape_ev});

    // async free of shape_strides temporary
    auto ctx = exec_q.get_context();
    exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(copy_and_cast_generic_ev);
        cgh.host_task(
            [ctx, shape_strides]() { sycl::free(shape_strides, ctx); });
    });

    return std::make_pair(
        keep_args_alive(exec_q, {src, dst}, {copy_and_cast_generic_ev}),
        copy_and_cast_generic_ev);
}

void init_copy_and_cast_usm_to_usm_dispatch_tables(void)
{
    using namespace dpctl::tensor::detail;

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

    using dpctl::tensor::kernels::copy_and_cast::CopyAndCast2DFactory;
    DispatchTableBuilder<copy_and_cast_2d_fn_ptr_t, CopyAndCast2DFactory,
                         num_types>
        dtb_2d;
    dtb_2d.populate_dispatch_table(copy_and_cast_2d_dispatch_table);
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
