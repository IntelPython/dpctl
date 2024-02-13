//===-- where.cpp - Implementation of where  --*-C++-*-/===//
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
/// This file defines Python API for implementation functions of
/// dpctl.tensor.where
//===----------------------------------------------------------------------===//

#include "dpctl4pybind11.hpp"
#include <complex>
#include <cstdint>
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sycl/sycl.hpp>
#include <utility>

#include "kernels/where.hpp"
#include "simplify_iteration_space.hpp"
#include "utils/memory_overlap.hpp"
#include "utils/offset_utils.hpp"
#include "utils/output_validation.hpp"
#include "utils/type_dispatch.hpp"
#include "where.hpp"

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::kernels::search::where_contig_impl_fn_ptr_t;
using dpctl::tensor::kernels::search::where_strided_impl_fn_ptr_t;

static where_contig_impl_fn_ptr_t where_contig_dispatch_table[td_ns::num_types]
                                                             [td_ns::num_types];
static where_strided_impl_fn_ptr_t
    where_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

using dpctl::utils::keep_args_alive;

std::pair<sycl::event, sycl::event>
py_where(const dpctl::tensor::usm_ndarray &condition,
         const dpctl::tensor::usm_ndarray &x1,
         const dpctl::tensor::usm_ndarray &x2,
         const dpctl::tensor::usm_ndarray &dst,
         sycl::queue &exec_q,
         const std::vector<sycl::event> &depends)
{

    if (!dpctl::utils::queues_are_compatible(exec_q, {x1, x2, condition, dst}))
    {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    int nd = condition.get_ndim();
    int x1_nd = x1.get_ndim();
    int x2_nd = x2.get_ndim();
    int dst_nd = dst.get_ndim();

    if (nd != x1_nd || nd != x2_nd) {
        throw py::value_error(
            "Input arrays are not of appropriate dimension for where kernel.");
    }

    if (nd != dst_nd) {
        throw py::value_error(
            "Destination is not of appropriate dimension for where kernel.");
    }

    const py::ssize_t *x1_shape = x1.get_shape_raw();
    const py::ssize_t *x2_shape = x2.get_shape_raw();
    const py::ssize_t *dst_shape = dst.get_shape_raw();
    const py::ssize_t *cond_shape = condition.get_shape_raw();

    bool shapes_equal(true);
    size_t nelems(1);
    for (int i = 0; i < nd; ++i) {
        const auto &sh_i = dst_shape[i];
        nelems *= static_cast<size_t>(sh_i);
        shapes_equal = shapes_equal && (x1_shape[i] == sh_i) &&
                       (x2_shape[i] == sh_i) && (cond_shape[i] == sh_i);
    }

    if (!shapes_equal) {
        throw py::value_error("Axes are not of matching shapes.");
    }

    if (nelems == 0) {
        return std::make_pair(sycl::event{}, sycl::event{});
    }

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(dst, condition) || overlap(dst, x1) || overlap(dst, x2)) {
        throw py::value_error("Destination array overlaps with input.");
    }

    int x1_typenum = x1.get_typenum();
    int x2_typenum = x2.get_typenum();
    int cond_typenum = condition.get_typenum();
    int dst_typenum = dst.get_typenum();

    auto const &array_types = td_ns::usm_ndarray_types();
    int cond_typeid = array_types.typenum_to_lookup_id(cond_typenum);
    int x1_typeid = array_types.typenum_to_lookup_id(x1_typenum);
    int x2_typeid = array_types.typenum_to_lookup_id(x2_typenum);
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    if (x1_typeid != x2_typeid || x1_typeid != dst_typeid) {
        throw py::value_error("Value arrays must have the same data type");
    }

    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(dst, nelems);

    char *cond_data = condition.get_data();
    char *x1_data = x1.get_data();
    char *x2_data = x2.get_data();
    char *dst_data = dst.get_data();

    bool is_x1_c_contig = x1.is_c_contiguous();
    bool is_x1_f_contig = x1.is_f_contiguous();

    bool is_x2_c_contig = x2.is_c_contiguous();
    bool is_x2_f_contig = x2.is_f_contiguous();

    bool is_cond_c_contig = condition.is_c_contiguous();
    bool is_cond_f_contig = condition.is_f_contiguous();

    bool is_dst_c_contig = dst.is_c_contiguous();
    bool is_dst_f_contig = dst.is_f_contiguous();

    bool all_c_contig = (is_x1_c_contig && is_x2_c_contig && is_cond_c_contig &&
                         is_dst_c_contig);
    bool all_f_contig = (is_x1_f_contig && is_x2_f_contig && is_cond_f_contig &&
                         is_dst_f_contig);

    if (all_c_contig || all_f_contig) {
        auto contig_fn = where_contig_dispatch_table[x1_typeid][cond_typeid];

        auto where_ev = contig_fn(exec_q, nelems, cond_data, x1_data, x2_data,
                                  dst_data, depends);
        sycl::event ht_ev =
            keep_args_alive(exec_q, {x1, x2, dst, condition}, {where_ev});

        return std::make_pair(ht_ev, where_ev);
    }

    auto const &cond_strides = condition.get_strides_vector();
    auto const &x1_strides = x1.get_strides_vector();
    auto const &x2_strides = x2.get_strides_vector();
    auto const &dst_strides = dst.get_strides_vector();

    using shT = std::vector<py::ssize_t>;
    shT simplified_shape;
    shT simplified_cond_strides;
    shT simplified_x1_strides;
    shT simplified_x2_strides;
    shT simplified_dst_strides;
    py::ssize_t cond_offset(0);
    py::ssize_t x1_offset(0);
    py::ssize_t x2_offset(0);
    py::ssize_t dst_offset(0);

    dpctl::tensor::py_internal::simplify_iteration_space_4(
        nd, x1_shape, cond_strides, x1_strides, x2_strides, dst_strides,
        // outputs
        simplified_shape, simplified_cond_strides, simplified_x1_strides,
        simplified_x2_strides, simplified_dst_strides, cond_offset, x1_offset,
        x2_offset, dst_offset);

    auto fn = where_strided_dispatch_table[x1_typeid][cond_typeid];

    std::vector<sycl::event> host_task_events;
    host_task_events.reserve(2);

    using dpctl::tensor::offset_utils::device_allocate_and_pack;
    auto ptr_size_event_tuple = device_allocate_and_pack<py::ssize_t>(
        exec_q, host_task_events,
        // common shape and strides
        simplified_shape, simplified_cond_strides, simplified_x1_strides,
        simplified_x2_strides, simplified_dst_strides);
    py::ssize_t *packed_shape_strides = std::get<0>(ptr_size_event_tuple);
    sycl::event copy_shape_strides_ev = std::get<2>(ptr_size_event_tuple);

    std::vector<sycl::event> all_deps;
    all_deps.reserve(depends.size() + 1);
    all_deps.insert(all_deps.end(), depends.begin(), depends.end());
    all_deps.push_back(copy_shape_strides_ev);

    assert(all_deps.size() == depends.size() + 1);

    sycl::event where_ev = fn(exec_q, nelems, nd, cond_data, x1_data, x2_data,
                              dst_data, packed_shape_strides, cond_offset,
                              x1_offset, x2_offset, dst_offset, all_deps);

    // free packed temporaries
    sycl::event temporaries_cleanup_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(where_ev);
        const auto &ctx = exec_q.get_context();
        cgh.host_task([packed_shape_strides, ctx]() {
            sycl::free(packed_shape_strides, ctx);
        });
    });

    host_task_events.push_back(temporaries_cleanup_ev);

    sycl::event arg_cleanup_ev =
        keep_args_alive(exec_q, {x1, x2, condition, dst}, host_task_events);

    return std::make_pair(arg_cleanup_ev, where_ev);
}

void init_where_dispatch_tables(void)
{
    using namespace td_ns;
    using dpctl::tensor::kernels::search::WhereContigFactory;
    DispatchTableBuilder<where_contig_impl_fn_ptr_t, WhereContigFactory,
                         num_types>
        dtb1;
    dtb1.populate_dispatch_table(where_contig_dispatch_table);

    using dpctl::tensor::kernels::search::WhereStridedFactory;
    DispatchTableBuilder<where_strided_impl_fn_ptr_t, WhereStridedFactory,
                         num_types>
        dtb2;
    dtb2.populate_dispatch_table(where_strided_dispatch_table);
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
