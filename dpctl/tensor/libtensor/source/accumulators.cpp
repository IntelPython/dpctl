//===-- ------------ Implementation of _tensor_impl module  ----*-C++-*-/===//
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
/// This file defines functions of dpctl.tensor._tensor_impl extensions
//===----------------------------------------------------------------------===//

#include "dpctl4pybind11.hpp"
#include <CL/sycl.hpp>
#include <cstdint>
#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <utility>
#include <vector>

#include "kernels/accumulators.hpp"
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

// Computation of positions of masked elements

namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::kernels::accumulators::mask_positions_contig_impl_fn_ptr_t;
static mask_positions_contig_impl_fn_ptr_t
    mask_positions_contig_i64_dispatch_vector[td_ns::num_types];
static mask_positions_contig_impl_fn_ptr_t
    mask_positions_contig_i32_dispatch_vector[td_ns::num_types];

using dpctl::tensor::kernels::accumulators::
    mask_positions_strided_impl_fn_ptr_t;
static mask_positions_strided_impl_fn_ptr_t
    mask_positions_strided_i64_dispatch_vector[td_ns::num_types];
static mask_positions_strided_impl_fn_ptr_t
    mask_positions_strided_i32_dispatch_vector[td_ns::num_types];

void populate_mask_positions_dispatch_vectors(void)
{
    using dpctl::tensor::kernels::accumulators::
        MaskPositionsContigFactoryForInt64;
    td_ns::DispatchVectorBuilder<mask_positions_contig_impl_fn_ptr_t,
                                 MaskPositionsContigFactoryForInt64,
                                 td_ns::num_types>
        dvb1;
    dvb1.populate_dispatch_vector(mask_positions_contig_i64_dispatch_vector);

    using dpctl::tensor::kernels::accumulators::
        MaskPositionsContigFactoryForInt32;
    td_ns::DispatchVectorBuilder<mask_positions_contig_impl_fn_ptr_t,
                                 MaskPositionsContigFactoryForInt32,
                                 td_ns::num_types>
        dvb2;
    dvb2.populate_dispatch_vector(mask_positions_contig_i32_dispatch_vector);

    using dpctl::tensor::kernels::accumulators::
        MaskPositionsStridedFactoryForInt64;
    td_ns::DispatchVectorBuilder<mask_positions_strided_impl_fn_ptr_t,
                                 MaskPositionsStridedFactoryForInt64,
                                 td_ns::num_types>
        dvb3;
    dvb3.populate_dispatch_vector(mask_positions_strided_i64_dispatch_vector);

    using dpctl::tensor::kernels::accumulators::
        MaskPositionsStridedFactoryForInt32;
    td_ns::DispatchVectorBuilder<mask_positions_strided_impl_fn_ptr_t,
                                 MaskPositionsStridedFactoryForInt32,
                                 td_ns::num_types>
        dvb4;
    dvb4.populate_dispatch_vector(mask_positions_strided_i32_dispatch_vector);

    return;
}

size_t py_mask_positions(dpctl::tensor::usm_ndarray mask,
                         dpctl::tensor::usm_ndarray cumsum,
                         sycl::queue exec_q,
                         std::vector<sycl::event> const &depends)
{
    // cumsum is 1D
    if (cumsum.get_ndim() != 1) {
        throw py::value_error("Result array must be one-dimensional.");
    }

    if (!cumsum.is_c_contiguous()) {
        throw py::value_error("Expecting `cumsum` array must be C-contiguous.");
    }

    // cumsum.shape == (mask.size,)
    auto mask_size = mask.get_size();
    auto cumsum_size = cumsum.get_shape(0);
    if (cumsum_size != mask_size) {
        throw py::value_error("Inconsistent dimensions");
    }

    if (!dpctl::utils::queues_are_compatible(exec_q, {mask, cumsum})) {
        // FIXME: use ExecutionPlacementError
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    if (mask_size == 0) {
        return 0;
    }

    int mask_typenum = mask.get_typenum();
    int cumsum_typenum = cumsum.get_typenum();

    // mask can be any type
    const char *mask_data = mask.get_data();
    char *cumsum_data = cumsum.get_data();

    auto const &array_types = td_ns::usm_ndarray_types();

    int mask_typeid = array_types.typenum_to_lookup_id(mask_typenum);
    int cumsum_typeid = array_types.typenum_to_lookup_id(cumsum_typenum);

    // cumsum must be int32_t/int64_t only
    constexpr int int32_typeid = static_cast<int>(td_ns::typenum_t::INT32);
    constexpr int int64_typeid = static_cast<int>(td_ns::typenum_t::INT64);
    if (cumsum_typeid != int32_typeid && cumsum_typeid != int64_typeid) {
        throw py::value_error(
            "Cumulative sum array must have int32 or int64 data-type.");
    }

    const bool use_i32 = (cumsum_typeid == int32_typeid);

    if (mask.is_c_contiguous()) {
        auto fn = (use_i32)
                      ? mask_positions_contig_i32_dispatch_vector[mask_typeid]
                      : mask_positions_contig_i64_dispatch_vector[mask_typeid];

        return fn(exec_q, mask_size, mask_data, cumsum_data, depends);
    }

    const py::ssize_t *shape = mask.get_shape_raw();
    auto const &strides_vector = mask.get_strides_vector();

    using shT = std::vector<py::ssize_t>;
    shT compact_shape;
    shT compact_strides;

    int mask_nd = mask.get_ndim();
    int nd = mask_nd;

    dpctl::tensor::py_internal::compact_iteration_space(
        nd, shape, strides_vector, compact_shape, compact_strides);

    // Strided implementation
    auto strided_fn =
        (use_i32) ? mask_positions_strided_i32_dispatch_vector[mask_typeid]
                  : mask_positions_strided_i64_dispatch_vector[mask_typeid];
    std::vector<sycl::event> host_task_events;

    using dpctl::tensor::offset_utils::device_allocate_and_pack;
    const auto &ptr_size_event_tuple = device_allocate_and_pack<py::ssize_t>(
        exec_q, host_task_events, compact_shape, compact_strides);
    py::ssize_t *shape_strides = std::get<0>(ptr_size_event_tuple);
    if (shape_strides == nullptr) {
        sycl::event::wait(host_task_events);
        throw std::runtime_error("Unexpected error");
    }
    sycl::event copy_shape_ev = std::get<2>(ptr_size_event_tuple);

    if (2 * static_cast<size_t>(nd) != std::get<1>(ptr_size_event_tuple)) {
        copy_shape_ev.wait();
        sycl::event::wait(host_task_events);
        sycl::free(shape_strides, exec_q);
        throw std::runtime_error("Unexpected error");
    }

    std::vector<sycl::event> dependent_events;
    dependent_events.reserve(depends.size() + 1);
    dependent_events.insert(dependent_events.end(), copy_shape_ev);
    dependent_events.insert(dependent_events.end(), depends.begin(),
                            depends.end());

    size_t total_set = strided_fn(exec_q, mask_size, mask_data, nd,
                                  shape_strides, cumsum_data, dependent_events);

    sycl::event::wait(host_task_events);
    sycl::free(shape_strides, exec_q);

    return total_set;
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
