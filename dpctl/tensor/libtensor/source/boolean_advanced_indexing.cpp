//===-- boolean_advanced_indexing.cpp -                       --*-C++-*-/===//
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
/// This file defines implementation functions of dpctl.tensor.place and
/// dpctl.tensor.extract, dpctl.tensor.nonzero
//===----------------------------------------------------------------------===//

#include "dpctl4pybind11.hpp"
#include <CL/sycl.hpp>
#include <cstdint>
#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <utility>
#include <vector>

#include "boolean_advanced_indexing.hpp"
#include "kernels/boolean_advanced_indexing.hpp"
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

/* @brief Split shape/strides into dir1 (complementary to axis_start <= i <
 * axis_end) and dir2 (along given set of axes)
 */
template <typename shT>
void _split_iteration_space(const shT &shape_vec,
                            const shT &strides_vec,
                            int axis_start,
                            int axis_end,
                            shT &dir1_shape_vec,
                            shT &dir2_shape_vec,
                            shT &dir1_strides_vec,
                            shT &dir2_strides_vec)
{
    int nd = static_cast<int>(shape_vec.size());
    int dir2_sz = axis_end - axis_start;
    int dir1_sz = nd - dir2_sz;

    assert(dir1_sz > 0);
    assert(dir2_sz > 0);

    dir1_shape_vec.resize(dir1_sz);
    dir2_shape_vec.resize(dir2_sz);

    std::copy(shape_vec.begin(), shape_vec.begin() + axis_start,
              dir1_shape_vec.begin());
    std::copy(shape_vec.begin() + axis_end, shape_vec.end(),
              dir1_shape_vec.begin() + axis_start);

    std::copy(shape_vec.begin() + axis_start, shape_vec.begin() + axis_end,
              dir2_shape_vec.begin());

    dir1_strides_vec.resize(dir1_sz);
    dir2_strides_vec.resize(dir2_sz);

    std::copy(strides_vec.begin(), strides_vec.begin() + axis_start,
              dir1_strides_vec.begin());
    std::copy(strides_vec.begin() + axis_end, strides_vec.end(),
              dir1_strides_vec.begin() + axis_start);

    std::copy(strides_vec.begin() + axis_start, strides_vec.begin() + axis_end,
              dir2_strides_vec.begin());

    return;
}

// Computation of positions of masked elements

namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::kernels::indexing::mask_positions_contig_impl_fn_ptr_t;
static mask_positions_contig_impl_fn_ptr_t
    mask_positions_contig_dispatch_vector[td_ns::num_types];

using dpctl::tensor::kernels::indexing::mask_positions_strided_impl_fn_ptr_t;
static mask_positions_strided_impl_fn_ptr_t
    mask_positions_strided_dispatch_vector[td_ns::num_types];

void populate_mask_positions_dispatch_vectors(void)
{
    using dpctl::tensor::kernels::indexing::MaskPositionsContigFactory;
    td_ns::DispatchVectorBuilder<mask_positions_contig_impl_fn_ptr_t,
                                 MaskPositionsContigFactory, td_ns::num_types>
        dvb1;
    dvb1.populate_dispatch_vector(mask_positions_contig_dispatch_vector);

    using dpctl::tensor::kernels::indexing::MaskPositionsStridedFactory;
    td_ns::DispatchVectorBuilder<mask_positions_strided_impl_fn_ptr_t,
                                 MaskPositionsStridedFactory, td_ns::num_types>
        dvb2;
    dvb2.populate_dispatch_vector(mask_positions_strided_dispatch_vector);

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

    // cumsum must be int64_t only
    constexpr int int64_typeid = static_cast<int>(td_ns::typenum_t::INT64);
    if (cumsum_typeid != int64_typeid) {
        throw py::value_error(
            "Cumulative sum array must have int64 data-type.");
    }

    if (mask.is_c_contiguous()) {
        auto fn = mask_positions_contig_dispatch_vector[mask_typeid];

        return fn(exec_q, mask_size, mask_data, cumsum_data, depends);
    }

    const py::ssize_t *shape = mask.get_shape_raw();
    auto const &strides_vector = mask.get_strides_vector();

    using shT = std::vector<py::ssize_t>;
    shT simplified_shape;
    shT simplified_strides;
    py::ssize_t offset(0);

    int mask_nd = mask.get_ndim();
    int nd = mask_nd;

    dpctl::tensor::py_internal::simplify_iteration_space_1(
        nd, shape, strides_vector, simplified_shape, simplified_strides,
        offset);

    if (nd == 1 && simplified_strides[0] == 1) {
        auto fn = mask_positions_contig_dispatch_vector[mask_typeid];

        return fn(exec_q, mask_size, mask_data, cumsum_data, depends);
    }

    // Strided implementation
    auto strided_fn = mask_positions_strided_dispatch_vector[mask_typeid];
    std::vector<sycl::event> host_task_events;

    using dpctl::tensor::offset_utils::device_allocate_and_pack;
    const auto &ptr_size_event_tuple = device_allocate_and_pack<py::ssize_t>(
        exec_q, host_task_events, simplified_shape, simplified_strides);
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

    size_t total_set = strided_fn(exec_q, mask_size, mask_data, nd, offset,
                                  shape_strides, cumsum_data, dependent_events);

    sycl::event::wait(host_task_events);
    sycl::free(shape_strides, exec_q);

    return total_set;
}

// Masked extraction

using dpctl::tensor::kernels::indexing::
    masked_extract_all_slices_strided_impl_fn_ptr_t;

static masked_extract_all_slices_strided_impl_fn_ptr_t
    masked_extract_all_slices_strided_impl_dispatch_vector[td_ns::num_types];

using dpctl::tensor::kernels::indexing::
    masked_extract_some_slices_strided_impl_fn_ptr_t;

static masked_extract_some_slices_strided_impl_fn_ptr_t
    masked_extract_some_slices_strided_impl_dispatch_vector[td_ns::num_types];

void populate_masked_extract_dispatch_vectors(void)
{
    using dpctl::tensor::kernels::indexing::MaskExtractAllSlicesStridedFactory;
    td_ns::DispatchVectorBuilder<
        masked_extract_all_slices_strided_impl_fn_ptr_t,
        MaskExtractAllSlicesStridedFactory, td_ns::num_types>
        dvb1;
    dvb1.populate_dispatch_vector(
        masked_extract_all_slices_strided_impl_dispatch_vector);

    using dpctl::tensor::kernels::indexing::MaskExtractSomeSlicesStridedFactory;
    td_ns::DispatchVectorBuilder<
        masked_extract_some_slices_strided_impl_fn_ptr_t,
        MaskExtractSomeSlicesStridedFactory, td_ns::num_types>
        dvb2;
    dvb2.populate_dispatch_vector(
        masked_extract_some_slices_strided_impl_dispatch_vector);
}

std::pair<sycl::event, sycl::event>
py_extract(dpctl::tensor::usm_ndarray src,
           dpctl::tensor::usm_ndarray cumsum,
           int axis_start, // axis_start <= mask_i < axis_end
           int axis_end,
           dpctl::tensor::usm_ndarray dst,
           sycl::queue exec_q,
           std::vector<sycl::event> const &depends)
{
    int src_nd = src.get_ndim();
    if ((axis_start < 0 || axis_end > src_nd || axis_start >= axis_end)) {
        throw py::value_error("Specified axes_start and axes_end are invalid.");
    }
    int mask_span_sz = axis_end - axis_start;

    int dst_nd = dst.get_ndim();
    if (src_nd != dst_nd + (mask_span_sz - 1)) {
        throw py::value_error("Number of dimensions of source and destination "
                              "arrays is not consistent");
    }

    if (!cumsum.is_c_contiguous() || cumsum.get_ndim() != 1) {
        throw py::value_error("cumsum array must be a C-contiguous vector");
    }

    if (!dpctl::utils::queues_are_compatible(exec_q, {src, cumsum, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    py::ssize_t cumsum_sz = cumsum.get_size();

    const py::ssize_t *src_shape = src.get_shape_raw();
    const py::ssize_t *dst_shape = dst.get_shape_raw();
    bool same_ortho_dims(true);
    size_t ortho_nelems(1); // number of orthogonal iterations

    for (auto i = 0; i < axis_start; ++i) {
        auto src_sh_i = src_shape[i];
        ortho_nelems *= src_sh_i;
        same_ortho_dims = same_ortho_dims && (src_sh_i == dst_shape[i]);
    }
    for (auto i = axis_end; i < src_nd; ++i) {
        auto src_sh_i = src_shape[i];
        ortho_nelems *= src_sh_i;
        same_ortho_dims =
            same_ortho_dims && (src_sh_i == dst_shape[i - (mask_span_sz - 1)]);
    }

    size_t masked_src_nelems(1);
    size_t masked_dst_nelems(dst_shape[axis_start]);
    for (auto i = axis_start; i < axis_end; ++i) {
        masked_src_nelems *= src_shape[i];
    }

    // masked_dst_nelems is number of set elements in the mask, or last element
    // in cumsum
    if (!same_ortho_dims ||
        (masked_src_nelems != static_cast<size_t>(cumsum_sz))) {
        throw py::value_error("Inconsistent array dimensions");
    }

    // ensure that dst is sufficiently ample
    auto dst_offsets = dst.get_minmax_offsets();
    // destination must be ample enough to accomodate all elements
    {
        size_t range =
            static_cast<size_t>(dst_offsets.second - dst_offsets.first);
        if (range + 1 < static_cast<size_t>(ortho_nelems * masked_dst_nelems)) {
            throw py::value_error(
                "Memory addressed by the destination array can not "
                "accomodate all the "
                "array elements.");
        }
    }

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    // check that dst does not intersect with src, not with cumsum.
    if (overlap(dst, cumsum) || overlap(dst, src)) {
        throw py::value_error("Destination array overlaps with inputs");
    }

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();
    int cumsum_typenum = cumsum.get_typenum();

    auto const &array_types = td_ns::usm_ndarray_types();
    int src_typeid = array_types.typenum_to_lookup_id(src_typenum);
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);
    int cumsum_typeid = array_types.typenum_to_lookup_id(cumsum_typenum);

    constexpr int int64_typeid = static_cast<int>(td_ns::typenum_t::INT64);
    if (cumsum_typeid != int64_typeid) {
        throw py::value_error(
            "Unexact data type of cumsum array, expecting 'int64'");
    }

    if (src_typeid != dst_typeid) {
        throw py::value_error(
            "Destination array must have the same elemental data types");
    }

    char *src_data_p = src.get_data();
    char *dst_data_p = dst.get_data();
    char *cumsum_data_p = cumsum.get_data();

    auto src_shape_vec = src.get_shape_vector();
    auto src_strides_vec = src.get_strides_vector();

    auto dst_shape_vec = dst.get_shape_vector();
    auto dst_strides_vec = dst.get_strides_vector();

    sycl::event extract_ev;
    std::vector<sycl::event> host_task_events{};
    if (axis_start == 0 && axis_end == src_nd) {
        // empty orthogonal directions
        auto fn =
            masked_extract_all_slices_strided_impl_dispatch_vector[src_typeid];

        using dpctl::tensor::offset_utils::device_allocate_and_pack;
        const auto &ptr_size_event_tuple1 =
            device_allocate_and_pack<py::ssize_t>(
                exec_q, host_task_events, src_shape_vec, src_strides_vec);
        py::ssize_t *packed_src_shape_strides =
            std::get<0>(ptr_size_event_tuple1);
        if (packed_src_shape_strides == nullptr) {
            throw std::runtime_error("Unable to allocated device memory");
        }
        sycl::event copy_src_shape_strides_ev =
            std::get<2>(ptr_size_event_tuple1);

        assert(dst_shape_vec.size() == 1);
        assert(dst_strides_vec.size() == 1);

        std::vector<sycl::event> all_deps;
        all_deps.reserve(depends.size() + 1);
        all_deps.insert(all_deps.end(), depends.begin(), depends.end());
        all_deps.push_back(copy_src_shape_strides_ev);

        assert(all_deps.size() == depends.size() + 1);

        extract_ev = fn(exec_q, cumsum_sz, src_data_p, cumsum_data_p,
                        dst_data_p, src_nd, packed_src_shape_strides,
                        dst_shape_vec[0], dst_strides_vec[0], all_deps);

        sycl::event cleanup_tmp_allocations_ev =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(extract_ev);
                auto ctx = exec_q.get_context();
                cgh.host_task([ctx, packed_src_shape_strides] {
                    sycl::free(packed_src_shape_strides, ctx);
                });
            });
        host_task_events.push_back(cleanup_tmp_allocations_ev);
    }
    else {
        // non-empty othogonal directions
        auto fn =
            masked_extract_some_slices_strided_impl_dispatch_vector[src_typeid];

        int masked_src_nd = mask_span_sz;
        int ortho_nd = src_nd - masked_src_nd;

        using shT = std::vector<py::ssize_t>;

        shT ortho_src_shape;
        shT masked_src_shape;
        shT ortho_src_strides;
        shT masked_src_strides;
        _split_iteration_space(src_shape_vec, src_strides_vec, axis_start,
                               axis_end, ortho_src_shape,
                               masked_src_shape, // 4 vectors modified
                               ortho_src_strides, masked_src_strides);

        shT ortho_dst_shape;
        shT masked_dst_shape;
        shT ortho_dst_strides;
        shT masked_dst_strides;
        _split_iteration_space(dst_shape_vec, dst_strides_vec, axis_start,
                               axis_start + 1, ortho_dst_shape,
                               masked_dst_shape, // 4 vectors modified
                               ortho_dst_strides, masked_dst_strides);

        assert(ortho_src_shape.size() == static_cast<size_t>(ortho_nd));
        assert(ortho_dst_shape.size() == static_cast<size_t>(ortho_nd));
        assert(std::equal(ortho_src_shape.begin(), ortho_src_shape.end(),
                          ortho_dst_shape.begin()));

        std::vector<py::ssize_t> simplified_ortho_shape;
        std::vector<py::ssize_t> simplified_ortho_src_strides;
        std::vector<py::ssize_t> simplified_ortho_dst_strides;

        const py::ssize_t *_shape = ortho_src_shape.data();

        py::ssize_t ortho_src_offset(0);
        py::ssize_t ortho_dst_offset(0);

        dpctl::tensor::py_internal::simplify_iteration_space(
            ortho_nd, _shape, ortho_src_strides, ortho_dst_strides,
            // output
            simplified_ortho_shape, simplified_ortho_src_strides,
            simplified_ortho_dst_strides, ortho_src_offset, ortho_dst_offset);

        using dpctl::tensor::offset_utils::device_allocate_and_pack;
        const auto &ptr_size_event_tuple1 =
            device_allocate_and_pack<py::ssize_t>(
                exec_q, host_task_events, simplified_ortho_shape,
                simplified_ortho_src_strides, simplified_ortho_dst_strides);
        py::ssize_t *packed_ortho_src_dst_shape_strides =
            std::get<0>(ptr_size_event_tuple1);
        if (packed_ortho_src_dst_shape_strides == nullptr) {
            throw std::runtime_error("Unable to allocate device memory");
        }
        sycl::event copy_shape_strides_ev1 = std::get<2>(ptr_size_event_tuple1);

        const auto &ptr_size_event_tuple2 =
            device_allocate_and_pack<py::ssize_t>(
                exec_q, host_task_events, masked_src_shape, masked_src_strides);
        py::ssize_t *packed_masked_src_shape_strides =
            std::get<0>(ptr_size_event_tuple2);
        if (packed_masked_src_shape_strides == nullptr) {
            copy_shape_strides_ev1.wait();
            sycl::free(packed_ortho_src_dst_shape_strides, exec_q);
            throw std::runtime_error("Unable to allocate device memory");
        }
        sycl::event copy_shape_strides_ev2 = std::get<2>(ptr_size_event_tuple2);

        assert(masked_dst_shape.size() == 1);
        assert(masked_dst_strides.size() == 1);

        std::vector<sycl::event> all_deps;
        all_deps.reserve(depends.size() + 2);
        all_deps.insert(all_deps.end(), depends.begin(), depends.end());
        all_deps.push_back(copy_shape_strides_ev1);
        all_deps.push_back(copy_shape_strides_ev2);

        assert(all_deps.size() == depends.size() + 2);

        // OrthogIndexerT orthog_src_dst_indexer_, MaskedIndexerT
        // masked_src_indexer_, MaskedIndexerT masked_dst_indexer_
        extract_ev = fn(exec_q, ortho_nelems, masked_src_nelems, src_data_p,
                        cumsum_data_p, dst_data_p,
                        // data to build orthog_src_dst_indexer
                        ortho_nd, packed_ortho_src_dst_shape_strides,
                        ortho_src_offset, ortho_dst_offset,
                        // data to build masked_src_indexer
                        masked_src_nd, packed_masked_src_shape_strides,
                        // data to build masked_dst_indexer,
                        masked_dst_shape[0], masked_dst_strides[0], all_deps);

        sycl::event cleanup_tmp_allocations_ev =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(extract_ev);
                auto ctx = exec_q.get_context();
                cgh.host_task([ctx, packed_ortho_src_dst_shape_strides,
                               packed_masked_src_shape_strides] {
                    sycl::free(packed_ortho_src_dst_shape_strides, ctx);
                    sycl::free(packed_masked_src_shape_strides, ctx);
                });
            });
        host_task_events.push_back(cleanup_tmp_allocations_ev);
    }

    host_task_events.push_back(extract_ev);

    sycl::event py_obj_management_host_task_ev = dpctl::utils::keep_args_alive(
        exec_q, {src, cumsum, dst}, host_task_events);

    return std::make_pair(py_obj_management_host_task_ev, extract_ev);
}

// Masked placement

using dpctl::tensor::kernels::indexing::
    masked_place_all_slices_strided_impl_fn_ptr_t;

static masked_place_all_slices_strided_impl_fn_ptr_t
    masked_place_all_slices_strided_impl_dispatch_vector[td_ns::num_types];

using dpctl::tensor::kernels::indexing::
    masked_place_some_slices_strided_impl_fn_ptr_t;

static masked_place_some_slices_strided_impl_fn_ptr_t
    masked_place_some_slices_strided_impl_dispatch_vector[td_ns::num_types];

void populate_masked_place_dispatch_vectors(void)
{
    using dpctl::tensor::kernels::indexing::MaskPlaceAllSlicesStridedFactory;
    td_ns::DispatchVectorBuilder<masked_place_all_slices_strided_impl_fn_ptr_t,
                                 MaskPlaceAllSlicesStridedFactory,
                                 td_ns::num_types>
        dvb1;
    dvb1.populate_dispatch_vector(
        masked_place_all_slices_strided_impl_dispatch_vector);

    using dpctl::tensor::kernels::indexing::MaskPlaceSomeSlicesStridedFactory;
    td_ns::DispatchVectorBuilder<masked_place_some_slices_strided_impl_fn_ptr_t,
                                 MaskPlaceSomeSlicesStridedFactory,
                                 td_ns::num_types>
        dvb2;
    dvb2.populate_dispatch_vector(
        masked_place_some_slices_strided_impl_dispatch_vector);
}

/*
 * @brief Copy dst[i, ortho_id] = rhs[cumsum[i] - 1, ortho_id]  if cumsum[i] ==
 * ((i > 0) ? cumsum[i-1] + 1 : 1)
 */
std::pair<sycl::event, sycl::event>
py_place(dpctl::tensor::usm_ndarray dst,
         dpctl::tensor::usm_ndarray cumsum,
         int axis_start, // axis_start <= mask_i < axis_end
         int axis_end,
         dpctl::tensor::usm_ndarray rhs,
         sycl::queue exec_q,
         std::vector<sycl::event> const &depends)
{
    int dst_nd = dst.get_ndim();
    if ((axis_start < 0 || axis_end > dst_nd || axis_start >= axis_end)) {
        throw py::value_error("Specified axes_start and axes_end are invalid.");
    }
    int mask_span_sz = axis_end - axis_start;

    int rhs_nd = rhs.get_ndim();
    if (dst_nd != rhs_nd + (mask_span_sz - 1)) {
        throw py::value_error("Number of dimensions of source and destination "
                              "arrays is not consistent");
    }

    if (!cumsum.is_c_contiguous() || cumsum.get_ndim() != 1) {
        throw py::value_error("cumsum array must be a C-contiguous vector");
    }

    if (!dpctl::utils::queues_are_compatible(exec_q, {dst, cumsum, rhs})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    py::ssize_t cumsum_sz = cumsum.get_size();

    const py::ssize_t *dst_shape = dst.get_shape_raw();
    const py::ssize_t *rhs_shape = rhs.get_shape_raw();
    bool same_ortho_dims(true);
    size_t ortho_nelems(1); // number of orthogonal iterations

    for (auto i = 0; i < axis_start; ++i) {
        auto dst_sh_i = dst_shape[i];
        ortho_nelems *= dst_sh_i;
        same_ortho_dims = same_ortho_dims && (dst_sh_i == rhs_shape[i]);
    }
    for (auto i = axis_end; i < dst_nd; ++i) {
        auto dst_sh_i = dst_shape[i];
        ortho_nelems *= dst_sh_i;
        same_ortho_dims =
            same_ortho_dims && (dst_sh_i == rhs_shape[i - (mask_span_sz - 1)]);
    }

    size_t masked_dst_nelems(1);
    for (auto i = axis_start; i < axis_end; ++i) {
        masked_dst_nelems *= dst_shape[i];
    }

    if (!same_ortho_dims ||
        (masked_dst_nelems != static_cast<size_t>(cumsum_sz))) {
        throw py::value_error("Inconsistent array dimensions");
    }

    // ensure that dst is sufficiently ample
    auto dst_offsets = dst.get_minmax_offsets();
    // destination must be ample enough to accomodate all elements
    {
        size_t range =
            static_cast<size_t>(dst_offsets.second - dst_offsets.first);
        if (range + 1 < static_cast<size_t>(ortho_nelems * masked_dst_nelems)) {
            throw py::value_error(
                "Memory addressed by the destination array can not "
                "accomodate all the "
                "array elements.");
        }
    }

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    // check that dst does not intersect with src, not with cumsum.
    if (overlap(dst, rhs) || overlap(dst, cumsum)) {
        throw py::value_error("Destination array overlaps with inputs");
    }

    int dst_typenum = dst.get_typenum();
    int rhs_typenum = rhs.get_typenum();
    int cumsum_typenum = cumsum.get_typenum();

    auto const &array_types = td_ns::usm_ndarray_types();
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);
    int rhs_typeid = array_types.typenum_to_lookup_id(rhs_typenum);
    int cumsum_typeid = array_types.typenum_to_lookup_id(cumsum_typenum);

    constexpr int int64_typeid = static_cast<int>(td_ns::typenum_t::INT64);
    if (cumsum_typeid != int64_typeid) {
        throw py::value_error(
            "Unexact data type of cumsum array, expecting 'int64'");
    }

    // FIXME: should types be the same?
    if (dst_typeid != rhs_typeid) {
        throw py::value_error(
            "Destination array must have the same elemental data types");
    }

    char *dst_data_p = dst.get_data();
    char *rhs_data_p = rhs.get_data();
    char *cumsum_data_p = cumsum.get_data();

    auto dst_shape_vec = dst.get_shape_vector();
    auto dst_strides_vec = dst.get_strides_vector();

    auto rhs_shape_vec = rhs.get_shape_vector();
    auto rhs_strides_vec = rhs.get_strides_vector();

    sycl::event extract_ev;
    std::vector<sycl::event> host_task_events{};
    if (axis_start == 0 && axis_end == dst_nd) {
        // empty orthogonal directions
        auto fn =
            masked_place_all_slices_strided_impl_dispatch_vector[dst_typeid];

        using dpctl::tensor::offset_utils::device_allocate_and_pack;
        const auto &ptr_size_event_tuple1 =
            device_allocate_and_pack<py::ssize_t>(
                exec_q, host_task_events, dst_shape_vec, dst_strides_vec);
        py::ssize_t *packed_dst_shape_strides =
            std::get<0>(ptr_size_event_tuple1);
        if (packed_dst_shape_strides == nullptr) {
            throw std::runtime_error("Unable to allocate device memory");
        }
        sycl::event copy_dst_shape_strides_ev =
            std::get<2>(ptr_size_event_tuple1);

        assert(rhs_shape_vec.size() == 1);
        assert(rhs_strides_vec.size() == 1);

        std::vector<sycl::event> all_deps;
        all_deps.reserve(depends.size() + 1);
        all_deps.insert(all_deps.end(), depends.begin(), depends.end());
        all_deps.push_back(copy_dst_shape_strides_ev);

        assert(all_deps.size() == depends.size() + 1);

        extract_ev = fn(exec_q, cumsum_sz, dst_data_p, cumsum_data_p,
                        rhs_data_p, dst_nd, packed_dst_shape_strides,
                        rhs_shape_vec[0], rhs_strides_vec[0], all_deps);

        sycl::event cleanup_tmp_allocations_ev =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(extract_ev);
                auto ctx = exec_q.get_context();
                cgh.host_task([ctx, packed_dst_shape_strides] {
                    sycl::free(packed_dst_shape_strides, ctx);
                });
            });
        host_task_events.push_back(cleanup_tmp_allocations_ev);
    }
    else {
        // non-empty othogonal directions
        auto fn =
            masked_place_some_slices_strided_impl_dispatch_vector[dst_typeid];

        int masked_dst_nd = mask_span_sz;
        int ortho_nd = dst_nd - masked_dst_nd;

        using shT = std::vector<py::ssize_t>;

        shT ortho_dst_shape;
        shT masked_dst_shape;
        shT ortho_dst_strides;
        shT masked_dst_strides;
        _split_iteration_space(dst_shape_vec, dst_strides_vec, axis_start,
                               axis_end, ortho_dst_shape,
                               masked_dst_shape, // 4 vectors modified
                               ortho_dst_strides, masked_dst_strides);

        shT ortho_rhs_shape;
        shT masked_rhs_shape;
        shT ortho_rhs_strides;
        shT masked_rhs_strides;
        _split_iteration_space(rhs_shape_vec, rhs_strides_vec, axis_start,
                               axis_start + 1, ortho_rhs_shape,
                               masked_rhs_shape, // 4 vectors modified
                               ortho_rhs_strides, masked_rhs_strides);

        assert(ortho_dst_shape.size() == static_cast<size_t>(ortho_nd));
        assert(ortho_rhs_shape.size() == static_cast<size_t>(ortho_nd));
        assert(std::equal(ortho_dst_shape.begin(), ortho_dst_shape.end(),
                          ortho_rhs_shape.begin()));

        std::vector<py::ssize_t> simplified_ortho_shape;
        std::vector<py::ssize_t> simplified_ortho_dst_strides;
        std::vector<py::ssize_t> simplified_ortho_rhs_strides;

        const py::ssize_t *_shape = ortho_dst_shape.data();

        py::ssize_t ortho_dst_offset(0);
        py::ssize_t ortho_rhs_offset(0);

        dpctl::tensor::py_internal::simplify_iteration_space(
            ortho_nd, _shape, ortho_dst_strides, ortho_rhs_strides,
            simplified_ortho_shape, simplified_ortho_dst_strides,
            simplified_ortho_rhs_strides, ortho_dst_offset, ortho_rhs_offset);

        using dpctl::tensor::offset_utils::device_allocate_and_pack;
        const auto &ptr_size_event_tuple1 =
            device_allocate_and_pack<py::ssize_t>(
                exec_q, host_task_events, simplified_ortho_shape,
                simplified_ortho_dst_strides, simplified_ortho_rhs_strides);
        py::ssize_t *packed_ortho_dst_rhs_shape_strides =
            std::get<0>(ptr_size_event_tuple1);
        if (packed_ortho_dst_rhs_shape_strides == nullptr) {
            throw std::runtime_error("Unable to allocate device memory");
        }
        sycl::event copy_shape_strides_ev1 = std::get<2>(ptr_size_event_tuple1);

        auto ptr_size_event_tuple2 = device_allocate_and_pack<py::ssize_t>(
            exec_q, host_task_events, masked_dst_shape, masked_dst_strides);
        py::ssize_t *packed_masked_dst_shape_strides =
            std::get<0>(ptr_size_event_tuple2);
        if (packed_masked_dst_shape_strides == nullptr) {
            copy_shape_strides_ev1.wait();
            sycl::free(packed_ortho_dst_rhs_shape_strides, exec_q);
            throw std::runtime_error("Unable to allocate device memory");
        }
        sycl::event copy_shape_strides_ev2 = std::get<2>(ptr_size_event_tuple2);

        assert(masked_rhs_shape.size() == 1);
        assert(masked_rhs_strides.size() == 1);

        std::vector<sycl::event> all_deps;
        all_deps.reserve(depends.size() + 2);
        all_deps.insert(all_deps.end(), depends.begin(), depends.end());
        all_deps.push_back(copy_shape_strides_ev1);
        all_deps.push_back(copy_shape_strides_ev2);

        assert(all_deps.size() == depends.size() + 2);

        extract_ev = fn(exec_q, ortho_nelems, masked_dst_nelems, dst_data_p,
                        cumsum_data_p, rhs_data_p,
                        // data to build orthog_dst_rhs_indexer
                        ortho_nd, packed_ortho_dst_rhs_shape_strides,
                        ortho_dst_offset, ortho_rhs_offset,
                        // data to build masked_dst_indexer
                        masked_dst_nd, packed_masked_dst_shape_strides,
                        // data to build masked_dst_indexer,
                        masked_rhs_shape[0], masked_rhs_strides[0], all_deps);

        sycl::event cleanup_tmp_allocations_ev =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(extract_ev);
                auto ctx = exec_q.get_context();
                cgh.host_task([ctx, packed_ortho_dst_rhs_shape_strides,
                               packed_masked_dst_shape_strides] {
                    sycl::free(packed_ortho_dst_rhs_shape_strides, ctx);
                    sycl::free(packed_masked_dst_shape_strides, ctx);
                });
            });
        host_task_events.push_back(cleanup_tmp_allocations_ev);
    }

    host_task_events.push_back(extract_ev);

    sycl::event py_obj_management_host_task_ev = dpctl::utils::keep_args_alive(
        exec_q, {dst, cumsum, rhs}, host_task_events);

    return std::make_pair(py_obj_management_host_task_ev, extract_ev);
}

// Non-zero

std::pair<sycl::event, sycl::event> py_nonzero(
    dpctl::tensor::usm_ndarray cumsum,  // int64 input array, 1D, C-contiguous
    dpctl::tensor::usm_ndarray indexes, // int64 2D output array, C-contiguous
    std::vector<py::ssize_t>
        mask_shape, // shape of array from which cumsum was computed
    sycl::queue exec_q,
    std::vector<sycl::event> const &depends)
{
    if (!dpctl::utils::queues_are_compatible(exec_q, {cumsum, indexes})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    int cumsum_nd = cumsum.get_ndim();
    if (cumsum_nd != 1 || !cumsum.is_c_contiguous()) {
        throw py::value_error("Cumsum array must be a C-contiguous vector");
    }

    int indexes_nd = indexes.get_ndim();
    if (indexes_nd != 2 || !indexes.is_c_contiguous()) {
        throw py::value_error("Index array must be a C-contiguous matrix");
    }

    size_t _ndim = mask_shape.size();
    if (_ndim > std::numeric_limits<int>::max()) {
        throw py::value_error("Shape is too large");
    }
    int ndim = static_cast<int>(_ndim);

    const py::ssize_t *indexes_shape = indexes.get_shape_raw();

    if (ndim != indexes_shape[0]) {
        throw py::value_error(
            "Length of shape must equal width of index matrix");
    }

    auto cumsum_sz = cumsum.get_size();
    py::ssize_t shape_nelems =
        std::accumulate(mask_shape.begin(), mask_shape.end(), py::ssize_t(1),
                        std::multiplies<py::ssize_t>());

    if (cumsum_sz != shape_nelems) {
        throw py::value_error("Shape and cumsum size are not constent");
    }

    py::ssize_t nz_elems = indexes_shape[1];

    int indexes_typenum = indexes.get_typenum();
    auto const &array_types = td_ns::usm_ndarray_types();
    int indexes_typeid = array_types.typenum_to_lookup_id(indexes_typenum);

    int cumsum_typenum = cumsum.get_typenum();
    int cumsum_typeid = array_types.typenum_to_lookup_id(cumsum_typenum);

    // cumsum must be int64_t only
    constexpr int int64_typeid = static_cast<int>(td_ns::typenum_t::INT64);
    if (cumsum_typeid != int64_typeid || indexes_typeid != int64_typeid) {
        throw py::value_error(
            "Cumulative sum array and index array must have int64 data-type");
    }

    if (cumsum_sz == 0) {
        return std::make_pair(sycl::event(), sycl::event());
    }

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(cumsum, indexes)) {
        throw py::value_error("Arrays are expected to ave no memory overlap");
    }

    // ensure that dst is sufficiently ample
    auto indexes_offsets = indexes.get_minmax_offsets();
    // destination must be ample enough to accomodate all elements
    {
        size_t range =
            static_cast<size_t>(indexes_offsets.second - indexes_offsets.first);
        if (range + 1 < static_cast<size_t>(nz_elems * _ndim)) {
            throw py::value_error(
                "Memory addressed by the destination array can not "
                "accomodate all the array elements.");
        }
    }

    std::vector<sycl::event> host_task_events;
    host_task_events.reserve(2);

    using dpctl::tensor::offset_utils::device_allocate_and_pack;
    const auto &mask_shape_copying_tuple =
        device_allocate_and_pack<py::ssize_t>(exec_q, host_task_events,
                                              mask_shape);
    py::ssize_t *src_shape_device_ptr = std::get<0>(mask_shape_copying_tuple);
    if (src_shape_device_ptr == nullptr) {
        sycl::event::wait(host_task_events);
        throw std::runtime_error("Device allocation failed");
    }
    sycl::event copy_ev = std::get<2>(mask_shape_copying_tuple);

    std::vector<sycl::event> all_deps;
    all_deps.reserve(depends.size() + 1);

    all_deps.insert(all_deps.end(), depends.begin(), depends.end());
    all_deps.push_back(copy_ev);

    using dpctl::tensor::kernels::indexing::non_zero_indexes_impl;

    sycl::event non_zero_indexes_ev =
        non_zero_indexes_impl<std::int64_t, std::int64_t>(
            exec_q, cumsum_sz, nz_elems, ndim, cumsum.get_data(),
            indexes.get_data(), src_shape_device_ptr, all_deps);

    sycl::event temporaries_cleanup_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(non_zero_indexes_ev);
        auto ctx = exec_q.get_context();
        cgh.host_task([ctx, src_shape_device_ptr] {
            sycl::free(src_shape_device_ptr, ctx);
        });
    });
    host_task_events.push_back(temporaries_cleanup_ev);

    sycl::event py_obj_management_host_task_ev = dpctl::utils::keep_args_alive(
        exec_q, {cumsum, indexes}, host_task_events);

    return std::make_pair(py_obj_management_host_task_ev, non_zero_indexes_ev);
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
