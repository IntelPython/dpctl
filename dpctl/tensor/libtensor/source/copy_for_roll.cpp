//===----------- Implementation of _tensor_impl module  ---------*-C++-*-/===//
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
/// This file defines functions of dpctl.tensor._tensor_impl extensions
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

#include "copy_for_roll.hpp"
#include "dpctl4pybind11.hpp"
#include "kernels/copy_and_cast.hpp"
#include "utils/output_validation.hpp"
#include "utils/type_dispatch.hpp"
#include <pybind11/pybind11.h>

#include "simplify_iteration_space.hpp"

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::kernels::copy_and_cast::copy_for_roll_contig_fn_ptr_t;
using dpctl::tensor::kernels::copy_and_cast::
    copy_for_roll_ndshift_strided_fn_ptr_t;
using dpctl::tensor::kernels::copy_and_cast::copy_for_roll_strided_fn_ptr_t;
using dpctl::utils::keep_args_alive;

// define static vector
static copy_for_roll_strided_fn_ptr_t
    copy_for_roll_strided_dispatch_vector[td_ns::num_types];

static copy_for_roll_contig_fn_ptr_t
    copy_for_roll_contig_dispatch_vector[td_ns::num_types];

static copy_for_roll_ndshift_strided_fn_ptr_t
    copy_for_roll_ndshift_dispatch_vector[td_ns::num_types];

/*
 * Copies src into dst (same data type) of different shapes by using flat
 * iterations.
 *
 * Equivalent to the following loop:
 *
 * for i for range(src.size):
 *     dst[np.multi_index(i, dst.shape)] = src[np.multi_index(i, src.shape)]
 */
std::pair<sycl::event, sycl::event>
copy_usm_ndarray_for_roll_1d(const dpctl::tensor::usm_ndarray &src,
                             const dpctl::tensor::usm_ndarray &dst,
                             py::ssize_t shift,
                             sycl::queue &exec_q,
                             const std::vector<sycl::event> &depends)
{
    int src_nd = src.get_ndim();
    int dst_nd = dst.get_ndim();

    // Must have the same number of dimensions
    if (src_nd != dst_nd) {
        throw py::value_error(
            "copy_usm_ndarray_for_roll_1d requires src and dst to "
            "have the same number of dimensions.");
    }

    const py::ssize_t *src_shape_ptr = src.get_shape_raw();
    const py::ssize_t *dst_shape_ptr = dst.get_shape_raw();

    if (!std::equal(src_shape_ptr, src_shape_ptr + src_nd, dst_shape_ptr)) {
        throw py::value_error(
            "copy_usm_ndarray_for_roll_1d requires src and dst to "
            "have the same shape.");
    }

    py::ssize_t src_nelems = src.get_size();

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    // typenames must be the same
    if (src_typenum != dst_typenum) {
        throw py::value_error(
            "copy_usm_ndarray_for_roll_1d requires src and dst to "
            "have the same type.");
    }

    if (src_nelems == 0) {
        return std::make_pair(sycl::event(), sycl::event());
    }

    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(dst, src_nelems);

    // check same contexts
    if (!dpctl::utils::queues_are_compatible(exec_q, {src, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    if (src_nelems == 1) {
        // handle special case of 1-element array
        int src_elemsize = src.get_elemsize();
        const char *src_data = src.get_data();
        char *dst_data = dst.get_data();
        sycl::event copy_ev =
            exec_q.copy<char>(src_data, dst_data, src_elemsize);
        return std::make_pair(keep_args_alive(exec_q, {src, dst}, {copy_ev}),
                              copy_ev);
    }

    auto array_types = td_ns::usm_ndarray_types();
    int type_id = array_types.typenum_to_lookup_id(src_typenum);

    const bool is_src_c_contig = src.is_c_contiguous();
    const bool is_src_f_contig = src.is_f_contiguous();

    const bool is_dst_c_contig = dst.is_c_contiguous();
    const bool is_dst_f_contig = dst.is_f_contiguous();

    const bool both_c_contig = is_src_c_contig && is_dst_c_contig;
    const bool both_f_contig = is_src_f_contig && is_dst_f_contig;

    // normalize shift parameter to be 0 <= offset < src_nelems
    size_t offset =
        (shift > 0) ? (shift % src_nelems) : src_nelems + (shift % src_nelems);

    const char *src_data = src.get_data();
    char *dst_data = dst.get_data();

    if (both_c_contig || both_f_contig) {
        auto fn = copy_for_roll_contig_dispatch_vector[type_id];

        if (fn != nullptr) {
            constexpr py::ssize_t zero_offset = 0;

            sycl::event copy_for_roll_ev =
                fn(exec_q, offset, src_nelems, src_data, zero_offset, dst_data,
                   zero_offset, depends);

            sycl::event ht_ev =
                keep_args_alive(exec_q, {src, dst}, {copy_for_roll_ev});

            return std::make_pair(ht_ev, copy_for_roll_ev);
        }
    }

    auto const &src_strides = src.get_strides_vector();
    auto const &dst_strides = dst.get_strides_vector();

    using shT = std::vector<py::ssize_t>;
    shT simplified_shape;
    shT simplified_src_strides;
    shT simplified_dst_strides;
    py::ssize_t src_offset(0);
    py::ssize_t dst_offset(0);

    int nd = src_nd;
    const py::ssize_t *shape = src_shape_ptr;

    // nd, simplified_* and *_offset are modified by reference
    dpctl::tensor::py_internal::simplify_iteration_space(
        nd, shape, src_strides, dst_strides,
        // output
        simplified_shape, simplified_src_strides, simplified_dst_strides,
        src_offset, dst_offset);

    if (nd == 1 && simplified_src_strides[0] == 1 &&
        simplified_dst_strides[0] == 1) {
        auto fn = copy_for_roll_contig_dispatch_vector[type_id];

        if (fn != nullptr) {

            sycl::event copy_for_roll_ev =
                fn(exec_q, offset, src_nelems, src_data, src_offset, dst_data,
                   dst_offset, depends);

            sycl::event ht_ev =
                keep_args_alive(exec_q, {src, dst}, {copy_for_roll_ev});

            return std::make_pair(ht_ev, copy_for_roll_ev);
        }
    }

    auto fn = copy_for_roll_strided_dispatch_vector[type_id];

    std::vector<sycl::event> host_task_events;
    host_task_events.reserve(2);

    // shape_strides = [src_shape, src_strides, dst_strides]
    using dpctl::tensor::offset_utils::device_allocate_and_pack;
    const auto &ptr_size_event_tuple = device_allocate_and_pack<py::ssize_t>(
        exec_q, host_task_events, simplified_shape, simplified_src_strides,
        simplified_dst_strides);

    py::ssize_t *shape_strides = std::get<0>(ptr_size_event_tuple);
    if (shape_strides == nullptr) {
        throw std::runtime_error("Unable to allocate device memory");
    }
    sycl::event copy_shape_ev = std::get<2>(ptr_size_event_tuple);

    std::vector<sycl::event> all_deps(depends.size() + 1);
    all_deps.push_back(copy_shape_ev);
    all_deps.insert(std::end(all_deps), std::begin(depends), std::end(depends));

    sycl::event copy_for_roll_event =
        fn(exec_q, offset, src_nelems, src_nd, shape_strides, src_data,
           src_offset, dst_data, dst_offset, all_deps);

    auto temporaries_cleanup_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(copy_for_roll_event);
        const auto &ctx = exec_q.get_context();
        cgh.host_task(
            [shape_strides, ctx]() { sycl::free(shape_strides, ctx); });
    });

    host_task_events.push_back(temporaries_cleanup_ev);

    return std::make_pair(keep_args_alive(exec_q, {src, dst}, host_task_events),
                          copy_for_roll_event);
}

std::pair<sycl::event, sycl::event>
copy_usm_ndarray_for_roll_nd(const dpctl::tensor::usm_ndarray &src,
                             const dpctl::tensor::usm_ndarray &dst,
                             const std::vector<py::ssize_t> &shifts,
                             sycl::queue &exec_q,
                             const std::vector<sycl::event> &depends)
{
    int src_nd = src.get_ndim();
    int dst_nd = dst.get_ndim();

    // Must have the same number of dimensions
    if (src_nd != dst_nd) {
        throw py::value_error(
            "copy_usm_ndarray_for_roll_nd requires src and dst to "
            "have the same number of dimensions.");
    }

    if (static_cast<size_t>(src_nd) != shifts.size()) {
        throw py::value_error(
            "copy_usm_ndarray_for_roll_nd requires shifts to "
            "contain an integral shift for each array dimension.");
    }

    const py::ssize_t *src_shape_ptr = src.get_shape_raw();
    const py::ssize_t *dst_shape_ptr = dst.get_shape_raw();

    if (!std::equal(src_shape_ptr, src_shape_ptr + src_nd, dst_shape_ptr)) {
        throw py::value_error(
            "copy_usm_ndarray_for_roll_nd requires src and dst to "
            "have the same shape.");
    }

    py::ssize_t src_nelems = src.get_size();

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    // typenames must be the same
    if (src_typenum != dst_typenum) {
        throw py::value_error(
            "copy_usm_ndarray_for_reshape requires src and dst to "
            "have the same type.");
    }

    if (src_nelems == 0) {
        return std::make_pair(sycl::event(), sycl::event());
    }

    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(dst, src_nelems);

    // check for compatible queues
    if (!dpctl::utils::queues_are_compatible(exec_q, {src, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    if (src_nelems == 1) {
        // handle special case of 1-element array
        int src_elemsize = src.get_elemsize();
        const char *src_data = src.get_data();
        char *dst_data = dst.get_data();
        sycl::event copy_ev =
            exec_q.copy<char>(src_data, dst_data, src_elemsize);
        return std::make_pair(keep_args_alive(exec_q, {src, dst}, {copy_ev}),
                              copy_ev);
    }

    auto array_types = td_ns::usm_ndarray_types();
    int type_id = array_types.typenum_to_lookup_id(src_typenum);

    std::vector<py::ssize_t> normalized_shifts{};
    normalized_shifts.reserve(src_nd);

    for (int i = 0; i < src_nd; ++i) {
        // normalize shift parameter to be 0 <= offset < dim
        py::ssize_t dim = src_shape_ptr[i];
        size_t offset =
            (shifts[i] > 0) ? (shifts[i] % dim) : dim + (shifts[i] % dim);

        normalized_shifts.push_back(offset);
    }

    const char *src_data = src.get_data();
    char *dst_data = dst.get_data();

    auto const &src_strides = src.get_strides_vector();
    auto const &dst_strides = dst.get_strides_vector();
    auto const &common_shape = src.get_shape_vector();

    constexpr py::ssize_t src_offset = 0;
    constexpr py::ssize_t dst_offset = 0;

    auto fn = copy_for_roll_ndshift_dispatch_vector[type_id];

    std::vector<sycl::event> host_task_events;
    host_task_events.reserve(2);

    // shape_strides = [src_shape, src_strides, dst_strides]
    using dpctl::tensor::offset_utils::device_allocate_and_pack;
    const auto &ptr_size_event_tuple = device_allocate_and_pack<py::ssize_t>(
        exec_q, host_task_events, common_shape, src_strides, dst_strides,
        normalized_shifts);

    py::ssize_t *shape_strides_shifts = std::get<0>(ptr_size_event_tuple);
    if (shape_strides_shifts == nullptr) {
        throw std::runtime_error("Unable to allocate device memory");
    }
    sycl::event copy_shape_ev = std::get<2>(ptr_size_event_tuple);

    std::vector<sycl::event> all_deps(depends.size() + 1);
    all_deps.push_back(copy_shape_ev);
    all_deps.insert(std::end(all_deps), std::begin(depends), std::end(depends));

    sycl::event copy_for_roll_event =
        fn(exec_q, src_nelems, src_nd, shape_strides_shifts, src_data,
           src_offset, dst_data, dst_offset, all_deps);

    auto temporaries_cleanup_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(copy_for_roll_event);
        const auto &ctx = exec_q.get_context();
        cgh.host_task([shape_strides_shifts, ctx]() {
            sycl::free(shape_strides_shifts, ctx);
        });
    });

    host_task_events.push_back(temporaries_cleanup_ev);

    return std::make_pair(keep_args_alive(exec_q, {src, dst}, host_task_events),
                          copy_for_roll_event);
}

void init_copy_for_roll_dispatch_vectors(void)
{
    using namespace td_ns;
    using dpctl::tensor::kernels::copy_and_cast::CopyForRollStridedFactory;

    DispatchVectorBuilder<copy_for_roll_strided_fn_ptr_t,
                          CopyForRollStridedFactory, num_types>
        dvb1;
    dvb1.populate_dispatch_vector(copy_for_roll_strided_dispatch_vector);

    using dpctl::tensor::kernels::copy_and_cast::CopyForRollContigFactory;
    DispatchVectorBuilder<copy_for_roll_contig_fn_ptr_t,
                          CopyForRollContigFactory, num_types>
        dvb2;
    dvb2.populate_dispatch_vector(copy_for_roll_contig_dispatch_vector);

    using dpctl::tensor::kernels::copy_and_cast::CopyForRollNDShiftFactory;
    DispatchVectorBuilder<copy_for_roll_ndshift_strided_fn_ptr_t,
                          CopyForRollNDShiftFactory, num_types>
        dvb3;
    dvb3.populate_dispatch_vector(copy_for_roll_ndshift_dispatch_vector);
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
