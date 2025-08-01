//===----------- Implementation of _tensor_impl module  ---------*-C++-*-/===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2025 Intel Corporation
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

#include <stdexcept>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

#include "dpctl4pybind11.hpp"
#include <pybind11/pybind11.h>

#include "copy_for_reshape.hpp"
#include "kernels/copy_and_cast.hpp"
#include "utils/offset_utils.hpp"
#include "utils/output_validation.hpp"
#include "utils/sycl_alloc_utils.hpp"
#include "utils/type_dispatch.hpp"

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::kernels::copy_and_cast::copy_for_reshape_fn_ptr_t;
using dpctl::utils::keep_args_alive;

// define static vector
static copy_for_reshape_fn_ptr_t
    copy_for_reshape_generic_dispatch_vector[td_ns::num_types];

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
copy_usm_ndarray_for_reshape(const dpctl::tensor::usm_ndarray &src,
                             const dpctl::tensor::usm_ndarray &dst,
                             sycl::queue &exec_q,
                             const std::vector<sycl::event> &depends)
{
    py::ssize_t src_nelems = src.get_size();
    py::ssize_t dst_nelems = dst.get_size();

    // Must have the same number of elements
    if (src_nelems != dst_nelems) {
        throw py::value_error(
            "copy_usm_ndarray_for_reshape requires src and dst to "
            "have the same number of elements.");
    }

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
            exec_q.copy<char>(src_data, dst_data, src_elemsize, depends);
        return std::make_pair(keep_args_alive(exec_q, {src, dst}, {copy_ev}),
                              copy_ev);
    }

    // dimensions may be different
    int src_nd = src.get_ndim();
    int dst_nd = dst.get_ndim();

    auto array_types = td_ns::usm_ndarray_types();
    int type_id = array_types.typenum_to_lookup_id(src_typenum);

    auto fn = copy_for_reshape_generic_dispatch_vector[type_id];

    auto src_shape = src.get_shape_vector();
    auto src_strides = src.get_strides_vector();

    auto dst_shape = dst.get_shape_vector();
    auto dst_strides = dst.get_strides_vector();

    std::vector<sycl::event> host_task_events;
    host_task_events.reserve(2);

    // shape_strides = [src_shape, src_strides, dst_shape, dst_strides]
    using dpctl::tensor::offset_utils::device_allocate_and_pack;
    auto ptr_size_event_tuple = device_allocate_and_pack<py::ssize_t>(
        exec_q, host_task_events, src_shape, src_strides, dst_shape,
        dst_strides);
    auto copy_shape_ev = std::get<2>(ptr_size_event_tuple);
    auto shape_strides_owner = std::move(std::get<0>(ptr_size_event_tuple));
    const py::ssize_t *shape_strides = shape_strides_owner.get();

    const char *src_data = src.get_data();
    char *dst_data = dst.get_data();

    std::vector<sycl::event> all_deps(depends.size() + 1);
    all_deps.push_back(copy_shape_ev);
    all_deps.insert(std::end(all_deps), std::begin(depends), std::end(depends));

    sycl::event copy_for_reshape_event =
        fn(exec_q, src_nelems, src_nd, dst_nd, shape_strides, src_data,
           dst_data, all_deps);

    sycl::event temporaries_cleanup_ev =
        dpctl::tensor::alloc_utils::async_smart_free(
            exec_q, {copy_for_reshape_event}, shape_strides_owner);

    host_task_events.push_back(temporaries_cleanup_ev);

    return std::make_pair(keep_args_alive(exec_q, {src, dst}, host_task_events),
                          copy_for_reshape_event);
}

void init_copy_for_reshape_dispatch_vectors(void)
{
    using namespace td_ns;
    using dpctl::tensor::kernels::copy_and_cast::CopyForReshapeGenericFactory;

    DispatchVectorBuilder<copy_for_reshape_fn_ptr_t,
                          CopyForReshapeGenericFactory, num_types>
        dvb;
    dvb.populate_dispatch_vector(copy_for_reshape_generic_dispatch_vector);
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
