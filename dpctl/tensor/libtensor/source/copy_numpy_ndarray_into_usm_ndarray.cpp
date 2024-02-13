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
/// This file defines functions of dpctl.tensor._tensor_impl extensions
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <sycl/sycl.hpp>
#include <vector>

#include "dpctl4pybind11.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "kernels/copy_and_cast.hpp"
#include "utils/output_validation.hpp"
#include "utils/type_dispatch.hpp"

#include "copy_numpy_ndarray_into_usm_ndarray.hpp"
#include "simplify_iteration_space.hpp"

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

using dpctl::tensor::kernels::copy_and_cast::
    copy_and_cast_from_host_blocking_fn_ptr_t;

static copy_and_cast_from_host_blocking_fn_ptr_t
    copy_and_cast_from_host_blocking_dispatch_table[td_ns::num_types]
                                                   [td_ns::num_types];

void copy_numpy_ndarray_into_usm_ndarray(
    const py::array &npy_src,
    const dpctl::tensor::usm_ndarray &dst,
    sycl::queue &exec_q,
    const std::vector<sycl::event> &depends)
{
    int src_ndim = npy_src.ndim();
    int dst_ndim = dst.get_ndim();

    if (src_ndim != dst_ndim) {
        throw py::value_error("Source ndarray and destination usm_ndarray have "
                              "different array ranks, "
                              "i.e. different number of indices needed to "
                              "address array elements.");
    }

    const py::ssize_t *src_shape = npy_src.shape();
    const py::ssize_t *dst_shape = dst.get_shape_raw();
    bool shapes_equal(true);
    size_t src_nelems(1);
    for (int i = 0; i < src_ndim; ++i) {
        shapes_equal = shapes_equal && (src_shape[i] == dst_shape[i]);
        src_nelems *= static_cast<size_t>(src_shape[i]);
    }

    if (!shapes_equal) {
        throw py::value_error("Source ndarray and destination usm_ndarray have "
                              "difference shapes.");
    }

    if (src_nelems == 0) {
        // nothing to do
        return;
    }

    auto dst_offsets = dst.get_minmax_offsets();
    // destination must be ample enough to accommodate all elements of source
    // array
    {
        size_t range =
            static_cast<size_t>(dst_offsets.second - dst_offsets.first);
        if (range + 1 < src_nelems) {
            throw py::value_error(
                "Destination array can not accommodate all the "
                "elements of source array.");
        }
    }

    if (!dpctl::utils::queues_are_compatible(exec_q, {dst})) {
        throw py::value_error("Execution queue is not compatible with the "
                              "allocation queue");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    // here we assume that NumPy's type numbers agree with ours for types
    // supported in both
    int src_typenum =
        py::detail::array_descriptor_proxy(npy_src.dtype().ptr())->type_num;
    int dst_typenum = dst.get_typenum();

    auto array_types = td_ns::usm_ndarray_types();
    int src_type_id = array_types.typenum_to_lookup_id(src_typenum);
    int dst_type_id = array_types.typenum_to_lookup_id(dst_typenum);

    py::buffer_info src_pybuf = npy_src.request();
    const char *const src_data = static_cast<const char *const>(src_pybuf.ptr);
    char *dst_data = dst.get_data();

    int src_flags = npy_src.flags();

    // check for applicability of special cases:
    //      (same type && (both C-contiguous || both F-contiguous)
    bool both_c_contig =
        ((src_flags & py::array::c_style) && dst.is_c_contiguous());
    bool both_f_contig =
        ((src_flags & py::array::f_style) && dst.is_f_contiguous());
    if (both_c_contig || both_f_contig) {
        if (src_type_id == dst_type_id) {
            int src_elem_size = npy_src.itemsize();

            sycl::event copy_ev =
                exec_q.memcpy(static_cast<void *>(dst_data),
                              static_cast<const void *>(src_data),
                              src_nelems * src_elem_size, depends);

            // wait for copy_ev to complete
            copy_ev.wait_and_throw();

            return;
        }
    }

    auto const &dst_strides =
        dst.get_strides_vector(); // N.B.: strides in elements

    using shT = std::vector<py::ssize_t>;
    shT simplified_shape;
    shT simplified_src_strides;
    shT simplified_dst_strides;
    py::ssize_t src_offset(0);
    py::ssize_t dst_offset(0);

    int nd = src_ndim;
    const py::ssize_t *shape = src_shape;

    const py::ssize_t *src_strides_p =
        npy_src.strides();                         // N.B.: strides in bytes
    py::ssize_t src_itemsize = npy_src.itemsize(); // item size in bytes

    bool is_src_c_contig = ((src_flags & py::array::c_style) != 0);
    bool is_src_f_contig = ((src_flags & py::array::f_style) != 0);

    shT src_strides_in_elems;
    if (src_strides_p) {
        src_strides_in_elems.resize(nd);
        // copy and convert strides from bytes to elements
        std::transform(
            src_strides_p, src_strides_p + nd, std::begin(src_strides_in_elems),
            [src_itemsize](py::ssize_t el) { return el / src_itemsize; });
    }
    else {
        if (is_src_c_contig) {
            src_strides_in_elems =
                dpctl::tensor::c_contiguous_strides(nd, src_shape);
        }
        else if (is_src_f_contig) {
            src_strides_in_elems =
                dpctl::tensor::f_contiguous_strides(nd, src_shape);
        }
        else {
            throw py::value_error("NumPy source array has null strides but is "
                                  "neither C- nor F-contiguous.");
        }
    }

    // nd, simplified_* vectors and offsets are modified by reference
    simplify_iteration_space(nd, shape, src_strides_in_elems, dst_strides,
                             // outputs
                             simplified_shape, simplified_src_strides,
                             simplified_dst_strides, src_offset, dst_offset);

    assert(simplified_shape.size() == static_cast<size_t>(nd));
    assert(simplified_src_strides.size() == static_cast<size_t>(nd));
    assert(simplified_dst_strides.size() == static_cast<size_t>(nd));

    // handle nd == 0
    if (nd == 0) {
        nd = 1;
        simplified_shape.reserve(nd);
        simplified_shape.push_back(1);

        simplified_src_strides.reserve(nd);
        simplified_src_strides.push_back(1);

        simplified_dst_strides.reserve(nd);
        simplified_dst_strides.push_back(1);
    }

    // Minimum and maximum element offsets for source np.ndarray
    py::ssize_t npy_src_min_nelem_offset(src_offset);
    py::ssize_t npy_src_max_nelem_offset(src_offset);
    for (int i = 0; i < nd; ++i) {
        if (simplified_src_strides[i] < 0) {
            npy_src_min_nelem_offset +=
                simplified_src_strides[i] * (simplified_shape[i] - 1);
        }
        else {
            npy_src_max_nelem_offset +=
                simplified_src_strides[i] * (simplified_shape[i] - 1);
        }
    }

    std::vector<sycl::event> host_task_events;
    host_task_events.reserve(1);

    // Copy shape strides into device memory
    using dpctl::tensor::offset_utils::device_allocate_and_pack;
    const auto &ptr_size_event_tuple = device_allocate_and_pack<py::ssize_t>(
        exec_q, host_task_events, simplified_shape, simplified_src_strides,
        simplified_dst_strides);
    py::ssize_t *shape_strides = std::get<0>(ptr_size_event_tuple);
    if (shape_strides == nullptr) {
        throw std::runtime_error("Unable to allocate device memory");
    }
    const sycl::event &copy_shape_ev = std::get<2>(ptr_size_event_tuple);

    // Get implementation function pointer
    auto copy_and_cast_from_host_blocking_fn =
        copy_and_cast_from_host_blocking_dispatch_table[dst_type_id]
                                                       [src_type_id];

    copy_and_cast_from_host_blocking_fn(
        exec_q, src_nelems, nd, shape_strides, src_data, src_offset,
        npy_src_min_nelem_offset, npy_src_max_nelem_offset, dst_data,
        dst_offset, depends, {copy_shape_ev});

    sycl::free(shape_strides, exec_q);

    return;
}

void init_copy_numpy_ndarray_into_usm_ndarray_dispatch_tables(void)
{
    using namespace td_ns;
    using dpctl::tensor::kernels::copy_and_cast::CopyAndCastFromHostFactory;

    DispatchTableBuilder<copy_and_cast_from_host_blocking_fn_ptr_t,
                         CopyAndCastFromHostFactory, num_types>
        dtb_copy_from_numpy;

    dtb_copy_from_numpy.populate_dispatch_table(
        copy_and_cast_from_host_blocking_dispatch_table);
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
