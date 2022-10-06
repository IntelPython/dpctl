//===----------- Implementation of _tensor_impl module  ---------*-C++-*-/===//
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
#include <utility>
#include <vector>

#include "copy_for_reshape.hpp"
#include "dpctl4pybind11.hpp"
#include "kernels/copy_and_cast.hpp"
#include "utils/type_dispatch.hpp"
#include <pybind11/pybind11.h>

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

namespace _ns = dpctl::tensor::detail;

using dpctl::tensor::kernels::copy_and_cast::copy_for_reshape_fn_ptr_t;
using dpctl::utils::keep_args_alive;

// define static vector
static copy_for_reshape_fn_ptr_t
    copy_for_reshape_generic_dispatch_vector[_ns::num_types];

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
copy_usm_ndarray_for_reshape(dpctl::tensor::usm_ndarray src,
                             dpctl::tensor::usm_ndarray dst,
                             py::ssize_t shift,
                             sycl::queue exec_q,
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

    // destination must be ample enough to accomodate all elements
    {
        auto dst_offsets = dst.get_minmax_offsets();
        py::ssize_t range =
            static_cast<py::ssize_t>(dst_offsets.second - dst_offsets.first);
        if (range + 1 < src_nelems) {
            throw py::value_error(
                "Destination array can not accomodate all the "
                "elements of source array.");
        }
    }

    // check same contexts
    sycl::queue src_q = src.get_queue();
    sycl::queue dst_q = dst.get_queue();

    if (!dpctl::utils::queues_are_compatible(exec_q, {src_q, dst_q})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    if (src_nelems == 1) {
        // handle special case of 1-element array
        int src_elemsize = src.get_elemsize();
        char *src_data = src.get_data();
        char *dst_data = dst.get_data();
        sycl::event copy_ev =
            exec_q.copy<char>(src_data, dst_data, src_elemsize);
        return std::make_pair(keep_args_alive(exec_q, {src, dst}, {copy_ev}),
                              copy_ev);
    }

    // dimensions may be different
    int src_nd = src.get_ndim();
    int dst_nd = dst.get_ndim();

    const py::ssize_t *src_shape = src.get_shape_raw();
    const py::ssize_t *dst_shape = dst.get_shape_raw();

    auto array_types = dpctl::tensor::detail::usm_ndarray_types();
    int type_id = array_types.typenum_to_lookup_id(src_typenum);

    auto fn = copy_for_reshape_generic_dispatch_vector[type_id];

    // packed_shape_strides = [src_shape, src_strides, dst_shape, dst_strides]
    py::ssize_t *packed_shapes_strides =
        sycl::malloc_device<py::ssize_t>(2 * (src_nd + dst_nd), exec_q);

    if (packed_shapes_strides == nullptr) {
        throw std::runtime_error("Unabled to allocate device memory");
    }

    using usm_host_allocatorT =
        sycl::usm_allocator<py::ssize_t, sycl::usm::alloc::host>;
    using shT = std::vector<py::ssize_t, usm_host_allocatorT>;
    usm_host_allocatorT allocator(exec_q);
    std::shared_ptr<shT> packed_host_shapes_strides_shp =
        std::make_shared<shT>(2 * (src_nd + dst_nd), allocator);

    std::copy(src_shape, src_shape + src_nd,
              packed_host_shapes_strides_shp->begin());
    std::copy(dst_shape, dst_shape + dst_nd,
              packed_host_shapes_strides_shp->begin() + 2 * src_nd);

    const py::ssize_t *src_strides = src.get_strides_raw();
    if (src_strides == nullptr) {
        if (src.is_c_contiguous()) {
            const auto &src_contig_strides =
                c_contiguous_strides(src_nd, src_shape);
            std::copy(src_contig_strides.begin(), src_contig_strides.end(),
                      packed_host_shapes_strides_shp->begin() + src_nd);
        }
        else if (src.is_f_contiguous()) {
            const auto &src_contig_strides =
                f_contiguous_strides(src_nd, src_shape);
            std::copy(src_contig_strides.begin(), src_contig_strides.end(),
                      packed_host_shapes_strides_shp->begin() + src_nd);
        }
        else {
            sycl::free(packed_shapes_strides, exec_q);
            throw std::runtime_error(
                "Invalid src array encountered: in copy_for_reshape function");
        }
    }
    else {
        std::copy(src_strides, src_strides + src_nd,
                  packed_host_shapes_strides_shp->begin() + src_nd);
    }

    const py::ssize_t *dst_strides = dst.get_strides_raw();
    if (dst_strides == nullptr) {
        if (dst.is_c_contiguous()) {
            const auto &dst_contig_strides =
                c_contiguous_strides(dst_nd, dst_shape);
            std::copy(dst_contig_strides.begin(), dst_contig_strides.end(),
                      packed_host_shapes_strides_shp->begin() + 2 * src_nd +
                          dst_nd);
        }
        else if (dst.is_f_contiguous()) {
            const auto &dst_contig_strides =
                f_contiguous_strides(dst_nd, dst_shape);
            std::copy(dst_contig_strides.begin(), dst_contig_strides.end(),
                      packed_host_shapes_strides_shp->begin() + 2 * src_nd +
                          dst_nd);
        }
        else {
            sycl::free(packed_shapes_strides, exec_q);
            throw std::runtime_error(
                "Invalid dst array encountered: in copy_for_reshape function");
        }
    }
    else {
        std::copy(dst_strides, dst_strides + dst_nd,
                  packed_host_shapes_strides_shp->begin() + 2 * src_nd +
                      dst_nd);
    }

    // copy packed shapes and strides from host to devices
    sycl::event packed_shape_strides_copy_ev = exec_q.copy<py::ssize_t>(
        packed_host_shapes_strides_shp->data(), packed_shapes_strides,
        packed_host_shapes_strides_shp->size());
    exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(packed_shape_strides_copy_ev);
        cgh.host_task([packed_host_shapes_strides_shp] {
            // Capturing shared pointer ensures that the underlying vector is
            // not destroyed until after its data are copied into packed USM
            // vector
        });
    });

    char *src_data = src.get_data();
    char *dst_data = dst.get_data();

    std::vector<sycl::event> all_deps(depends.size() + 1);
    all_deps.push_back(packed_shape_strides_copy_ev);
    all_deps.insert(std::end(all_deps), std::begin(depends), std::end(depends));

    sycl::event copy_for_reshape_event =
        fn(exec_q, shift, src_nelems, src_nd, dst_nd, packed_shapes_strides,
           src_data, dst_data, all_deps);

    exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(copy_for_reshape_event);
        auto ctx = exec_q.get_context();
        cgh.host_task([packed_shapes_strides, ctx]() {
            sycl::free(packed_shapes_strides, ctx);
        });
    });

    return std::make_pair(
        keep_args_alive(exec_q, {src, dst}, {copy_for_reshape_event}),
        copy_for_reshape_event);
}

void init_copy_for_reshape_dispatch_vectors(void)
{
    using namespace dpctl::tensor::detail;
    using dpctl::tensor::kernels::copy_and_cast::CopyForReshapeGenericFactory;

    DispatchVectorBuilder<copy_for_reshape_fn_ptr_t,
                          CopyForReshapeGenericFactory, num_types>
        dvb;
    dvb.populate_dispatch_vector(copy_for_reshape_generic_dispatch_vector);
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
