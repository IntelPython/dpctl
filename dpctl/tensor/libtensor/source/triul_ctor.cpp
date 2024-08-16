//===-- ------------ Implementation of _tensor_impl module  ----*-C++-*-/===//
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
//===--------------------------------------------------------------------===//
///
/// \file
/// This file defines functions of dpctl.tensor._tensor_impl extensions
//===--------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

#include "dpctl4pybind11.hpp"
#include <pybind11/pybind11.h>

#include "kernels/constructors.hpp"
#include "simplify_iteration_space.hpp"
#include "utils/memory_overlap.hpp"
#include "utils/offset_utils.hpp"
#include "utils/output_validation.hpp"
#include "utils/sycl_alloc_utils.hpp"
#include "utils/type_dispatch.hpp"

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

using dpctl::utils::keep_args_alive;

using dpctl::tensor::kernels::constructors::tri_fn_ptr_t;

static tri_fn_ptr_t tril_generic_dispatch_vector[td_ns::num_types];
static tri_fn_ptr_t triu_generic_dispatch_vector[td_ns::num_types];

std::pair<sycl::event, sycl::event>
usm_ndarray_triul(sycl::queue &exec_q,
                  const dpctl::tensor::usm_ndarray &src,
                  const dpctl::tensor::usm_ndarray &dst,
                  char part,
                  py::ssize_t k = 0,
                  const std::vector<sycl::event> &depends = {})
{
    // array dimensions must be the same
    int src_nd = src.get_ndim();
    int dst_nd = dst.get_ndim();
    if (src_nd != dst_nd) {
        throw py::value_error("Array dimensions are not the same.");
    }

    if (src_nd < 2) {
        throw py::value_error("Array dimensions less than 2.");
    }

    // shapes must be the same
    const py::ssize_t *src_shape = src.get_shape_raw();
    const py::ssize_t *dst_shape = dst.get_shape_raw();

    bool shapes_equal(true);
    size_t src_nelems(1);

    for (int i = 0; shapes_equal && i < src_nd; ++i) {
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

    char *src_data = src.get_data();
    char *dst_data = dst.get_data();

    // check that arrays do not overlap, and concurrent copying is safe.
    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(src, dst)) {
        // TODO: could use a temporary, but this is done by the caller
        throw py::value_error("Arrays index overlapping segments of memory");
    }

    auto array_types = td_ns::usm_ndarray_types();

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();
    int src_typeid = array_types.typenum_to_lookup_id(src_typenum);
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    if (dst_typeid != src_typeid) {
        throw py::value_error("Array dtype are not the same.");
    }

    // check same queues
    if (!dpctl::utils::queues_are_compatible(exec_q, {src, dst})) {
        throw py::value_error(
            "Execution queue context is not the same as allocation contexts");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    auto src_strides = src.get_strides_vector();
    auto dst_strides = dst.get_strides_vector();

    using shT = std::vector<py::ssize_t>;
    shT simplified_shape;
    shT simplified_src_strides;
    shT simplified_dst_strides;
    py::ssize_t src_offset(0);
    py::ssize_t dst_offset(0);

    int nd = src_nd - 2;
    const py::ssize_t *shape = src_shape;

    const shT iter_src_strides(std::begin(src_strides),
                               std::begin(src_strides) + nd);
    const shT iter_dst_strides(std::begin(dst_strides),
                               std::begin(dst_strides) + nd);

    simplify_iteration_space(nd, shape, iter_src_strides, iter_dst_strides,
                             // output
                             simplified_shape, simplified_src_strides,
                             simplified_dst_strides, src_offset, dst_offset);

    if (src_offset != 0 || dst_offset != 0) {
        throw py::value_error("Reversed slice for dst is not supported");
    }

    nd += 2;

    using usm_host_allocatorT =
        dpctl::tensor::alloc_utils::usm_host_allocator<py::ssize_t>;
    using usmshT = std::vector<py::ssize_t, usm_host_allocatorT>;

    usm_host_allocatorT allocator(exec_q);
    auto shp_host_shape_and_strides =
        std::make_shared<usmshT>(3 * nd, allocator);

    std::copy(simplified_shape.begin(), simplified_shape.end(),
              shp_host_shape_and_strides->begin());
    (*shp_host_shape_and_strides)[nd - 2] = src_shape[src_nd - 2];
    (*shp_host_shape_and_strides)[nd - 1] = src_shape[src_nd - 1];

    std::copy(simplified_src_strides.begin(), simplified_src_strides.end(),
              shp_host_shape_and_strides->begin() + nd);
    (*shp_host_shape_and_strides)[2 * nd - 2] = src_strides[src_nd - 2];
    (*shp_host_shape_and_strides)[2 * nd - 1] = src_strides[src_nd - 1];

    std::copy(simplified_dst_strides.begin(), simplified_dst_strides.end(),
              shp_host_shape_and_strides->begin() + 2 * nd);
    (*shp_host_shape_and_strides)[3 * nd - 2] = dst_strides[src_nd - 2];
    (*shp_host_shape_and_strides)[3 * nd - 1] = dst_strides[src_nd - 1];

    py::ssize_t *dev_shape_and_strides =
        sycl::malloc_device<py::ssize_t>(3 * nd, exec_q);
    if (dev_shape_and_strides == nullptr) {
        throw std::runtime_error("Unabled to allocate device memory");
    }
    const sycl::event &copy_shape_and_strides = exec_q.copy<py::ssize_t>(
        shp_host_shape_and_strides->data(), dev_shape_and_strides, 3 * nd);

    py::ssize_t inner_range = src_shape[src_nd - 1] * src_shape[src_nd - 2];
    py::ssize_t outer_range = src_nelems / inner_range;

    sycl::event tri_ev;
    if (part == 'l') {
        auto fn = tril_generic_dispatch_vector[src_typeid];
        tri_ev =
            fn(exec_q, inner_range, outer_range, src_data, dst_data, nd,
               dev_shape_and_strides, k, depends, {copy_shape_and_strides});
    }
    else {
        auto fn = triu_generic_dispatch_vector[src_typeid];
        tri_ev =
            fn(exec_q, inner_range, outer_range, src_data, dst_data, nd,
               dev_shape_and_strides, k, depends, {copy_shape_and_strides});
    }

    const auto &temporaries_cleanup_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(tri_ev);
        const auto &ctx = exec_q.get_context();
        using dpctl::tensor::alloc_utils::sycl_free_noexcept;
        cgh.host_task(
            [shp_host_shape_and_strides, dev_shape_and_strides, ctx]() {
                // capture of shp_host_shape_and_strides ensure the underlying
                // vector exists for the entire execution of copying kernel
                sycl_free_noexcept(dev_shape_and_strides, ctx);
            });
    });

    return std::make_pair(
        keep_args_alive(exec_q, {src, dst}, {temporaries_cleanup_ev}), tri_ev);
}

void init_triul_ctor_dispatch_vectors(void)
{

    using namespace td_ns;
    using dpctl::tensor::kernels::constructors::TrilGenericFactory;
    using dpctl::tensor::kernels::constructors::TriuGenericFactory;

    DispatchVectorBuilder<tri_fn_ptr_t, TrilGenericFactory, num_types> dvb1;
    dvb1.populate_dispatch_vector(tril_generic_dispatch_vector);

    DispatchVectorBuilder<tri_fn_ptr_t, TriuGenericFactory, num_types> dvb2;
    dvb2.populate_dispatch_vector(triu_generic_dispatch_vector);
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
