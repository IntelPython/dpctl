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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <thread>
#include <type_traits>

#include "dpctl4pybind11.hpp"
#include "kernels/constructors.hpp"
#include "kernels/copy_and_cast.hpp"
#include "utils/strided_iters.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

#include "copy_and_cast_usm_to_usm.hpp"
#include "copy_for_reshape.hpp"
#include "copy_numpy_ndarray_into_usm_ndarray.hpp"
#include "linear_sequences.hpp"
#include "simplify_iteration_space.hpp"

namespace py = pybind11;
namespace _ns = dpctl::tensor::detail;

namespace
{

using dpctl::tensor::c_contiguous_strides;
using dpctl::tensor::f_contiguous_strides;

using dpctl::utils::keep_args_alive;

using dpctl::tensor::py_internal::copy_usm_ndarray_into_usm_ndarray;
using dpctl::tensor::py_internal::simplify_iteration_space;

/* =========================== Copy for reshape ============================= */

using dpctl::tensor::py_internal::copy_usm_ndarray_for_reshape;

/* ============= Copy from numpy.ndarray to usm_ndarray ==================== */

using dpctl::tensor::py_internal::copy_numpy_ndarray_into_usm_ndarray;

/* ============= linear-sequence ==================== */

using dpctl::tensor::py_internal::usm_ndarray_linear_sequence_affine;
using dpctl::tensor::py_internal::usm_ndarray_linear_sequence_step;

/* ================ Full ================== */

using dpctl::tensor::kernels::constructors::full_contig_fn_ptr_t;

static full_contig_fn_ptr_t full_contig_dispatch_vector[_ns::num_types];

std::pair<sycl::event, sycl::event>
usm_ndarray_full(py::object py_value,
                 dpctl::tensor::usm_ndarray dst,
                 sycl::queue exec_q,
                 const std::vector<sycl::event> &depends = {})
{
    // start, end should be coercible into data type of dst

    py::ssize_t dst_nelems = dst.get_size();

    if (dst_nelems == 0) {
        // nothing to do
        return std::make_pair(sycl::event(), sycl::event());
    }

    sycl::queue dst_q = dst.get_queue();
    if (!dpctl::utils::queues_are_compatible(exec_q, {dst_q})) {
        throw py::value_error(
            "Execution queue is not compatible with the allocation queue");
    }

    auto array_types = dpctl::tensor::detail::usm_ndarray_types();
    int dst_typenum = dst.get_typenum();
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    char *dst_data = dst.get_data();
    sycl::event full_event;

    if (dst_nelems == 1 || dst.is_c_contiguous() || dst.is_f_contiguous()) {
        auto fn = full_contig_dispatch_vector[dst_typeid];

        sycl::event full_contig_event =
            fn(exec_q, static_cast<size_t>(dst_nelems), py_value, dst_data,
               depends);

        return std::make_pair(
            keep_args_alive(exec_q, {dst}, {full_contig_event}),
            full_contig_event);
    }
    else {
        throw std::runtime_error(
            "Only population of contiguous usm_ndarray objects is supported.");
    }
}

/* ================ Eye ================== */

using dpctl::tensor::kernels::constructors::eye_fn_ptr_t;

static eye_fn_ptr_t eye_dispatch_vector[_ns::num_types];

std::pair<sycl::event, sycl::event>
eye(py::ssize_t k,
    dpctl::tensor::usm_ndarray dst,
    sycl::queue exec_q,
    const std::vector<sycl::event> &depends = {})
{
    // dst must be 2D

    if (dst.get_ndim() != 2) {
        throw py::value_error(
            "usm_ndarray_eye: Expecting 2D array to populate");
    }

    sycl::queue dst_q = dst.get_queue();
    if (!dpctl::utils::queues_are_compatible(exec_q, {dst_q})) {
        throw py::value_error("Execution queue is not compatible with the "
                              "allocation queue");
    }

    auto array_types = dpctl::tensor::detail::usm_ndarray_types();
    int dst_typenum = dst.get_typenum();
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    const py::ssize_t nelem = dst.get_size();
    const py::ssize_t rows = dst.get_shape(0);
    const py::ssize_t cols = dst.get_shape(1);
    if (rows == 0 || cols == 0) {
        // nothing to do
        return std::make_pair(sycl::event{}, sycl::event{});
    }

    bool is_dst_c_contig = dst.is_c_contiguous();
    bool is_dst_f_contig = dst.is_f_contiguous();
    if (!is_dst_c_contig && !is_dst_f_contig) {
        throw py::value_error("USM array is not contiguous");
    }

    py::ssize_t start;
    if (is_dst_c_contig) {
        start = (k < 0) ? -k * cols : k;
    }
    else {
        start = (k < 0) ? -k : k * rows;
    }

    const py::ssize_t *strides = dst.get_strides_raw();
    py::ssize_t step;
    if (strides == nullptr) {
        step = (is_dst_c_contig) ? cols + 1 : rows + 1;
    }
    else {
        step = strides[0] + strides[1];
    }

    const py::ssize_t length = std::min({rows, cols, rows + k, cols - k});
    const py::ssize_t end = start + step * (length - 1);

    char *dst_data = dst.get_data();
    sycl::event eye_event;

    auto fn = eye_dispatch_vector[dst_typeid];

    eye_event = fn(exec_q, static_cast<size_t>(nelem), start, end, step,
                   dst_data, depends);

    return std::make_pair(keep_args_alive(exec_q, {dst}, {eye_event}),
                          eye_event);
}

/* =========================== Tril and triu ============================== */

using dpctl::tensor::kernels::constructors::tri_fn_ptr_t;

static tri_fn_ptr_t tril_generic_dispatch_vector[_ns::num_types];
static tri_fn_ptr_t triu_generic_dispatch_vector[_ns::num_types];

std::pair<sycl::event, sycl::event>
tri(sycl::queue &exec_q,
    dpctl::tensor::usm_ndarray src,
    dpctl::tensor::usm_ndarray dst,
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
    auto src_offsets = src.get_minmax_offsets();
    auto dst_offsets = dst.get_minmax_offsets();
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

    auto array_types = dpctl::tensor::detail::usm_ndarray_types();

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();
    int src_typeid = array_types.typenum_to_lookup_id(src_typenum);
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    if (dst_typeid != src_typeid) {
        throw py::value_error("Array dtype are not the same.");
    }

    // check same contexts
    sycl::queue src_q = src.get_queue();
    sycl::queue dst_q = dst.get_queue();

    if (!dpctl::utils::queues_are_compatible(exec_q, {src_q, dst_q})) {
        throw py::value_error(
            "Execution queue context is not the same as allocation contexts");
    }

    using shT = std::vector<py::ssize_t>;
    shT src_strides(src_nd);

    bool is_src_c_contig = src.is_c_contiguous();
    bool is_src_f_contig = src.is_f_contiguous();

    const py::ssize_t *src_strides_raw = src.get_strides_raw();
    if (src_strides_raw == nullptr) {
        if (is_src_c_contig) {
            src_strides = c_contiguous_strides(src_nd, src_shape);
        }
        else if (is_src_f_contig) {
            src_strides = f_contiguous_strides(src_nd, src_shape);
        }
        else {
            throw std::runtime_error("Source array has null strides but has "
                                     "neither C- nor F- contiguous flag set");
        }
    }
    else {
        std::copy(src_strides_raw, src_strides_raw + src_nd,
                  src_strides.begin());
    }

    shT dst_strides(src_nd);

    bool is_dst_c_contig = dst.is_c_contiguous();
    bool is_dst_f_contig = dst.is_f_contiguous();

    const py::ssize_t *dst_strides_raw = dst.get_strides_raw();
    if (dst_strides_raw == nullptr) {
        if (is_dst_c_contig) {
            dst_strides =
                dpctl::tensor::c_contiguous_strides(src_nd, src_shape);
        }
        else if (is_dst_f_contig) {
            dst_strides =
                dpctl::tensor::f_contiguous_strides(src_nd, src_shape);
        }
        else {
            throw std::runtime_error("Source array has null strides but has "
                                     "neither C- nor F- contiguous flag set");
        }
    }
    else {
        std::copy(dst_strides_raw, dst_strides_raw + dst_nd,
                  dst_strides.begin());
    }

    shT simplified_shape;
    shT simplified_src_strides;
    shT simplified_dst_strides;
    py::ssize_t src_offset(0);
    py::ssize_t dst_offset(0);

    constexpr py::ssize_t src_itemsize = 1; // item size in elements
    constexpr py::ssize_t dst_itemsize = 1; // item size in elements

    int nd = src_nd - 2;
    const py::ssize_t *shape = src_shape;
    const py::ssize_t *p_src_strides = src_strides.data();
    const py::ssize_t *p_dst_strides = dst_strides.data();

    simplify_iteration_space(nd, shape, p_src_strides, src_itemsize,
                             is_src_c_contig, is_src_f_contig, p_dst_strides,
                             dst_itemsize, is_dst_c_contig, is_dst_f_contig,
                             simplified_shape, simplified_src_strides,
                             simplified_dst_strides, src_offset, dst_offset);

    if (src_offset != 0 || dst_offset != 0) {
        throw py::value_error("Reversed slice for dst is not supported");
    }

    nd += 2;

    using usm_host_allocatorT =
        sycl::usm_allocator<py::ssize_t, sycl::usm::alloc::host>;
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
        sycl::malloc_device<ssize_t>(3 * nd, exec_q);
    if (dev_shape_and_strides == nullptr) {
        throw std::runtime_error("Unabled to allocate device memory");
    }
    sycl::event copy_shape_and_strides = exec_q.copy<ssize_t>(
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

    exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on({tri_ev});
        auto ctx = exec_q.get_context();
        cgh.host_task(
            [shp_host_shape_and_strides, dev_shape_and_strides, ctx]() {
                // capture of shp_host_shape_and_strides ensure the underlying
                // vector exists for the entire execution of copying kernel
                sycl::free(dev_shape_and_strides, ctx);
            });
    });

    return std::make_pair(keep_args_alive(exec_q, {src, dst}, {tri_ev}),
                          tri_ev);
}

// populate dispatch tables
void init_dispatch_tables(void)
{
    dpctl::tensor::py_internal::init_copy_and_cast_usm_to_usm_dispatch_tables();
    dpctl::tensor::py_internal::
        init_copy_numpy_ndarray_into_usm_ndarray_dispatch_tables();
    return;
}

// populate dispatch vectors
void init_dispatch_vectors(void)
{
    dpctl::tensor::py_internal::init_copy_for_reshape_dispatch_vectors();
    dpctl::tensor::py_internal::init_linear_sequences_dispatch_vectors();

    using namespace dpctl::tensor::detail;
    using dpctl::tensor::kernels::constructors::EyeFactory;
    using dpctl::tensor::kernels::constructors::FullContigFactory;
    using dpctl::tensor::kernels::constructors::TrilGenericFactory;
    using dpctl::tensor::kernels::constructors::TriuGenericFactory;

    DispatchVectorBuilder<full_contig_fn_ptr_t, FullContigFactory, num_types>
        dvb3;
    dvb3.populate_dispatch_vector(full_contig_dispatch_vector);

    DispatchVectorBuilder<eye_fn_ptr_t, EyeFactory, num_types> dvb4;
    dvb4.populate_dispatch_vector(eye_dispatch_vector);

    DispatchVectorBuilder<tri_fn_ptr_t, TrilGenericFactory, num_types> dvb5;
    dvb5.populate_dispatch_vector(tril_generic_dispatch_vector);

    DispatchVectorBuilder<tri_fn_ptr_t, TriuGenericFactory, num_types> dvb6;
    dvb6.populate_dispatch_vector(triu_generic_dispatch_vector);

    return;
}

std::string get_default_device_fp_type(sycl::device d)
{
    if (d.has(sycl::aspect::fp64)) {
        return "f8";
    }
    else {
        return "f4";
    }
}

std::string get_default_device_int_type(sycl::device)
{
    return "i8";
}

std::string get_default_device_complex_type(sycl::device d)
{
    if (d.has(sycl::aspect::fp64)) {
        return "c16";
    }
    else {
        return "c8";
    }
}

std::string get_default_device_bool_type(sycl::device)
{
    return "b1";
}

} // namespace

PYBIND11_MODULE(_tensor_impl, m)
{
    init_dispatch_tables();
    init_dispatch_vectors();

    m.def(
        "_contract_iter", &contract_iter<py::ssize_t, py::value_error>,
        "Simplifies iteration of array of given shape & stride. Returns "
        "a triple: shape, stride and offset for the new iterator of possible "
        "smaller dimension, which traverses the same elements as the original "
        "iterator, possibly in a different order.");

    m.def("_copy_usm_ndarray_into_usm_ndarray",
          &copy_usm_ndarray_into_usm_ndarray,
          "Copies from usm_ndarray `src` into usm_ndarray `dst` of the same "
          "shape. "
          "Returns a tuple of events: (host_task_event, compute_task_event)",
          py::arg("src"), py::arg("dst"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());

    m.def(
        "_contract_iter2", &contract_iter2<py::ssize_t, py::value_error>,
        "Simplifies iteration over elements of pair of arrays of given shape "
        "with strides stride1 and stride2. Returns "
        "a 5-tuple: shape, stride and offset for the new iterator of possible "
        "smaller dimension for each array, which traverses the same elements "
        "as the original "
        "iterator, possibly in a different order.");

    m.def("_copy_usm_ndarray_for_reshape", &copy_usm_ndarray_for_reshape,
          "Copies from usm_ndarray `src` into usm_ndarray `dst` with the same "
          "number of elements using underlying 'C'-contiguous order for flat "
          "traversal with shift. "
          "Returns a tuple of events: (ht_event, comp_event)",
          py::arg("src"), py::arg("dst"), py::arg("shift"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    m.def("_linspace_step", &usm_ndarray_linear_sequence_step,
          "Fills input 1D contiguous usm_ndarray `dst` with linear sequence "
          "specified by "
          "starting point `start` and step `dt`. "
          "Returns a tuple of events: (ht_event, comp_event)",
          py::arg("start"), py::arg("dt"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    m.def("_linspace_affine", &usm_ndarray_linear_sequence_affine,
          "Fills input 1D contiguous usm_ndarray `dst` with linear sequence "
          "specified by "
          "starting point `start` and end point `end`. "
          "Returns a tuple of events: (ht_event, comp_event)",
          py::arg("start"), py::arg("end"), py::arg("dst"),
          py::arg("include_endpoint"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());

    m.def("_copy_numpy_ndarray_into_usm_ndarray",
          &copy_numpy_ndarray_into_usm_ndarray,
          "Copy fom numpy array `src` into usm_ndarray `dst` synchronously.",
          py::arg("src"), py::arg("dst"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());

    m.def("_full_usm_ndarray", &usm_ndarray_full,
          "Populate usm_ndarray `dst` with given fill_value.",
          py::arg("fill_value"), py::arg("dst"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());

    m.def("_eye", &eye,
          "Fills input 2D contiguous usm_ndarray `dst` with "
          "zeros outside of the diagonal "
          "specified by "
          "the diagonal index `k` "
          "which is filled with ones."
          "Returns a tuple of events: (ht_event, comp_event)",
          py::arg("k"), py::arg("dst"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());

    m.def("default_device_fp_type", [](sycl::queue q) -> std::string {
        return get_default_device_fp_type(q.get_device());
    });
    m.def("default_device_fp_type_device", [](sycl::device dev) -> std::string {
        return get_default_device_fp_type(dev);
    });

    m.def("default_device_int_type", [](sycl::queue q) -> std::string {
        return get_default_device_int_type(q.get_device());
    });
    m.def("default_device_int_type_device",
          [](sycl::device dev) -> std::string {
              return get_default_device_int_type(dev);
          });

    m.def("default_device_bool_type", [](sycl::queue q) -> std::string {
        return get_default_device_bool_type(q.get_device());
    });
    m.def("default_device_bool_type_device",
          [](sycl::device dev) -> std::string {
              return get_default_device_bool_type(dev);
          });

    m.def("default_device_complex_type", [](sycl::queue q) -> std::string {
        return get_default_device_complex_type(q.get_device());
    });
    m.def("default_device_complex_type_device",
          [](sycl::device dev) -> std::string {
              return get_default_device_complex_type(dev);
          });
    m.def(
        "_tril",
        [](dpctl::tensor::usm_ndarray src, dpctl::tensor::usm_ndarray dst,
           py::ssize_t k, sycl::queue exec_q,
           const std::vector<sycl::event> depends)
            -> std::pair<sycl::event, sycl::event> {
            return tri(exec_q, src, dst, 'l', k, depends);
        },
        "Tril helper function.", py::arg("src"), py::arg("dst"),
        py::arg("k") = 0, py::arg("sycl_queue"),
        py::arg("depends") = py::list());

    m.def(
        "_triu",
        [](dpctl::tensor::usm_ndarray src, dpctl::tensor::usm_ndarray dst,
           py::ssize_t k, sycl::queue exec_q,
           const std::vector<sycl::event> depends)
            -> std::pair<sycl::event, sycl::event> {
            return tri(exec_q, src, dst, 'u', k, depends);
        },
        "Triu helper function.", py::arg("src"), py::arg("dst"),
        py::arg("k") = 0, py::arg("sycl_queue"),
        py::arg("depends") = py::list());
}
