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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <thread>
#include <type_traits>

#include "dpctl4pybind11.hpp"

#include "boolean_advanced_indexing.hpp"
#include "boolean_reductions.hpp"
#include "copy_and_cast_usm_to_usm.hpp"
#include "copy_for_reshape.hpp"
#include "copy_numpy_ndarray_into_usm_ndarray.hpp"
#include "device_support_queries.hpp"
#include "eye_ctor.hpp"
#include "full_ctor.hpp"
#include "integer_advanced_indexing.hpp"
#include "linear_sequences.hpp"
#include "simplify_iteration_space.hpp"
#include "triul_ctor.hpp"
#include "utils/memory_overlap.hpp"
#include "utils/strided_iters.hpp"
#include "where.hpp"

namespace py = pybind11;

namespace
{

using dpctl::tensor::c_contiguous_strides;
using dpctl::tensor::f_contiguous_strides;

using dpctl::tensor::overlap::MemoryOverlap;

using dpctl::tensor::py_internal::copy_usm_ndarray_into_usm_ndarray;

/* =========================== Copy for reshape ============================= */

using dpctl::tensor::py_internal::copy_usm_ndarray_for_reshape;

/* ============= Copy from numpy.ndarray to usm_ndarray ==================== */

using dpctl::tensor::py_internal::copy_numpy_ndarray_into_usm_ndarray;

/* ============= linear-sequence ==================== */

using dpctl::tensor::py_internal::usm_ndarray_linear_sequence_affine;
using dpctl::tensor::py_internal::usm_ndarray_linear_sequence_step;

/* ================ Full ================== */

using dpctl::tensor::py_internal::usm_ndarray_full;

/* ========== Any and All ========= */
using dpctl::tensor::py_internal::py_all;
using dpctl::tensor::py_internal::py_any;

/* ============== Advanced Indexing ============= */
using dpctl::tensor::py_internal::usm_ndarray_put;
using dpctl::tensor::py_internal::usm_ndarray_take;

using dpctl::tensor::py_internal::py_extract;
using dpctl::tensor::py_internal::py_mask_positions;
using dpctl::tensor::py_internal::py_nonzero;
using dpctl::tensor::py_internal::py_place;

/* ================ Eye ================== */

using dpctl::tensor::py_internal::usm_ndarray_eye;

/* =========================== Tril and triu ============================== */

using dpctl::tensor::py_internal::usm_ndarray_triul;

/* =========================== Where ============================== */

using dpctl::tensor::py_internal::py_where;

// populate dispatch tables
void init_dispatch_tables(void)
{
    using namespace dpctl::tensor::py_internal;

    init_copy_and_cast_usm_to_usm_dispatch_tables();
    init_copy_numpy_ndarray_into_usm_ndarray_dispatch_tables();
    init_advanced_indexing_dispatch_tables();
    init_where_dispatch_tables();
    return;
}

// populate dispatch vectors
void init_dispatch_vectors(void)
{
    using namespace dpctl::tensor::py_internal;

    init_copy_for_reshape_dispatch_vectors();
    init_linear_sequences_dispatch_vectors();
    init_full_ctor_dispatch_vectors();
    init_eye_ctor_dispatch_vectors();
    init_triul_ctor_dispatch_vectors();

    populate_boolean_reduction_dispatch_vectors();
    populate_mask_positions_dispatch_vectors();
    populate_masked_extract_dispatch_vectors();
    populate_masked_place_dispatch_vectors();

    return;
}

} // namespace

PYBIND11_MODULE(_tensor_impl, m)
{
    init_dispatch_tables();
    init_dispatch_vectors();

    using dpctl::tensor::strides::contract_iter;
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

    using dpctl::tensor::strides::contract_iter2;
    m.def(
        "_contract_iter2", &contract_iter2<py::ssize_t, py::value_error>,
        "Simplifies iteration over elements of pair of arrays of given shape "
        "with strides stride1 and stride2. Returns "
        "a 5-tuple: shape, stride and offset for the new iterator of possible "
        "smaller dimension for each array, which traverses the same elements "
        "as the original "
        "iterator, possibly in a different order.");

    using dpctl::tensor::strides::contract_iter3;
    m.def(
        "_contract_iter3", &contract_iter3<py::ssize_t, py::value_error>,
        "Simplifies iteration over elements of 3-tuple of arrays of given "
        "shape "
        "with strides stride1, stride2, and stride3. Returns "
        "a 7-tuple: shape, stride and offset for the new iterator of possible "
        "smaller dimension for each array, which traverses the same elements "
        "as the original "
        "iterator, possibly in a different order.");

    using dpctl::tensor::strides::contract_iter4;
    m.def(
        "_contract_iter4", &contract_iter4<py::ssize_t, py::value_error>,
        "Simplifies iteration over elements of 4-tuple of arrays of given "
        "shape "
        "with strides stride1, stride2, stride3, and stride4. Returns "
        "a 9-tuple: shape, stride and offset for the new iterator of possible "
        "smaller dimension for each array, which traverses the same elements "
        "as the original "
        "iterator, possibly in a different order.");

    static constexpr char orderC = 'C';
    m.def(
        "_ravel_multi_index",
        [](const std::vector<py::ssize_t> &mi,
           const std::vector<py::ssize_t> &shape, char order = 'C') {
            if (order == orderC) {
                return dpctl::tensor::py_internal::_ravel_multi_index_c(mi,
                                                                        shape);
            }
            else {
                return dpctl::tensor::py_internal::_ravel_multi_index_f(mi,
                                                                        shape);
            }
        },
        "");

    m.def(
        "_unravel_index",
        [](py::ssize_t flat_index, const std::vector<py::ssize_t> &shape,
           char order = 'C') {
            if (order == orderC) {
                return dpctl::tensor::py_internal::_unravel_index_c(flat_index,
                                                                    shape);
            }
            else {
                return dpctl::tensor::py_internal::_unravel_index_f(flat_index,
                                                                    shape);
            }
        },
        "");

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

    m.def("_take", &usm_ndarray_take,
          "Takes elements at usm_ndarray indices `ind` and axes starting "
          "at axis `axis_start` from array `src` and copies them "
          "into usm_ndarray `dst` synchronously."
          "Returns a tuple of events: (hev, ev)",
          py::arg("src"), py::arg("ind"), py::arg("dst"), py::arg("axis_start"),
          py::arg("mode"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());

    m.def("_put", &usm_ndarray_put,
          "Puts elements at usm_ndarray indices `ind` and axes starting "
          "at axis `axis_start` into array `dst` from "
          "usm_ndarray `val` synchronously."
          "Returns a tuple of events: (hev, ev)",
          py::arg("dst"), py::arg("ind"), py::arg("val"), py::arg("axis_start"),
          py::arg("mode"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());

    m.def("_eye", &usm_ndarray_eye,
          "Fills input 2D contiguous usm_ndarray `dst` with "
          "zeros outside of the diagonal "
          "specified by "
          "the diagonal index `k` "
          "which is filled with ones."
          "Returns a tuple of events: (ht_event, comp_event)",
          py::arg("k"), py::arg("dst"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());

    m.def("default_device_fp_type",
          dpctl::tensor::py_internal::default_device_fp_type,
          "Gives default floating point type supported by device.",
          py::arg("dev"));

    m.def("default_device_int_type",
          dpctl::tensor::py_internal::default_device_int_type,
          "Gives default integer type supported by device.", py::arg("dev"));

    m.def("default_device_bool_type",
          dpctl::tensor::py_internal::default_device_bool_type,
          "Gives default boolean type supported by device.", py::arg("dev"));

    m.def("default_device_complex_type",
          dpctl::tensor::py_internal::default_device_complex_type,
          "Gives default complex floating point type support by device.",
          py::arg("dev"));

    auto tril_fn = [](dpctl::tensor::usm_ndarray src,
                      dpctl::tensor::usm_ndarray dst, py::ssize_t k,
                      sycl::queue exec_q,
                      const std::vector<sycl::event> depends)
        -> std::pair<sycl::event, sycl::event> {
        return usm_ndarray_triul(exec_q, src, dst, 'l', k, depends);
    };
    m.def("_tril", tril_fn, "Tril helper function.", py::arg("src"),
          py::arg("dst"), py::arg("k") = 0, py::arg("sycl_queue"),
          py::arg("depends") = py::list());

    auto triu_fn = [](dpctl::tensor::usm_ndarray src,
                      dpctl::tensor::usm_ndarray dst, py::ssize_t k,
                      sycl::queue exec_q,
                      const std::vector<sycl::event> depends)
        -> std::pair<sycl::event, sycl::event> {
        return usm_ndarray_triul(exec_q, src, dst, 'u', k, depends);
    };
    m.def("_triu", triu_fn, "Triu helper function.", py::arg("src"),
          py::arg("dst"), py::arg("k") = 0, py::arg("sycl_queue"),
          py::arg("depends") = py::list());

    m.def("mask_positions", &py_mask_positions, "", py::arg("mask"),
          py::arg("cumsum"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());

    m.def("_extract", &py_extract, "", py::arg("src"), py::arg("cumsum"),
          py::arg("axis_start"), py::arg("axis_end"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    auto overlap = [](dpctl::tensor::usm_ndarray x1,
                      dpctl::tensor::usm_ndarray x2) -> bool {
        auto const &overlap = MemoryOverlap();
        return overlap(x1, x2);
    };
    m.def("_array_overlap", overlap,
          "Determines if the memory regions indexed by each array overlap",
          py::arg("array1"), py::arg("array2"));

    m.def("_place", &py_place, "", py::arg("dst"), py::arg("cumsum"),
          py::arg("axis_start"), py::arg("axis_end"), py::arg("rhs"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    m.def("_nonzero", &py_nonzero, "", py::arg("cumsum"), py::arg("indexes"),
          py::arg("mask_shape"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());

    m.def("_where", &py_where, "", py::arg("condition"), py::arg("x1"),
          py::arg("x2"), py::arg("dst"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());

    m.def("_all", py_all, "", py::arg("src"),
          py::arg("trailing_dims_to_reduce"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    m.def("_any", py_any, "", py::arg("src"),
          py::arg("trailing_dims_to_reduce"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());
}
