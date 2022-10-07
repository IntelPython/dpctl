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

#include "copy_and_cast_usm_to_usm.hpp"
#include "copy_for_reshape.hpp"
#include "copy_numpy_ndarray_into_usm_ndarray.hpp"
#include "eye_ctor.hpp"
#include "full_ctor.hpp"
#include "linear_sequences.hpp"
#include "triul_ctor.hpp"
#include "utils/strided_iters.hpp"

namespace py = pybind11;

namespace
{

using dpctl::tensor::c_contiguous_strides;
using dpctl::tensor::f_contiguous_strides;

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

/* ================ Eye ================== */

using dpctl::tensor::py_internal::usm_ndarray_eye;

/* =========================== Tril and triu ============================== */

using dpctl::tensor::py_internal::usm_ndarray_triul;

// populate dispatch tables
void init_dispatch_tables(void)
{
    using namespace dpctl::tensor::py_internal;

    init_copy_and_cast_usm_to_usm_dispatch_tables();
    init_copy_numpy_ndarray_into_usm_ndarray_dispatch_tables();
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

    m.def("_eye", &usm_ndarray_eye,
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
            return usm_ndarray_triul(exec_q, src, dst, 'l', k, depends);
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
            return usm_ndarray_triul(exec_q, src, dst, 'u', k, depends);
        },
        "Triu helper function.", py::arg("src"), py::arg("dst"),
        py::arg("k") = 0, py::arg("sycl_queue"),
        py::arg("depends") = py::list());
}
