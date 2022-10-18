//===-- ------------ Implementation of _tensor_impl module  ----*-C++-*-/===//
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
//===--------------------------------------------------------------------===//
///
/// \file
/// This file defines functions of dpctl.tensor._tensor_impl extensions
//===--------------------------------------------------------------------===//

#include "dpctl4pybind11.hpp"
#include <CL/sycl.hpp>
#include <complex>
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <utility>
#include <vector>

#include "kernels/constructors.hpp"
#include "utils/strided_iters.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

#include "linear_sequences.hpp"

namespace py = pybind11;
namespace _ns = dpctl::tensor::detail;

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

using dpctl::utils::keep_args_alive;

using dpctl::tensor::kernels::constructors::lin_space_step_fn_ptr_t;

static lin_space_step_fn_ptr_t lin_space_step_dispatch_vector[_ns::num_types];

using dpctl::tensor::kernels::constructors::lin_space_affine_fn_ptr_t;

static lin_space_affine_fn_ptr_t
    lin_space_affine_dispatch_vector[_ns::num_types];

std::pair<sycl::event, sycl::event>
usm_ndarray_linear_sequence_step(py::object start,
                                 py::object dt,
                                 dpctl::tensor::usm_ndarray dst,
                                 sycl::queue exec_q,
                                 const std::vector<sycl::event> &depends)
{
    // dst must be 1D and C-contiguous
    // start, end should be coercible into data type of dst

    if (dst.get_ndim() != 1) {
        throw py::value_error(
            "usm_ndarray_linspace: Expecting 1D array to populate");
    }

    if (!dst.is_c_contiguous()) {
        throw py::value_error(
            "usm_ndarray_linspace: Non-contiguous arrays are not supported");
    }

    sycl::queue dst_q = dst.get_queue();
    if (!dpctl::utils::queues_are_compatible(exec_q, {dst_q})) {
        throw py::value_error(
            "Execution queue is not compatible with the allocation queue");
    }

    auto array_types = dpctl::tensor::detail::usm_ndarray_types();
    int dst_typenum = dst.get_typenum();
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    py::ssize_t len = dst.get_shape(0);
    if (len == 0) {
        // nothing to do
        return std::make_pair(sycl::event{}, sycl::event{});
    }

    char *dst_data = dst.get_data();
    sycl::event linspace_step_event;

    auto fn = lin_space_step_dispatch_vector[dst_typeid];

    linspace_step_event =
        fn(exec_q, static_cast<size_t>(len), start, dt, dst_data, depends);

    return std::make_pair(keep_args_alive(exec_q, {dst}, {linspace_step_event}),
                          linspace_step_event);
}

std::pair<sycl::event, sycl::event>
usm_ndarray_linear_sequence_affine(py::object start,
                                   py::object end,
                                   dpctl::tensor::usm_ndarray dst,
                                   bool include_endpoint,
                                   sycl::queue exec_q,
                                   const std::vector<sycl::event> &depends)
{
    // dst must be 1D and C-contiguous
    // start, end should be coercible into data type of dst

    if (dst.get_ndim() != 1) {
        throw py::value_error(
            "usm_ndarray_linspace: Expecting 1D array to populate");
    }

    if (!dst.is_c_contiguous()) {
        throw py::value_error(
            "usm_ndarray_linspace: Non-contiguous arrays are not supported");
    }

    sycl::queue dst_q = dst.get_queue();
    if (!dpctl::utils::queues_are_compatible(exec_q, {dst_q})) {
        throw py::value_error(
            "Execution queue context is not the same as allocation context");
    }

    auto array_types = dpctl::tensor::detail::usm_ndarray_types();
    int dst_typenum = dst.get_typenum();
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    py::ssize_t len = dst.get_shape(0);
    if (len == 0) {
        // nothing to do
        return std::make_pair(sycl::event{}, sycl::event{});
    }

    char *dst_data = dst.get_data();
    sycl::event linspace_affine_event;

    auto fn = lin_space_affine_dispatch_vector[dst_typeid];

    linspace_affine_event = fn(exec_q, static_cast<size_t>(len), start, end,
                               include_endpoint, dst_data, depends);

    return std::make_pair(
        keep_args_alive(exec_q, {dst}, {linspace_affine_event}),
        linspace_affine_event);
}

void init_linear_sequences_dispatch_vectors(void)
{
    using namespace dpctl::tensor::detail;
    using dpctl::tensor::kernels::constructors::LinSpaceAffineFactory;
    using dpctl::tensor::kernels::constructors::LinSpaceStepFactory;

    DispatchVectorBuilder<lin_space_step_fn_ptr_t, LinSpaceStepFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(lin_space_step_dispatch_vector);

    DispatchVectorBuilder<lin_space_affine_fn_ptr_t, LinSpaceAffineFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(lin_space_affine_dispatch_vector);
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
