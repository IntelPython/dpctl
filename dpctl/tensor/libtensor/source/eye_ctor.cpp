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

#include <CL/sycl.hpp>
#include <utility>
#include <vector>

#include "dpctl4pybind11.hpp"
#include <pybind11/pybind11.h>

#include "eye_ctor.hpp"
#include "kernels/constructors.hpp"
#include "utils/type_dispatch.hpp"

namespace py = pybind11;
namespace _ns = dpctl::tensor::detail;

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

using dpctl::utils::keep_args_alive;

using dpctl::tensor::kernels::constructors::eye_fn_ptr_t;
static eye_fn_ptr_t eye_dispatch_vector[_ns::num_types];

std::pair<sycl::event, sycl::event>
usm_ndarray_eye(py::ssize_t k,
                dpctl::tensor::usm_ndarray dst,
                sycl::queue exec_q,
                const std::vector<sycl::event> &depends)
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

void init_eye_ctor_dispatch_vectors(void)
{
    using namespace dpctl::tensor::detail;
    using dpctl::tensor::kernels::constructors::EyeFactory;

    DispatchVectorBuilder<eye_fn_ptr_t, EyeFactory, num_types> dvb;
    dvb.populate_dispatch_vector(eye_dispatch_vector);

    return;
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
