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

#include "full_ctor.hpp"

namespace py = pybind11;
namespace _ns = dpctl::tensor::detail;

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

using dpctl::tensor::kernels::constructors::lin_space_step_fn_ptr_t;
using dpctl::utils::keep_args_alive;

using dpctl::tensor::kernels::constructors::full_contig_fn_ptr_t;

static full_contig_fn_ptr_t full_contig_dispatch_vector[_ns::num_types];

std::pair<sycl::event, sycl::event>
usm_ndarray_full(py::object py_value,
                 dpctl::tensor::usm_ndarray dst,
                 sycl::queue exec_q,
                 const std::vector<sycl::event> &depends)
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

void init_full_ctor_dispatch_vectors(void)
{
    using namespace dpctl::tensor::detail;
    using dpctl::tensor::kernels::constructors::FullContigFactory;

    DispatchVectorBuilder<full_contig_fn_ptr_t, FullContigFactory, num_types>
        dvb;
    dvb.populate_dispatch_vector(full_contig_dispatch_vector);

    return;
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
