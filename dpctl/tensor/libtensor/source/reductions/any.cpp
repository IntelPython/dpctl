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

#include "dpctl4pybind11.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sycl/sycl.hpp>
#include <vector>

#include "kernels/boolean_reductions.hpp"
#include "reduction_over_axis.hpp"
#include "utils/type_dispatch.hpp"

namespace py = pybind11;

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

namespace td_ns = dpctl::tensor::type_dispatch;

namespace impl
{
using dpctl::tensor::kernels::boolean_reduction_strided_impl_fn_ptr;
static boolean_reduction_strided_impl_fn_ptr
    any_reduction_strided_dispatch_vector[td_ns::num_types];

using dpctl::tensor::kernels::boolean_reduction_contig_impl_fn_ptr;
static boolean_reduction_contig_impl_fn_ptr
    any_reduction_axis1_contig_dispatch_vector[td_ns::num_types];

static boolean_reduction_contig_impl_fn_ptr
    any_reduction_axis0_contig_dispatch_vector[td_ns::num_types];

void populate_any_dispatch_vectors(void)
{
    using td_ns::DispatchVectorBuilder;

    using dpctl::tensor::kernels::boolean_reduction_strided_impl_fn_ptr;

    using dpctl::tensor::kernels::AnyStridedFactory;
    DispatchVectorBuilder<boolean_reduction_strided_impl_fn_ptr,
                          AnyStridedFactory, td_ns::num_types>
        any_dvb1;
    any_dvb1.populate_dispatch_vector(any_reduction_strided_dispatch_vector);

    using dpctl::tensor::kernels::boolean_reduction_contig_impl_fn_ptr;

    using dpctl::tensor::kernels::AnyAxis1ContigFactory;
    DispatchVectorBuilder<boolean_reduction_contig_impl_fn_ptr,
                          AnyAxis1ContigFactory, td_ns::num_types>
        any_dvb2;
    any_dvb2.populate_dispatch_vector(
        any_reduction_axis1_contig_dispatch_vector);

    using dpctl::tensor::kernels::AnyAxis0ContigFactory;
    DispatchVectorBuilder<boolean_reduction_contig_impl_fn_ptr,
                          AnyAxis0ContigFactory, td_ns::num_types>
        any_dvb3;
    any_dvb3.populate_dispatch_vector(
        any_reduction_axis0_contig_dispatch_vector);
};

} // namespace impl

void init_any(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;
    {
        impl::populate_any_dispatch_vectors();
        using impl::any_reduction_axis0_contig_dispatch_vector;
        using impl::any_reduction_axis1_contig_dispatch_vector;
        using impl::any_reduction_strided_dispatch_vector;

        auto any_pyapi = [&](const arrayT &src, int trailing_dims_to_reduce,
                             const arrayT &dst, sycl::queue &exec_q,
                             const event_vecT &depends = {}) {
            return py_boolean_reduction(
                src, trailing_dims_to_reduce, dst, exec_q, depends,
                any_reduction_axis1_contig_dispatch_vector,
                any_reduction_axis0_contig_dispatch_vector,
                any_reduction_strided_dispatch_vector);
        };
        m.def("_any", any_pyapi, "", py::arg("src"),
              py::arg("trailing_dims_to_reduce"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());
    }
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
