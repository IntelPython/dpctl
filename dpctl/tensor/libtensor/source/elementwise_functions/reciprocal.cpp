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
/// This file defines functions of dpctl.tensor._tensor_impl extensions,
/// specifically functions for elementwise operations.
//===----------------------------------------------------------------------===//

#include "dpctl4pybind11.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sycl/sycl.hpp>
#include <vector>

#include "elementwise_functions.hpp"
#include "reciprocal.hpp"
#include "utils/type_dispatch.hpp"

#include "kernels/elementwise_functions/common.hpp"
#include "kernels/elementwise_functions/reciprocal.hpp"

namespace py = pybind11;

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

namespace td_ns = dpctl::tensor::type_dispatch;

namespace ew_cmn_ns = dpctl::tensor::kernels::elementwise_common;
using ew_cmn_ns::unary_contig_impl_fn_ptr_t;
using ew_cmn_ns::unary_strided_impl_fn_ptr_t;

// U42: ==== REAL   (x)
namespace impl
{

namespace reciprocal_fn_ns = dpctl::tensor::kernels::reciprocal;

static unary_contig_impl_fn_ptr_t
    reciprocal_contig_dispatch_vector[td_ns::num_types];
static int reciprocal_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    reciprocal_strided_dispatch_vector[td_ns::num_types];

void populate_reciprocal_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = reciprocal_fn_ns;

    using fn_ns::ReciprocalContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, ReciprocalContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(reciprocal_contig_dispatch_vector);

    using fn_ns::ReciprocalStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, ReciprocalStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(reciprocal_strided_dispatch_vector);

    using fn_ns::ReciprocalTypeMapFactory;
    DispatchVectorBuilder<int, ReciprocalTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(reciprocal_output_typeid_vector);
};

} // namespace impl

void init_reciprocal(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;
    {
        impl::populate_reciprocal_dispatch_vectors();
        using impl::reciprocal_contig_dispatch_vector;
        using impl::reciprocal_output_typeid_vector;
        using impl::reciprocal_strided_dispatch_vector;

        auto reciprocal_pyapi = [&](const arrayT &src, const arrayT &dst,
                                    sycl::queue &exec_q,
                                    const event_vecT &depends = {}) {
            return py_unary_ufunc(src, dst, exec_q, depends,
                                  reciprocal_output_typeid_vector,
                                  reciprocal_contig_dispatch_vector,
                                  reciprocal_strided_dispatch_vector);
        };
        m.def("_reciprocal", reciprocal_pyapi, "", py::arg("src"),
              py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());

        auto reciprocal_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype,
                                              reciprocal_output_typeid_vector);
        };
        m.def("_reciprocal_result_type", reciprocal_result_type_pyapi);
    }
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
