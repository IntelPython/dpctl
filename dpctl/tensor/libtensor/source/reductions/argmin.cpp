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

#include "kernels/reductions.hpp"
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

using dpctl::tensor::kernels::search_strided_impl_fn_ptr;
static search_strided_impl_fn_ptr
    argmin_over_axis_strided_temps_dispatch_table[td_ns::num_types]
                                                 [td_ns::num_types];

using dpctl::tensor::kernels::search_contig_impl_fn_ptr;
static search_contig_impl_fn_ptr
    argmin_over_axis1_contig_temps_dispatch_table[td_ns::num_types]
                                                 [td_ns::num_types];
using dpctl::tensor::kernels::search_contig_impl_fn_ptr;
static search_contig_impl_fn_ptr
    argmin_over_axis0_contig_temps_dispatch_table[td_ns::num_types]
                                                 [td_ns::num_types];

void populate_argmin_over_axis_dispatch_tables(void)
{
    using dpctl::tensor::kernels::search_strided_impl_fn_ptr;
    using td_ns::DispatchTableBuilder;

    using dpctl::tensor::kernels::ArgminOverAxisTempsStridedFactory;
    DispatchTableBuilder<search_strided_impl_fn_ptr,
                         ArgminOverAxisTempsStridedFactory, td_ns::num_types>
        dtb1;
    dtb1.populate_dispatch_table(argmin_over_axis_strided_temps_dispatch_table);

    using dpctl::tensor::kernels::ArgminOverAxis1TempsContigFactory;
    DispatchTableBuilder<search_contig_impl_fn_ptr,
                         ArgminOverAxis1TempsContigFactory, td_ns::num_types>
        dtb2;
    dtb2.populate_dispatch_table(argmin_over_axis1_contig_temps_dispatch_table);

    using dpctl::tensor::kernels::ArgminOverAxis0TempsContigFactory;
    DispatchTableBuilder<search_contig_impl_fn_ptr,
                         ArgminOverAxis0TempsContigFactory, td_ns::num_types>
        dtb3;
    dtb3.populate_dispatch_table(argmin_over_axis0_contig_temps_dispatch_table);
}

} // namespace impl

void init_argmin(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;
    {
        using impl::populate_argmin_over_axis_dispatch_tables;
        populate_argmin_over_axis_dispatch_tables();
        using impl::argmin_over_axis0_contig_temps_dispatch_table;
        using impl::argmin_over_axis1_contig_temps_dispatch_table;
        using impl::argmin_over_axis_strided_temps_dispatch_table;

        auto argmin_pyapi = [&](const arrayT &src, int trailing_dims_to_reduce,
                                const arrayT &dst, sycl::queue &exec_q,
                                const event_vecT &depends = {}) {
            using dpctl::tensor::py_internal::py_search_over_axis;
            return py_search_over_axis(
                src, trailing_dims_to_reduce, dst, exec_q, depends,
                argmin_over_axis_strided_temps_dispatch_table,
                argmin_over_axis0_contig_temps_dispatch_table,
                argmin_over_axis1_contig_temps_dispatch_table);
        };
        m.def("_argmin_over_axis", argmin_pyapi, "", py::arg("src"),
              py::arg("trailing_dims_to_reduce"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());
    }
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
