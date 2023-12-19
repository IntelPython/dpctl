//===-- ------------ Implementation of _tensor_impl module  ----*-C++-*-/===//
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
//===--------------------------------------------------------------------===//
///
/// \file
/// This file defines functions of dpctl.tensor._tensor_impl extensions
//===--------------------------------------------------------------------===//

#include "dpctl4pybind11.hpp"
#include <CL/sycl.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "kernels/reductions.hpp"
#include "utils/type_dispatch.hpp"

#include "reduction_atomic_support.hpp"
#include "reduction_over_axis.hpp"

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

using dpctl::tensor::kernels::reduction_strided_impl_fn_ptr;
static reduction_strided_impl_fn_ptr
    max_over_axis_strided_atomic_dispatch_table[td_ns::num_types]
                                               [td_ns::num_types];
static reduction_strided_impl_fn_ptr
    max_over_axis_strided_temps_dispatch_table[td_ns::num_types]
                                              [td_ns::num_types];

using dpctl::tensor::kernels::reduction_contig_impl_fn_ptr;
static reduction_contig_impl_fn_ptr
    max_over_axis1_contig_atomic_dispatch_table[td_ns::num_types]
                                               [td_ns::num_types];
static reduction_contig_impl_fn_ptr
    max_over_axis0_contig_atomic_dispatch_table[td_ns::num_types]
                                               [td_ns::num_types];
static reduction_contig_impl_fn_ptr
    max_over_axis1_contig_temps_dispatch_table[td_ns::num_types]
                                              [td_ns::num_types];
static reduction_contig_impl_fn_ptr
    max_over_axis0_contig_temps_dispatch_table[td_ns::num_types]
                                              [td_ns::num_types];

void populate_max_over_axis_dispatch_tables(void)
{
    using td_ns::DispatchTableBuilder;

    using dpctl::tensor::kernels::MaxOverAxisAtomicStridedFactory;
    DispatchTableBuilder<reduction_strided_impl_fn_ptr,
                         MaxOverAxisAtomicStridedFactory, td_ns::num_types>
        dtb1;
    dtb1.populate_dispatch_table(max_over_axis_strided_atomic_dispatch_table);

    using dpctl::tensor::kernels::MaxOverAxisTempsStridedFactory;
    DispatchTableBuilder<reduction_strided_impl_fn_ptr,
                         MaxOverAxisTempsStridedFactory, td_ns::num_types>
        dtb2;
    dtb2.populate_dispatch_table(max_over_axis_strided_temps_dispatch_table);

    using dpctl::tensor::kernels::MaxOverAxis1AtomicContigFactory;
    DispatchTableBuilder<reduction_contig_impl_fn_ptr,
                         MaxOverAxis1AtomicContigFactory, td_ns::num_types>
        dtb3;
    dtb3.populate_dispatch_table(max_over_axis1_contig_atomic_dispatch_table);

    using dpctl::tensor::kernels::MaxOverAxis0AtomicContigFactory;
    DispatchTableBuilder<reduction_contig_impl_fn_ptr,
                         MaxOverAxis0AtomicContigFactory, td_ns::num_types>
        dtb4;
    dtb4.populate_dispatch_table(max_over_axis0_contig_atomic_dispatch_table);

    using dpctl::tensor::kernels::MaxOverAxis1TempsContigFactory;
    DispatchTableBuilder<reduction_contig_impl_fn_ptr,
                         MaxOverAxis1TempsContigFactory, td_ns::num_types>
        dtb5;
    dtb5.populate_dispatch_table(max_over_axis1_contig_temps_dispatch_table);

    using dpctl::tensor::kernels::MaxOverAxis0TempsContigFactory;
    DispatchTableBuilder<reduction_contig_impl_fn_ptr,
                         MaxOverAxis0TempsContigFactory, td_ns::num_types>
        dtb6;
    dtb6.populate_dispatch_table(max_over_axis0_contig_temps_dispatch_table);
}

using atomic_support::atomic_support_fn_ptr_t;
static atomic_support_fn_ptr_t max_atomic_support_vector[td_ns::num_types];

void populate_max_atomic_support_dispatch_vector(void)
{
    using td_ns::DispatchVectorBuilder;

    using atomic_support::MaxAtomicSupportFactory;
    DispatchVectorBuilder<atomic_support_fn_ptr_t, MaxAtomicSupportFactory,
                          td_ns::num_types>
        dvb;
    dvb.populate_dispatch_vector(max_atomic_support_vector);
}

} // namespace impl

void init_max(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;
    {
        using impl::populate_max_over_axis_dispatch_tables;
        populate_max_over_axis_dispatch_tables();
        using impl::max_over_axis0_contig_atomic_dispatch_table;
        using impl::max_over_axis0_contig_temps_dispatch_table;
        using impl::max_over_axis1_contig_atomic_dispatch_table;
        using impl::max_over_axis1_contig_temps_dispatch_table;
        using impl::max_over_axis_strided_atomic_dispatch_table;
        using impl::max_over_axis_strided_temps_dispatch_table;

        using impl::populate_max_atomic_support_dispatch_vector;
        populate_max_atomic_support_dispatch_vector();
        using impl::max_atomic_support_vector;

        auto max_pyapi = [&](const arrayT &src, int trailing_dims_to_reduce,
                             const arrayT &dst, sycl::queue &exec_q,
                             const event_vecT &depends = {}) {
            using dpctl::tensor::py_internal::py_reduction_over_axis;
            return py_reduction_over_axis(
                src, trailing_dims_to_reduce, dst, exec_q, depends,
                max_over_axis_strided_atomic_dispatch_table,
                max_over_axis0_contig_atomic_dispatch_table,
                max_over_axis1_contig_atomic_dispatch_table,
                max_over_axis_strided_temps_dispatch_table,
                max_over_axis0_contig_temps_dispatch_table,
                max_over_axis1_contig_temps_dispatch_table,
                max_atomic_support_vector);
        };
        m.def("_max_over_axis", max_pyapi, "", py::arg("src"),
              py::arg("trailing_dims_to_reduce"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());
    }
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl