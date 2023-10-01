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

#include <CL/sycl.hpp>
#include <cstdint>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dpctl4pybind11.hpp"
#include "kernels/reductions.hpp"
#include "reduction_over_axis.hpp"
#include "simplify_iteration_space.hpp"
#include "utils/type_dispatch.hpp"

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

namespace td_ns = dpctl::tensor::type_dispatch;
// Max
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

void populate_max_over_axis_dispatch_tables(void)
{
    using dpctl::tensor::kernels::reduction_contig_impl_fn_ptr;
    using dpctl::tensor::kernels::reduction_strided_impl_fn_ptr;
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
}

} // namespace impl

// Min
namespace impl
{

using dpctl::tensor::kernels::reduction_strided_impl_fn_ptr;
static reduction_strided_impl_fn_ptr
    min_over_axis_strided_atomic_dispatch_table[td_ns::num_types]
                                               [td_ns::num_types];
static reduction_strided_impl_fn_ptr
    min_over_axis_strided_temps_dispatch_table[td_ns::num_types]
                                              [td_ns::num_types];

using dpctl::tensor::kernels::reduction_contig_impl_fn_ptr;
static reduction_contig_impl_fn_ptr
    min_over_axis1_contig_atomic_dispatch_table[td_ns::num_types]
                                               [td_ns::num_types];
static reduction_contig_impl_fn_ptr
    min_over_axis0_contig_atomic_dispatch_table[td_ns::num_types]
                                               [td_ns::num_types];

void populate_min_over_axis_dispatch_tables(void)
{
    using dpctl::tensor::kernels::reduction_contig_impl_fn_ptr;
    using dpctl::tensor::kernels::reduction_strided_impl_fn_ptr;
    using td_ns::DispatchTableBuilder;

    using dpctl::tensor::kernels::MinOverAxisAtomicStridedFactory;
    DispatchTableBuilder<reduction_strided_impl_fn_ptr,
                         MinOverAxisAtomicStridedFactory, td_ns::num_types>
        dtb1;
    dtb1.populate_dispatch_table(min_over_axis_strided_atomic_dispatch_table);

    using dpctl::tensor::kernels::MinOverAxisTempsStridedFactory;
    DispatchTableBuilder<reduction_strided_impl_fn_ptr,
                         MinOverAxisTempsStridedFactory, td_ns::num_types>
        dtb2;
    dtb2.populate_dispatch_table(min_over_axis_strided_temps_dispatch_table);

    using dpctl::tensor::kernels::MinOverAxis1AtomicContigFactory;
    DispatchTableBuilder<reduction_contig_impl_fn_ptr,
                         MinOverAxis1AtomicContigFactory, td_ns::num_types>
        dtb3;
    dtb3.populate_dispatch_table(min_over_axis1_contig_atomic_dispatch_table);

    using dpctl::tensor::kernels::MinOverAxis0AtomicContigFactory;
    DispatchTableBuilder<reduction_contig_impl_fn_ptr,
                         MinOverAxis0AtomicContigFactory, td_ns::num_types>
        dtb4;
    dtb4.populate_dispatch_table(min_over_axis0_contig_atomic_dispatch_table);
}

} // namespace impl

// Sum
namespace impl
{

using dpctl::tensor::kernels::reduction_strided_impl_fn_ptr;
static reduction_strided_impl_fn_ptr
    sum_over_axis_strided_atomic_dispatch_table[td_ns::num_types]
                                               [td_ns::num_types];
static reduction_strided_impl_fn_ptr
    sum_over_axis_strided_temps_dispatch_table[td_ns::num_types]
                                              [td_ns::num_types];

using dpctl::tensor::kernels::reduction_contig_impl_fn_ptr;
static reduction_contig_impl_fn_ptr
    sum_over_axis1_contig_atomic_dispatch_table[td_ns::num_types]
                                               [td_ns::num_types];
static reduction_contig_impl_fn_ptr
    sum_over_axis0_contig_atomic_dispatch_table[td_ns::num_types]
                                               [td_ns::num_types];

void populate_sum_over_axis_dispatch_tables(void)
{
    using dpctl::tensor::kernels::reduction_contig_impl_fn_ptr;
    using dpctl::tensor::kernels::reduction_strided_impl_fn_ptr;
    using namespace td_ns;

    using dpctl::tensor::kernels::SumOverAxisAtomicStridedFactory;
    DispatchTableBuilder<reduction_strided_impl_fn_ptr,
                         SumOverAxisAtomicStridedFactory, num_types>
        dtb1;
    dtb1.populate_dispatch_table(sum_over_axis_strided_atomic_dispatch_table);

    using dpctl::tensor::kernels::SumOverAxisTempsStridedFactory;
    DispatchTableBuilder<reduction_strided_impl_fn_ptr,
                         SumOverAxisTempsStridedFactory, num_types>
        dtb2;
    dtb2.populate_dispatch_table(sum_over_axis_strided_temps_dispatch_table);

    using dpctl::tensor::kernels::SumOverAxis1AtomicContigFactory;
    DispatchTableBuilder<reduction_contig_impl_fn_ptr,
                         SumOverAxis1AtomicContigFactory, num_types>
        dtb3;
    dtb3.populate_dispatch_table(sum_over_axis1_contig_atomic_dispatch_table);

    using dpctl::tensor::kernels::SumOverAxis0AtomicContigFactory;
    DispatchTableBuilder<reduction_contig_impl_fn_ptr,
                         SumOverAxis0AtomicContigFactory, num_types>
        dtb4;
    dtb4.populate_dispatch_table(sum_over_axis0_contig_atomic_dispatch_table);
}

} // namespace impl

// Product
namespace impl
{

using dpctl::tensor::kernels::reduction_strided_impl_fn_ptr;
static reduction_strided_impl_fn_ptr
    prod_over_axis_strided_atomic_dispatch_table[td_ns::num_types]
                                                [td_ns::num_types];
static reduction_strided_impl_fn_ptr
    prod_over_axis_strided_temps_dispatch_table[td_ns::num_types]
                                               [td_ns::num_types];

using dpctl::tensor::kernels::reduction_contig_impl_fn_ptr;
static reduction_contig_impl_fn_ptr
    prod_over_axis1_contig_atomic_dispatch_table[td_ns::num_types]
                                                [td_ns::num_types];
static reduction_contig_impl_fn_ptr
    prod_over_axis0_contig_atomic_dispatch_table[td_ns::num_types]
                                                [td_ns::num_types];

void populate_prod_over_axis_dispatch_tables(void)
{
    using dpctl::tensor::kernels::reduction_contig_impl_fn_ptr;
    using dpctl::tensor::kernels::reduction_strided_impl_fn_ptr;
    using namespace td_ns;

    using dpctl::tensor::kernels::ProductOverAxisAtomicStridedFactory;
    DispatchTableBuilder<reduction_strided_impl_fn_ptr,
                         ProductOverAxisAtomicStridedFactory, num_types>
        dtb1;
    dtb1.populate_dispatch_table(prod_over_axis_strided_atomic_dispatch_table);

    using dpctl::tensor::kernels::ProductOverAxisTempsStridedFactory;
    DispatchTableBuilder<reduction_strided_impl_fn_ptr,
                         ProductOverAxisTempsStridedFactory, num_types>
        dtb2;
    dtb2.populate_dispatch_table(prod_over_axis_strided_temps_dispatch_table);

    using dpctl::tensor::kernels::ProductOverAxis1AtomicContigFactory;
    DispatchTableBuilder<reduction_contig_impl_fn_ptr,
                         ProductOverAxis1AtomicContigFactory, num_types>
        dtb3;
    dtb3.populate_dispatch_table(prod_over_axis1_contig_atomic_dispatch_table);

    using dpctl::tensor::kernels::ProductOverAxis0AtomicContigFactory;
    DispatchTableBuilder<reduction_contig_impl_fn_ptr,
                         ProductOverAxis0AtomicContigFactory, num_types>
        dtb4;
    dtb4.populate_dispatch_table(prod_over_axis0_contig_atomic_dispatch_table);
}

} // namespace impl

// Argmax
namespace impl
{

using dpctl::tensor::kernels::search_reduction_strided_impl_fn_ptr;
static search_reduction_strided_impl_fn_ptr
    argmax_over_axis_strided_temps_dispatch_table[td_ns::num_types]
                                                 [td_ns::num_types];

void populate_argmax_over_axis_dispatch_tables(void)
{
    using dpctl::tensor::kernels::search_reduction_strided_impl_fn_ptr;
    using td_ns::DispatchTableBuilder;

    using dpctl::tensor::kernels::ArgmaxOverAxisTempsStridedFactory;
    DispatchTableBuilder<search_reduction_strided_impl_fn_ptr,
                         ArgmaxOverAxisTempsStridedFactory, td_ns::num_types>
        dtb1;
    dtb1.populate_dispatch_table(argmax_over_axis_strided_temps_dispatch_table);
}

} // namespace impl

// Argmin
namespace impl
{

using dpctl::tensor::kernels::search_reduction_strided_impl_fn_ptr;
static search_reduction_strided_impl_fn_ptr
    argmin_over_axis_strided_temps_dispatch_table[td_ns::num_types]
                                                 [td_ns::num_types];

void populate_argmin_over_axis_dispatch_tables(void)
{
    using dpctl::tensor::kernels::search_reduction_strided_impl_fn_ptr;
    using td_ns::DispatchTableBuilder;

    using dpctl::tensor::kernels::ArgminOverAxisTempsStridedFactory;
    DispatchTableBuilder<search_reduction_strided_impl_fn_ptr,
                         ArgminOverAxisTempsStridedFactory, td_ns::num_types>
        dtb1;
    dtb1.populate_dispatch_table(argmin_over_axis_strided_temps_dispatch_table);
}

} // namespace impl

namespace py = pybind11;

void init_reduction_functions(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;

    namespace impl = dpctl::tensor::py_internal::impl;

    // MAX
    {
        using dpctl::tensor::py_internal::impl::
            populate_max_over_axis_dispatch_tables;
        populate_max_over_axis_dispatch_tables();
        using impl::max_over_axis0_contig_atomic_dispatch_table;
        using impl::max_over_axis1_contig_atomic_dispatch_table;
        using impl::max_over_axis_strided_atomic_dispatch_table;
        using impl::max_over_axis_strided_temps_dispatch_table;

        auto max_pyapi = [&](const arrayT &src, int trailing_dims_to_reduce,
                             const arrayT &dst, sycl::queue &exec_q,
                             const event_vecT &depends = {}) {
            using dpctl::tensor::py_internal::py_reduction_over_axis;
            return py_reduction_over_axis(
                src, trailing_dims_to_reduce, dst, exec_q, depends,
                max_over_axis_strided_atomic_dispatch_table,
                max_over_axis_strided_temps_dispatch_table,
                max_over_axis0_contig_atomic_dispatch_table,
                max_over_axis1_contig_atomic_dispatch_table);
        };
        m.def("_max_over_axis", max_pyapi, "", py::arg("src"),
              py::arg("trailing_dims_to_reduce"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());
    }

    // MIN
    {
        using dpctl::tensor::py_internal::impl::
            populate_min_over_axis_dispatch_tables;
        populate_min_over_axis_dispatch_tables();
        using impl::min_over_axis0_contig_atomic_dispatch_table;
        using impl::min_over_axis1_contig_atomic_dispatch_table;
        using impl::min_over_axis_strided_atomic_dispatch_table;
        using impl::min_over_axis_strided_temps_dispatch_table;

        auto min_pyapi = [&](const arrayT &src, int trailing_dims_to_reduce,
                             const arrayT &dst, sycl::queue &exec_q,
                             const event_vecT &depends = {}) {
            using dpctl::tensor::py_internal::py_reduction_over_axis;
            return py_reduction_over_axis(
                src, trailing_dims_to_reduce, dst, exec_q, depends,
                min_over_axis_strided_atomic_dispatch_table,
                min_over_axis_strided_temps_dispatch_table,
                min_over_axis0_contig_atomic_dispatch_table,
                min_over_axis1_contig_atomic_dispatch_table);
        };
        m.def("_min_over_axis", min_pyapi, "", py::arg("src"),
              py::arg("trailing_dims_to_reduce"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());
    }

    // SUM
    {
        using dpctl::tensor::py_internal::impl::
            populate_sum_over_axis_dispatch_tables;
        populate_sum_over_axis_dispatch_tables();
        using impl::sum_over_axis0_contig_atomic_dispatch_table;
        using impl::sum_over_axis1_contig_atomic_dispatch_table;
        using impl::sum_over_axis_strided_atomic_dispatch_table;
        using impl::sum_over_axis_strided_temps_dispatch_table;

        auto sum_pyapi = [&](const arrayT &src, int trailing_dims_to_reduce,
                             const arrayT &dst, sycl::queue &exec_q,
                             const event_vecT &depends = {}) {
            using dpctl::tensor::py_internal::py_reduction_over_axis;
            return py_reduction_over_axis(
                src, trailing_dims_to_reduce, dst, exec_q, depends,
                sum_over_axis_strided_atomic_dispatch_table,
                sum_over_axis_strided_temps_dispatch_table,
                sum_over_axis0_contig_atomic_dispatch_table,
                sum_over_axis1_contig_atomic_dispatch_table);
        };
        m.def("_sum_over_axis", sum_pyapi, "", py::arg("src"),
              py::arg("trailing_dims_to_reduce"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto sum_dtype_supported =
            [&](const py::dtype &input_dtype, const py::dtype &output_dtype,
                const std::string &dst_usm_type, sycl::queue &q) {
                using dpctl::tensor::py_internal::py_reduction_dtype_supported;
                return py_reduction_dtype_supported(
                    input_dtype, output_dtype, dst_usm_type, q,
                    sum_over_axis_strided_atomic_dispatch_table,
                    sum_over_axis_strided_temps_dispatch_table);
            };
        m.def("_sum_over_axis_dtype_supported", sum_dtype_supported, "",
              py::arg("arg_dtype"), py::arg("out_dtype"),
              py::arg("dst_usm_type"), py::arg("sycl_queue"));
    }

    // PROD
    {
        using dpctl::tensor::py_internal::impl::
            populate_prod_over_axis_dispatch_tables;
        populate_prod_over_axis_dispatch_tables();
        using impl::prod_over_axis0_contig_atomic_dispatch_table;
        using impl::prod_over_axis1_contig_atomic_dispatch_table;
        using impl::prod_over_axis_strided_atomic_dispatch_table;
        using impl::prod_over_axis_strided_temps_dispatch_table;

        auto prod_pyapi = [&](const arrayT &src, int trailing_dims_to_reduce,
                              const arrayT &dst, sycl::queue &exec_q,
                              const event_vecT &depends = {}) {
            using dpctl::tensor::py_internal::py_reduction_over_axis;
            return py_reduction_over_axis(
                src, trailing_dims_to_reduce, dst, exec_q, depends,
                prod_over_axis_strided_atomic_dispatch_table,
                prod_over_axis_strided_temps_dispatch_table,
                prod_over_axis0_contig_atomic_dispatch_table,
                prod_over_axis1_contig_atomic_dispatch_table);
        };
        m.def("_prod_over_axis", prod_pyapi, "", py::arg("src"),
              py::arg("trailing_dims_to_reduce"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto prod_dtype_supported =
            [&](const py::dtype &input_dtype, const py::dtype &output_dtype,
                const std::string &dst_usm_type, sycl::queue &q) {
                using dpctl::tensor::py_internal::py_reduction_dtype_supported;
                return py_reduction_dtype_supported(
                    input_dtype, output_dtype, dst_usm_type, q,
                    prod_over_axis_strided_atomic_dispatch_table,
                    prod_over_axis_strided_temps_dispatch_table);
            };
        m.def("_prod_over_axis_dtype_supported", prod_dtype_supported, "",
              py::arg("arg_dtype"), py::arg("out_dtype"),
              py::arg("dst_usm_type"), py::arg("sycl_queue"));
    }

    // ARGMAX
    {
        using dpctl::tensor::py_internal::impl::
            populate_argmax_over_axis_dispatch_tables;
        populate_argmax_over_axis_dispatch_tables();
        using impl::argmax_over_axis_strided_temps_dispatch_table;

        auto argmax_pyapi = [&](const arrayT &src, int trailing_dims_to_reduce,
                                const arrayT &dst, sycl::queue &exec_q,
                                const event_vecT &depends = {}) {
            using dpctl::tensor::py_internal::py_search_over_axis;
            return py_search_over_axis(
                src, trailing_dims_to_reduce, dst, exec_q, depends,
                argmax_over_axis_strided_temps_dispatch_table);
        };
        m.def("_argmax_over_axis", argmax_pyapi, "", py::arg("src"),
              py::arg("trailing_dims_to_reduce"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());
    }

    // ARGMIN
    {
        using dpctl::tensor::py_internal::impl::
            populate_argmin_over_axis_dispatch_tables;
        populate_argmin_over_axis_dispatch_tables();
        using impl::argmin_over_axis_strided_temps_dispatch_table;

        auto argmin_pyapi = [&](const arrayT &src, int trailing_dims_to_reduce,
                                const arrayT &dst, sycl::queue &exec_q,
                                const event_vecT &depends = {}) {
            using dpctl::tensor::py_internal::py_search_over_axis;
            return py_search_over_axis(
                src, trailing_dims_to_reduce, dst, exec_q, depends,
                argmin_over_axis_strided_temps_dispatch_table);
        };
        m.def("_argmin_over_axis", argmin_pyapi, "", py::arg("src"),
              py::arg("trailing_dims_to_reduce"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());
    }
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
