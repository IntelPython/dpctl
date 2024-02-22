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
#include "reduction_atomic_support.hpp"
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

using dpctl::tensor::kernels::reduction_strided_impl_fn_ptr;
static reduction_strided_impl_fn_ptr
    all_reduction_strided_dispatch_vector[td_ns::num_types];

using dpctl::tensor::kernels::reduction_contig_impl_fn_ptr;
static reduction_contig_impl_fn_ptr
    all_reduction_axis1_contig_dispatch_vector[td_ns::num_types];
static reduction_contig_impl_fn_ptr
    all_reduction_axis0_contig_dispatch_vector[td_ns::num_types];

template <typename fnT, typename srcTy> struct AllStridedFactory
{
    fnT get() const
    {
        using dstTy = std::int32_t;
        using ReductionOpT = sycl::logical_and<dstTy>;
        return dpctl::tensor::kernels::
            reduction_over_group_with_atomics_strided_impl<srcTy, dstTy,
                                                           ReductionOpT>;
    }
};

template <typename fnT, typename srcTy> struct AllAxis1ContigFactory
{
    fnT get() const
    {
        using dstTy = std::int32_t;
        using ReductionOpT = sycl::logical_and<dstTy>;
        return dpctl::tensor::kernels::
            reduction_axis1_over_group_with_atomics_contig_impl<srcTy, dstTy,
                                                                ReductionOpT>;
    }
};

template <typename fnT, typename srcTy> struct AllAxis0ContigFactory
{
    fnT get() const
    {
        using dstTy = std::int32_t;
        using ReductionOpT = sycl::logical_and<dstTy>;
        return dpctl::tensor::kernels::
            reduction_axis0_over_group_with_atomics_contig_impl<srcTy, dstTy,
                                                                ReductionOpT>;
    }
};

void populate_all_dispatch_vectors(void)
{
    using td_ns::DispatchVectorBuilder;

    DispatchVectorBuilder<reduction_strided_impl_fn_ptr, AllStridedFactory,
                          td_ns::num_types>
        all_dvb1;
    all_dvb1.populate_dispatch_vector(all_reduction_strided_dispatch_vector);

    DispatchVectorBuilder<reduction_contig_impl_fn_ptr, AllAxis1ContigFactory,
                          td_ns::num_types>
        all_dvb2;
    all_dvb2.populate_dispatch_vector(
        all_reduction_axis1_contig_dispatch_vector);

    DispatchVectorBuilder<reduction_contig_impl_fn_ptr, AllAxis0ContigFactory,
                          td_ns::num_types>
        all_dvb3;
    all_dvb3.populate_dispatch_vector(
        all_reduction_axis0_contig_dispatch_vector);
};

using atomic_support::atomic_support_fn_ptr_t;
using atomic_support::check_atomic_support;
static atomic_support_fn_ptr_t all_atomic_support =
    check_atomic_support<std::int32_t>;

} // namespace impl

void init_all(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;
    {
        impl::populate_all_dispatch_vectors();
        using impl::all_reduction_axis0_contig_dispatch_vector;
        using impl::all_reduction_axis1_contig_dispatch_vector;
        using impl::all_reduction_strided_dispatch_vector;

        using impl::all_atomic_support;

        auto all_pyapi = [&](const arrayT &src, int trailing_dims_to_reduce,
                             const arrayT &dst, sycl::queue &exec_q,
                             const event_vecT &depends = {}) {
            return py_boolean_reduction(
                src, trailing_dims_to_reduce, dst, exec_q, depends,
                all_reduction_axis1_contig_dispatch_vector,
                all_reduction_axis0_contig_dispatch_vector,
                all_reduction_strided_dispatch_vector, all_atomic_support);
        };
        m.def("_all", all_pyapi, "", py::arg("src"),
              py::arg("trailing_dims_to_reduce"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());
    }
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
