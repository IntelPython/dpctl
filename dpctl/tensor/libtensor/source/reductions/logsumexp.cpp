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
#include <type_traits>
#include <vector>

#include "kernels/reductions.hpp"
#include "reduction_over_axis.hpp"
#include "utils/type_dispatch_building.hpp"

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
    logsumexp_over_axis_strided_temps_dispatch_table[td_ns::num_types]
                                                    [td_ns::num_types];

using dpctl::tensor::kernels::reduction_contig_impl_fn_ptr;
static reduction_contig_impl_fn_ptr
    logsumexp_over_axis1_contig_temps_dispatch_table[td_ns::num_types]
                                                    [td_ns::num_types];
static reduction_contig_impl_fn_ptr
    logsumexp_over_axis0_contig_temps_dispatch_table[td_ns::num_types]
                                                    [td_ns::num_types];

template <typename argTy, typename outTy>
struct TypePairSupportDataForLogSumExpReductionTemps
{

    static constexpr bool is_defined = std::disjunction< // disjunction is C++17
                                                         // feature, supported
                                                         // by DPC++ input bool
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, sycl::half>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, double>,

        // input int8_t
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, sycl::half>,
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, double>,

        // input uint8_t
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, sycl::half>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, double>,

        // input int16_t
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, double>,

        // input uint16_t
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, double>,

        // input int32_t
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, double>,

        // input uint32_t
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, double>,

        // input int64_t
        td_ns::TypePairDefinedEntry<argTy, std::int64_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::int64_t, outTy, double>,

        // input uint64_t
        td_ns::TypePairDefinedEntry<argTy, std::uint64_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::uint64_t, outTy, double>,

        // input half
        td_ns::TypePairDefinedEntry<argTy, sycl::half, outTy, sycl::half>,
        td_ns::TypePairDefinedEntry<argTy, sycl::half, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, sycl::half, outTy, double>,

        // input float
        td_ns::TypePairDefinedEntry<argTy, float, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, float, outTy, double>,

        // input double
        td_ns::TypePairDefinedEntry<argTy, double, outTy, double>,

        // fall-through
        td_ns::NotDefinedEntry>::is_defined;
};

template <typename fnT, typename srcTy, typename dstTy>
struct LogSumExpOverAxisTempsStridedFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForLogSumExpReductionTemps<
                          srcTy, dstTy>::is_defined)
        {
            using ReductionOpT = su_ns::LogSumExp<dstTy>;
            return dpctl::tensor::kernels::
                reduction_over_group_temps_strided_impl<srcTy, dstTy,
                                                        ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct LogSumExpOverAxis1TempsContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForLogSumExpReductionTemps<
                          srcTy, dstTy>::is_defined)
        {
            using ReductionOpT = su_ns::LogSumExp<dstTy>;
            return dpctl::tensor::kernels::
                reduction_axis1_over_group_temps_contig_impl<srcTy, dstTy,
                                                             ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct LogSumExpOverAxis0TempsContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForLogSumExpReductionTemps<
                          srcTy, dstTy>::is_defined)
        {
            using ReductionOpT = su_ns::LogSumExp<dstTy>;
            return dpctl::tensor::kernels::
                reduction_axis0_over_group_temps_contig_impl<srcTy, dstTy,
                                                             ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

void populate_logsumexp_over_axis_dispatch_tables(void)
{
    using dpctl::tensor::kernels::reduction_contig_impl_fn_ptr;
    using dpctl::tensor::kernels::reduction_strided_impl_fn_ptr;
    using namespace td_ns;

    DispatchTableBuilder<reduction_strided_impl_fn_ptr,
                         LogSumExpOverAxisTempsStridedFactory, num_types>
        dtb1;
    dtb1.populate_dispatch_table(
        logsumexp_over_axis_strided_temps_dispatch_table);

    DispatchTableBuilder<reduction_contig_impl_fn_ptr,
                         LogSumExpOverAxis1TempsContigFactory, td_ns::num_types>
        dtb2;
    dtb2.populate_dispatch_table(
        logsumexp_over_axis1_contig_temps_dispatch_table);

    DispatchTableBuilder<reduction_contig_impl_fn_ptr,
                         LogSumExpOverAxis0TempsContigFactory, td_ns::num_types>
        dtb3;
    dtb3.populate_dispatch_table(
        logsumexp_over_axis0_contig_temps_dispatch_table);
}

} // namespace impl

void init_logsumexp(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;
    {
        using impl::populate_logsumexp_over_axis_dispatch_tables;
        populate_logsumexp_over_axis_dispatch_tables();
        using impl::logsumexp_over_axis0_contig_temps_dispatch_table;
        using impl::logsumexp_over_axis1_contig_temps_dispatch_table;
        using impl::logsumexp_over_axis_strided_temps_dispatch_table;

        using dpctl::tensor::kernels::reduction_contig_impl_fn_ptr;
        using dpctl::tensor::kernels::reduction_strided_impl_fn_ptr;

        auto logsumexp_pyapi = [&](const arrayT &src,
                                   int trailing_dims_to_reduce,
                                   const arrayT &dst, sycl::queue &exec_q,
                                   const event_vecT &depends = {}) {
            using dpctl::tensor::py_internal::py_tree_reduction_over_axis;
            return py_tree_reduction_over_axis(
                src, trailing_dims_to_reduce, dst, exec_q, depends,
                logsumexp_over_axis_strided_temps_dispatch_table,
                logsumexp_over_axis0_contig_temps_dispatch_table,
                logsumexp_over_axis1_contig_temps_dispatch_table);
        };
        m.def("_logsumexp_over_axis", logsumexp_pyapi, "", py::arg("src"),
              py::arg("trailing_dims_to_reduce"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto logsumexp_dtype_supported = [&](const py::dtype &input_dtype,
                                             const py::dtype &output_dtype) {
            using dpctl::tensor::py_internal::py_tree_reduction_dtype_supported;
            return py_tree_reduction_dtype_supported(
                input_dtype, output_dtype,
                logsumexp_over_axis_strided_temps_dispatch_table);
        };
        m.def("_logsumexp_over_axis_dtype_supported", logsumexp_dtype_supported,
              "", py::arg("arg_dtype"), py::arg("out_dtype"));
    }
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
