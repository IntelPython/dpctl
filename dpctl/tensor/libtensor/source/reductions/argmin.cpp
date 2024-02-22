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

template <typename argTy, typename outTy>
struct TypePairSupportForArgminReductionTemps
{

    static constexpr bool is_defined = std::disjunction< // disjunction is C++17
                                                         // feature, supported
                                                         // by DPC++ input bool
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::int64_t>,
        // input int8_t
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, std::int64_t>,

        // input uint8_t
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::int64_t>,

        // input int16_t
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, std::int64_t>,

        // input uint16_t
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::int64_t>,

        // input int32_t
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, std::int64_t>,
        // input uint32_t
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, std::int64_t>,

        // input int64_t
        td_ns::TypePairDefinedEntry<argTy, std::int64_t, outTy, std::int64_t>,

        // input uint32_t
        td_ns::TypePairDefinedEntry<argTy, std::uint64_t, outTy, std::int64_t>,

        // input half
        td_ns::TypePairDefinedEntry<argTy, sycl::half, outTy, std::int64_t>,

        // input float
        td_ns::TypePairDefinedEntry<argTy, float, outTy, std::int64_t>,

        // input double
        td_ns::TypePairDefinedEntry<argTy, double, outTy, std::int64_t>,

        // input std::complex
        td_ns::TypePairDefinedEntry<argTy,
                                    std::complex<float>,
                                    outTy,
                                    std::int64_t>,

        td_ns::TypePairDefinedEntry<argTy,
                                    std::complex<double>,
                                    outTy,
                                    std::int64_t>,

        // fall-through
        td_ns::NotDefinedEntry>::is_defined;
};

template <typename fnT, typename srcTy, typename dstTy>
struct ArgminOverAxisTempsStridedFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportForArgminReductionTemps<srcTy,
                                                             dstTy>::is_defined)
        {
            if constexpr (std::is_integral_v<srcTy> &&
                          !std::is_same_v<srcTy, bool>) {
                // op for values
                using ReductionOpT = sycl::minimum<srcTy>;
                // op for indices
                using IndexOpT = sycl::minimum<dstTy>;
                return dpctl::tensor::kernels::
                    search_over_group_temps_strided_impl<
                        srcTy, dstTy, ReductionOpT, IndexOpT>;
            }
            else {
                // op for values
                using ReductionOpT = su_ns::Minimum<srcTy>;
                // op for indices
                using IndexOpT = sycl::minimum<dstTy>;
                return dpctl::tensor::kernels::
                    search_over_group_temps_strided_impl<
                        srcTy, dstTy, ReductionOpT, IndexOpT>;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct ArgminOverAxis1TempsContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportForArgminReductionTemps<srcTy,
                                                             dstTy>::is_defined)
        {
            if constexpr (std::is_integral_v<srcTy> &&
                          !std::is_same_v<srcTy, bool>) {
                // op for values
                using ReductionOpT = sycl::minimum<srcTy>;
                // op for indices
                using IndexOpT = sycl::minimum<dstTy>;
                return dpctl::tensor::kernels::
                    search_axis1_over_group_temps_contig_impl<
                        srcTy, dstTy, ReductionOpT, IndexOpT>;
            }
            else {
                // op for values
                using ReductionOpT = su_ns::Minimum<srcTy>;
                // op for indices
                using IndexOpT = sycl::minimum<dstTy>;
                return dpctl::tensor::kernels::
                    search_axis1_over_group_temps_contig_impl<
                        srcTy, dstTy, ReductionOpT, IndexOpT>;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct ArgminOverAxis0TempsContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportForArgminReductionTemps<srcTy,
                                                             dstTy>::is_defined)
        {
            if constexpr (std::is_integral_v<srcTy> &&
                          !std::is_same_v<srcTy, bool>) {
                // op for values
                using ReductionOpT = sycl::minimum<srcTy>;
                // op for indices
                using IndexOpT = sycl::minimum<dstTy>;
                return dpctl::tensor::kernels::
                    search_axis0_over_group_temps_contig_impl<
                        srcTy, dstTy, ReductionOpT, IndexOpT>;
            }
            else {
                // op for values
                using ReductionOpT = su_ns::Minimum<srcTy>;
                // op for indices
                using IndexOpT = sycl::minimum<dstTy>;
                return dpctl::tensor::kernels::
                    search_axis0_over_group_temps_contig_impl<
                        srcTy, dstTy, ReductionOpT, IndexOpT>;
            }
        }
        else {
            return nullptr;
        }
    }
};

void populate_argmin_over_axis_dispatch_tables(void)
{
    using dpctl::tensor::kernels::search_strided_impl_fn_ptr;
    using td_ns::DispatchTableBuilder;

    DispatchTableBuilder<search_strided_impl_fn_ptr,
                         ArgminOverAxisTempsStridedFactory, td_ns::num_types>
        dtb1;
    dtb1.populate_dispatch_table(argmin_over_axis_strided_temps_dispatch_table);

    DispatchTableBuilder<search_contig_impl_fn_ptr,
                         ArgminOverAxis1TempsContigFactory, td_ns::num_types>
        dtb2;
    dtb2.populate_dispatch_table(argmin_over_axis1_contig_temps_dispatch_table);

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
