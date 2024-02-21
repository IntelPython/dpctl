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
#include "utils/type_dispatch_building.hpp"

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

/* @brief Types supported by max reduction code based on atomic_ref */
template <typename argTy, typename outTy>
struct TypePairSupportDataForMaxReductionAtomic
{

    /* value is true if a kernel for <argTy, outTy> must be instantiated, false
     * otherwise */
    // disjunction is C++17 feature, supported by DPC++
    static constexpr bool is_defined = std::disjunction<
        // input int32
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, std::int32_t>,
        // input uint32
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, std::uint32_t>,
        // input int64
        td_ns::TypePairDefinedEntry<argTy, std::int64_t, outTy, std::int64_t>,
        // input uint64
        td_ns::TypePairDefinedEntry<argTy, std::uint64_t, outTy, std::uint64_t>,
        // input float
        td_ns::TypePairDefinedEntry<argTy, float, outTy, float>,
        // input double
        td_ns::TypePairDefinedEntry<argTy, double, outTy, double>,
        // fall-through
        td_ns::NotDefinedEntry>::is_defined;
};

template <typename argTy, typename outTy>
struct TypePairSupportDataForMaxReductionTemps
{

    // disjunction is C++17 feature, supported by DPC++
    static constexpr bool is_defined = std::disjunction<
        // input bool
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, bool>,
        // input int8_t
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, std::int8_t>,
        // input uint8_t
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::uint8_t>,

        // input int16_t
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, std::int16_t>,
        // input uint16_t
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::uint16_t>,

        // input int32_t
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, std::int32_t>,
        // input uint32_t
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, std::uint32_t>,

        // input int64_t
        td_ns::TypePairDefinedEntry<argTy, std::int64_t, outTy, std::int64_t>,

        // input uint32_t
        td_ns::TypePairDefinedEntry<argTy, std::uint64_t, outTy, std::uint64_t>,

        // input half
        td_ns::TypePairDefinedEntry<argTy, sycl::half, outTy, sycl::half>,

        // input float
        td_ns::TypePairDefinedEntry<argTy, float, outTy, float>,

        // input double
        td_ns::TypePairDefinedEntry<argTy, double, outTy, double>,

        // input std::complex
        td_ns::TypePairDefinedEntry<argTy,
                                    std::complex<float>,
                                    outTy,
                                    std::complex<float>>,

        td_ns::TypePairDefinedEntry<argTy,
                                    std::complex<double>,
                                    outTy,
                                    std::complex<double>>,

        // fall-through
        td_ns::NotDefinedEntry>::is_defined;
};

template <typename fnT, typename srcTy, typename dstTy>
struct MaxOverAxisAtomicStridedFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForMaxReductionAtomic<
                          srcTy, dstTy>::is_defined)
        {
            if constexpr (std::is_floating_point<dstTy>::value) {
                using ReductionOpT = su_ns::Maximum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_over_group_with_atomics_strided_impl<
                        srcTy, dstTy, ReductionOpT>;
            }
            else {
                using ReductionOpT = sycl::maximum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_over_group_with_atomics_strided_impl<
                        srcTy, dstTy, ReductionOpT>;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct MaxOverAxisTempsStridedFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForMaxReductionTemps<
                          srcTy, dstTy>::is_defined) {
            if constexpr (std::is_integral_v<dstTy> &&
                          !std::is_same_v<dstTy, bool>) {
                using ReductionOpT = sycl::maximum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_over_group_temps_strided_impl<srcTy, dstTy,
                                                            ReductionOpT>;
            }
            else {
                using ReductionOpT = su_ns::Maximum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_over_group_temps_strided_impl<srcTy, dstTy,
                                                            ReductionOpT>;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct MaxOverAxis1AtomicContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForMaxReductionAtomic<
                          srcTy, dstTy>::is_defined)
        {
            if constexpr (std::is_floating_point<dstTy>::value) {
                using ReductionOpT = su_ns::Maximum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_axis1_over_group_with_atomics_contig_impl<
                        srcTy, dstTy, ReductionOpT>;
            }
            else {
                using ReductionOpT = sycl::maximum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_axis1_over_group_with_atomics_contig_impl<
                        srcTy, dstTy, ReductionOpT>;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct MaxOverAxis0AtomicContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForMaxReductionAtomic<
                          srcTy, dstTy>::is_defined)
        {
            if constexpr (std::is_floating_point<dstTy>::value) {
                using ReductionOpT = su_ns::Maximum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_axis0_over_group_with_atomics_contig_impl<
                        srcTy, dstTy, ReductionOpT>;
            }
            else {
                using ReductionOpT = sycl::maximum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_axis0_over_group_with_atomics_contig_impl<
                        srcTy, dstTy, ReductionOpT>;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct MaxOverAxis1TempsContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForMaxReductionTemps<
                          srcTy, dstTy>::is_defined) {
            if constexpr (std::is_integral_v<dstTy> &&
                          !std::is_same_v<dstTy, bool>) {
                using ReductionOpT = sycl::maximum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_axis1_over_group_temps_contig_impl<srcTy, dstTy,
                                                                 ReductionOpT>;
            }
            else {
                using ReductionOpT = su_ns::Maximum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_axis1_over_group_temps_contig_impl<srcTy, dstTy,
                                                                 ReductionOpT>;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct MaxOverAxis0TempsContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForMaxReductionTemps<
                          srcTy, dstTy>::is_defined) {
            if constexpr (std::is_integral_v<dstTy> &&
                          !std::is_same_v<dstTy, bool>) {
                using ReductionOpT = sycl::maximum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_axis0_over_group_temps_contig_impl<srcTy, dstTy,
                                                                 ReductionOpT>;
            }
            else {
                using ReductionOpT = su_ns::Maximum<dstTy>;
                return dpctl::tensor::kernels::
                    reduction_axis0_over_group_temps_contig_impl<srcTy, dstTy,
                                                                 ReductionOpT>;
            }
        }
        else {
            return nullptr;
        }
    }
};

void populate_max_over_axis_dispatch_tables(void)
{
    using td_ns::DispatchTableBuilder;

    DispatchTableBuilder<reduction_strided_impl_fn_ptr,
                         MaxOverAxisAtomicStridedFactory, td_ns::num_types>
        dtb1;
    dtb1.populate_dispatch_table(max_over_axis_strided_atomic_dispatch_table);

    DispatchTableBuilder<reduction_strided_impl_fn_ptr,
                         MaxOverAxisTempsStridedFactory, td_ns::num_types>
        dtb2;
    dtb2.populate_dispatch_table(max_over_axis_strided_temps_dispatch_table);

    DispatchTableBuilder<reduction_contig_impl_fn_ptr,
                         MaxOverAxis1AtomicContigFactory, td_ns::num_types>
        dtb3;
    dtb3.populate_dispatch_table(max_over_axis1_contig_atomic_dispatch_table);

    DispatchTableBuilder<reduction_contig_impl_fn_ptr,
                         MaxOverAxis0AtomicContigFactory, td_ns::num_types>
        dtb4;
    dtb4.populate_dispatch_table(max_over_axis0_contig_atomic_dispatch_table);

    DispatchTableBuilder<reduction_contig_impl_fn_ptr,
                         MaxOverAxis1TempsContigFactory, td_ns::num_types>
        dtb5;
    dtb5.populate_dispatch_table(max_over_axis1_contig_temps_dispatch_table);

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
