//===-- ------------ Implementation of _tensor_impl module  ----*-C++-*-/===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2025 Intel Corporation
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
#include "utils/sycl_utils.hpp"
#include "utils/type_dispatch_building.hpp"
#include "utils/type_utils.hpp"

#include "reduction_atomic_support.hpp"
#include "reduction_over_axis.hpp"

namespace py = pybind11;

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

namespace su_ns = dpctl::tensor::sycl_utils;
namespace td_ns = dpctl::tensor::type_dispatch;
namespace tu_ns = dpctl::tensor::type_utils;

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
static reduction_contig_impl_fn_ptr
    prod_over_axis1_contig_temps_dispatch_table[td_ns::num_types]
                                               [td_ns::num_types];
static reduction_contig_impl_fn_ptr
    prod_over_axis0_contig_temps_dispatch_table[td_ns::num_types]
                                               [td_ns::num_types];

/* @brief Types supported by plus-reduction code based on atomic_ref */
template <typename argTy, typename outTy>
struct TypePairSupportDataForProductReductionAtomic
{

    /* value if true a kernel for <argTy, outTy> must be instantiated, false
     * otherwise */
    static constexpr bool is_defined = std::disjunction<
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::uint64_t>,
        // input int8
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, std::int64_t>,
        // input uint8
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::uint64_t>,
        // input int16
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, std::int64_t>,
        // input uint16
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::uint64_t>,
        // input int32
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, std::int64_t>,
        // input uint32
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, std::uint64_t>,
        // input int64
        td_ns::TypePairDefinedEntry<argTy, std::int64_t, outTy, std::int64_t>,
        // input uint64
        td_ns::TypePairDefinedEntry<argTy, std::uint64_t, outTy, std::uint64_t>,
        // fall-through
        td_ns::NotDefinedEntry>::is_defined;
};

template <typename argTy, typename outTy>
struct TypePairSupportDataForProductReductionTemps
{

    static constexpr bool is_defined = std::disjunction<
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, bool>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::int8_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::uint8_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::int16_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::uint16_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::uint64_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, double>,

        // input int8_t
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, std::int8_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, std::int16_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, double>,

        // input uint8_t
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::uint8_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::int16_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::uint16_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::uint64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, double>,

        // input int16_t
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, std::int16_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, double>,

        // input uint16_t
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::uint16_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::uint64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, double>,

        // input int32_t
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, double>,

        // input uint32_t
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, std::uint64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, double>,

        // input int64_t
        td_ns::TypePairDefinedEntry<argTy, std::int64_t, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int64_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::int64_t, outTy, double>,

        // input uint32_t
        td_ns::TypePairDefinedEntry<argTy, std::uint64_t, outTy, std::uint64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint64_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::uint64_t, outTy, double>,

        // input half
        td_ns::TypePairDefinedEntry<argTy, sycl::half, outTy, sycl::half>,
        td_ns::TypePairDefinedEntry<argTy, sycl::half, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, sycl::half, outTy, double>,
        td_ns::
            TypePairDefinedEntry<argTy, sycl::half, outTy, std::complex<float>>,
        td_ns::TypePairDefinedEntry<argTy,
                                    sycl::half,
                                    outTy,
                                    std::complex<double>>,

        // input float
        td_ns::TypePairDefinedEntry<argTy, float, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, float, outTy, double>,
        td_ns::TypePairDefinedEntry<argTy, float, outTy, std::complex<float>>,
        td_ns::TypePairDefinedEntry<argTy, float, outTy, std::complex<double>>,

        // input double
        td_ns::TypePairDefinedEntry<argTy, double, outTy, double>,
        td_ns::TypePairDefinedEntry<argTy, double, outTy, std::complex<double>>,

        // input std::complex
        td_ns::TypePairDefinedEntry<argTy,
                                    std::complex<float>,
                                    outTy,
                                    std::complex<float>>,
        td_ns::TypePairDefinedEntry<argTy,
                                    std::complex<float>,
                                    outTy,
                                    std::complex<double>>,

        td_ns::TypePairDefinedEntry<argTy,
                                    std::complex<double>,
                                    outTy,
                                    std::complex<double>>,

        // fall-through
        td_ns::NotDefinedEntry>::is_defined;
};

template <typename fnT, typename srcTy, typename dstTy>
struct ProductOverAxisAtomicStridedFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForProductReductionAtomic<
                          srcTy, dstTy>::is_defined)
        {
            using ReductionOpT = sycl::multiplies<dstTy>;
            return dpctl::tensor::kernels::
                reduction_over_group_with_atomics_strided_impl<srcTy, dstTy,
                                                               ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct ProductOverAxisTempsStridedFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForProductReductionTemps<
                          srcTy, dstTy>::is_defined)
        {
            using ReductionOpT = std::conditional_t<
                std::is_same_v<dstTy, bool>, sycl::logical_and<dstTy>,
                std::conditional_t<tu_ns::is_complex_v<dstTy>,
                                   su_ns::Multiplies<dstTy>,
                                   sycl::multiplies<dstTy>>>;
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
struct ProductOverAxis1AtomicContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForProductReductionAtomic<
                          srcTy, dstTy>::is_defined)
        {
            using ReductionOpT = sycl::multiplies<dstTy>;
            return dpctl::tensor::kernels::
                reduction_axis1_over_group_with_atomics_contig_impl<
                    srcTy, dstTy, ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct ProductOverAxis0AtomicContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForProductReductionAtomic<
                          srcTy, dstTy>::is_defined)
        {
            using ReductionOpT = sycl::multiplies<dstTy>;
            return dpctl::tensor::kernels::
                reduction_axis0_over_group_with_atomics_contig_impl<
                    srcTy, dstTy, ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct ProductOverAxis1TempsContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForProductReductionTemps<
                          srcTy, dstTy>::is_defined)
        {
            using ReductionOpT = std::conditional_t<
                std::is_same_v<dstTy, bool>, sycl::logical_and<dstTy>,
                std::conditional_t<tu_ns::is_complex_v<dstTy>,
                                   su_ns::Multiplies<dstTy>,
                                   sycl::multiplies<dstTy>>>;
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
struct ProductOverAxis0TempsContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForProductReductionTemps<
                          srcTy, dstTy>::is_defined)
        {
            using ReductionOpT = std::conditional_t<
                std::is_same_v<dstTy, bool>, sycl::logical_and<dstTy>,
                std::conditional_t<tu_ns::is_complex_v<dstTy>,
                                   su_ns::Multiplies<dstTy>,
                                   sycl::multiplies<dstTy>>>;
            return dpctl::tensor::kernels::
                reduction_axis0_over_group_temps_contig_impl<srcTy, dstTy,
                                                             ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

void populate_prod_over_axis_dispatch_tables(void)
{
    using dpctl::tensor::kernels::reduction_contig_impl_fn_ptr;
    using dpctl::tensor::kernels::reduction_strided_impl_fn_ptr;
    using namespace td_ns;

    DispatchTableBuilder<reduction_strided_impl_fn_ptr,
                         ProductOverAxisAtomicStridedFactory, num_types>
        dtb1;
    dtb1.populate_dispatch_table(prod_over_axis_strided_atomic_dispatch_table);

    DispatchTableBuilder<reduction_strided_impl_fn_ptr,
                         ProductOverAxisTempsStridedFactory, num_types>
        dtb2;
    dtb2.populate_dispatch_table(prod_over_axis_strided_temps_dispatch_table);

    DispatchTableBuilder<reduction_contig_impl_fn_ptr,
                         ProductOverAxis1AtomicContigFactory, num_types>
        dtb3;
    dtb3.populate_dispatch_table(prod_over_axis1_contig_atomic_dispatch_table);

    DispatchTableBuilder<reduction_contig_impl_fn_ptr,
                         ProductOverAxis0AtomicContigFactory, num_types>
        dtb4;
    dtb4.populate_dispatch_table(prod_over_axis0_contig_atomic_dispatch_table);

    DispatchTableBuilder<reduction_contig_impl_fn_ptr,
                         ProductOverAxis1TempsContigFactory, td_ns::num_types>
        dtb5;
    dtb5.populate_dispatch_table(prod_over_axis1_contig_temps_dispatch_table);

    DispatchTableBuilder<reduction_contig_impl_fn_ptr,
                         ProductOverAxis0TempsContigFactory, td_ns::num_types>
        dtb6;
    dtb6.populate_dispatch_table(prod_over_axis0_contig_temps_dispatch_table);
}

using atomic_support::atomic_support_fn_ptr_t;
static atomic_support_fn_ptr_t prod_atomic_support_vector[td_ns::num_types];

void populate_prod_atomic_support_dispatch_vector(void)
{
    using td_ns::DispatchVectorBuilder;

    using atomic_support::ProductAtomicSupportFactory;
    DispatchVectorBuilder<atomic_support_fn_ptr_t, ProductAtomicSupportFactory,
                          td_ns::num_types>
        dvb;
    dvb.populate_dispatch_vector(prod_atomic_support_vector);
}

} // namespace impl

void init_prod(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;
    {
        using impl::populate_prod_over_axis_dispatch_tables;
        populate_prod_over_axis_dispatch_tables();
        using impl::prod_over_axis0_contig_atomic_dispatch_table;
        using impl::prod_over_axis0_contig_temps_dispatch_table;
        using impl::prod_over_axis1_contig_atomic_dispatch_table;
        using impl::prod_over_axis1_contig_temps_dispatch_table;
        using impl::prod_over_axis_strided_atomic_dispatch_table;
        using impl::prod_over_axis_strided_temps_dispatch_table;

        using impl::populate_prod_atomic_support_dispatch_vector;
        populate_prod_atomic_support_dispatch_vector();
        using impl::prod_atomic_support_vector;

        auto prod_pyapi = [&](const arrayT &src, int trailing_dims_to_reduce,
                              const arrayT &dst, sycl::queue &exec_q,
                              const event_vecT &depends = {}) {
            using dpctl::tensor::py_internal::py_reduction_over_axis;
            return py_reduction_over_axis(
                src, trailing_dims_to_reduce, dst, exec_q, depends,
                prod_over_axis_strided_atomic_dispatch_table,
                prod_over_axis0_contig_atomic_dispatch_table,
                prod_over_axis1_contig_atomic_dispatch_table,
                prod_over_axis_strided_temps_dispatch_table,
                prod_over_axis0_contig_temps_dispatch_table,
                prod_over_axis1_contig_temps_dispatch_table,
                prod_atomic_support_vector);
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
                    prod_over_axis_strided_temps_dispatch_table,
                    prod_atomic_support_vector);
            };
        m.def("_prod_over_axis_dtype_supported", prod_dtype_supported, "",
              py::arg("arg_dtype"), py::arg("out_dtype"),
              py::arg("dst_usm_type"), py::arg("sycl_queue"));
    }
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
