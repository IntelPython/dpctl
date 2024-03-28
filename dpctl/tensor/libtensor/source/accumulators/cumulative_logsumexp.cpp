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

#include "accumulate_over_axis.hpp"
#include "kernels/accumulators.hpp"
#include "utils/sycl_utils.hpp"
#include "utils/type_dispatch_building.hpp"

namespace py = pybind11;

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

namespace su_ns = dpctl::tensor::sycl_utils;
namespace td_ns = dpctl::tensor::type_dispatch;

namespace impl
{

using dpctl::tensor::kernels::accumulators::accumulate_1d_contig_impl_fn_ptr_t;
static accumulate_1d_contig_impl_fn_ptr_t
    cumlogsumexp_1d_contig_dispatch_table[td_ns::num_types][td_ns::num_types];

using dpctl::tensor::kernels::accumulators::accumulate_strided_impl_fn_ptr_t;
static accumulate_strided_impl_fn_ptr_t
    cumlogsumexp_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

static accumulate_1d_contig_impl_fn_ptr_t
    cumlogsumexp_1d_include_initial_contig_dispatch_table[td_ns::num_types]
                                                         [td_ns::num_types];

static accumulate_strided_impl_fn_ptr_t
    cumlogsumexp_include_initial_strided_dispatch_table[td_ns::num_types]
                                                       [td_ns::num_types];

template <typename argTy, typename outTy>
struct TypePairSupportDataForLogSumExpAccumulation
{
    static constexpr bool is_defined = std::disjunction<
        // disjunction is C++17 feature, supported by DPC++ input bool
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
struct CumLogSumExp1DContigFactory
{
    fnT get()
    {
        if constexpr (TypePairSupportDataForLogSumExpAccumulation<
                          srcTy, dstTy>::is_defined)
        {
            using ScanOpT = su_ns::LogSumExp<dstTy>;
            constexpr bool include_initial = false;
            if constexpr (std::is_same_v<srcTy, dstTy>) {
                using dpctl::tensor::kernels::accumulators::NoOpTransformer;
                fnT fn = dpctl::tensor::kernels::accumulators::
                    accumulate_1d_contig_impl<srcTy, dstTy,
                                              NoOpTransformer<dstTy>, ScanOpT,
                                              include_initial>;
                return fn;
            }
            else {
                using dpctl::tensor::kernels::accumulators::CastTransformer;
                fnT fn = dpctl::tensor::kernels::accumulators::
                    accumulate_1d_contig_impl<srcTy, dstTy,
                                              CastTransformer<srcTy, dstTy>,
                                              ScanOpT, include_initial>;
                return fn;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct CumLogSumExp1DIncludeInitialContigFactory
{
    fnT get()
    {
        if constexpr (TypePairSupportDataForLogSumExpAccumulation<
                          srcTy, dstTy>::is_defined)
        {
            using ScanOpT = su_ns::LogSumExp<dstTy>;
            constexpr bool include_initial = true;
            if constexpr (std::is_same_v<srcTy, dstTy>) {
                using dpctl::tensor::kernels::accumulators::NoOpTransformer;
                fnT fn = dpctl::tensor::kernels::accumulators::
                    accumulate_1d_contig_impl<srcTy, dstTy,
                                              NoOpTransformer<dstTy>, ScanOpT,
                                              include_initial>;
                return fn;
            }
            else {
                using dpctl::tensor::kernels::accumulators::CastTransformer;
                fnT fn = dpctl::tensor::kernels::accumulators::
                    accumulate_1d_contig_impl<srcTy, dstTy,
                                              CastTransformer<srcTy, dstTy>,
                                              ScanOpT, include_initial>;
                return fn;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct CumLogSumExpStridedFactory
{
    fnT get()
    {
        if constexpr (TypePairSupportDataForLogSumExpAccumulation<
                          srcTy, dstTy>::is_defined)
        {
            using ScanOpT = su_ns::LogSumExp<dstTy>;
            constexpr bool include_initial = false;
            if constexpr (std::is_same_v<srcTy, dstTy>) {
                using dpctl::tensor::kernels::accumulators::NoOpTransformer;
                fnT fn = dpctl::tensor::kernels::accumulators::
                    accumulate_strided_impl<srcTy, dstTy,
                                            NoOpTransformer<dstTy>, ScanOpT,
                                            include_initial>;
                return fn;
            }
            else {
                using dpctl::tensor::kernels::accumulators::CastTransformer;
                fnT fn = dpctl::tensor::kernels::accumulators::
                    accumulate_strided_impl<srcTy, dstTy,
                                            CastTransformer<srcTy, dstTy>,
                                            ScanOpT, include_initial>;
                return fn;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct CumLogSumExpIncludeInitialStridedFactory
{
    fnT get()
    {
        if constexpr (TypePairSupportDataForLogSumExpAccumulation<
                          srcTy, dstTy>::is_defined)
        {
            using ScanOpT = su_ns::LogSumExp<dstTy>;
            constexpr bool include_initial = true;
            if constexpr (std::is_same_v<srcTy, dstTy>) {
                using dpctl::tensor::kernels::accumulators::NoOpTransformer;
                fnT fn = dpctl::tensor::kernels::accumulators::
                    accumulate_strided_impl<srcTy, dstTy,
                                            NoOpTransformer<dstTy>, ScanOpT,
                                            include_initial>;
                return fn;
            }
            else {
                using dpctl::tensor::kernels::accumulators::CastTransformer;
                fnT fn = dpctl::tensor::kernels::accumulators::
                    accumulate_strided_impl<srcTy, dstTy,
                                            CastTransformer<srcTy, dstTy>,
                                            ScanOpT, include_initial>;
                return fn;
            }
        }
        else {
            return nullptr;
        }
    }
};

void populate_cumlogsumexp_dispatch_tables(void)
{
    td_ns::DispatchTableBuilder<accumulate_1d_contig_impl_fn_ptr_t,
                                CumLogSumExp1DContigFactory, td_ns::num_types>
        dtb1;
    dtb1.populate_dispatch_table(cumlogsumexp_1d_contig_dispatch_table);

    td_ns::DispatchTableBuilder<accumulate_strided_impl_fn_ptr_t,
                                CumLogSumExpStridedFactory, td_ns::num_types>
        dtb2;
    dtb2.populate_dispatch_table(cumlogsumexp_strided_dispatch_table);

    td_ns::DispatchTableBuilder<accumulate_1d_contig_impl_fn_ptr_t,
                                CumLogSumExp1DIncludeInitialContigFactory,
                                td_ns::num_types>
        dtb3;
    dtb3.populate_dispatch_table(
        cumlogsumexp_1d_include_initial_contig_dispatch_table);

    td_ns::DispatchTableBuilder<accumulate_strided_impl_fn_ptr_t,
                                CumLogSumExpIncludeInitialStridedFactory,
                                td_ns::num_types>
        dtb4;
    dtb4.populate_dispatch_table(
        cumlogsumexp_include_initial_strided_dispatch_table);

    return;
}

} // namespace impl

void init_cumulative_logsumexp(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;

    using impl::populate_cumlogsumexp_dispatch_tables;
    populate_cumlogsumexp_dispatch_tables();

    using impl::cumlogsumexp_1d_contig_dispatch_table;
    using impl::cumlogsumexp_strided_dispatch_table;
    auto cumlogsumexp_pyapi = [&](const arrayT &src,
                                  int trailing_dims_to_accumulate,
                                  const arrayT &dst, sycl::queue &exec_q,
                                  const event_vecT &depends = {}) {
        using dpctl::tensor::py_internal::py_accumulate_over_axis;
        return py_accumulate_over_axis(src, trailing_dims_to_accumulate, dst,
                                       exec_q, depends,
                                       cumlogsumexp_strided_dispatch_table,
                                       cumlogsumexp_1d_contig_dispatch_table);
    };
    m.def("_cumlogsumexp_over_axis", cumlogsumexp_pyapi, "", py::arg("src"),
          py::arg("trailing_dims_to_accumulate"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    using impl::cumlogsumexp_1d_include_initial_contig_dispatch_table;
    using impl::cumlogsumexp_include_initial_strided_dispatch_table;
    auto cumlogsumexp_include_initial_pyapi =
        [&](const arrayT &src, const arrayT &dst, sycl::queue &exec_q,
            const event_vecT &depends = {}) {
            using dpctl::tensor::py_internal::
                py_accumulate_final_axis_include_initial;
            return py_accumulate_final_axis_include_initial(
                src, dst, exec_q, depends,
                cumlogsumexp_include_initial_strided_dispatch_table,
                cumlogsumexp_1d_include_initial_contig_dispatch_table);
        };
    m.def("_cumlogsumexp_final_axis_include_initial",
          cumlogsumexp_include_initial_pyapi, "", py::arg("src"),
          py::arg("dst"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());

    auto cumlogsumexp_dtype_supported = [&](const py::dtype &input_dtype,
                                            const py::dtype &output_dtype) {
        using dpctl::tensor::py_internal::py_accumulate_dtype_supported;
        return py_accumulate_dtype_supported(
            input_dtype, output_dtype, cumlogsumexp_strided_dispatch_table);
    };
    m.def("_cumlogsumexp_dtype_supported", cumlogsumexp_dtype_supported, "",
          py::arg("arg_dtype"), py::arg("out_dtype"));
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
