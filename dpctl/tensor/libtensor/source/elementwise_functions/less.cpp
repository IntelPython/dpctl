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
#include <CL/sycl.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "elementwise_functions.hpp"
#include "less.hpp"
#include "utils/type_dispatch.hpp"

#include "kernels/elementwise_functions/common.hpp"
#include "kernels/elementwise_functions/less.hpp"

namespace py = pybind11;

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

namespace td_ns = dpctl::tensor::type_dispatch;

namespace ew_cmn_ns = dpctl::tensor::kernels::elementwise_common;
using ew_cmn_ns::binary_contig_impl_fn_ptr_t;
using ew_cmn_ns::binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t;
using ew_cmn_ns::binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t;
using ew_cmn_ns::binary_strided_impl_fn_ptr_t;

// B13: ===== LESS (x1, x2)
namespace impl
{
namespace less_fn_ns = dpctl::tensor::kernels::less;

static binary_contig_impl_fn_ptr_t less_contig_dispatch_table[td_ns::num_types]
                                                             [td_ns::num_types];
static int less_output_id_table[td_ns::num_types][td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    less_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

void populate_less_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = less_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::LessTypeMapFactory;
    DispatchTableBuilder<int, LessTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(less_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::LessStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t, LessStridedFactory,
                         num_types>
        dtb2;
    dtb2.populate_dispatch_table(less_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::LessContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, LessContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(less_contig_dispatch_table);
};

} // namespace impl

void init_less(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;
    {
        impl::populate_less_dispatch_tables();
        using impl::less_contig_dispatch_table;
        using impl::less_output_id_table;
        using impl::less_strided_dispatch_table;

        auto less_pyapi = [&](const arrayT &src1, const arrayT &src2,
                              const arrayT &dst, sycl::queue &exec_q,
                              const event_vecT &depends = {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends, less_output_id_table,
                // function pointers to handle operation on contiguous arrays
                // (pointers may be nullptr)
                less_contig_dispatch_table,
                // function pointers to handle operation on strided arrays (most
                // general case)
                less_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t>{},
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t>{});
        };
        auto less_result_type_pyapi = [&](const py::dtype &dtype1,
                                          const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               less_output_id_table);
        };
        m.def("_less", less_pyapi, "", py::arg("src1"), py::arg("src2"),
              py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_less_result_type", less_result_type_pyapi, "");
    }
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
