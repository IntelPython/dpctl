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
#include <utility>

#include "elementwise_functions.hpp"
#include "multiply.hpp"
#include "utils/type_dispatch.hpp"

#include "kernels/elementwise_functions/common.hpp"
#include "kernels/elementwise_functions/common_inplace.hpp"
#include "kernels/elementwise_functions/multiply.hpp"

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

using ew_cmn_ns::binary_inplace_contig_impl_fn_ptr_t;
using ew_cmn_ns::binary_inplace_row_matrix_broadcast_impl_fn_ptr_t;
using ew_cmn_ns::binary_inplace_strided_impl_fn_ptr_t;

// B19: ===== MULTIPLY (x1, x2)
namespace impl
{

namespace multiply_fn_ns = dpctl::tensor::kernels::multiply;

static binary_contig_impl_fn_ptr_t
    multiply_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static int multiply_output_id_table[td_ns::num_types][td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    multiply_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

// mul(matrix, row)
static binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t
    multiply_contig_matrix_contig_row_broadcast_dispatch_table
        [td_ns::num_types][td_ns::num_types];

// mul(row, matrix)
static binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t
    multiply_contig_row_contig_matrix_broadcast_dispatch_table
        [td_ns::num_types][td_ns::num_types];

static binary_inplace_contig_impl_fn_ptr_t
    multiply_inplace_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static binary_inplace_strided_impl_fn_ptr_t
    multiply_inplace_strided_dispatch_table[td_ns::num_types][td_ns::num_types];
static binary_inplace_row_matrix_broadcast_impl_fn_ptr_t
    multiply_inplace_row_matrix_dispatch_table[td_ns::num_types]
                                              [td_ns::num_types];

void populate_multiply_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = multiply_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::MultiplyTypeMapFactory;
    DispatchTableBuilder<int, MultiplyTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(multiply_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::MultiplyStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t, MultiplyStridedFactory,
                         num_types>
        dtb2;
    dtb2.populate_dispatch_table(multiply_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::MultiplyContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, MultiplyContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(multiply_contig_dispatch_table);

    // function pointers for operation on contiguous matrix, contiguous row
    // with contiguous matrix output
    using fn_ns::MultiplyContigMatrixContigRowBroadcastFactory;
    DispatchTableBuilder<
        binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t,
        MultiplyContigMatrixContigRowBroadcastFactory, num_types>
        dtb4;
    dtb4.populate_dispatch_table(
        multiply_contig_matrix_contig_row_broadcast_dispatch_table);

    // function pointers for operation on contiguous row, contiguous matrix
    // with contiguous matrix output
    using fn_ns::MultiplyContigRowContigMatrixBroadcastFactory;
    DispatchTableBuilder<
        binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t,
        MultiplyContigRowContigMatrixBroadcastFactory, num_types>
        dtb5;
    dtb5.populate_dispatch_table(
        multiply_contig_row_contig_matrix_broadcast_dispatch_table);

    // function pointers for inplace operation on general strided arrays
    using fn_ns::MultiplyInplaceStridedFactory;
    DispatchTableBuilder<binary_inplace_strided_impl_fn_ptr_t,
                         MultiplyInplaceStridedFactory, num_types>
        dtb6;
    dtb6.populate_dispatch_table(multiply_inplace_strided_dispatch_table);

    // function pointers for inplace operation on contiguous inputs and output
    using fn_ns::MultiplyInplaceContigFactory;
    DispatchTableBuilder<binary_inplace_contig_impl_fn_ptr_t,
                         MultiplyInplaceContigFactory, num_types>
        dtb7;
    dtb7.populate_dispatch_table(multiply_inplace_contig_dispatch_table);

    // function pointers for inplace operation on contiguous matrix
    // and contiguous row
    using fn_ns::MultiplyInplaceRowMatrixBroadcastFactory;
    DispatchTableBuilder<binary_inplace_row_matrix_broadcast_impl_fn_ptr_t,
                         MultiplyInplaceRowMatrixBroadcastFactory, num_types>
        dtb8;
    dtb8.populate_dispatch_table(multiply_inplace_row_matrix_dispatch_table);
};

} // namespace impl

void init_multiply(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;
    {
        impl::populate_multiply_dispatch_tables();
        using impl::multiply_contig_dispatch_table;
        using impl::multiply_contig_matrix_contig_row_broadcast_dispatch_table;
        using impl::multiply_contig_row_contig_matrix_broadcast_dispatch_table;
        using impl::multiply_output_id_table;
        using impl::multiply_strided_dispatch_table;

        auto multiply_pyapi = [&](const arrayT &src1, const arrayT &src2,
                                  const arrayT &dst, sycl::queue &exec_q,
                                  const event_vecT &depends = {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends, multiply_output_id_table,
                // function pointers to handle operation on contiguous
                // arrays (pointers may be nullptr)
                multiply_contig_dispatch_table,
                // function pointers to handle operation on strided arrays
                // (most general case)
                multiply_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix
                // and c-contig row with broadcasting (may be nullptr)
                multiply_contig_matrix_contig_row_broadcast_dispatch_table,
                // function pointers to handle operation of c-contig matrix
                // and c-contig row with broadcasting (may be nullptr)
                multiply_contig_row_contig_matrix_broadcast_dispatch_table);
        };
        auto multiply_result_type_pyapi = [&](const py::dtype &dtype1,
                                              const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               multiply_output_id_table);
        };
        m.def("_multiply", multiply_pyapi, "", py::arg("src1"), py::arg("src2"),
              py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_multiply_result_type", multiply_result_type_pyapi, "");

        using impl::multiply_inplace_contig_dispatch_table;
        using impl::multiply_inplace_row_matrix_dispatch_table;
        using impl::multiply_inplace_strided_dispatch_table;

        auto multiply_inplace_pyapi = [&](const arrayT &src, const arrayT &dst,
                                          sycl::queue &exec_q,
                                          const event_vecT &depends = {}) {
            return py_binary_inplace_ufunc(
                src, dst, exec_q, depends, multiply_output_id_table,
                // function pointers to handle inplace operation on
                // contiguous arrays (pointers may be nullptr)
                multiply_inplace_contig_dispatch_table,
                // function pointers to handle inplace operation on strided
                // arrays (most general case)
                multiply_inplace_strided_dispatch_table,
                // function pointers to handle inplace operation on
                // c-contig matrix with c-contig row with broadcasting
                // (may be nullptr)
                multiply_inplace_row_matrix_dispatch_table);
        };
        m.def("_multiply_inplace", multiply_inplace_pyapi, "", py::arg("lhs"),
              py::arg("rhs"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
    }
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
