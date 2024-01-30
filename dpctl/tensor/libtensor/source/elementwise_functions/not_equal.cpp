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
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sycl/sycl.hpp>
#include <vector>

#include "elementwise_functions.hpp"
#include "not_equal.hpp"
#include "utils/type_dispatch.hpp"

#include "kernels/elementwise_functions/common.hpp"
#include "kernels/elementwise_functions/not_equal.hpp"

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

// B20: ===== NOT_EQUAL (x1, x2)
namespace impl
{
namespace not_equal_fn_ns = dpctl::tensor::kernels::not_equal;

static binary_contig_impl_fn_ptr_t
    not_equal_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static int not_equal_output_id_table[td_ns::num_types][td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    not_equal_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

void populate_not_equal_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = not_equal_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::NotEqualTypeMapFactory;
    DispatchTableBuilder<int, NotEqualTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(not_equal_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::NotEqualStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t, NotEqualStridedFactory,
                         num_types>
        dtb2;
    dtb2.populate_dispatch_table(not_equal_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::NotEqualContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, NotEqualContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(not_equal_contig_dispatch_table);
};

} // namespace impl

void init_not_equal(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;
    {
        impl::populate_not_equal_dispatch_tables();
        using impl::not_equal_contig_dispatch_table;
        using impl::not_equal_output_id_table;
        using impl::not_equal_strided_dispatch_table;

        auto not_equal_pyapi = [&](const arrayT &src1, const arrayT &src2,
                                   const arrayT &dst, sycl::queue &exec_q,
                                   const event_vecT &depends = {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends, not_equal_output_id_table,
                // function pointers to handle operation on contiguous arrays
                // (pointers may be nullptr)
                not_equal_contig_dispatch_table,
                // function pointers to handle operation on strided arrays (most
                // general case)
                not_equal_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t>{},
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t>{});
        };
        auto not_equal_result_type_pyapi = [&](const py::dtype &dtype1,
                                               const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               not_equal_output_id_table);
        };
        m.def("_not_equal", not_equal_pyapi, "", py::arg("src1"),
              py::arg("src2"), py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_not_equal_result_type", not_equal_result_type_pyapi, "");
    }
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
