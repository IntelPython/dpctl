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
/// This file defines functions of dpctl.tensor._tensor_sorting_impl
/// extension.
//===--------------------------------------------------------------------===//

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpctl4pybind11.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "utils/memory_overlap.hpp"
#include "utils/offset_utils.hpp"
#include "utils/output_validation.hpp"
#include "utils/sycl_alloc_utils.hpp"
#include "utils/type_dispatch.hpp"

#include "kernels/dpctl_tensor_types.hpp"
#include "kernels/sorting/radix_sort.hpp"
#include "kernels/sorting/sort_impl_fn_ptr_t.hpp"

#include "py_argsort_common.hpp"
#include "radix_argsort.hpp"
#include "radix_sort_support.hpp"

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

namespace td_ns = dpctl::tensor::type_dispatch;
namespace impl_ns = dpctl::tensor::kernels::radix_sort_details;

using dpctl::tensor::kernels::sort_contig_fn_ptr_t;

static sort_contig_fn_ptr_t
    ascending_radix_argsort_contig_dispatch_table[td_ns::num_types]
                                                 [td_ns::num_types];
static sort_contig_fn_ptr_t
    descending_radix_argsort_contig_dispatch_table[td_ns::num_types]
                                                  [td_ns::num_types];

namespace
{

template <bool is_ascending, typename T, typename I>
sycl::event argsort_axis1_contig_caller(sycl::queue &q,
                                        std::size_t iter_nelems,
                                        std::size_t sort_nelems,
                                        const char *arg_cp,
                                        char *res_cp,
                                        ssize_t iter_arg_offset,
                                        ssize_t iter_res_offset,
                                        ssize_t sort_arg_offset,
                                        ssize_t sort_res_offset,
                                        const std::vector<sycl::event> &depends)
{
    using dpctl::tensor::kernels::radix_argsort_axis1_contig_impl;

    return radix_argsort_axis1_contig_impl<T, I>(
        q, is_ascending, iter_nelems, sort_nelems, arg_cp, res_cp,
        iter_arg_offset, iter_res_offset, sort_arg_offset, sort_res_offset,
        depends);
}

} // end of anonymous namespace

template <typename fnT, typename argTy, typename IndexTy>
struct AscendingRadixArgSortContigFactory
{
    fnT get()
    {
        if constexpr (RadixSortSupportVector<argTy>::is_defined &&
                      (std::is_same_v<IndexTy, std::int64_t> ||
                       std::is_same_v<IndexTy, std::int32_t>))
        {
            return argsort_axis1_contig_caller<
                /*ascending*/ true, argTy, IndexTy>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename argTy, typename IndexTy>
struct DescendingRadixArgSortContigFactory
{
    fnT get()
    {
        if constexpr (RadixSortSupportVector<argTy>::is_defined &&
                      (std::is_same_v<IndexTy, std::int64_t> ||
                       std::is_same_v<IndexTy, std::int32_t>))
        {
            return argsort_axis1_contig_caller<
                /*ascending*/ false, argTy, IndexTy>;
        }
        else {
            return nullptr;
        }
    }
};

void init_radix_argsort_dispatch_tables(void)
{
    using dpctl::tensor::kernels::sort_contig_fn_ptr_t;

    td_ns::DispatchTableBuilder<sort_contig_fn_ptr_t,
                                AscendingRadixArgSortContigFactory,
                                td_ns::num_types>
        dtb1;
    dtb1.populate_dispatch_table(ascending_radix_argsort_contig_dispatch_table);

    td_ns::DispatchTableBuilder<sort_contig_fn_ptr_t,
                                DescendingRadixArgSortContigFactory,
                                td_ns::num_types>
        dtb2;
    dtb2.populate_dispatch_table(
        descending_radix_argsort_contig_dispatch_table);
}

void init_radix_argsort_functions(py::module_ m)
{
    dpctl::tensor::py_internal::init_radix_argsort_dispatch_tables();

    auto py_radix_argsort_ascending =
        [](const dpctl::tensor::usm_ndarray &src,
           const int trailing_dims_to_sort,
           const dpctl::tensor::usm_ndarray &dst, sycl::queue &exec_q,
           const std::vector<sycl::event> &depends)
        -> std::pair<sycl::event, sycl::event> {
        return dpctl::tensor::py_internal::py_argsort(
            src, trailing_dims_to_sort, dst, exec_q, depends,
            dpctl::tensor::py_internal::
                ascending_radix_argsort_contig_dispatch_table);
    };
    m.def("_radix_argsort_ascending", py_radix_argsort_ascending,
          py::arg("src"), py::arg("trailing_dims_to_sort"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    auto py_radix_argsort_descending =
        [](const dpctl::tensor::usm_ndarray &src,
           const int trailing_dims_to_sort,
           const dpctl::tensor::usm_ndarray &dst, sycl::queue &exec_q,
           const std::vector<sycl::event> &depends)
        -> std::pair<sycl::event, sycl::event> {
        return dpctl::tensor::py_internal::py_argsort(
            src, trailing_dims_to_sort, dst, exec_q, depends,
            dpctl::tensor::py_internal::
                descending_radix_argsort_contig_dispatch_table);
    };
    m.def("_radix_argsort_descending", py_radix_argsort_descending,
          py::arg("src"), py::arg("trailing_dims_to_sort"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    return;
}

} // namespace py_internal
} // end of namespace tensor
} // end of namespace dpctl
