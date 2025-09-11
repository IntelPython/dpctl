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

#include "dpctl4pybind11.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sycl/sycl.hpp>

#include "utils/math_utils.hpp"
#include "utils/memory_overlap.hpp"
#include "utils/output_validation.hpp"
#include "utils/rich_comparisons.hpp"
#include "utils/type_dispatch.hpp"

#include "kernels/sorting/merge_sort.hpp"
#include "kernels/sorting/sort_impl_fn_ptr_t.hpp"

#include "merge_argsort.hpp"
#include "py_argsort_common.hpp"

namespace td_ns = dpctl::tensor::type_dispatch;

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

using dpctl::tensor::kernels::sort_contig_fn_ptr_t;
static sort_contig_fn_ptr_t
    ascending_argsort_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static sort_contig_fn_ptr_t
    descending_argsort_contig_dispatch_table[td_ns::num_types]
                                            [td_ns::num_types];

template <typename fnT, typename argTy, typename IndexTy>
struct AscendingArgSortContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<IndexTy, std::int64_t> ||
                      std::is_same_v<IndexTy, std::int32_t>)
        {
            using dpctl::tensor::rich_comparisons::AscendingSorter;
            using Comp = typename AscendingSorter<argTy>::type;

            using dpctl::tensor::kernels::stable_argsort_axis1_contig_impl;
            return stable_argsort_axis1_contig_impl<argTy, IndexTy, Comp>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename argTy, typename IndexTy>
struct DescendingArgSortContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<IndexTy, std::int64_t> ||
                      std::is_same_v<IndexTy, std::int32_t>)
        {
            using dpctl::tensor::rich_comparisons::DescendingSorter;
            using Comp = typename DescendingSorter<argTy>::type;

            using dpctl::tensor::kernels::stable_argsort_axis1_contig_impl;
            return stable_argsort_axis1_contig_impl<argTy, IndexTy, Comp>;
        }
        else {
            return nullptr;
        }
    }
};

void init_merge_argsort_dispatch_tables(void)
{
    using dpctl::tensor::kernels::sort_contig_fn_ptr_t;

    td_ns::DispatchTableBuilder<sort_contig_fn_ptr_t,
                                AscendingArgSortContigFactory, td_ns::num_types>
        dtb1;
    dtb1.populate_dispatch_table(ascending_argsort_contig_dispatch_table);

    td_ns::DispatchTableBuilder<
        sort_contig_fn_ptr_t, DescendingArgSortContigFactory, td_ns::num_types>
        dtb2;
    dtb2.populate_dispatch_table(descending_argsort_contig_dispatch_table);
}

void init_merge_argsort_functions(py::module_ m)
{
    dpctl::tensor::py_internal::init_merge_argsort_dispatch_tables();

    auto py_argsort_ascending = [](const dpctl::tensor::usm_ndarray &src,
                                   const int trailing_dims_to_sort,
                                   const dpctl::tensor::usm_ndarray &dst,
                                   sycl::queue &exec_q,
                                   const std::vector<sycl::event> &depends)
        -> std::pair<sycl::event, sycl::event> {
        return dpctl::tensor::py_internal::py_argsort(
            src, trailing_dims_to_sort, dst, exec_q, depends,
            dpctl::tensor::py_internal::
                ascending_argsort_contig_dispatch_table);
    };
    m.def("_argsort_ascending", py_argsort_ascending, py::arg("src"),
          py::arg("trailing_dims_to_sort"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    auto py_argsort_descending = [](const dpctl::tensor::usm_ndarray &src,
                                    const int trailing_dims_to_sort,
                                    const dpctl::tensor::usm_ndarray &dst,
                                    sycl::queue &exec_q,
                                    const std::vector<sycl::event> &depends)
        -> std::pair<sycl::event, sycl::event> {
        return dpctl::tensor::py_internal::py_argsort(
            src, trailing_dims_to_sort, dst, exec_q, depends,
            dpctl::tensor::py_internal::
                descending_argsort_contig_dispatch_table);
    };
    m.def("_argsort_descending", py_argsort_descending, py::arg("src"),
          py::arg("trailing_dims_to_sort"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    return;
}

} // end of namespace py_internal
} // end of namespace tensor
} // end of namespace dpctl
