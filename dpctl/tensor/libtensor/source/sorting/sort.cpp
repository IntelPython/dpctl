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
/// This file defines functions of dpctl.tensor._tensor_sorting_impl
/// extension.
//===--------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include "dpctl4pybind11.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "utils/math_utils.hpp"
#include "utils/memory_overlap.hpp"
#include "utils/output_validation.hpp"
#include "utils/type_dispatch.hpp"

#include "kernels/sorting/merge_sort.hpp"
#include "rich_comparisons.hpp"
#include "sort.hpp"

#include "py_sort_common.hpp"

namespace td_ns = dpctl::tensor::type_dispatch;

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

using dpctl::tensor::kernels::sort_contig_fn_ptr_t;
static sort_contig_fn_ptr_t
    ascending_sort_contig_dispatch_vector[td_ns::num_types];
static sort_contig_fn_ptr_t
    descending_sort_contig_dispatch_vector[td_ns::num_types];

template <typename fnT, typename argTy> struct AscendingSortContigFactory
{
    fnT get()
    {
        using Comp = typename AscendingSorter<argTy>::type;

        using dpctl::tensor::kernels::stable_sort_axis1_contig_impl;
        return stable_sort_axis1_contig_impl<argTy, Comp>;
    }
};

template <typename fnT, typename argTy> struct DescendingSortContigFactory
{
    fnT get()
    {
        using Comp = typename DescendingSorter<argTy>::type;
        using dpctl::tensor::kernels::stable_sort_axis1_contig_impl;
        return stable_sort_axis1_contig_impl<argTy, Comp>;
    }
};

void init_sort_dispatch_vectors(void)
{
    using dpctl::tensor::kernels::sort_contig_fn_ptr_t;

    td_ns::DispatchVectorBuilder<sort_contig_fn_ptr_t,
                                 AscendingSortContigFactory, td_ns::num_types>
        dtv1;
    dtv1.populate_dispatch_vector(ascending_sort_contig_dispatch_vector);

    td_ns::DispatchVectorBuilder<sort_contig_fn_ptr_t,
                                 DescendingSortContigFactory, td_ns::num_types>
        dtv2;
    dtv2.populate_dispatch_vector(descending_sort_contig_dispatch_vector);
}

void init_sort_functions(py::module_ m)
{
    dpctl::tensor::py_internal::init_sort_dispatch_vectors();

    auto py_sort_ascending = [](const dpctl::tensor::usm_ndarray &src,
                                const int trailing_dims_to_sort,
                                const dpctl::tensor::usm_ndarray &dst,
                                sycl::queue &exec_q,
                                const std::vector<sycl::event> &depends)
        -> std::pair<sycl::event, sycl::event> {
        return dpctl::tensor::py_internal::py_sort(
            src, trailing_dims_to_sort, dst, exec_q, depends,
            dpctl::tensor::py_internal::ascending_sort_contig_dispatch_vector);
    };
    m.def("_sort_ascending", py_sort_ascending, py::arg("src"),
          py::arg("trailing_dims_to_sort"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    auto py_sort_descending = [](const dpctl::tensor::usm_ndarray &src,
                                 const int trailing_dims_to_sort,
                                 const dpctl::tensor::usm_ndarray &dst,
                                 sycl::queue &exec_q,
                                 const std::vector<sycl::event> &depends)
        -> std::pair<sycl::event, sycl::event> {
        return dpctl::tensor::py_internal::py_sort(
            src, trailing_dims_to_sort, dst, exec_q, depends,
            dpctl::tensor::py_internal::descending_sort_contig_dispatch_vector);
    };
    m.def("_sort_descending", py_sort_descending, py::arg("src"),
          py::arg("trailing_dims_to_sort"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    return;
}

} // end of namespace py_internal
} // end of namespace tensor
} // end of namespace dpctl
