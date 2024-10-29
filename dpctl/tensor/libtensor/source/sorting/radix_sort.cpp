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

#include <cstdint>
#include <exception>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpctl4pybind11.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "utils/memory_overlap.hpp"
#include "utils/offset_utils.hpp"
#include "utils/output_validation.hpp"
#include "utils/type_dispatch.hpp"

#include "kernels/dpctl_tensor_types.hpp"
#include "kernels/sorting/radix_sort.hpp"
#include "kernels/sorting/sort_impl_fn_ptr_t.hpp"

#include "py_sort_common.hpp"
#include "radix_sort.hpp"
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
    ascending_radix_sort_contig_dispatch_vector[td_ns::num_types];
static sort_contig_fn_ptr_t
    descending_radix_sort_contig_dispatch_vector[td_ns::num_types];

namespace
{

template <bool is_ascending, typename T>
sycl::event sort_axis1_contig_caller(sycl::queue &q,
                                     size_t iter_nelems,
                                     size_t sort_nelems,
                                     const char *arg_cp,
                                     char *res_cp,
                                     ssize_t iter_arg_offset,
                                     ssize_t iter_res_offset,
                                     ssize_t sort_arg_offset,
                                     ssize_t sort_res_offset,
                                     const std::vector<sycl::event> &depends)
{
    using dpctl::tensor::kernels::radix_sort_axis1_contig_impl;

    return radix_sort_axis1_contig_impl<T>(
        q, is_ascending, iter_nelems, sort_nelems, arg_cp, res_cp,
        iter_arg_offset, iter_res_offset, sort_arg_offset, sort_res_offset,
        depends);
}

} // end of anonymous namespace

template <typename fnT, typename argTy> struct AscendingRadixSortContigFactory
{
    fnT get()
    {
        if constexpr (RadixSortSupportVector<argTy>::is_defined) {
            return sort_axis1_contig_caller</*ascending*/ true, argTy>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename argTy> struct DescendingRadixSortContigFactory
{
    fnT get()
    {
        if constexpr (RadixSortSupportVector<argTy>::is_defined) {
            return sort_axis1_contig_caller</*ascending*/ false, argTy>;
        }
        else {
            return nullptr;
        }
    }
};

void init_radix_sort_dispatch_vectors(void)
{
    using dpctl::tensor::kernels::sort_contig_fn_ptr_t;

    td_ns::DispatchVectorBuilder<
        sort_contig_fn_ptr_t, AscendingRadixSortContigFactory, td_ns::num_types>
        dtv1;
    dtv1.populate_dispatch_vector(ascending_radix_sort_contig_dispatch_vector);

    td_ns::DispatchVectorBuilder<sort_contig_fn_ptr_t,
                                 DescendingRadixSortContigFactory,
                                 td_ns::num_types>
        dtv2;
    dtv2.populate_dispatch_vector(descending_radix_sort_contig_dispatch_vector);
}

bool py_radix_sort_defined(int typenum)
{
    const auto &array_types = td_ns::usm_ndarray_types();

    try {
        int type_id = array_types.typenum_to_lookup_id(typenum);
        return (nullptr !=
                ascending_radix_sort_contig_dispatch_vector[type_id]);
    } catch (const std::exception &e) {
        return false;
    }
}

void init_radix_sort_functions(py::module_ m)
{
    dpctl::tensor::py_internal::init_radix_sort_dispatch_vectors();

    auto py_radix_sort_ascending = [](const dpctl::tensor::usm_ndarray &src,
                                      const int trailing_dims_to_sort,
                                      const dpctl::tensor::usm_ndarray &dst,
                                      sycl::queue &exec_q,
                                      const std::vector<sycl::event> &depends)
        -> std::pair<sycl::event, sycl::event> {
        return dpctl::tensor::py_internal::py_sort(
            src, trailing_dims_to_sort, dst, exec_q, depends,
            dpctl::tensor::py_internal::
                ascending_radix_sort_contig_dispatch_vector);
    };
    m.def("_radix_sort_ascending", py_radix_sort_ascending, py::arg("src"),
          py::arg("trailing_dims_to_sort"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    auto py_radix_sort_descending = [](const dpctl::tensor::usm_ndarray &src,
                                       const int trailing_dims_to_sort,
                                       const dpctl::tensor::usm_ndarray &dst,
                                       sycl::queue &exec_q,
                                       const std::vector<sycl::event> &depends)
        -> std::pair<sycl::event, sycl::event> {
        return dpctl::tensor::py_internal::py_sort(
            src, trailing_dims_to_sort, dst, exec_q, depends,
            dpctl::tensor::py_internal::
                descending_radix_sort_contig_dispatch_vector);
    };
    m.def("_radix_sort_descending", py_radix_sort_descending, py::arg("src"),
          py::arg("trailing_dims_to_sort"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    m.def("_radix_sort_dtype_supported", py_radix_sort_defined);

    return;
}

} // namespace py_internal
} // end of namespace tensor
} // end of namespace dpctl
