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

#include "dpctl4pybind11.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sycl/sycl.hpp>

#include "utils/math_utils.hpp"
#include "utils/memory_overlap.hpp"
#include "utils/output_validation.hpp"
#include "utils/type_dispatch.hpp"

#include "kernels/sorting.hpp"
#include "sort.hpp"
#include "sorting_common.hpp"

namespace td_ns = dpctl::tensor::type_dispatch;

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

template <typename sorting_contig_impl_fnT>
std::pair<sycl::event, sycl::event>
py_sort(const dpctl::tensor::usm_ndarray &src,
        const int trailing_dims_to_sort,
        const dpctl::tensor::usm_ndarray &dst,
        sycl::queue &exec_q,
        const std::vector<sycl::event> &depends,
        const sorting_contig_impl_fnT &stable_sort_contig_fns)
{
    int src_nd = src.get_ndim();
    int dst_nd = dst.get_ndim();
    if (src_nd != dst_nd) {
        throw py::value_error("The input and output arrays must have "
                              "the same array ranks");
    }
    int iteration_nd = src_nd - trailing_dims_to_sort;
    if (trailing_dims_to_sort <= 0 || iteration_nd < 0) {
        throw py::value_error("Trailing_dim_to_sort must be positive, but no "
                              "greater than rank of the array being sorted");
    }

    const py::ssize_t *src_shape_ptr = src.get_shape_raw();
    const py::ssize_t *dst_shape_ptr = dst.get_shape_raw();

    bool same_shapes = true;
    size_t iter_nelems(1);

    for (int i = 0; same_shapes && (i < iteration_nd); ++i) {
        auto src_shape_i = src_shape_ptr[i];
        same_shapes = same_shapes && (src_shape_i == dst_shape_ptr[i]);
        iter_nelems *= static_cast<size_t>(src_shape_i);
    }

    size_t sort_nelems(1);
    for (int i = iteration_nd; same_shapes && (i < src_nd); ++i) {
        auto src_shape_i = src_shape_ptr[i];
        same_shapes = same_shapes && (src_shape_i == dst_shape_ptr[i]);
        sort_nelems *= static_cast<size_t>(src_shape_i);
    }

    if (!same_shapes) {
        throw py::value_error(
            "Destination shape does not match the input shape");
    }

    if (!dpctl::utils::queues_are_compatible(exec_q, {src, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    if ((iter_nelems == 0) || (sort_nelems == 0)) {
        // Nothing to do
        return std::make_pair(sycl::event(), sycl::event());
    }

    // check that dst and src do not overlap
    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(src, dst)) {
        throw py::value_error("Arrays index overlapping segments of memory");
    }

    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(
        dst, sort_nelems * iter_nelems);

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    const auto &array_types = td_ns::usm_ndarray_types();
    int src_typeid = array_types.typenum_to_lookup_id(src_typenum);
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    if (src_typeid != dst_typeid) {
        throw py::value_error("Both input arrays must have "
                              "the same value data type");
    }

    // handle special case when both reduction and iteration are 1D contiguous
    bool is_src_c_contig = src.is_c_contiguous();
    bool is_dst_c_contig = dst.is_c_contiguous();

    if (is_src_c_contig && is_dst_c_contig) {
        using dpctl::tensor::kernels::stable_sort_axis1_contig_impl;

        static constexpr py::ssize_t zero_offset = py::ssize_t(0);

        auto fn = stable_sort_contig_fns[src_typeid];

        sycl::event comp_ev =
            fn(exec_q, iter_nelems, sort_nelems, src.get_data(), dst.get_data(),
               zero_offset, zero_offset, zero_offset, zero_offset, depends);

        sycl::event keep_args_alive_ev =
            dpctl::utils::keep_args_alive(exec_q, {src, dst}, {comp_ev});

        return std::make_pair(keep_args_alive_ev, comp_ev);
    }

    return std::make_pair(sycl::event(), sycl::event());
}

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
