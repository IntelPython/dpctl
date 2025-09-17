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
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

#include "dpctl4pybind11.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kernels/sorting/isin.hpp"
#include "utils/memory_overlap.hpp"
#include "utils/output_validation.hpp"
#include "utils/sycl_alloc_utils.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

#include "simplify_iteration_space.hpp"

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;
namespace tu_ns = dpctl::tensor::type_utils;

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

namespace detail
{

using dpctl::tensor::kernels::isin_contig_impl_fp_ptr_t;

static isin_contig_impl_fp_ptr_t
    isin_contig_impl_dispatch_vector[td_ns::num_types];

template <typename fnT, typename argTy> struct IsinContigFactory
{
    constexpr IsinContigFactory() {}

    fnT get() const
    {
        using dpctl::tensor::kernels::isin_contig_impl;
        return isin_contig_impl<argTy>;
    }
};

using dpctl::tensor::kernels::isin_strided_impl_fp_ptr_t;

static isin_strided_impl_fp_ptr_t
    isin_strided_impl_dispatch_vector[td_ns::num_types];

template <typename fnT, typename argTy> struct IsinStridedFactory
{
    constexpr IsinStridedFactory() {}

    fnT get() const
    {
        using dpctl::tensor::kernels::isin_strided_impl;
        return isin_strided_impl<argTy>;
    }
};

void init_isin_dispatch_vector(void)
{

    // Contiguous input function dispatch
    td_ns::DispatchVectorBuilder<isin_contig_impl_fp_ptr_t, IsinContigFactory,
                                 td_ns::num_types>
        dvb1;
    dvb1.populate_dispatch_vector(isin_contig_impl_dispatch_vector);

    // Strided input function dispatch
    td_ns::DispatchVectorBuilder<isin_strided_impl_fp_ptr_t, IsinStridedFactory,
                                 td_ns::num_types>
        dvb2;
    dvb2.populate_dispatch_vector(isin_strided_impl_dispatch_vector);
}

} // namespace detail

/*! @brief search for needle from needles in sorted hay */
std::pair<sycl::event, sycl::event>
py_isin(const dpctl::tensor::usm_ndarray &needles,
        const dpctl::tensor::usm_ndarray &hay,
        const dpctl::tensor::usm_ndarray &dst,
        sycl::queue &exec_q,
        const bool invert,
        const std::vector<sycl::event> &depends)
{
    const int hay_nd = hay.get_ndim();
    const int needles_nd = needles.get_ndim();
    const int dst_nd = dst.get_ndim();

    if (hay_nd != 1 || needles_nd != dst_nd) {
        throw py::value_error("Array dimensions mismatch");
    }

    // check that needle and dst have the same shape
    std::size_t needles_nelems(1);
    bool same_shape(true);

    const std::size_t hay_nelems = static_cast<std::size_t>(hay.get_shape(0));

    const py::ssize_t *needles_shape_ptr = needles.get_shape_raw();
    const py::ssize_t *dst_shape_ptr = needles.get_shape_raw();

    for (int i = 0; (i < needles_nd) && same_shape; ++i) {
        const auto needles_sh_i = needles_shape_ptr[i];
        const auto dst_sh_i = dst_shape_ptr[i];

        same_shape = same_shape && (needles_sh_i == dst_sh_i);
        needles_nelems *= static_cast<std::size_t>(needles_sh_i);
    }

    if (!same_shape) {
        throw py::value_error(
            "Array of values to search for and array of their "
            "dst do not have the same shape");
    }

    // check that dst is ample enough
    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(dst,
                                                               needles_nelems);

    // check that dst is writable
    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    // check that queues are compatible
    if (!dpctl::utils::queues_are_compatible(exec_q, {hay, needles, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    // if output array overlaps with input arrays, race condition results
    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(dst, hay) || overlap(dst, needles)) {
        throw py::value_error("Destination array overlaps with input.");
    }

    const int hay_typenum = hay.get_typenum();
    const int needles_typenum = needles.get_typenum();
    const int dst_typenum = dst.get_typenum();

    auto const &array_types = td_ns::usm_ndarray_types();
    const int hay_typeid = array_types.typenum_to_lookup_id(hay_typenum);
    const int needles_typeid =
        array_types.typenum_to_lookup_id(needles_typenum);
    const int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    // check hay and needle have the same data-type
    if (needles_typeid != hay_typeid) {
        throw py::value_error(
            "Hay array and needles array must have the same data types");
    }
    // check that dst has boolean data type
    const auto dst_typenum_t_v = static_cast<td_ns::typenum_t>(dst_typeid);
    if (dst_typenum_t_v != td_ns::typenum_t::BOOL) {
        throw py::value_error("dst array must have data-type bool");
    }

    if (needles_nelems == 0) {
        // Nothing to do
        return std::make_pair(sycl::event{}, sycl::event{});
    }

    // if all inputs are contiguous call contiguous implementations
    // otherwise call strided implementation
    const bool hay_is_c_contig = hay.is_c_contiguous();
    const bool hay_is_f_contig = hay.is_f_contiguous();

    const bool needles_is_c_contig = needles.is_c_contiguous();
    const bool needles_is_f_contig = needles.is_f_contiguous();

    const bool dst_is_c_contig = dst.is_c_contiguous();
    const bool dst_is_f_contig = dst.is_f_contiguous();

    const bool all_c_contig =
        (hay_is_c_contig && needles_is_c_contig && dst_is_c_contig);
    const bool all_f_contig =
        (hay_is_f_contig && needles_is_f_contig && dst_is_f_contig);

    const char *hay_data = hay.get_data();
    const char *needles_data = needles.get_data();

    char *dst_data = dst.get_data();

    if (all_c_contig || all_f_contig) {
        auto fn = detail::isin_contig_impl_dispatch_vector[hay_typeid];

        static constexpr py::ssize_t zero_offset(0);

        sycl::event comp_ev = fn(exec_q, invert, hay_nelems, needles_nelems,
                                 hay_data, zero_offset, needles_data,
                                 zero_offset, dst_data, zero_offset, depends);

        return std::make_pair(dpctl::utils::keep_args_alive(
                                  exec_q, {hay, needles, dst}, {comp_ev}),
                              comp_ev);
    }

    // strided case

    const auto &needles_strides = needles.get_strides_vector();
    const auto &dst_strides = dst.get_strides_vector();

    int simplified_nd = needles_nd;

    using shT = std::vector<py::ssize_t>;
    shT simplified_common_shape;
    shT simplified_needles_strides;
    shT simplified_dst_strides;
    py::ssize_t needles_offset(0);
    py::ssize_t dst_offset(0);

    if (simplified_nd == 0) {
        // needles and dst have same nd
        simplified_nd = 1;
        simplified_common_shape.push_back(1);
        simplified_needles_strides.push_back(0);
        simplified_dst_strides.push_back(0);
    }
    else {
        dpctl::tensor::py_internal::simplify_iteration_space(
            // modified by refernce
            simplified_nd,
            // read-only inputs
            needles_shape_ptr, needles_strides, dst_strides,
            // output, modified by reference
            simplified_common_shape, simplified_needles_strides,
            simplified_dst_strides, needles_offset, dst_offset);
    }
    std::vector<sycl::event> host_task_events;
    host_task_events.reserve(2);

    using dpctl::tensor::offset_utils::device_allocate_and_pack;

    auto ptr_size_event_tuple = device_allocate_and_pack<py::ssize_t>(
        exec_q, host_task_events,
        // vectors being packed
        simplified_common_shape, simplified_needles_strides,
        simplified_dst_strides);
    auto packed_shape_strides_owner =
        std::move(std::get<0>(ptr_size_event_tuple));
    const sycl::event &copy_shape_strides_ev =
        std::get<2>(ptr_size_event_tuple);
    const py::ssize_t *packed_shape_strides = packed_shape_strides_owner.get();

    std::vector<sycl::event> all_deps;
    all_deps.reserve(depends.size() + 1);
    all_deps.insert(all_deps.end(), depends.begin(), depends.end());
    all_deps.push_back(copy_shape_strides_ev);

    auto strided_fn = detail::isin_strided_impl_dispatch_vector[hay_typeid];

    if (!strided_fn) {
        throw std::runtime_error(
            "No implementation for data types of input arrays");
    }

    static constexpr py::ssize_t zero_offset(0);
    py::ssize_t hay_step = hay.get_strides_vector()[0];

    const sycl::event &comp_ev = strided_fn(
        exec_q, invert, hay_nelems, needles_nelems, hay_data, zero_offset,
        hay_step, needles_data, needles_offset, dst_data, dst_offset,
        simplified_nd, packed_shape_strides, all_deps);

    // free packed temporaries
    sycl::event temporaries_cleanup_ev =
        dpctl::tensor::alloc_utils::async_smart_free(
            exec_q, {comp_ev}, packed_shape_strides_owner);

    host_task_events.push_back(temporaries_cleanup_ev);
    const sycl::event &ht_ev = dpctl::utils::keep_args_alive(
        exec_q, {hay, needles, dst}, host_task_events);

    return std::make_pair(ht_ev, comp_ev);
}

void init_isin_functions(py::module_ m)
{
    dpctl::tensor::py_internal::detail::init_isin_dispatch_vector();

    using dpctl::tensor::py_internal::py_isin;
    m.def("_isin", &py_isin, py::arg("needles"), py::arg("hay"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("invert"),
          py::arg("depends") = py::list());
}

} // end of namespace py_internal
} // end of namespace tensor
} // namespace dpctl
