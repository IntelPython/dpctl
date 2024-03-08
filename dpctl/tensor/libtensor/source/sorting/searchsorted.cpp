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
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kernels/sorting/searchsorted.hpp"
#include "utils/memory_overlap.hpp"
#include "utils/output_validation.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"
#include <dpctl4pybind11.hpp>

#include "rich_comparisons.hpp"
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

using dpctl::tensor::kernels::searchsorted_contig_impl_fp_ptr_t;

static searchsorted_contig_impl_fp_ptr_t
    left_side_searchsorted_contig_impl[td_ns::num_types][td_ns::num_types];

static searchsorted_contig_impl_fp_ptr_t
    right_side_searchsorted_contig_impl[td_ns::num_types][td_ns::num_types];

template <typename fnT, typename argTy, typename indTy>
struct LeftSideSearchSortedContigFactory
{
    constexpr LeftSideSearchSortedContigFactory() {}

    fnT get() const
    {
        if constexpr (std::is_same_v<indTy, std::int32_t> ||
                      std::is_same_v<indTy, std::int64_t>)
        {
            constexpr bool left_side_search(true);
            using dpctl::tensor::kernels::searchsorted_contig_impl;

            using Compare = typename AscendingSorter<argTy>::type;

            return searchsorted_contig_impl<argTy, indTy, left_side_search,
                                            Compare>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename argTy, typename indTy>
struct RightSideSearchSortedContigFactory
{
    constexpr RightSideSearchSortedContigFactory() {}

    fnT get() const
    {
        if constexpr (std::is_same_v<indTy, std::int32_t> ||
                      std::is_same_v<indTy, std::int64_t>)
        {
            constexpr bool right_side_search(false);
            using dpctl::tensor::kernels::searchsorted_contig_impl;

            using Compare = typename AscendingSorter<argTy>::type;

            return searchsorted_contig_impl<argTy, indTy, right_side_search,
                                            Compare>;
        }
        else {
            return nullptr;
        }
    }
};

using dpctl::tensor::kernels::searchsorted_strided_impl_fp_ptr_t;

static searchsorted_strided_impl_fp_ptr_t
    left_side_searchsorted_strided_impl[td_ns::num_types][td_ns::num_types];

static searchsorted_strided_impl_fp_ptr_t
    right_side_searchsorted_strided_impl[td_ns::num_types][td_ns::num_types];

template <typename fnT, typename argTy, typename indTy>
struct LeftSideSearchSortedStridedFactory
{
    constexpr LeftSideSearchSortedStridedFactory() {}

    fnT get() const
    {
        if constexpr (std::is_same_v<indTy, std::int32_t> ||
                      std::is_same_v<indTy, std::int64_t>)
        {
            constexpr bool left_side_search(true);
            using dpctl::tensor::kernels::searchsorted_strided_impl;

            using Compare = typename AscendingSorter<argTy>::type;

            return searchsorted_strided_impl<argTy, indTy, left_side_search,
                                             Compare>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename argTy, typename indTy>
struct RightSideSearchSortedStridedFactory
{
    constexpr RightSideSearchSortedStridedFactory() {}

    fnT get() const
    {
        if constexpr (std::is_same_v<indTy, std::int32_t> ||
                      std::is_same_v<indTy, std::int64_t>)
        {
            constexpr bool right_side_search(false);
            using dpctl::tensor::kernels::searchsorted_strided_impl;

            using Compare = typename AscendingSorter<argTy>::type;

            return searchsorted_strided_impl<argTy, indTy, right_side_search,
                                             Compare>;
        }
        else {
            return nullptr;
        }
    }
};

void init_searchsorted_dispatch_table(void)
{

    // Contiguous input function dispatch
    td_ns::DispatchTableBuilder<searchsorted_contig_impl_fp_ptr_t,
                                LeftSideSearchSortedContigFactory,
                                td_ns::num_types>
        dtb1;
    dtb1.populate_dispatch_table(left_side_searchsorted_contig_impl);

    td_ns::DispatchTableBuilder<searchsorted_contig_impl_fp_ptr_t,
                                RightSideSearchSortedContigFactory,
                                td_ns::num_types>
        dtb2;
    dtb2.populate_dispatch_table(right_side_searchsorted_contig_impl);

    // Strided input function dispatch
    td_ns::DispatchTableBuilder<searchsorted_strided_impl_fp_ptr_t,
                                LeftSideSearchSortedStridedFactory,
                                td_ns::num_types>
        dtb3;
    dtb3.populate_dispatch_table(left_side_searchsorted_strided_impl);

    td_ns::DispatchTableBuilder<searchsorted_strided_impl_fp_ptr_t,
                                RightSideSearchSortedStridedFactory,
                                td_ns::num_types>
        dtb4;
    dtb4.populate_dispatch_table(right_side_searchsorted_strided_impl);
}

} // namespace detail

/*! @brief search for needle from needles in sorted hay */
std::pair<sycl::event, sycl::event>
py_searchsorted(const dpctl::tensor::usm_ndarray &hay,
                const dpctl::tensor::usm_ndarray &needles,
                const dpctl::tensor::usm_ndarray &positions,
                sycl::queue &exec_q,
                const bool search_left_side,
                const std::vector<sycl::event> &depends)
{
    const int hay_nd = hay.get_ndim();
    const int needles_nd = needles.get_ndim();
    const int positions_nd = positions.get_ndim();

    if (hay_nd != 1 || needles_nd != positions_nd) {
        throw py::value_error("Array dimensions mismatch");
    }

    // check that needle and positions have the same shape
    size_t needles_nelems(1);
    bool same_shape(true);

    const size_t hay_nelems = static_cast<size_t>(hay.get_shape(0));

    const py::ssize_t *needles_shape_ptr = needles.get_shape_raw();
    const py::ssize_t *positions_shape_ptr = needles.get_shape_raw();

    for (int i = 0; (i < needles_nd) && same_shape; ++i) {
        const auto needles_sh_i = needles_shape_ptr[i];
        const auto positions_sh_i = positions_shape_ptr[i];

        same_shape = same_shape && (needles_sh_i == positions_sh_i);
        needles_nelems *= static_cast<size_t>(needles_sh_i);
    }

    if (!same_shape) {
        throw py::value_error(
            "Array of values to search for and array of their "
            "positions do not have the same shape");
    }

    // check that positions is ample enough
    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(positions,
                                                               needles_nelems);

    // check that positions is writable
    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(positions);

    // check that queues are compatible
    if (!dpctl::utils::queues_are_compatible(exec_q, {hay, needles, positions}))
    {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    // if output array overlaps with input arrays, race condition results
    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(positions, hay) || overlap(positions, needles)) {
        throw py::value_error("Destination array overlaps with input.");
    }

    const int hay_typenum = hay.get_typenum();
    const int needles_typenum = needles.get_typenum();
    const int positions_typenum = positions.get_typenum();

    auto const &array_types = td_ns::usm_ndarray_types();
    const int hay_typeid = array_types.typenum_to_lookup_id(hay_typenum);
    const int needles_typeid =
        array_types.typenum_to_lookup_id(needles_typenum);
    const int positions_typeid =
        array_types.typenum_to_lookup_id(positions_typenum);

    // check hay and needle have the same data-type
    if (needles_typeid != hay_typeid) {
        throw py::value_error(
            "Hay array and needles array must have the same data types");
    }
    // check that positions has indexing data-type (int32, or int64)
    const auto positions_typenum_t_v =
        static_cast<td_ns::typenum_t>(positions_typeid);
    if (positions_typenum_t_v != td_ns::typenum_t::INT32 &&
        positions_typenum_t_v != td_ns::typenum_t::INT64)
    {
        throw py::value_error(
            "Positions array must have data-type int32, or int64");
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

    const bool positions_is_c_contig = positions.is_c_contiguous();
    const bool positions_is_f_contig = positions.is_f_contiguous();

    const bool all_c_contig =
        (hay_is_c_contig && needles_is_c_contig && positions_is_c_contig);
    const bool all_f_contig =
        (hay_is_f_contig && needles_is_f_contig && positions_is_f_contig);

    const char *hay_data = hay.get_data();
    const char *needles_data = needles.get_data();

    char *positions_data = positions.get_data();

    if (all_c_contig || all_f_contig) {
        auto fn =
            (search_left_side)
                ? detail::left_side_searchsorted_contig_impl[hay_typeid]
                                                            [positions_typeid]
                : detail::right_side_searchsorted_contig_impl[hay_typeid]
                                                             [positions_typeid];

        if (fn) {
            constexpr py::ssize_t zero_offset(0);

            sycl::event comp_ev =
                fn(exec_q, hay_nelems, needles_nelems, hay_data, zero_offset,
                   needles_data, zero_offset, positions_data, zero_offset,
                   depends);

            return std::make_pair(
                dpctl::utils::keep_args_alive(exec_q, {hay, needles, positions},
                                              {comp_ev}),
                comp_ev);
        }
    }

    // strided case

    const auto &needles_strides = needles.get_strides_vector();
    const auto &positions_strides = positions.get_strides_vector();

    int simplified_nd = needles_nd;

    using shT = std::vector<py::ssize_t>;

    shT simplified_common_shape;
    shT simplified_needles_strides;
    shT simplified_positions_strides;
    py::ssize_t needles_offset(0);
    py::ssize_t positions_offset(0);

    dpctl::tensor::py_internal::simplify_iteration_space(
        // modified by refernce
        simplified_nd,
        // read-only inputs
        needles_shape_ptr, needles_strides, positions_strides,
        // output, modified by reference
        simplified_common_shape, simplified_needles_strides,
        simplified_positions_strides, needles_offset, positions_offset);

    std::vector<sycl::event> host_task_events;
    host_task_events.reserve(2);

    using dpctl::tensor::offset_utils::device_allocate_and_pack;

    auto ptr_size_event_tuple = device_allocate_and_pack<py::ssize_t>(
        exec_q, host_task_events,
        // vectors being packed
        simplified_common_shape, simplified_needles_strides,
        simplified_positions_strides);

    py::ssize_t *packed_shape_strides = std::get<0>(ptr_size_event_tuple);
    const sycl::event &copy_shape_strides_ev =
        std::get<2>(ptr_size_event_tuple);

    if (!packed_shape_strides) {
        throw std::runtime_error("USM-host allocation failure");
    }

    std::vector<sycl::event> all_deps;
    all_deps.reserve(depends.size() + 1);
    all_deps.insert(all_deps.end(), depends.begin(), depends.end());
    all_deps.push_back(copy_shape_strides_ev);

    auto strided_fn =
        (search_left_side)
            ? detail::left_side_searchsorted_strided_impl[hay_typeid]
                                                         [positions_typeid]
            : detail::right_side_searchsorted_strided_impl[hay_typeid]
                                                          [positions_typeid];

    if (!strided_fn) {
        throw std::runtime_error(
            "No implementation for data types of input arrays");
    }

    constexpr py::ssize_t zero_offset(0);
    py::ssize_t hay_step = hay.get_strides_vector()[0];

    sycl::event comp_ev = strided_fn(
        exec_q, hay_nelems, needles_nelems, hay_data, zero_offset, hay_step,
        needles_data, needles_offset, positions_data, positions_offset,
        simplified_nd, packed_shape_strides, all_deps);

    // free packed temporaries
    sycl::event temporaries_cleanup_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(comp_ev);
        const auto &ctx = exec_q.get_context();
        cgh.host_task([packed_shape_strides, ctx]() {
            sycl::free(packed_shape_strides, ctx);
        });
    });

    host_task_events.push_back(temporaries_cleanup_ev);

    return std::make_pair(
        dpctl::utils::keep_args_alive(exec_q, {hay, needles, positions},
                                      host_task_events),
        comp_ev);
}

/*! @brief search for needle from needles in sorted hay,
 *         hay[pos] <= needle < hay[pos + 1]
 */
std::pair<sycl::event, sycl::event>
py_searchsorted_left(const dpctl::tensor::usm_ndarray &hay,
                     const dpctl::tensor::usm_ndarray &needles,
                     const dpctl::tensor::usm_ndarray &positions,
                     sycl::queue &exec_q,
                     const std::vector<sycl::event> &depends)
{
    constexpr bool side_left(true);
    return py_searchsorted(hay, needles, positions, exec_q, side_left, depends);
}

/*! @brief search for needle from needles in sorted hay,
 *         hay[pos] < needle <= hay[pos + 1]
 */
std::pair<sycl::event, sycl::event>
py_searchsorted_right(const dpctl::tensor::usm_ndarray &hay,
                      const dpctl::tensor::usm_ndarray &needles,
                      const dpctl::tensor::usm_ndarray &positions,
                      sycl::queue &exec_q,
                      const std::vector<sycl::event> &depends)
{
    constexpr bool side_right(false);
    return py_searchsorted(hay, needles, positions, exec_q, side_right,
                           depends);
}

void init_searchsorted_functions(py::module_ m)
{
    dpctl::tensor::py_internal::detail::init_searchsorted_dispatch_table();

    using dpctl::tensor::py_internal::py_searchsorted_left;
    using dpctl::tensor::py_internal::py_searchsorted_right;

    m.def("_searchsorted_left", &py_searchsorted_left, py::arg("hay"),
          py::arg("needles"), py::arg("positions"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());
    m.def("_searchsorted_right", &py_searchsorted_right, py::arg("hay"),
          py::arg("needles"), py::arg("positions"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());
}

} // end of namespace py_internal
} // end of namespace tensor
} // namespace dpctl
