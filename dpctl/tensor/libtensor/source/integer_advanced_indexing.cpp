//===-- take_kernel_impl.cpp - Implementation of take  --*-C++-*-/===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2022 Intel Corporation
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
/// This file defines implementation functions of dpctl.tensor.take and
/// dpctl.tensor.put
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <algorithm>
#include <complex>
#include <cstdint>
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <utility>

#include "dpctl4pybind11.hpp"
#include "kernels/integer_advanced_indexing.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

#include "integer_advanced_indexing.hpp"

#define INDEXING_MODES 2
#define CLIP_MODE 0
#define WRAP_MODE 1

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

namespace _ns = dpctl::tensor::detail;

using dpctl::tensor::kernels::indexing::put_fn_ptr_t;
using dpctl::tensor::kernels::indexing::take_fn_ptr_t;

static take_fn_ptr_t take_dispatch_table[INDEXING_MODES][_ns::num_types]
                                        [_ns::num_types];

static put_fn_ptr_t put_dispatch_table[INDEXING_MODES][_ns::num_types]
                                      [_ns::num_types];

namespace py = pybind11;

using dpctl::tensor::c_contiguous_strides;
using dpctl::tensor::f_contiguous_strides;

using dpctl::utils::keep_args_alive;

std::vector<sycl::event> _populate_packed_shapes_strides_for_indexing(
    sycl::queue exec_q,
    std::vector<sycl::event> &host_task_events,
    py::ssize_t *device_orthog_shapes_strides,
    py::ssize_t *device_axes_shapes_strides,
    const py::ssize_t *inp_shape,
    const py::ssize_t *inp_strides,
    bool is_inp_c_contig,
    bool is_inp_f_contig,
    const py::ssize_t *arr_shape,
    const py::ssize_t *arr_strides,
    bool is_arr_c_contig,
    bool is_arr_f_contig,
    int axis_start,
    int k,
    int ind_nd,
    int inp_nd,
    int arr_nd)
{

    int orthog_sh_elems = std::max<int>(inp_nd - k, 1);
    int along_sh_elems = std::max<int>(ind_nd, 1);

    using usm_host_allocatorT =
        sycl::usm_allocator<py::ssize_t, sycl::usm::alloc::host>;
    using shT = std::vector<py::ssize_t, usm_host_allocatorT>;

    usm_host_allocatorT allocator(exec_q);
    std::shared_ptr<shT> packed_host_shapes_strides_shp =
        std::make_shared<shT>(3 * orthog_sh_elems, 0, allocator);

    std::shared_ptr<shT> packed_host_axes_shapes_strides_shp =
        std::make_shared<shT>(2 * k + along_sh_elems, 0, allocator);

    if (inp_nd > 0) {
        std::copy(inp_shape, inp_shape + axis_start,
                  packed_host_shapes_strides_shp->begin());
        std::copy(inp_shape + axis_start + k, inp_shape + inp_nd,
                  packed_host_shapes_strides_shp->begin() + axis_start);
        std::copy(inp_shape + axis_start, inp_shape + axis_start + k,
                  packed_host_axes_shapes_strides_shp->begin());

        // contract axes by using two copies
        if (inp_strides == nullptr) {
            if (is_inp_c_contig) {
                const auto &inp_contig_strides =
                    c_contiguous_strides(inp_nd, inp_shape);
                std::copy(inp_contig_strides.begin(),
                          inp_contig_strides.begin() + axis_start,
                          packed_host_shapes_strides_shp->begin() +
                              orthog_sh_elems);
                std::copy(inp_contig_strides.begin() + axis_start + k,
                          inp_contig_strides.end(),
                          packed_host_shapes_strides_shp->begin() +
                              orthog_sh_elems + axis_start);
                std::copy(inp_contig_strides.begin() + axis_start,
                          inp_contig_strides.begin() + axis_start + k,
                          packed_host_axes_shapes_strides_shp->begin() + k);
            }
            else if (is_inp_f_contig) {
                const auto &inp_contig_strides =
                    f_contiguous_strides(inp_nd, inp_shape);
                std::copy(inp_contig_strides.begin(),
                          inp_contig_strides.begin() + axis_start,
                          packed_host_shapes_strides_shp->begin() +
                              orthog_sh_elems);
                std::copy(inp_contig_strides.begin() + axis_start + k,
                          inp_contig_strides.end(),
                          packed_host_shapes_strides_shp->begin() +
                              orthog_sh_elems + axis_start);
                std::copy(inp_contig_strides.begin() + axis_start,
                          inp_contig_strides.begin() + axis_start + k,
                          packed_host_axes_shapes_strides_shp->begin() + k);
            }
            else {
                // FIXME: this pointer was not allocated in this function
                // the caller should be freeing it
                sycl::free(device_orthog_shapes_strides, exec_q);
                throw std::runtime_error("Invalid array encountered");
            }
        }
        else {
            std::copy(inp_strides, inp_strides + axis_start,
                      packed_host_shapes_strides_shp->begin() +
                          orthog_sh_elems);
            std::copy(inp_strides + axis_start + k, inp_strides + inp_nd,
                      packed_host_shapes_strides_shp->begin() +
                          orthog_sh_elems + axis_start);
            std::copy(inp_strides + axis_start, inp_strides + axis_start + k,
                      packed_host_axes_shapes_strides_shp->begin() + k);
        }

        if (arr_strides == nullptr) {
            if (is_arr_c_contig) {
                const auto &arr_contig_strides =
                    c_contiguous_strides(arr_nd, arr_shape);
                std::copy(arr_contig_strides.begin(),
                          arr_contig_strides.begin() + axis_start,
                          packed_host_shapes_strides_shp->begin() +
                              2 * orthog_sh_elems);
                std::copy(arr_contig_strides.begin() + axis_start + ind_nd,
                          arr_contig_strides.end(),
                          packed_host_shapes_strides_shp->begin() +
                              2 * orthog_sh_elems + axis_start);
                std::copy(arr_contig_strides.begin() + axis_start,
                          arr_contig_strides.begin() + axis_start + ind_nd,
                          packed_host_axes_shapes_strides_shp->begin() + 2 * k);
            }
            else if (is_arr_f_contig) {
                const auto &arr_contig_strides =
                    f_contiguous_strides(arr_nd, arr_shape);
                std::copy(arr_contig_strides.begin(),
                          arr_contig_strides.begin() + axis_start,
                          packed_host_shapes_strides_shp->begin() +
                              2 * orthog_sh_elems);
                std::copy(arr_contig_strides.begin() + axis_start + ind_nd,
                          arr_contig_strides.end(),
                          packed_host_shapes_strides_shp->begin() +
                              2 * orthog_sh_elems + axis_start);
                std::copy(arr_contig_strides.begin() + axis_start,
                          arr_contig_strides.begin() + axis_start + ind_nd,
                          packed_host_axes_shapes_strides_shp->begin() + 2 * k);
            }
            else {
                // FIXME: this pointer was not allocated in this function
                // the caller should be freeing it
                sycl::free(device_orthog_shapes_strides, exec_q);
                throw std::runtime_error("Invalid array encountered");
            }
        }
        else {
            std::copy(arr_strides, arr_strides + axis_start,
                      packed_host_shapes_strides_shp->begin() +
                          2 * orthog_sh_elems);
            std::copy(arr_strides + axis_start + ind_nd, arr_strides + arr_nd,
                      packed_host_shapes_strides_shp->begin() +
                          2 * orthog_sh_elems + axis_start);
            std::copy(arr_strides + axis_start,
                      arr_strides + axis_start + ind_nd,
                      packed_host_axes_shapes_strides_shp->begin() + 2 * k);
        }

        // copy packed shapes and strides from host to devices
        sycl::event device_orthog_shapes_strides_copy_ev =
            exec_q.copy<py::ssize_t>(packed_host_shapes_strides_shp->data(),
                                     device_orthog_shapes_strides,
                                     packed_host_shapes_strides_shp->size());

        sycl::event device_axes_shapes_strides_copy_ev =
            exec_q.copy<py::ssize_t>(
                packed_host_axes_shapes_strides_shp->data(),
                device_axes_shapes_strides,
                packed_host_axes_shapes_strides_shp->size());

        sycl::event clean_up_host_task_ev =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(device_axes_shapes_strides_copy_ev);
                cgh.depends_on(device_orthog_shapes_strides_copy_ev);
                cgh.host_task([packed_host_axes_shapes_strides_shp,
                               packed_host_shapes_strides_shp]() {});
            });
        host_task_events.push_back(clean_up_host_task_ev);

        std::vector<sycl::event> v = {device_orthog_shapes_strides_copy_ev,
                                      device_axes_shapes_strides_copy_ev};
        return v;
    }
    else {
        // no orthogonal dimensions
        sycl::event device_orthog_shapes_strides_fill_ev =
            exec_q.fill<py::ssize_t>(device_orthog_shapes_strides,
                                     py::ssize_t(0), 3);

        packed_host_axes_shapes_strides_shp->insert(
            packed_host_axes_shapes_strides_shp->end(), py::ssize_t(0), 2);
        if (arr_strides == nullptr) {
            if (is_arr_c_contig) {
                const auto &arr_contig_strides =
                    c_contiguous_strides(arr_nd, arr_shape);
                std::copy(arr_contig_strides.begin() + axis_start,
                          arr_contig_strides.begin() + axis_start + ind_nd,
                          packed_host_axes_shapes_strides_shp->begin() + 2);
            }
            else if (is_arr_f_contig) {
                const auto &arr_contig_strides =
                    f_contiguous_strides(arr_nd, arr_shape);
                std::copy(arr_contig_strides.begin() + axis_start,
                          arr_contig_strides.begin() + axis_start + ind_nd,
                          packed_host_axes_shapes_strides_shp->begin() + 2);
            }
            else {
                // FIXME: memory was not allocated in this function
                // it should be freed by the caller
                sycl::free(device_orthog_shapes_strides, exec_q);
                throw std::runtime_error("Invalid array encountered");
            }
        }
        else {
            std::copy(arr_strides + axis_start,
                      arr_strides + axis_start + ind_nd,
                      packed_host_axes_shapes_strides_shp->begin() + 2);
        }

        sycl::event device_axes_shapes_strides_copy_ev =
            exec_q.copy<py::ssize_t>(
                packed_host_axes_shapes_strides_shp->data(),
                device_axes_shapes_strides,
                packed_host_axes_shapes_strides_shp->size());

        sycl::event clean_up_host_task_ev =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(device_axes_shapes_strides_copy_ev);
                cgh.host_task([packed_host_axes_shapes_strides_shp]() {});
            });
        host_task_events.push_back(clean_up_host_task_ev);

        std::vector<sycl::event> v = {device_orthog_shapes_strides_fill_ev,
                                      device_axes_shapes_strides_copy_ev};
        return v;
    }
}

/* Utility to parse python object py_ind into vector of `usm_ndarray`s */
std::vector<dpctl::tensor::usm_ndarray> parse_py_ind(const sycl::queue &q,
                                                     py::object py_ind)
{
    size_t ind_count = py::len(py_ind);
    std::vector<dpctl::tensor::usm_ndarray> res;
    res.reserve(ind_count);

    bool nd_is_known = false;
    int nd = -1;
    for (size_t i = 0; i < ind_count; ++i) {
        py::object el_i = py_ind[py::cast(i)];
        dpctl::tensor::usm_ndarray arr_i =
            py::cast<dpctl::tensor::usm_ndarray>(el_i);
        if (!dpctl::utils::queues_are_compatible(q, {arr_i})) {
            throw py::value_error("Index allocation queue is not compatible "
                                  "with execution queue");
        }
        if (nd_is_known) {
            if (nd != arr_i.get_ndim()) {
                throw py::value_error(
                    "Indices must have the same number of dimensions.");
            }
        }
        else {
            nd_is_known = true;
            nd = arr_i.get_ndim();
        }
        res.push_back(arr_i);
    }

    return res;
}

std::pair<sycl::event, sycl::event>
usm_ndarray_take(dpctl::tensor::usm_ndarray src,
                 py::object py_ind,
                 dpctl::tensor::usm_ndarray dst,
                 int axis_start,
                 uint8_t mode,
                 sycl::queue exec_q,
                 const std::vector<sycl::event> &depends)
{
    std::vector<dpctl::tensor::usm_ndarray> ind = parse_py_ind(exec_q, py_ind);

    int k = ind.size();

    if (k == 0) {
        throw py::value_error("List of indices is empty.");
    }

    if (axis_start < 0) {
        throw py::value_error("Axis cannot be negative.");
    }

    if (mode != 0 && mode != 1) {
        throw py::value_error("Mode must be 0 or 1.");
    }

    const dpctl::tensor::usm_ndarray ind_rep = ind[0];

    int src_nd = src.get_ndim();
    int dst_nd = dst.get_ndim();
    int ind_nd = ind_rep.get_ndim();

    auto sh_elems = (src_nd > 0) ? src_nd : 1;

    if (axis_start + k > sh_elems) {
        throw py::value_error("Axes are out of range for array of dimension " +
                              std::to_string(src_nd));
    }
    if (src_nd == 0) {
        if (dst_nd != ind_nd) {
            throw py::value_error(
                "Destination is not of appropriate dimension for take kernel.");
        }
    }
    else {
        if (dst_nd != (src_nd - k + ind_nd)) {
            throw py::value_error(
                "Destination is not of appropriate dimension for take kernel.");
        }
    }

    const py::ssize_t *src_shape = src.get_shape_raw();
    const py::ssize_t *dst_shape = dst.get_shape_raw();

    int orthog_nd = std::max<int>(src_nd - k, 1);

    bool orthog_shapes_equal(true);
    size_t orthog_nelems(1);
    for (int i = 0; i < (src_nd - k); ++i) {
        auto idx1 = (i < axis_start) ? i : i + k;
        auto idx2 = (i < axis_start) ? i : i + ind_nd;

        orthog_nelems *= static_cast<size_t>(src_shape[idx1]);
        orthog_shapes_equal =
            orthog_shapes_equal && (src_shape[idx1] == dst_shape[idx2]);
    }

    if (!orthog_shapes_equal) {
        throw py::value_error(
            "Axes of basic indices are not of matching shapes.");
    }

    if (orthog_nelems == 0) {
        return std::make_pair(sycl::event{}, sycl::event{});
    }

    char *src_data = src.get_data();
    char *dst_data = dst.get_data();

    if (!dpctl::utils::queues_are_compatible(exec_q, {src, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    auto src_offsets = src.get_minmax_offsets();
    auto dst_offsets = dst.get_minmax_offsets();
    int src_elem_size = src.get_elemsize();
    int dst_elem_size = dst.get_elemsize();

    py::ssize_t src_offset = py::ssize_t(0);
    py::ssize_t dst_offset = py::ssize_t(0);

    bool memory_overlap =
        ((dst_data - src_data > src_offsets.second * src_elem_size -
                                    dst_offsets.first * dst_elem_size) &&
         (src_data - dst_data > dst_offsets.second * dst_elem_size -
                                    src_offsets.first * src_elem_size));
    if (memory_overlap) {
        throw py::value_error("Array memory overlap.");
    }

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    auto array_types = dpctl::tensor::detail::usm_ndarray_types();
    int src_type_id = array_types.typenum_to_lookup_id(src_typenum);
    int dst_type_id = array_types.typenum_to_lookup_id(dst_typenum);

    if (src_type_id != dst_type_id) {
        throw py::type_error("Array data types are not the same.");
    }

    const py::ssize_t *ind_shape = ind_rep.get_shape_raw();

    int ind_typenum = ind_rep.get_typenum();
    int ind_type_id = array_types.typenum_to_lookup_id(ind_typenum);

    size_t ind_nelems(1);
    for (int i = 0; i < ind_nd; ++i) {
        ind_nelems *= static_cast<size_t>(ind_shape[i]);

        if (!(ind_shape[i] == dst_shape[axis_start + i])) {
            throw py::value_error(
                "Indices shape does not match shape of axis in destination.");
        }
    }

    // destination must be ample enough to accommodate all elements
    {
        size_t range =
            static_cast<size_t>(dst_offsets.second - dst_offsets.first);
        if ((range + 1) < (orthog_nelems * ind_nelems)) {
            throw py::value_error(
                "Destination array can not accommodate all the "
                "elements of source array.");
        }
    }

    int ind_sh_elems = std::max<int>(ind_nd, 1);

    std::vector<char *> ind_ptrs;
    ind_ptrs.reserve(k);

    std::vector<py::ssize_t> ind_offsets;
    ind_offsets.reserve(k);

    std::vector<py::ssize_t> ind_sh_sts((k + 1) * ind_sh_elems, py::ssize_t(0));
    if (ind_nd > 0) {
        std::copy(ind_shape, ind_shape + ind_sh_elems, ind_sh_sts.begin());
    }
    for (int i = 0; i < k; ++i) {
        dpctl::tensor::usm_ndarray ind_ = ind[i];

        if (!dpctl::utils::queues_are_compatible(exec_q, {ind_})) {
            throw py::value_error(
                "Execution queue is not compatible with allocation queues");
        }

        // ndim, type, and shape are checked against the first array
        if (i > 0) {
            if (!(ind_.get_ndim() == ind_nd)) {
                throw py::value_error("Index dimensions are not the same");
            }

            if (!(ind_type_id ==
                  array_types.typenum_to_lookup_id(ind_.get_typenum()))) {
                throw py::type_error(
                    "Indices array data types are not all the same.");
            }

            const py::ssize_t *ind_shape_ = ind_.get_shape_raw();
            for (int dim = 0; dim < ind_nd; ++dim) {
                if (!(ind_shape[dim] == ind_shape_[dim])) {
                    throw py::value_error("Indices shapes are not all equal.");
                }
            }
        }

        // check for overlap with destination
        int ind_elem_size = ind_.get_elemsize();
        auto ind_mem_offsets = ind_.get_minmax_offsets();
        char *ind_data = ind_.get_data();
        bool ind_memory_overlap =
            ((dst_data - ind_data > ind_mem_offsets.second * ind_elem_size -
                                        dst_offsets.first * dst_elem_size) &&
             (ind_data - dst_data > dst_offsets.second * dst_elem_size -
                                        ind_mem_offsets.first * ind_elem_size));

        if (ind_memory_overlap) {
            throw py::value_error(
                "Arrays index overlapping segments of memory");
        }

        // strides are initialized to 0 for 0D indices, so skip here
        if (ind_nd > 0) {
            const py::ssize_t *ind_strides = ind_.get_strides_raw();
            if (ind_strides == nullptr) {
                if (ind_.is_c_contiguous()) {
                    const auto &ind_contig_strides_ =
                        c_contiguous_strides(ind_nd, ind_shape);
                    std::copy(ind_contig_strides_.begin(),
                              ind_contig_strides_.end(),
                              ind_sh_sts.begin() + (i + 1) * ind_nd);
                }
                else if (ind_.is_f_contiguous()) {
                    const auto &ind_contig_strides_ =
                        f_contiguous_strides(ind_nd, ind_shape);
                    std::copy(ind_contig_strides_.begin(),
                              ind_contig_strides_.end(),
                              ind_sh_sts.begin() + (i + 1) * ind_nd);
                }
                else {
                    throw std::runtime_error(
                        "Invalid ind array encountered in: take function");
                }
            }
            else {
                std::copy(ind_strides, ind_strides + ind_nd,
                          ind_sh_sts.begin() + (i + 1) * ind_nd);
            }
        }

        ind_ptrs.push_back(ind_data);
        ind_offsets.push_back(py::ssize_t(0));
    }

    char **packed_ind_ptrs = sycl::malloc_device<char *>(k, exec_q);

    if (packed_ind_ptrs == nullptr) {
        throw std::runtime_error(
            "Unable to allocate packed_ind_ptrs device memory");
    }

    // rearrange to past where indices shapes are checked
    // packed_ind_shapes_strides = [ind_shape,
    //                              ind[0] strides,
    //                              ...,
    //                              ind[k] strides]
    py::ssize_t *packed_ind_shapes_strides =
        sycl::malloc_device<py::ssize_t>((k + 1) * ind_sh_elems, exec_q);

    if (packed_ind_shapes_strides == nullptr) {
        sycl::free(packed_ind_ptrs, exec_q);
        throw std::runtime_error(
            "Unable to allocate packed_ind_shapes_strides device memory");
    }

    py::ssize_t *packed_ind_offsets =
        sycl::malloc_device<py::ssize_t>(k, exec_q);

    if (packed_ind_offsets == nullptr) {
        sycl::free(packed_ind_ptrs, exec_q);
        sycl::free(packed_ind_shapes_strides, exec_q);
        throw std::runtime_error(
            "Unable to allocate packed_ind_offsets device memory");
    }

    using usm_host_allocator_T =
        sycl::usm_allocator<char *, sycl::usm::alloc::host>;
    using ptrT = std::vector<char *, usm_host_allocator_T>;

    usm_host_allocator_T ptr_allocator(exec_q);
    std::shared_ptr<ptrT> host_ind_ptrs_shp =
        std::make_shared<ptrT>(k, ptr_allocator);

    using usm_host_allocatorT =
        sycl::usm_allocator<py::ssize_t, sycl::usm::alloc::host>;
    using shT = std::vector<py::ssize_t, usm_host_allocatorT>;

    usm_host_allocatorT ind_allocator(exec_q);
    std::shared_ptr<shT> host_ind_shapes_strides_shp =
        std::make_shared<shT>(ind_sh_elems * (k + 1), ind_allocator);

    std::shared_ptr<shT> host_ind_offsets_shp =
        std::make_shared<shT>(k, ind_allocator);

    std::copy(ind_sh_sts.begin(), ind_sh_sts.end(),
              host_ind_shapes_strides_shp->begin());
    std::copy(ind_ptrs.begin(), ind_ptrs.end(), host_ind_ptrs_shp->begin());
    std::copy(ind_offsets.begin(), ind_offsets.end(),
              host_ind_offsets_shp->begin());

    std::vector<sycl::event> host_task_events;
    host_task_events.reserve(5);

    sycl::event packed_ind_ptrs_copy_ev = exec_q.copy<char *>(
        host_ind_ptrs_shp->data(), packed_ind_ptrs, host_ind_ptrs_shp->size());

    sycl::event packed_ind_shapes_strides_copy_ev = exec_q.copy<py::ssize_t>(
        host_ind_shapes_strides_shp->data(), packed_ind_shapes_strides,
        host_ind_shapes_strides_shp->size());

    sycl::event packed_ind_offsets_copy_ev = exec_q.copy<py::ssize_t>(
        host_ind_offsets_shp->data(), packed_ind_offsets,
        host_ind_offsets_shp->size());

    sycl::event shared_ptr_cleanup_host_task =
        exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on({packed_ind_offsets_copy_ev,
                            packed_ind_shapes_strides_copy_ev,
                            packed_ind_ptrs_copy_ev});
            cgh.host_task([host_ind_offsets_shp, host_ind_shapes_strides_shp,
                           host_ind_ptrs_shp]() {});
        });
    host_task_events.push_back(shared_ptr_cleanup_host_task);

    std::vector<sycl::event> ind_pack_depends{packed_ind_ptrs_copy_ev,
                                              packed_ind_shapes_strides_copy_ev,
                                              packed_ind_offsets_copy_ev};

    bool is_src_c_contig = src.is_c_contiguous();
    bool is_src_f_contig = src.is_f_contiguous();

    bool is_dst_c_contig = dst.is_c_contiguous();
    bool is_dst_f_contig = dst.is_f_contiguous();

    const py::ssize_t *src_strides = src.get_strides_raw();
    const py::ssize_t *dst_strides = dst.get_strides_raw();

    // packed_shapes_strides = [src_shape[:axis] + src_shape[axis+k:],
    //                          src_strides[:axis] + src_strides[axis+k:],
    //                          dst_strides[:axis] + dst_strides[axis+k:]]
    py::ssize_t *packed_shapes_strides =
        sycl::malloc_device<py::ssize_t>(3 * sh_elems, exec_q);

    if (packed_shapes_strides == nullptr) {
        sycl::event::wait(host_task_events);
        sycl::free(packed_ind_ptrs, exec_q);
        sycl::free(packed_ind_shapes_strides, exec_q);
        sycl::free(packed_ind_offsets, exec_q);
        throw std::runtime_error(
            "Unable to allocate packed_shapes_strides device memory");
    }

    // packed_axes_shapes_strides = [src_shape[axis:axis+k],
    //                               src_strides[axis:axis+k,
    //                               dst_strides[axis:ind.ndim]]
    py::ssize_t *packed_axes_shapes_strides =
        sycl::malloc_device<py::ssize_t>((2 * k) + ind_sh_elems, exec_q);

    if (packed_axes_shapes_strides == nullptr) {
        sycl::event::wait(host_task_events);
        sycl::free(packed_ind_ptrs, exec_q);
        sycl::free(packed_ind_shapes_strides, exec_q);
        sycl::free(packed_ind_offsets, exec_q);
        sycl::free(packed_shapes_strides, exec_q);
        throw std::runtime_error(
            "Unable to allocate packed_axes_shapes_strides device memory");
    }

    std::vector<sycl::event> src_dst_pack_deps =
        _populate_packed_shapes_strides_for_indexing(
            exec_q, host_task_events, packed_shapes_strides,
            packed_axes_shapes_strides, src_shape, src_strides, is_src_c_contig,
            is_src_f_contig, dst_shape, dst_strides, is_dst_c_contig,
            is_dst_f_contig, axis_start, k, ind_nd, src_nd, dst_nd);

    std::vector<sycl::event> all_deps;
    all_deps.reserve(depends.size() + ind_pack_depends.size() +
                     src_dst_pack_deps.size());
    all_deps.insert(std::end(all_deps), std::begin(ind_pack_depends),
                    std::end(ind_pack_depends));
    all_deps.insert(std::end(all_deps), std::begin(src_dst_pack_deps),
                    std::end(src_dst_pack_deps));
    all_deps.insert(std::end(all_deps), std::begin(depends), std::end(depends));

    auto fn = take_dispatch_table[mode][src_type_id][ind_type_id];

    if (fn == nullptr) {
        sycl::event::wait(host_task_events);
        sycl::free(packed_ind_ptrs, exec_q);
        sycl::free(packed_ind_shapes_strides, exec_q);
        sycl::free(packed_ind_offsets, exec_q);
        sycl::free(packed_shapes_strides, exec_q);
        sycl::free(packed_axes_shapes_strides, exec_q);
        throw std::runtime_error("Indices must be integer type, got " +
                                 std::to_string(ind_type_id));
    }

    sycl::event take_generic_ev =
        fn(exec_q, orthog_nelems, ind_nelems, orthog_nd, ind_nd, k,
           packed_shapes_strides, packed_axes_shapes_strides,
           packed_ind_shapes_strides, src_data, dst_data, packed_ind_ptrs,
           src_offset, dst_offset, packed_ind_offsets, all_deps);

    // free packed temporaries
    sycl::event temporaries_cleanup_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(take_generic_ev);
        auto ctx = exec_q.get_context();
        cgh.host_task([packed_shapes_strides, packed_axes_shapes_strides,
                       packed_ind_shapes_strides, packed_ind_ptrs,
                       packed_ind_offsets, ctx]() {
            sycl::free(packed_shapes_strides, ctx);
            sycl::free(packed_axes_shapes_strides, ctx);
            sycl::free(packed_ind_shapes_strides, ctx);
            sycl::free(packed_ind_ptrs, ctx);
            sycl::free(packed_ind_offsets, ctx);
        });
    });

    sycl::event::wait({take_generic_ev, temporaries_cleanup_ev});
    sycl::event::wait(host_task_events);

    /*
    sycl::event host_task_ev = keep_args_alive(exec_q, {src, py_ind, dst},
                                               {temporaries_cleanup_ev});
    */

    return std::make_pair(sycl::event(), temporaries_cleanup_ev);
}

std::pair<sycl::event, sycl::event>
usm_ndarray_put(dpctl::tensor::usm_ndarray dst,
                py::object py_ind,
                dpctl::tensor::usm_ndarray val,
                int axis_start,
                uint8_t mode,
                sycl::queue exec_q,
                const std::vector<sycl::event> &depends)
{
    std::vector<dpctl::tensor::usm_ndarray> ind = parse_py_ind(exec_q, py_ind);
    int k = ind.size();

    if (k == 0) {
        // no indices to write to
        throw py::value_error("List of indices is empty.");
    }

    if (axis_start < 0) {
        throw py::value_error("Axis cannot be negative.");
    }

    if (mode != 0 && mode != 1) {
        throw py::value_error("Mode must be 0 or 1.");
    }

    if (!dst.is_writable()) {
        throw py::value_error("Output array is read-only.");
    }

    const dpctl::tensor::usm_ndarray ind_rep = ind[0];

    int dst_nd = dst.get_ndim();
    int val_nd = val.get_ndim();
    int ind_nd = ind_rep.get_ndim();

    auto sh_elems = (dst_nd > 0) ? dst_nd : 1;

    if (axis_start + k > sh_elems) {
        throw py::value_error("Axes are out of range for array of dimension " +
                              std::to_string(dst_nd));
    }
    if (dst_nd == 0) {
        if (val_nd != ind_nd) {
            throw py::value_error("Destination is not of appropriate dimension "
                                  "for put function.");
        }
    }
    else {
        if (val_nd != (dst_nd - k + ind_nd)) {
            throw py::value_error("Destination is not of appropriate dimension "
                                  "for put function.");
        }
    }

    size_t dst_nelems = dst.get_size();

    const py::ssize_t *dst_shape = dst.get_shape_raw();
    const py::ssize_t *val_shape = val.get_shape_raw();

    int orthog_nd = ((dst_nd - k) > 0) ? dst_nd - k : 1;

    bool orthog_shapes_equal(true);
    size_t orthog_nelems(1);
    for (int i = 0; i < (dst_nd - k); ++i) {
        auto idx1 = (i < axis_start) ? i : i + k;
        auto idx2 = (i < axis_start) ? i : i + ind_nd;

        orthog_nelems *= static_cast<size_t>(dst_shape[idx1]);
        orthog_shapes_equal =
            orthog_shapes_equal && (dst_shape[idx1] == val_shape[idx2]);
    }

    if (!orthog_shapes_equal) {
        throw py::value_error(
            "Axes of basic indices are not of matching shapes.");
    }

    if (orthog_nelems == 0) {
        return std::make_pair(sycl::event(), sycl::event());
    }

    char *dst_data = dst.get_data();
    char *val_data = val.get_data();

    auto dst_offsets = dst.get_minmax_offsets();
    auto val_offsets = val.get_minmax_offsets();
    int dst_elem_size = dst.get_elemsize();
    int val_elem_size = val.get_elemsize();

    if (!dpctl::utils::queues_are_compatible(exec_q, {dst, val})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }
    py::ssize_t dst_offset = py::ssize_t(0);
    py::ssize_t val_offset = py::ssize_t(0);

    bool memory_overlap =
        ((val_data - dst_data > dst_offsets.second * dst_elem_size -
                                    val_offsets.first * val_elem_size) &&
         (dst_data - val_data > val_offsets.second * val_elem_size -
                                    dst_offsets.first * dst_elem_size));
    if (memory_overlap) {
        throw py::value_error("Arrays index overlapping segments of memory");
    }

    // destination must be ample enough to accommodate all possible elements
    {
        size_t range =
            static_cast<size_t>(dst_offsets.second - dst_offsets.first);
        if ((range + 1) < dst_nelems) {
            throw py::value_error(
                "Destination array can not accommodate all the "
                "elements of source array.");
        }
    }

    int dst_typenum = dst.get_typenum();
    int val_typenum = val.get_typenum();

    auto array_types = dpctl::tensor::detail::usm_ndarray_types();
    int dst_type_id = array_types.typenum_to_lookup_id(dst_typenum);
    int val_type_id = array_types.typenum_to_lookup_id(val_typenum);

    if (dst_type_id != val_type_id) {
        throw py::type_error("Array data types are not the same.");
    }

    const py::ssize_t *ind_shape = ind_rep.get_shape_raw();

    int ind_typenum = ind_rep.get_typenum();
    int ind_type_id = array_types.typenum_to_lookup_id(ind_typenum);

    size_t ind_nelems(1);
    for (int i = 0; i < ind_nd; ++i) {
        ind_nelems *= static_cast<size_t>(ind_shape[i]);

        if (!(ind_shape[i] == val_shape[axis_start + i])) {
            throw py::value_error(
                "Indices shapes does not match shape of axis in vals.");
        }
    }

    auto ind_sh_elems = (ind_nd > 0) ? ind_nd : 1;

    std::vector<char *> ind_ptrs;
    ind_ptrs.reserve(k);
    std::vector<py::ssize_t> ind_offsets;
    ind_offsets.reserve(k);
    std::vector<py::ssize_t> ind_sh_sts((k + 1) * ind_sh_elems, py::ssize_t(0));
    if (ind_nd > 0) {
        std::copy(ind_shape, ind_shape + ind_sh_elems, ind_sh_sts.begin());
    }
    for (int i = 0; i < k; ++i) {
        dpctl::tensor::usm_ndarray ind_ = ind[i];

        if (!dpctl::utils::queues_are_compatible(exec_q, {ind_})) {
            throw py::value_error(
                "Execution queue is not compatible with allocation queues");
        }

        // ndim, type, and shape are checked against the first array
        if (i > 0) {
            if (!(ind_.get_ndim() == ind_nd)) {
                throw py::value_error("Index dimensions are not the same");
            }

            if (!(ind_type_id ==
                  array_types.typenum_to_lookup_id(ind_.get_typenum()))) {
                throw py::type_error(
                    "Indices array data types are not all the same.");
            }

            const py::ssize_t *ind_shape_ = ind_.get_shape_raw();
            for (int dim = 0; dim < ind_nd; ++dim) {
                if (!(ind_shape[dim] == ind_shape_[dim])) {
                    throw py::value_error("Indices shapes are not all equal.");
                }
            }
        }

        // check for overlap with destination
        int ind_elem_size = ind_.get_elemsize();
        auto ind_mem_offsets = ind_.get_minmax_offsets();
        char *ind_data = ind_.get_data();
        bool ind_memory_overlap =
            ((val_data - ind_data > ind_mem_offsets.second * ind_elem_size -
                                        val_offsets.first * val_elem_size) &&
             (ind_data - val_data > val_offsets.second * val_elem_size -
                                        ind_mem_offsets.first * ind_elem_size));

        if (ind_memory_overlap) {
            throw py::value_error(
                "Arrays index overlapping segments of memory");
        }

        // strides are initialized to 0 for 0D indices, so skip here
        if (ind_nd > 0) {
            const py::ssize_t *ind_strides = ind_.get_strides_raw();
            if (ind_strides == nullptr) {
                if (ind_.is_c_contiguous()) {
                    const auto &ind_contig_strides_ =
                        c_contiguous_strides(ind_nd, ind_shape);
                    std::copy(ind_contig_strides_.begin(),
                              ind_contig_strides_.end(),
                              ind_sh_sts.begin() + (i + 1) * ind_nd);
                }
                else if (ind_.is_f_contiguous()) {
                    const auto &ind_contig_strides_ =
                        f_contiguous_strides(ind_nd, ind_shape);
                    std::copy(ind_contig_strides_.begin(),
                              ind_contig_strides_.end(),
                              ind_sh_sts.begin() + (i + 1) * ind_nd);
                }
                else {
                    throw std::runtime_error(
                        "Invalid ind array encountered in: take function");
                }
            }
            else {
                std::copy(ind_strides, ind_strides + ind_nd,
                          ind_sh_sts.begin() + (i + 1) * ind_nd);
            }
        }

        ind_ptrs.push_back(ind_data);
        ind_offsets.push_back(py::ssize_t(0));
    }

    char **packed_ind_ptrs = sycl::malloc_device<char *>(k, exec_q);

    if (packed_ind_ptrs == nullptr) {
        throw std::runtime_error(
            "Unable to allocate packed_ind_ptrs device memory");
    }

    // packed_ind_shapes_strides = [ind_shape,
    //                              ind[0] strides,
    //                              ...,
    //                              ind[k] strides]
    py::ssize_t *packed_ind_shapes_strides =
        sycl::malloc_device<py::ssize_t>((k + 1) * ind_sh_elems, exec_q);

    if (packed_ind_shapes_strides == nullptr) {
        sycl::free(packed_ind_ptrs, exec_q);
        throw std::runtime_error(
            "Unable to allocate packed_ind_shapes_strides device memory");
    }

    py::ssize_t *packed_ind_offsets =
        sycl::malloc_device<py::ssize_t>(k, exec_q);

    if (packed_ind_offsets == nullptr) {
        sycl::free(packed_ind_ptrs, exec_q);
        sycl::free(packed_ind_shapes_strides, exec_q);
        throw std::runtime_error(
            "Unable to allocate packed_ind_offsets device memory");
    }

    using usm_host_allocator_T =
        sycl::usm_allocator<char *, sycl::usm::alloc::host>;
    using ptrT = std::vector<char *, usm_host_allocator_T>;

    usm_host_allocator_T ptr_allocator(exec_q);
    std::shared_ptr<ptrT> host_ind_ptrs_shp =
        std::make_shared<ptrT>(k, ptr_allocator);

    using usm_host_allocatorT =
        sycl::usm_allocator<py::ssize_t, sycl::usm::alloc::host>;
    using shT = std::vector<py::ssize_t, usm_host_allocatorT>;

    usm_host_allocatorT ind_allocator(exec_q);
    std::shared_ptr<shT> host_ind_shapes_strides_shp =
        std::make_shared<shT>(ind_sh_elems * (k + 1), ind_allocator);

    std::shared_ptr<shT> host_ind_offsets_shp =
        std::make_shared<shT>(k, ind_allocator);

    std::copy(ind_sh_sts.begin(), ind_sh_sts.end(),
              host_ind_shapes_strides_shp->begin());
    std::copy(ind_ptrs.begin(), ind_ptrs.end(), host_ind_ptrs_shp->begin());
    std::copy(ind_offsets.begin(), ind_offsets.end(),
              host_ind_offsets_shp->begin());

    std::vector<sycl::event> host_task_events;
    host_task_events.reserve(5);

    sycl::event packed_ind_ptrs_copy_ev = exec_q.copy<char *>(
        host_ind_ptrs_shp->data(), packed_ind_ptrs, host_ind_ptrs_shp->size());

    sycl::event packed_ind_shapes_strides_copy_ev = exec_q.copy<py::ssize_t>(
        host_ind_shapes_strides_shp->data(), packed_ind_shapes_strides,
        host_ind_shapes_strides_shp->size());

    sycl::event packed_ind_offsets_copy_ev = exec_q.copy<py::ssize_t>(
        host_ind_offsets_shp->data(), packed_ind_offsets,
        host_ind_offsets_shp->size());

    sycl::event shared_ptr_cleanup_host_task =
        exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on({packed_ind_offsets_copy_ev,
                            packed_ind_shapes_strides_copy_ev,
                            packed_ind_ptrs_copy_ev});
            cgh.host_task([host_ind_offsets_shp, host_ind_shapes_strides_shp,
                           host_ind_ptrs_shp]() {});
        });
    host_task_events.push_back(shared_ptr_cleanup_host_task);

    std::vector<sycl::event> ind_pack_depends{packed_ind_ptrs_copy_ev,
                                              packed_ind_shapes_strides_copy_ev,
                                              packed_ind_offsets_copy_ev};

    bool is_dst_c_contig = dst.is_c_contiguous();
    bool is_dst_f_contig = dst.is_f_contiguous();

    bool is_val_c_contig = val.is_c_contiguous();
    bool is_val_f_contig = val.is_f_contiguous();

    const py::ssize_t *dst_strides = dst.get_strides_raw();
    const py::ssize_t *val_strides = val.get_strides_raw();

    // packed_shapes_strides = [dst_shape[:axis] + dst_shape[axis+k:],
    //                          dst_strides[:axis] + dst_strides[axis+k:],
    //                          val_strides[:axis] + val_strides[axis+k:]]
    py::ssize_t *packed_shapes_strides =
        sycl::malloc_device<py::ssize_t>(3 * sh_elems, exec_q);

    if (packed_shapes_strides == nullptr) {
        sycl::event::wait(host_task_events);
        sycl::free(packed_ind_ptrs, exec_q);
        sycl::free(packed_ind_shapes_strides, exec_q);
        sycl::free(packed_ind_offsets, exec_q);
        throw std::runtime_error(
            "Unable to allocate packed_shapes_strides device memory");
    }

    // packed_axes_shapes_strides = [dst_shape[axis:k],
    //                               dst_strides[axis:k,
    //                               val_strides[axis:ind.ndim]]
    py::ssize_t *packed_axes_shapes_strides =
        sycl::malloc_device<py::ssize_t>((2 * k) + ind_sh_elems, exec_q);

    if (packed_axes_shapes_strides == nullptr) {
        sycl::event::wait(host_task_events);
        sycl::free(packed_shapes_strides, exec_q);
        sycl::free(packed_ind_ptrs, exec_q);
        sycl::free(packed_ind_shapes_strides, exec_q);
        sycl::free(packed_ind_offsets, exec_q);
        throw std::runtime_error(
            "Unable to allocate packed_axes_shapes_strides device memory");
    }

    std::vector<sycl::event> copy_shapes_strides_deps =
        _populate_packed_shapes_strides_for_indexing(
            exec_q, host_task_events, packed_shapes_strides,
            packed_axes_shapes_strides, dst_shape, dst_strides, is_dst_c_contig,
            is_dst_f_contig, val_shape, val_strides, is_val_c_contig,
            is_val_f_contig, axis_start, k, ind_nd, dst_nd, val_nd);

    std::vector<sycl::event> all_deps(depends.size() +
                                      copy_shapes_strides_deps.size() +
                                      ind_pack_depends.size());
    all_deps.insert(std::end(all_deps), std::begin(copy_shapes_strides_deps),
                    std::end(copy_shapes_strides_deps));
    all_deps.insert(std::end(all_deps), std::begin(ind_pack_depends),
                    std::end(ind_pack_depends));
    all_deps.insert(std::end(all_deps), std::begin(depends), std::end(depends));

    auto fn = put_dispatch_table[mode][dst_type_id][ind_type_id];

    if (fn == nullptr) {
        sycl::event::wait(host_task_events);
        sycl::free(packed_shapes_strides, exec_q);
        sycl::free(packed_axes_shapes_strides, exec_q);
        sycl::free(packed_ind_shapes_strides, exec_q);
        sycl::free(packed_ind_ptrs, exec_q);
        sycl::free(packed_ind_offsets, exec_q);

        throw std::runtime_error("Indices must be integer type, got " +
                                 std::to_string(ind_type_id));
    }

    sycl::event put_generic_ev =
        fn(exec_q, orthog_nelems, ind_nelems, orthog_nd, ind_nd, k,
           packed_shapes_strides, packed_axes_shapes_strides,
           packed_ind_shapes_strides, dst_data, val_data, packed_ind_ptrs,
           dst_offset, val_offset, packed_ind_offsets, all_deps);

    // free packed temporaries

    sycl::event temporaries_cleanup_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(put_generic_ev);
        auto ctx = exec_q.get_context();
        cgh.host_task([packed_shapes_strides, packed_axes_shapes_strides,
                       packed_ind_shapes_strides, packed_ind_ptrs,
                       packed_ind_offsets, ctx]() {
            sycl::free(packed_shapes_strides, ctx);
            sycl::free(packed_axes_shapes_strides, ctx);
            sycl::free(packed_ind_shapes_strides, ctx);
            sycl::free(packed_ind_ptrs, ctx);
            sycl::free(packed_ind_offsets, ctx);
        });
    });

    sycl::event::wait({put_generic_ev, temporaries_cleanup_ev});
    sycl::event::wait(host_task_events);

    /*
    sycl::event py_obj_cleanup_ev =
        keep_args_alive(exec_q, {dst, py_ind, val},
                        {put_generic_ev, temporaries_cleanup_ev});
    */

    return std::make_pair(sycl::event(), temporaries_cleanup_ev);
}

void init_advanced_indexing_dispatch_tables(void)
{
    using namespace dpctl::tensor::detail;

    using dpctl::tensor::kernels::indexing::TakeClipFactory;
    DispatchTableBuilder<take_fn_ptr_t, TakeClipFactory, num_types>
        dtb_takeclip;
    dtb_takeclip.populate_dispatch_table(take_dispatch_table[CLIP_MODE]);

    using dpctl::tensor::kernels::indexing::TakeWrapFactory;
    DispatchTableBuilder<take_fn_ptr_t, TakeWrapFactory, num_types>
        dtb_takewrap;
    dtb_takewrap.populate_dispatch_table(take_dispatch_table[WRAP_MODE]);

    using dpctl::tensor::kernels::indexing::PutClipFactory;
    DispatchTableBuilder<put_fn_ptr_t, PutClipFactory, num_types> dtb_putclip;
    dtb_putclip.populate_dispatch_table(put_dispatch_table[CLIP_MODE]);

    using dpctl::tensor::kernels::indexing::PutWrapFactory;
    DispatchTableBuilder<put_fn_ptr_t, PutWrapFactory, num_types> dtb_putwrap;
    dtb_putwrap.populate_dispatch_table(put_dispatch_table[WRAP_MODE]);
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
