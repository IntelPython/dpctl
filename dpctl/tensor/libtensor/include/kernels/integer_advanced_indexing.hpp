//=== indexing.hpp -  Implementation of indexing kernels ---*-C++-*--/===//
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
/// This file defines kernels for advanced tensor index operations.
//===----------------------------------------------------------------------===//

#pragma once
#include "utils/strided_iters.hpp"
#include "utils/type_utils.hpp"
#include <CL/sycl.hpp>
#include <algorithm>
#include <complex>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <type_traits>

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace indexing
{

namespace py = pybind11;

template <typename ProjectorT, typename Ty, typename indT> class take_kernel;
template <typename ProjectorT, typename Ty, typename indT> class put_kernel;

class ClipIndex
{
public:
    ClipIndex() = default;

    void operator()(py::ssize_t max_item, py::ssize_t &ind) const
    {
        max_item = std::max<py::ssize_t>(max_item, 1);
        ind = std::clamp<py::ssize_t>(ind, -max_item, max_item - 1);
        ind = (ind < 0) ? ind + max_item : ind;
        return;
    }
};

class WrapIndex
{
public:
    WrapIndex() = default;

    void operator()(py::ssize_t max_item, py::ssize_t &ind) const
    {
        max_item = std::max<py::ssize_t>(max_item, 1);
        ind = (ind < 0) ? ind % max_item + max_item : ind % max_item;
        return;
    }
};

template <typename ProjectorT, typename T, typename indT> class TakeFunctor
{
private:
    const char *src_ = nullptr;
    char *dst_ = nullptr;
    char **ind_ = nullptr;
    int nd_ = 0;
    int ind_nd_ = 0;
    int k_ = 0;
    size_t ind_nelems_ = 0;
    const py::ssize_t *orthog_shape_and_strides_ = nullptr;
    const py::ssize_t *axes_shape_and_strides_ = nullptr;
    const py::ssize_t *ind_shape_and_strides_ = nullptr;
    py::ssize_t src_offset_ = 0;
    py::ssize_t dst_offset_ = 0;
    const py::ssize_t *ind_offsets_ = nullptr;

public:
    TakeFunctor(const char *src_cp,
                char *dst_cp,
                char **ind_cp,
                int nd,
                int ind_nd,
                int k,
                size_t ind_nelems,
                const py::ssize_t *orthog_shape_and_strides,
                const py::ssize_t *axes_shape_and_strides,
                const py::ssize_t *ind_shape_and_strides,
                py::ssize_t src_offset,
                py::ssize_t dst_offset,
                const py::ssize_t *ind_offsets)
        : src_(src_cp), dst_(dst_cp), ind_(ind_cp), nd_(nd), ind_nd_(ind_nd),
          k_(k), ind_nelems_(ind_nelems),
          orthog_shape_and_strides_(orthog_shape_and_strides),
          axes_shape_and_strides_(axes_shape_and_strides),
          ind_shape_and_strides_(ind_shape_and_strides),
          src_offset_(src_offset), dst_offset_(dst_offset),
          ind_offsets_(ind_offsets)
    {
    }

    void operator()(sycl::id<1> id) const
    {
        const T *src = reinterpret_cast<const T *>(src_);
        T *dst = reinterpret_cast<T *>(dst_);

        py::ssize_t i_orthog = id / ind_nelems_;
        py::ssize_t i_along = id - (i_orthog * ind_nelems_);

        py::ssize_t src_orthog_idx(0);
        py::ssize_t dst_orthog_idx(0);
        CIndexer_vector<py::ssize_t> indxr(nd_);
        indxr.get_displacement<const py::ssize_t *, const py::ssize_t *>(
            static_cast<py::ssize_t>(i_orthog),
            orthog_shape_and_strides_,           // common shape
            orthog_shape_and_strides_ + nd_,     // src strides
            orthog_shape_and_strides_ + 2 * nd_, // dst strides
            src_orthog_idx,                      // modified by reference
            dst_orthog_idx);

        ProjectorT proj{};
        CIndexer_vector<py::ssize_t> ind_indxr(ind_nd_);
        for (int axis_idx = 0; axis_idx < k_; ++axis_idx) {
            py::ssize_t ind_arr_idx(0);
            ind_indxr.get_displacement<const py::ssize_t *>(
                static_cast<py::ssize_t>(i_along), ind_shape_and_strides_,
                ind_shape_and_strides_ + ((axis_idx + 1) * ind_nd_),
                ind_arr_idx);
            indT *ind_data = reinterpret_cast<indT *>(ind_[axis_idx]);
            py::ssize_t i = static_cast<py::ssize_t>(
                ind_data[ind_arr_idx + ind_offsets_[axis_idx]]);
            proj(axes_shape_and_strides_[axis_idx], i);
            src_orthog_idx += i * axes_shape_and_strides_[k_ + axis_idx];
        }
        py::ssize_t ind_dst_idx(0);
        ind_indxr.get_displacement<const py::ssize_t *>(
            static_cast<py::ssize_t>(i_along), ind_shape_and_strides_,
            axes_shape_and_strides_ + (2 * k_), ind_dst_idx);

        dst[dst_orthog_idx + ind_dst_idx + dst_offset_] =
            src[src_orthog_idx + src_offset_];
    }
};

typedef sycl::event (*take_fn_ptr_t)(sycl::queue,
                                     size_t,
                                     size_t,
                                     int,
                                     int,
                                     int,
                                     const py::ssize_t *,
                                     const py::ssize_t *,
                                     const py::ssize_t *,
                                     const char *,
                                     char *,
                                     char **,
                                     py::ssize_t,
                                     py::ssize_t,
                                     const py::ssize_t *,
                                     const std::vector<sycl::event> &);

template <typename ProjectorT, typename Ty, typename indT>
sycl::event take_impl(sycl::queue q,
                      size_t orthog_nelems,
                      size_t ind_nelems,
                      int nd,
                      int ind_nd,
                      int k,
                      const py::ssize_t *orthog_shape_and_strides,
                      const py::ssize_t *axes_shape_and_strides,
                      const py::ssize_t *ind_shape_and_strides,
                      const char *src_p,
                      char *dst_p,
                      char **ind_p,
                      py::ssize_t src_offset,
                      py::ssize_t dst_offset,
                      const py::ssize_t *ind_offsets,
                      const std::vector<sycl::event> &depends)
{
    dpctl::tensor::type_utils::validate_type_for_device<Ty>(q);

    sycl::event take_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        const size_t gws = orthog_nelems * ind_nelems;

        cgh.parallel_for<take_kernel<ProjectorT, Ty, indT>>(
            sycl::range<1>(gws),
            TakeFunctor<ProjectorT, Ty, indT>(
                src_p, dst_p, ind_p, nd, ind_nd, k, ind_nelems,
                orthog_shape_and_strides, axes_shape_and_strides,
                ind_shape_and_strides, src_offset, dst_offset, ind_offsets));
    });

    return take_ev;
}

template <typename ProjectorT, typename T, typename indT> class PutFunctor
{
private:
    char *dst_ = nullptr;
    const char *val_ = nullptr;
    char **ind_ = nullptr;
    int nd_ = 0;
    int ind_nd_ = 0;
    int k_ = 0;
    size_t ind_nelems_ = 0;
    const py::ssize_t *orthog_shape_and_strides_ = nullptr;
    const py::ssize_t *axes_shape_and_strides_ = nullptr;
    const py::ssize_t *ind_shape_and_strides_ = nullptr;
    py::ssize_t dst_offset_ = 0;
    py::ssize_t val_offset_ = 0;
    const py::ssize_t *ind_offsets_ = nullptr;

public:
    PutFunctor(char *dst_cp,
               const char *val_cp,
               char **ind_cp,
               int nd,
               int ind_nd,
               int k,
               size_t ind_nelems,
               const py::ssize_t *orthog_shape_and_strides,
               const py::ssize_t *axes_shape_and_strides,
               const py::ssize_t *ind_shape_and_strides,
               py::ssize_t dst_offset,
               py::ssize_t val_offset,
               const py::ssize_t *ind_offsets)
        : dst_(dst_cp), val_(val_cp), ind_(ind_cp), nd_(nd), ind_nd_(ind_nd),
          k_(k), ind_nelems_(ind_nelems),
          orthog_shape_and_strides_(orthog_shape_and_strides),
          axes_shape_and_strides_(axes_shape_and_strides),
          ind_shape_and_strides_(ind_shape_and_strides),
          dst_offset_(dst_offset), val_offset_(val_offset),
          ind_offsets_(ind_offsets)
    {
    }

    void operator()(sycl::id<1> id) const
    {
        T *dst = reinterpret_cast<T *>(dst_);
        const T *val = reinterpret_cast<const T *>(val_);

        py::ssize_t i_orthog = id / ind_nelems_;
        py::ssize_t i_along = id - (i_orthog * ind_nelems_);

        py::ssize_t dst_orthog_idx(0);
        py::ssize_t val_orthog_idx(0);
        CIndexer_vector<py::ssize_t> indxr(nd_);
        indxr.get_displacement<const py::ssize_t *, const py::ssize_t *>(
            static_cast<py::ssize_t>(i_orthog),
            orthog_shape_and_strides_,           // common shape
            orthog_shape_and_strides_ + nd_,     // dst strides
            orthog_shape_and_strides_ + 2 * nd_, // val strides
            dst_orthog_idx,                      // modified by reference
            val_orthog_idx);

        ProjectorT proj{};
        py::ssize_t ind_arr_idx(0);
        CIndexer_vector<py::ssize_t> ind_indxr(ind_nd_);
        for (int axis_idx = 0; axis_idx < k_; ++axis_idx) {
            ind_indxr.get_displacement<const py::ssize_t *>(
                static_cast<py::ssize_t>(i_along), ind_shape_and_strides_,
                ind_shape_and_strides_ + ((axis_idx + 1) * ind_nd_),
                ind_arr_idx);
            indT *ind_data = reinterpret_cast<indT *>(ind_[axis_idx]);
            py::ssize_t i = static_cast<py::ssize_t>(
                ind_data[ind_arr_idx + ind_offsets_[axis_idx]]);
            proj(axes_shape_and_strides_[axis_idx], i);
            dst_orthog_idx += i * axes_shape_and_strides_[k_ + axis_idx];
        }
        py::ssize_t ind_val_idx(0);
        ind_indxr.get_displacement<const py::ssize_t *>(
            static_cast<py::ssize_t>(i_along), ind_shape_and_strides_,
            axes_shape_and_strides_ + (2 * k_), ind_val_idx);

        dst[dst_orthog_idx + dst_offset_] =
            val[val_orthog_idx + ind_val_idx + val_offset_];
    }
};

typedef sycl::event (*put_fn_ptr_t)(sycl::queue,
                                    size_t,
                                    size_t,
                                    int,
                                    int,
                                    int,
                                    const py::ssize_t *,
                                    const py::ssize_t *,
                                    const py::ssize_t *,
                                    char *,
                                    const char *,
                                    char **,
                                    py::ssize_t,
                                    py::ssize_t,
                                    const py::ssize_t *,
                                    const std::vector<sycl::event> &);

template <typename ProjectorT, typename Ty, typename indT>
sycl::event put_impl(sycl::queue q,
                     size_t orthog_nelems,
                     size_t ind_nelems,
                     int nd,
                     int ind_nd,
                     int k,
                     const py::ssize_t *orthog_shape_and_strides,
                     const py::ssize_t *axes_shape_and_strides,
                     const py::ssize_t *ind_shape_and_strides,
                     char *dst_p,
                     const char *val_p,
                     char **ind_p,
                     py::ssize_t dst_offset,
                     py::ssize_t val_offset,
                     const py::ssize_t *ind_offsets,
                     const std::vector<sycl::event> &depends)
{
    dpctl::tensor::type_utils::validate_type_for_device<Ty>(q);

    sycl::event put_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        const size_t gws = orthog_nelems * ind_nelems;

        cgh.parallel_for<put_kernel<ProjectorT, Ty, indT>>(
            sycl::range<1>(gws),
            PutFunctor<ProjectorT, Ty, indT>(
                dst_p, val_p, ind_p, nd, ind_nd, k, ind_nelems,
                orthog_shape_and_strides, axes_shape_and_strides,
                ind_shape_and_strides, dst_offset, val_offset, ind_offsets));
    });

    return put_ev;
}

template <typename fnT, typename T, typename indT> struct TakeWrapFactory
{
    fnT get()
    {
        if constexpr (std::is_integral<indT>::value &&
                      !std::is_same<indT, bool>::value) {
            fnT fn = take_impl<WrapIndex, T, indT>;
            return fn;
        }
        else {
            fnT fn = nullptr;
            return fn;
        }
    }
};

template <typename fnT, typename T, typename indT> struct TakeClipFactory
{
    fnT get()
    {
        if constexpr (std::is_integral<indT>::value &&
                      !std::is_same<indT, bool>::value) {
            fnT fn = take_impl<ClipIndex, T, indT>;
            return fn;
        }
        else {
            fnT fn = nullptr;
            return fn;
        }
    }
};

template <typename fnT, typename T, typename indT> struct PutWrapFactory
{
    fnT get()
    {
        if constexpr (std::is_integral<indT>::value &&
                      !std::is_same<indT, bool>::value) {
            fnT fn = put_impl<WrapIndex, T, indT>;
            return fn;
        }
        else {
            fnT fn = nullptr;
            return fn;
        }
    }
};

template <typename fnT, typename T, typename indT> struct PutClipFactory
{
    fnT get()
    {
        if constexpr (std::is_integral<indT>::value &&
                      !std::is_same<indT, bool>::value) {
            fnT fn = put_impl<ClipIndex, T, indT>;
            return fn;
        }
        else {
            fnT fn = nullptr;
            return fn;
        }
    }
};

} // namespace indexing
} // namespace kernels
} // namespace tensor
} // namespace dpctl
