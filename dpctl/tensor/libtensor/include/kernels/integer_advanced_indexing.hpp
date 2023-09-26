//=== indexing.hpp -  Implementation of indexing kernels ---*-C++-*--/===//
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
/// This file defines kernels for advanced tensor index operations.
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl.hpp>
#include <algorithm>
#include <complex>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <type_traits>

#include "utils/offset_utils.hpp"
#include "utils/type_utils.hpp"

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace indexing
{

namespace py = pybind11;
using namespace dpctl::tensor::offset_utils;

template <typename ProjectorT,
          typename OrthogStrider,
          typename IndicesStrider,
          typename AxesStrider,
          typename T,
          typename indT>
class take_kernel;
template <typename ProjectorT,
          typename OrthogStrider,
          typename IndicesStrider,
          typename AxesStrider,
          typename T,
          typename indT>
class put_kernel;

class WrapIndex
{
public:
    WrapIndex() = default;

    void operator()(py::ssize_t max_item, py::ssize_t &ind) const
    {
        max_item = std::max<py::ssize_t>(max_item, 1);
        ind = std::clamp<py::ssize_t>(ind, -max_item, max_item - 1);
        ind = (ind < 0) ? ind + max_item : ind;
        return;
    }
};

class ClipIndex
{
public:
    ClipIndex() = default;

    void operator()(py::ssize_t max_item, py::ssize_t &ind) const
    {
        max_item = std::max<py::ssize_t>(max_item, 1);
        ind = std::clamp<py::ssize_t>(ind, 0, max_item - 1);
        return;
    }
};

template <typename ProjectorT,
          typename OrthogStrider,
          typename IndicesStrider,
          typename AxesStrider,
          typename T,
          typename indT>
class TakeFunctor
{
private:
    const char *src_ = nullptr;
    char *dst_ = nullptr;
    char **ind_ = nullptr;
    int k_ = 0;
    size_t ind_nelems_ = 0;
    const py::ssize_t *axes_shape_and_strides_ = nullptr;
    OrthogStrider orthog_strider;
    IndicesStrider ind_strider;
    AxesStrider axes_strider;

public:
    TakeFunctor(const char *src_cp,
                char *dst_cp,
                char **ind_cp,
                int k,
                size_t ind_nelems,
                const py::ssize_t *axes_shape_and_strides,
                OrthogStrider orthog_strider_,
                IndicesStrider ind_strider_,
                AxesStrider axes_strider_)
        : src_(src_cp), dst_(dst_cp), ind_(ind_cp), k_(k),
          ind_nelems_(ind_nelems),
          axes_shape_and_strides_(axes_shape_and_strides),
          orthog_strider(orthog_strider_), ind_strider(ind_strider_),
          axes_strider(axes_strider_)
    {
    }

    void operator()(sycl::id<1> id) const
    {
        const T *src = reinterpret_cast<const T *>(src_);
        T *dst = reinterpret_cast<T *>(dst_);

        py::ssize_t i_orthog = id / ind_nelems_;
        py::ssize_t i_along = id - (i_orthog * ind_nelems_);

        auto orthog_offsets = orthog_strider(i_orthog);

        py::ssize_t src_offset = orthog_offsets.get_first_offset();
        py::ssize_t dst_offset = orthog_offsets.get_second_offset();

        ProjectorT proj{};
        for (int axis_idx = 0; axis_idx < k_; ++axis_idx) {
            indT *ind_data = reinterpret_cast<indT *>(ind_[axis_idx]);

            py::ssize_t ind_offset = ind_strider(i_along, axis_idx);
            py::ssize_t i = static_cast<py::ssize_t>(ind_data[ind_offset]);

            proj(axes_shape_and_strides_[axis_idx], i);

            src_offset += i * axes_shape_and_strides_[k_ + axis_idx];
        }

        dst_offset += axes_strider(i_along);

        dst[dst_offset] = src[src_offset];
    }
};

typedef sycl::event (*take_fn_ptr_t)(sycl::queue &,
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
sycl::event take_impl(sycl::queue &q,
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

        TwoOffsets_StridedIndexer orthog_indexer{nd, src_offset, dst_offset,
                                                 orthog_shape_and_strides};
        NthStrideOffset indices_indexer{ind_nd, ind_offsets,
                                        ind_shape_and_strides};
        StridedIndexer axes_indexer{ind_nd, 0,
                                    axes_shape_and_strides + (2 * k)};

        const size_t gws = orthog_nelems * ind_nelems;

        cgh.parallel_for<
            take_kernel<ProjectorT, TwoOffsets_StridedIndexer, NthStrideOffset,
                        StridedIndexer, Ty, indT>>(
            sycl::range<1>(gws),
            TakeFunctor<ProjectorT, TwoOffsets_StridedIndexer, NthStrideOffset,
                        StridedIndexer, Ty, indT>(
                src_p, dst_p, ind_p, k, ind_nelems, axes_shape_and_strides,
                orthog_indexer, indices_indexer, axes_indexer));
    });

    return take_ev;
}

template <typename ProjectorT,
          typename OrthogStrider,
          typename IndicesStrider,
          typename AxesStrider,
          typename T,
          typename indT>
class PutFunctor
{
private:
    char *dst_ = nullptr;
    const char *val_ = nullptr;
    char **ind_ = nullptr;
    int k_ = 0;
    size_t ind_nelems_ = 0;
    const py::ssize_t *axes_shape_and_strides_ = nullptr;
    OrthogStrider orthog_strider;
    IndicesStrider ind_strider;
    AxesStrider axes_strider;

public:
    PutFunctor(char *dst_cp,
               const char *val_cp,
               char **ind_cp,
               int k,
               size_t ind_nelems,
               const py::ssize_t *axes_shape_and_strides,
               OrthogStrider orthog_strider_,
               IndicesStrider ind_strider_,
               AxesStrider axes_strider_)
        : dst_(dst_cp), val_(val_cp), ind_(ind_cp), k_(k),
          ind_nelems_(ind_nelems),
          axes_shape_and_strides_(axes_shape_and_strides),
          orthog_strider(orthog_strider_), ind_strider(ind_strider_),
          axes_strider(axes_strider_)
    {
    }

    void operator()(sycl::id<1> id) const
    {
        T *dst = reinterpret_cast<T *>(dst_);
        const T *val = reinterpret_cast<const T *>(val_);

        py::ssize_t i_orthog = id / ind_nelems_;
        py::ssize_t i_along = id - (i_orthog * ind_nelems_);

        auto orthog_offsets = orthog_strider(i_orthog);

        py::ssize_t dst_offset = orthog_offsets.get_first_offset();
        py::ssize_t val_offset = orthog_offsets.get_second_offset();

        ProjectorT proj{};
        for (int axis_idx = 0; axis_idx < k_; ++axis_idx) {
            indT *ind_data = reinterpret_cast<indT *>(ind_[axis_idx]);

            py::ssize_t ind_offset = ind_strider(i_along, axis_idx);
            py::ssize_t i = static_cast<py::ssize_t>(ind_data[ind_offset]);

            proj(axes_shape_and_strides_[axis_idx], i);

            dst_offset += i * axes_shape_and_strides_[k_ + axis_idx];
        }

        val_offset += axes_strider(i_along);

        dst[dst_offset] = val[val_offset];
    }
};

typedef sycl::event (*put_fn_ptr_t)(sycl::queue &,
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
sycl::event put_impl(sycl::queue &q,
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

        TwoOffsets_StridedIndexer orthog_indexer{nd, dst_offset, val_offset,
                                                 orthog_shape_and_strides};
        NthStrideOffset indices_indexer{ind_nd, ind_offsets,
                                        ind_shape_and_strides};
        StridedIndexer axes_indexer{ind_nd, 0,
                                    axes_shape_and_strides + (2 * k)};

        const size_t gws = orthog_nelems * ind_nelems;

        cgh.parallel_for<put_kernel<ProjectorT, TwoOffsets_StridedIndexer,
                                    NthStrideOffset, StridedIndexer, Ty, indT>>(
            sycl::range<1>(gws),
            PutFunctor<ProjectorT, TwoOffsets_StridedIndexer, NthStrideOffset,
                       StridedIndexer, Ty, indT>(
                dst_p, val_p, ind_p, k, ind_nelems, axes_shape_and_strides,
                orthog_indexer, indices_indexer, axes_indexer));
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
