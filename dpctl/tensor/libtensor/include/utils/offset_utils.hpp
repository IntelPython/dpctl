//===-- offset_utils.hpp - Indexer classes for strided iteration  ---*-C++-*-//
//                                                                        ===//
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
/// This file defines Indexer callable operator to compute element offset in
/// an array addressed by gloabl_id.
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl.hpp>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <tuple>
#include <vector>

#include "utils/strided_iters.hpp"

namespace py = pybind11;

namespace dpctl
{
namespace tensor
{
namespace offset_utils
{

namespace detail
{

struct sink_t
{
    sink_t(){};
    template <class T> sink_t(T &&){};
};

template <class V> std::size_t __accumulate_size(std::size_t &s, V &&v)
{
    return s += v.size();
}

template <class V, class U> sink_t __appender(V &lhs, U &&rhs)
{
    lhs.insert(lhs.end(), rhs.begin(), rhs.end());
    return {};
}

template <typename T, typename A, typename... Vs>
std::vector<T, A> concat(std::vector<T, A> lhs, Vs &&...vs)
{
    std::size_t s = lhs.size();
    {
        // limited scope ensures array is freed
        [[maybe_unused]] sink_t tmp[] = {__accumulate_size(s, vs)..., 0};
    }
    lhs.reserve(s);
    {
        // array of no-data objects ensures ordering of calls to the appender
        [[maybe_unused]] sink_t tmp[] = {
            __appender(lhs, std::forward<Vs>(vs))..., 0};
    }

    return std::move(lhs); // prevent return-value optimization
}

} // namespace detail

template <typename indT, typename... Vs>
std::tuple<indT *, size_t, sycl::event>
device_allocate_and_pack(sycl::queue q,
                         std::vector<sycl::event> &host_task_events,
                         Vs &&...vs)
{

    // memory transfer optimization, use USM-host for temporary speeds up
    // tranfer to device, especially on dGPUs
    using usm_host_allocatorT =
        sycl::usm_allocator<indT, sycl::usm::alloc::host>;
    using shT = std::vector<indT, usm_host_allocatorT>;

    usm_host_allocatorT usm_host_allocator(q);
    shT empty{0, usm_host_allocator};
    shT packed_shape_strides = detail::concat(empty, vs...);

    auto packed_shape_strides_owner =
        std::make_shared<shT>(std::move(packed_shape_strides));

    auto sz = packed_shape_strides_owner->size();
    indT *shape_strides = sycl::malloc_device<indT>(sz, q);

    if (shape_strides == nullptr) {
        return std::make_tuple(shape_strides, 0, sycl::event());
    }

    sycl::event copy_ev =
        q.copy<indT>(packed_shape_strides_owner->data(), shape_strides, sz);

    sycl::event cleanup_host_task_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(copy_ev);
        cgh.host_task([packed_shape_strides_owner] {
            // increment shared pointer ref-count to keep it alive
            // till copy operation completes;
        });
    });
    host_task_events.push_back(cleanup_host_task_ev);

    return std::make_tuple(shape_strides, sz, copy_ev);
}

struct NoOpIndexer
{
    size_t operator()(size_t gid) const
    {
        return gid;
    }
};

/* @brief Indexer with shape and strides arrays of same size are packed */
struct StridedIndexer
{
    StridedIndexer(int _nd,
                   py::ssize_t _offset,
                   py::ssize_t const *_packed_shape_strides)
        : nd(_nd), starting_offset(_offset),
          shape_strides(_packed_shape_strides)
    {
    }

    size_t operator()(py::ssize_t gid) const
    {
        return compute_offset(gid);
    }

    size_t operator()(size_t gid) const
    {
        return compute_offset(static_cast<py::ssize_t>(gid));
    }

private:
    int nd;
    py::ssize_t starting_offset;
    py::ssize_t const *shape_strides;

    size_t compute_offset(py::ssize_t gid) const
    {
        using dpctl::tensor::strides::CIndexer_vector;

        CIndexer_vector _ind(nd);
        py::ssize_t relative_offset(0);
        _ind.get_displacement<const py::ssize_t *, const py::ssize_t *>(
            gid,
            shape_strides,      // shape ptr
            shape_strides + nd, // strides ptr
            relative_offset);
        return starting_offset + relative_offset;
    }
};

/* @brief Indexer with shape, strides provided separately */
struct UnpackedStridedIndexer
{
    UnpackedStridedIndexer(int _nd,
                           py::ssize_t _offset,
                           py::ssize_t const *_shape,
                           py::ssize_t const *_strides)
        : nd(_nd), starting_offset(_offset), shape(_shape), strides(_strides)
    {
    }

    size_t operator()(py::ssize_t gid) const
    {
        return compute_offset(gid);
    }

    size_t operator()(size_t gid) const
    {
        return compute_offset(static_cast<py::ssize_t>(gid));
    }

private:
    int nd;
    py::ssize_t starting_offset;
    py::ssize_t const *shape;
    py::ssize_t const *strides;

    size_t compute_offset(py::ssize_t gid) const
    {
        using dpctl::tensor::strides::CIndexer_vector;

        CIndexer_vector _ind(nd);
        py::ssize_t relative_offset(0);
        _ind.get_displacement<const py::ssize_t *, const py::ssize_t *>(
            gid,
            shape,   // shape ptr
            strides, // strides ptr
            relative_offset);
        return starting_offset + relative_offset;
    }
};

struct Strided1DIndexer
{
    Strided1DIndexer(py::ssize_t _offset, py::ssize_t _size, py::ssize_t _step)
        : offset(_offset), size(static_cast<size_t>(_size)), step(_step)
    {
    }

    size_t operator()(size_t gid) const
    {
        // ensure 0 <= gid < size
        return static_cast<size_t>(offset +
                                   std::min<size_t>(gid, size - 1) * step);
    }

private:
    py::ssize_t offset = 0;
    size_t size = 1;
    py::ssize_t step = 1;
};

struct Strided1DCyclicIndexer
{
    Strided1DCyclicIndexer(py::ssize_t _offset,
                           py::ssize_t _size,
                           py::ssize_t _step)
        : offset(_offset), size(static_cast<size_t>(_size)), step(_step)
    {
    }

    size_t operator()(size_t gid) const
    {
        return static_cast<size_t>(offset + (gid % size) * step);
    }

private:
    py::ssize_t offset = 0;
    size_t size = 1;
    py::ssize_t step = 1;
};

template <typename displacementT> struct TwoOffsets
{
    TwoOffsets() : first_offset(0), second_offset(0) {}
    TwoOffsets(const displacementT &first_offset_,
               const displacementT &second_offset_)
        : first_offset(first_offset_), second_offset(second_offset_)
    {
    }

    displacementT get_first_offset() const
    {
        return first_offset;
    }
    displacementT get_second_offset() const
    {
        return second_offset;
    }

private:
    displacementT first_offset = 0;
    displacementT second_offset = 0;
};

struct TwoOffsets_StridedIndexer
{
    TwoOffsets_StridedIndexer(int common_nd,
                              py::ssize_t first_offset_,
                              py::ssize_t second_offset_,
                              py::ssize_t const *_packed_shape_strides)
        : nd(common_nd), starting_first_offset(first_offset_),
          starting_second_offset(second_offset_),
          shape_strides(_packed_shape_strides)
    {
    }

    TwoOffsets<py::ssize_t> operator()(py::ssize_t gid) const
    {
        return compute_offsets(gid);
    }

    TwoOffsets<py::ssize_t> operator()(size_t gid) const
    {
        return compute_offsets(static_cast<py::ssize_t>(gid));
    }

private:
    int nd;
    py::ssize_t starting_first_offset;
    py::ssize_t starting_second_offset;
    py::ssize_t const *shape_strides;

    TwoOffsets<py::ssize_t> compute_offsets(py::ssize_t gid) const
    {
        using dpctl::tensor::strides::CIndexer_vector;

        CIndexer_vector _ind(nd);
        py::ssize_t relative_first_offset(0);
        py::ssize_t relative_second_offset(0);
        _ind.get_displacement<const py::ssize_t *, const py::ssize_t *>(
            gid,
            shape_strides,          // shape ptr
            shape_strides + nd,     // strides ptr
            shape_strides + 2 * nd, // strides ptr
            relative_first_offset, relative_second_offset);
        return TwoOffsets<py::ssize_t>(
            starting_first_offset + relative_first_offset,
            starting_second_offset + relative_second_offset);
    }
};

struct TwoZeroOffsets_Indexer
{
    TwoZeroOffsets_Indexer() {}

    TwoOffsets<py::ssize_t> operator()(py::ssize_t) const
    {
        return TwoOffsets<py::ssize_t>();
    }
};

template <typename FirstIndexerT, typename SecondIndexerT>
struct TwoOffsets_CombinedIndexer
{
private:
    FirstIndexerT first_indexer_;
    SecondIndexerT second_indexer_;

public:
    TwoOffsets_CombinedIndexer(const FirstIndexerT &first_indexer,
                               const SecondIndexerT &second_indexer)
        : first_indexer_(first_indexer), second_indexer_(second_indexer)
    {
    }

    TwoOffsets<py::ssize_t> operator()(py::ssize_t gid) const
    {
        return TwoOffsets<py::ssize_t>(first_indexer_(gid),
                                       second_indexer_(gid));
    }
};

template <typename displacementT> struct ThreeOffsets
{
    ThreeOffsets() : first_offset(0), second_offset(0), third_offset(0) {}
    ThreeOffsets(const displacementT &first_offset_,
                 const displacementT &second_offset_,
                 const displacementT &third_offset_)
        : first_offset(first_offset_), second_offset(second_offset_),
          third_offset(third_offset_)
    {
    }

    displacementT get_first_offset() const
    {
        return first_offset;
    }
    displacementT get_second_offset() const
    {
        return second_offset;
    }
    displacementT get_third_offset() const
    {
        return third_offset;
    }

private:
    displacementT first_offset = 0;
    displacementT second_offset = 0;
    displacementT third_offset = 0;
};

struct ThreeOffsets_StridedIndexer
{
    ThreeOffsets_StridedIndexer(int common_nd,
                                py::ssize_t first_offset_,
                                py::ssize_t second_offset_,
                                py::ssize_t third_offset_,
                                py::ssize_t const *_packed_shape_strides)
        : nd(common_nd), starting_first_offset(first_offset_),
          starting_second_offset(second_offset_),
          starting_third_offset(third_offset_),
          shape_strides(_packed_shape_strides)
    {
    }

    ThreeOffsets<py::ssize_t> operator()(py::ssize_t gid) const
    {
        return compute_offsets(gid);
    }

    ThreeOffsets<py::ssize_t> operator()(size_t gid) const
    {
        return compute_offsets(static_cast<py::ssize_t>(gid));
    }

private:
    int nd;
    py::ssize_t starting_first_offset;
    py::ssize_t starting_second_offset;
    py::ssize_t starting_third_offset;
    py::ssize_t const *shape_strides;

    ThreeOffsets<py::ssize_t> compute_offsets(py::ssize_t gid) const
    {
        using dpctl::tensor::strides::CIndexer_vector;

        CIndexer_vector _ind(nd);
        py::ssize_t relative_first_offset(0);
        py::ssize_t relative_second_offset(0);
        py::ssize_t relative_third_offset(0);
        _ind.get_displacement<const py::ssize_t *, const py::ssize_t *>(
            gid,
            shape_strides,          // shape ptr
            shape_strides + nd,     // strides ptr
            shape_strides + 2 * nd, // strides ptr
            shape_strides + 3 * nd, // strides ptr
            relative_first_offset, relative_second_offset,
            relative_third_offset);
        return ThreeOffsets<py::ssize_t>(
            starting_first_offset + relative_first_offset,
            starting_second_offset + relative_second_offset,
            starting_third_offset + relative_third_offset);
    }
};

struct ThreeZeroOffsets_Indexer
{
    ThreeZeroOffsets_Indexer() {}

    ThreeOffsets<py::ssize_t> operator()(py::ssize_t) const
    {
        return ThreeOffsets<py::ssize_t>();
    }
};

template <typename displacementT> struct FourOffsets
{
    FourOffsets()
        : first_offset(0), second_offset(0), third_offset(0), fourth_offset(0)
    {
    }
    FourOffsets(const displacementT &first_offset_,
                const displacementT &second_offset_,
                const displacementT &third_offset_,
                const displacementT &fourth_offset_)
        : first_offset(first_offset_), second_offset(second_offset_),
          third_offset(third_offset_), fourth_offset(fourth_offset_)
    {
    }

    displacementT get_first_offset() const
    {
        return first_offset;
    }
    displacementT get_second_offset() const
    {
        return second_offset;
    }
    displacementT get_third_offset() const
    {
        return third_offset;
    }
    displacementT get_fourth_offset() const
    {
        return fourth_offset;
    }

private:
    displacementT first_offset = 0;
    displacementT second_offset = 0;
    displacementT third_offset = 0;
    displacementT fourth_offset = 0;
};

struct FourOffsets_StridedIndexer
{
    FourOffsets_StridedIndexer(int common_nd,
                               py::ssize_t first_offset_,
                               py::ssize_t second_offset_,
                               py::ssize_t third_offset_,
                               py::ssize_t fourth_offset_,
                               py::ssize_t const *_packed_shape_strides)
        : nd(common_nd), starting_first_offset(first_offset_),
          starting_second_offset(second_offset_),
          starting_third_offset(third_offset_),
          starting_fourth_offset(fourth_offset_),
          shape_strides(_packed_shape_strides)
    {
    }

    FourOffsets<py::ssize_t> operator()(py::ssize_t gid) const
    {
        return compute_offsets(gid);
    }

    FourOffsets<py::ssize_t> operator()(size_t gid) const
    {
        return compute_offsets(static_cast<py::ssize_t>(gid));
    }

private:
    int nd;
    py::ssize_t starting_first_offset;
    py::ssize_t starting_second_offset;
    py::ssize_t starting_third_offset;
    py::ssize_t starting_fourth_offset;
    py::ssize_t const *shape_strides;

    FourOffsets<py::ssize_t> compute_offsets(py::ssize_t gid) const
    {
        using dpctl::tensor::strides::CIndexer_vector;

        CIndexer_vector _ind(nd);
        py::ssize_t relative_first_offset(0);
        py::ssize_t relative_second_offset(0);
        py::ssize_t relative_third_offset(0);
        py::ssize_t relative_fourth_offset(0);
        _ind.get_displacement<const py::ssize_t *, const py::ssize_t *>(
            gid,
            shape_strides,          // shape ptr
            shape_strides + nd,     // strides ptr
            shape_strides + 2 * nd, // strides ptr
            shape_strides + 3 * nd, // strides ptr
            shape_strides + 4 * nd, // strides ptr
            relative_first_offset, relative_second_offset,
            relative_third_offset, relative_fourth_offset);
        return FourOffsets<py::ssize_t>(
            starting_first_offset + relative_first_offset,
            starting_second_offset + relative_second_offset,
            starting_third_offset + relative_third_offset,
            starting_fourth_offset + relative_fourth_offset);
    }
};

struct FourZeroOffsets_Indexer
{
    FourZeroOffsets_Indexer() {}

    FourOffsets<py::ssize_t> operator()(py::ssize_t) const
    {
        return FourOffsets<py::ssize_t>();
    }
};

struct NthStrideOffset
{
    NthStrideOffset(int common_nd,
                    py::ssize_t const *_offsets,
                    py::ssize_t const *_packed_shape_strides)
        : _ind(common_nd), nd(common_nd), offsets(_offsets),
          shape_strides(_packed_shape_strides)
    {
    }

    size_t operator()(py::ssize_t gid, int n) const
    {
        py::ssize_t relative_offset(0);
        _ind.get_displacement<const py::ssize_t *, const py::ssize_t *>(
            gid, shape_strides, shape_strides + ((n + 1) * nd),
            relative_offset);

        return relative_offset + offsets[n];
    }

private:
    dpctl::tensor::strides::CIndexer_vector<py::ssize_t> _ind;

    int nd;
    py::ssize_t const *offsets;
    py::ssize_t const *shape_strides;
};

template <int nd> struct FixedDimStridedIndexer
{
    FixedDimStridedIndexer(const std::array<py::ssize_t, nd> _shape,
                           const std::array<py::ssize_t, nd> _strides,
                           py::ssize_t _offset)
        : _ind(_shape), strides(_strides), starting_offset(_offset)
    {
    }
    size_t operator()(size_t gid) const
    {
        dpctl::tensor::strides::CIndexer_array<nd, py::ssize_t> local_indexer(
            std::move(_ind));
        local_indexer.set(gid);
        auto mi = local_indexer.get();

        py::ssize_t relative_offset = 0;

#pragma unroll
        for (int i = 0; i < nd; ++i) {
            relative_offset += mi[i] * strides[i];
        }
        return starting_offset + relative_offset;
    }

private:
    dpctl::tensor::strides::CIndexer_array<nd, py::ssize_t> _ind;

    const std::array<py::ssize_t, nd> strides;
    py::ssize_t starting_offset;
};

template <int nd> struct TwoOffsets_FixedDimStridedIndexer
{
    TwoOffsets_FixedDimStridedIndexer(
        const std::array<py::ssize_t, nd> _shape,
        const std::array<py::ssize_t, nd> _strides1,
        const std::array<py::ssize_t, nd> _strides2,
        py::ssize_t _offset1,
        py::ssize_t _offset2)
        : _ind(_shape), strides1(_strides1), strides2(_strides2),
          starting_offset1(_offset1), starting_offset2(_offset2)
    {
    }

    TwoOffsets<py::ssize_t> operator()(size_t gid) const
    {
        dpctl::tensor::strides::CIndexer_array<nd, py::ssize_t> local_indexer(
            std::move(_ind));
        local_indexer.set(gid);
        auto mi = local_indexer.get();

        py::ssize_t relative_offset1 = 0;
#pragma unroll
        for (int i = 0; i < nd; ++i) {
            relative_offset1 += mi[i] * strides1[i];
        }

        py::ssize_t relative_offset2 = 0;
#pragma unroll
        for (int i = 0; i < nd; ++i) {
            relative_offset2 += mi[i] * strides2[i];
        }

        return TwoOffsets<py::ssize_t>(starting_offset1 + relative_offset1,
                                       starting_offset2 + relative_offset2);
    }

private:
    dpctl::tensor::strides::CIndexer_array<nd, py::ssize_t> _ind;

    const std::array<py::ssize_t, nd> strides1;
    const std::array<py::ssize_t, nd> strides2;
    py::ssize_t starting_offset1;
    py::ssize_t starting_offset2;
};

} // namespace offset_utils
} // namespace tensor
} // namespace dpctl
