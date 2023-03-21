#pragma once

#include <CL/sycl.hpp>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <tuple>
#include <vector>

#include "utils/strided_iters.hpp"

namespace py = pybind11;

namespace concat_impl
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

} // namespace concat_impl

template <typename T, typename A, typename... Vs>
std::vector<T, A> concat(std::vector<T, A> lhs, Vs &&...vs)
{
    using concat_impl::__accumulate_size;
    using concat_impl::__appender;
    using concat_impl::sink_t;
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

namespace dpctl
{
namespace tensor
{
namespace offset_utils
{

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
    shT packed_shape_strides = concat(empty, vs...);

    auto packed_shape_strides_owner =
        std::make_shared<shT>(std::move(packed_shape_strides));

    auto sz = packed_shape_strides_owner->size();
    indT *shape_strides = sycl::malloc_device<indT>(sz, q);

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

struct StridedIndexer
{
    StridedIndexer(int _nd,
                   py::ssize_t _offset,
                   py::ssize_t const *_packed_shape_strides)
        : nd(_nd), starting_offset(_offset),
          shape_strides(_packed_shape_strides)
    {
    }

    size_t operator()(size_t gid) const
    {
        CIndexer_vector _ind(nd);
        py::ssize_t relative_offset(0);
        _ind.get_displacement<const py::ssize_t *, const py::ssize_t *>(
            static_cast<py::ssize_t>(gid),
            shape_strides,      // shape ptr
            shape_strides + nd, // strides ptr
            relative_offset);
        return starting_offset + relative_offset;
    }

private:
    int nd;
    py::ssize_t starting_offset;
    py::ssize_t const *shape_strides;
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
    TwoOffsets(displacementT first_offset_, displacementT second_offset_)
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
        CIndexer_vector _ind(nd);
        py::ssize_t relative_first_offset(0);
        py::ssize_t relative_second_offset(0);
        _ind.get_displacement<const py::ssize_t *, const py::ssize_t *>(
            gid,
            shape_strides,          // shape ptr
            shape_strides + nd,     // src strides ptr
            shape_strides + 2 * nd, // src strides ptr
            relative_first_offset, relative_second_offset);
        return TwoOffsets<py::ssize_t>(
            starting_first_offset + relative_first_offset,
            starting_second_offset + relative_second_offset);
    }

private:
    int nd;
    py::ssize_t starting_first_offset;
    py::ssize_t starting_second_offset;
    py::ssize_t const *shape_strides;
};

struct TwoZeroOffsets_Indexer
{
    TwoZeroOffsets_Indexer() {}

    TwoOffsets<py::ssize_t> operator()(py::ssize_t) const
    {
        return TwoOffsets<py::ssize_t>();
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
            gid, shape_strides, shape_strides + ((n + 1) * nd), relative_offset);

        return relative_offset + offsets[n];
    }

private:
    CIndexer_vector<py::ssize_t> _ind;

    int nd;
    py::ssize_t const *offsets;
    py::ssize_t const *shape_strides;
};

template <int nd> struct StridedIndexerArray
{
    StridedIndexerArray(const std::array<py::ssize_t, nd> _shape,
                        const std::array<py::ssize_t, nd> _strides,
                        py::ssize_t _offset)
        : _ind(_shape), strides(_strides), starting_offset(_offset)
    {
    }
    size_t operator()(size_t gid) const
    {
        py::ssize_t relative_offset = 0;
        CIndexer_array<nd, py::ssize_t> local_indxr(std::move(_ind));

        local_indxr.set(gid);
        auto mi = local_indxr.get();
#pragma unroll
        for (int i = 0; i < nd; ++i) {
            relative_offset += mi[i] * strides[i];
        }
        return starting_offset + relative_offset;
    }

private:
    CIndexer_array<nd, py::ssize_t> _ind;

    const std::array<py::ssize_t, nd> strides;
    py::ssize_t starting_offset;
};

template <int nd> struct TwoOffsets_StridedIndexerArray
{
    TwoOffsets_StridedIndexerArray(const std::array<py::ssize_t, nd> _shape,
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
        py::ssize_t relative_offset1 = 0;
        py::ssize_t relative_offset2 = 0;

        CIndexer_array<nd, py::ssize_t> local_indxr(std::move(_ind));
        local_indxr.set(gid);
        auto mi = local_indxr.get();
#pragma unroll
        for (int i = 0; i < nd; ++i) {
            relative_offset1 += mi[i] * strides1[i];
        }
#pragma unroll
        for (int i = 0; i < nd; ++i) {
            relative_offset2 += mi[i] * strides2[i];
        }
        return TwoOffsets<py::ssize_t>(starting_offset1 + relative_offset1,
                                       starting_offset2 + relative_offset2);
    }

private:
    CIndexer_array<nd, py::ssize_t> _ind;

    const std::array<py::ssize_t, nd> strides1;
    const std::array<py::ssize_t, nd> strides2;
    py::ssize_t starting_offset1;
    py::ssize_t starting_offset2;
};

} // namespace offset_utils
} // namespace tensor
} // namespace dpctl