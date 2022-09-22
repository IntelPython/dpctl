//=== copy_and_cast.hpp - Implementation of copy-and-cast kernels *-C++-*/===//
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
/// This file defines kernels for tensor copying and value casting.
//===----------------------------------------------------------------------===//

#pragma once
#include "utils/strided_iters.hpp"
#include "utils/type_utils.hpp"
#include <CL/sycl.hpp>
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
namespace copy_and_cast
{

namespace py = pybind11;

template <typename srcT, typename dstT> class copy_cast_generic_kernel;
template <typename srcT, typename dstT> class copy_cast_from_host_kernel;
template <typename srcT, typename dstT, int nd> class copy_cast_spec_kernel;
template <typename Ty> class copy_for_reshape_generic_kernel;

template <typename srcT, typename dstT> class Caster
{
public:
    Caster() = default;
    void operator()(char *src,
                    std::ptrdiff_t src_offset,
                    char *dst,
                    std::ptrdiff_t dst_offset) const
    {
        using dpctl::tensor::type_utils::convert_impl;

        srcT *src_ = reinterpret_cast<srcT *>(src) + src_offset;
        dstT *dst_ = reinterpret_cast<dstT *>(dst) + dst_offset;
        *dst_ = convert_impl<dstT, srcT>(*src_);
    }
};

template <typename CastFnT> class GenericCopyFunctor
{
private:
    char *src_ = nullptr;
    char *dst_ = nullptr;
    py::ssize_t *shape_strides_ = nullptr;
    int nd_ = 0;
    py::ssize_t src_offset0 = 0;
    py::ssize_t dst_offset0 = 0;

public:
    GenericCopyFunctor(char *src_cp,
                       char *dst_cp,
                       py::ssize_t *shape_strides,
                       int nd,
                       py::ssize_t src_offset,
                       py::ssize_t dst_offset)
        : src_(src_cp), dst_(dst_cp), shape_strides_(shape_strides), nd_(nd),
          src_offset0(src_offset), dst_offset0(dst_offset)
    {
    }

    void operator()(sycl::id<1> wiid) const
    {
        py::ssize_t src_offset(0);
        py::ssize_t dst_offset(0);
        CIndexer_vector<py::ssize_t> indxr(nd_);
        indxr.get_displacement<const py::ssize_t *, const py::ssize_t *>(
            static_cast<py::ssize_t>(wiid.get(0)),
            const_cast<const py::ssize_t *>(shape_strides_), // common shape
            const_cast<const py::ssize_t *>(shape_strides_ +
                                            nd_), // src strides
            const_cast<const py::ssize_t *>(shape_strides_ +
                                            2 * nd_), // dst strides
            src_offset,                               // modified by reference
            dst_offset                                // modified by reference
        );
        CastFnT fn{};
        fn(src_, src_offset0 + src_offset, dst_, dst_offset0 + dst_offset);
    }
};

template <int nd, typename CastFnT> class NDSpecializedCopyFunctor
{
private:
    char *src_ = nullptr;
    char *dst_ = nullptr;
    CIndexer_array<nd, py::ssize_t> indxr;
    const std::array<py::ssize_t, nd> src_strides_;
    const std::array<py::ssize_t, nd> dst_strides_;
    static const int nd_ = nd;
    py::ssize_t src_offset0 = 0;
    py::ssize_t dst_offset0 = 0;

public:
    NDSpecializedCopyFunctor(char *src_cp, // USM pointer
                             char *dst_cp, // USM pointer
                             const std::array<py::ssize_t, nd> shape,
                             const std::array<py::ssize_t, nd> src_strides,
                             const std::array<py::ssize_t, nd> dst_strides,
                             py::ssize_t src_offset,
                             py::ssize_t dst_offset)
        : src_(src_cp), dst_(dst_cp), indxr(shape), src_strides_(src_strides),
          dst_strides_(dst_strides), src_offset0(src_offset),
          dst_offset0(dst_offset)
    {
    }

    void operator()(sycl::id<1> wiid) const
    {
        py::ssize_t src_offset = 0;
        py::ssize_t dst_offset = 0;
        CIndexer_array<nd, py::ssize_t> local_indxr(std::move(indxr));

        local_indxr.set(wiid.get(0));
        auto mi = local_indxr.get();
        for (int i = 0; i < nd; ++i)
            src_offset += mi[i] * src_strides_[i];
        for (int i = 0; i < nd; ++i)
            dst_offset += mi[i] * dst_strides_[i];

        CastFnT fn{};
        fn(src_, src_offset0 + src_offset, dst_, dst_offset0 + dst_offset);
    }
};

typedef sycl::event (*copy_and_cast_generic_fn_ptr_t)(
    sycl::queue,
    size_t,
    int,
    py::ssize_t *,
    char *,
    py::ssize_t,
    char *,
    py::ssize_t,
    const std::vector<sycl::event> &,
    const std::vector<sycl::event> &);

template <typename dstTy, typename srcTy>
sycl::event
copy_and_cast_generic_impl(sycl::queue q,
                           size_t nelems,
                           int nd,
                           py::ssize_t *shape_and_strides,
                           char *src_p,
                           py::ssize_t src_offset,
                           char *dst_p,
                           py::ssize_t dst_offset,
                           const std::vector<sycl::event> &depends,
                           const std::vector<sycl::event> &additional_depends)
{
    sycl::event copy_and_cast_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.depends_on(additional_depends);
        cgh.parallel_for<copy_cast_generic_kernel<srcTy, dstTy>>(
            sycl::range<1>(nelems),
            GenericCopyFunctor<Caster<srcTy, dstTy>>(
                src_p, dst_p, shape_and_strides, nd, src_offset, dst_offset));
    });

    return copy_and_cast_ev;
}

template <typename fnT, typename D, typename S> struct CopyAndCastGenericFactory
{
    fnT get()
    {
        fnT f = copy_and_cast_generic_impl<D, S>;
        return f;
    }
};

// Specialization of copy_and_cast for 1D arrays

typedef sycl::event (*copy_and_cast_1d_fn_ptr_t)(
    sycl::queue,
    size_t,
    const std::array<py::ssize_t, 1>,
    const std::array<py::ssize_t, 1>,
    const std::array<py::ssize_t, 1>,
    char *,
    py::ssize_t,
    char *,
    py::ssize_t,
    const std::vector<sycl::event> &);

typedef sycl::event (*copy_and_cast_2d_fn_ptr_t)(
    sycl::queue,
    size_t,
    const std::array<py::ssize_t, 2>,
    const std::array<py::ssize_t, 2>,
    const std::array<py::ssize_t, 2>,
    char *,
    py::ssize_t,
    char *,
    py::ssize_t,
    const std::vector<sycl::event> &);

template <typename dstTy, typename srcTy, int nd>
sycl::event
copy_and_cast_nd_specialized_impl(sycl::queue q,
                                  size_t nelems,
                                  const std::array<py::ssize_t, nd> shape,
                                  const std::array<py::ssize_t, nd> src_strides,
                                  const std::array<py::ssize_t, nd> dst_strides,
                                  char *src_p,
                                  py::ssize_t src_offset,
                                  char *dst_p,
                                  py::ssize_t dst_offset,
                                  const std::vector<sycl::event> &depends)
{
    sycl::event copy_and_cast_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.parallel_for<copy_cast_spec_kernel<srcTy, dstTy, nd>>(
            sycl::range<1>(nelems),
            NDSpecializedCopyFunctor<nd, Caster<srcTy, dstTy>>(
                src_p, dst_p, shape, src_strides, dst_strides, src_offset,
                dst_offset));
    });

    return copy_and_cast_ev;
}

template <typename fnT, typename D, typename S> struct CopyAndCast1DFactory
{
    fnT get()
    {
        fnT f = copy_and_cast_nd_specialized_impl<D, S, 1>;
        return f;
    }
};

template <typename fnT, typename D, typename S> struct CopyAndCast2DFactory
{
    fnT get()
    {
        fnT f = copy_and_cast_nd_specialized_impl<D, S, 2>;
        return f;
    }
};

// ====================== Copying from host to USM

template <typename srcT, typename dstT, typename AccessorT>
class CasterForAccessor
{
public:
    CasterForAccessor() = default;
    void operator()(AccessorT src,
                    std::ptrdiff_t src_offset,
                    char *dst,
                    std::ptrdiff_t dst_offset) const
    {
        using dpctl::tensor::type_utils::convert_impl;

        dstT *dst_ = reinterpret_cast<dstT *>(dst) + dst_offset;
        *dst_ = convert_impl<dstT, srcT>(src[src_offset]);
    }
};

template <typename CastFnT, typename AccessorT> class GenericCopyFromHostFunctor
{
private:
    AccessorT src_acc_;
    char *dst_ = nullptr;
    py::ssize_t *shape_strides_ = nullptr;
    int nd_ = 0;
    py::ssize_t src_offset0 = 0;
    py::ssize_t dst_offset0 = 0;

public:
    GenericCopyFromHostFunctor(AccessorT src_acc,
                               char *dst_cp,
                               py::ssize_t *shape_strides,
                               int nd,
                               py::ssize_t src_offset,
                               py::ssize_t dst_offset)
        : src_acc_(src_acc), dst_(dst_cp), shape_strides_(shape_strides),
          nd_(nd), src_offset0(src_offset), dst_offset0(dst_offset)
    {
    }

    void operator()(sycl::id<1> wiid) const
    {
        py::ssize_t src_offset(0);
        py::ssize_t dst_offset(0);
        CIndexer_vector<py::ssize_t> indxr(nd_);
        indxr.get_displacement<const py::ssize_t *, const py::ssize_t *>(
            static_cast<py::ssize_t>(wiid.get(0)),
            const_cast<const py::ssize_t *>(shape_strides_), // common shape
            const_cast<const py::ssize_t *>(shape_strides_ +
                                            nd_), // src strides
            const_cast<const py::ssize_t *>(shape_strides_ +
                                            2 * nd_), // dst strides
            src_offset,                               // modified by reference
            dst_offset                                // modified by reference
        );
        CastFnT fn{};
        fn(src_acc_, src_offset0 + src_offset, dst_, dst_offset0 + dst_offset);
    }
};

typedef void (*copy_and_cast_from_host_blocking_fn_ptr_t)(
    sycl::queue,
    size_t,
    int,
    py::ssize_t *,
    const char *,
    py::ssize_t,
    py::ssize_t,
    py::ssize_t,
    char *,
    py::ssize_t,
    const std::vector<sycl::event> &,
    const std::vector<sycl::event> &);

template <typename dstTy, typename srcTy>
void copy_and_cast_from_host_impl(
    sycl::queue q,
    size_t nelems,
    int nd,
    py::ssize_t *shape_and_strides,
    const char *host_src_p,
    py::ssize_t src_offset,
    py::ssize_t src_min_nelem_offset,
    py::ssize_t src_max_nelem_offset,
    char *dst_p,
    py::ssize_t dst_offset,
    const std::vector<sycl::event> &depends,
    const std::vector<sycl::event> &additional_depends)
{
    py::ssize_t nelems_range = src_max_nelem_offset - src_min_nelem_offset + 1;
    sycl::buffer<srcTy, 1> npy_buf(
        reinterpret_cast<const srcTy *>(host_src_p) + src_min_nelem_offset,
        sycl::range<1>(nelems_range), {sycl::property::buffer::use_host_ptr{}});

    sycl::event copy_and_cast_from_host_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.depends_on(additional_depends);

        sycl::accessor npy_acc(npy_buf, cgh, sycl::read_only);

        cgh.parallel_for<copy_cast_from_host_kernel<srcTy, dstTy>>(
            sycl::range<1>(nelems),
            GenericCopyFromHostFunctor<
                CasterForAccessor<srcTy, dstTy, decltype(npy_acc)>,
                decltype(npy_acc)>(npy_acc, dst_p, shape_and_strides, nd,
                                   src_offset - src_min_nelem_offset,
                                   dst_offset));
    });

    copy_and_cast_from_host_ev.wait_and_throw();

    return;
}

template <typename fnT, typename D, typename S>
struct CopyAndCastFromHostFactory
{
    fnT get()
    {
        fnT f = copy_and_cast_from_host_impl<D, S>;
        return f;
    }
};

// =============== Copying for reshape ================== //

template <typename Ty> class GenericCopyForReshapeFunctor
{
private:
    py::ssize_t offset = 0;
    py::ssize_t size = 1;
    int src_nd = -1;
    int dst_nd = -1;
    // USM array of size 2*(src_nd + dst_nd)
    //   [ src_shape; src_strides; dst_shape; dst_strides ]
    const py::ssize_t *src_dst_shapes_and_strides = nullptr;
    Ty *src_p = nullptr;
    Ty *dst_p = nullptr;

public:
    GenericCopyForReshapeFunctor(py::ssize_t shift,
                                 py::ssize_t nelems,
                                 int src_ndim,
                                 int dst_ndim,
                                 const py::ssize_t *packed_shapes_and_strides,
                                 char *src_ptr,
                                 char *dst_ptr)
        : offset(shift), size(nelems), src_nd(src_ndim), dst_nd(dst_ndim),
          src_dst_shapes_and_strides(packed_shapes_and_strides),
          src_p(reinterpret_cast<Ty *>(src_ptr)),
          dst_p(reinterpret_cast<Ty *>(dst_ptr))
    {
    }

    void operator()(sycl::id<1> wiid) const
    {
        py::ssize_t this_src_offset(0);
        CIndexer_vector<py::ssize_t> src_indxr(src_nd);

        src_indxr.get_displacement<const py::ssize_t *, const py::ssize_t *>(
            static_cast<py::ssize_t>(wiid.get(0)),
            const_cast<const py::ssize_t *>(
                src_dst_shapes_and_strides), // src shape
            const_cast<const py::ssize_t *>(src_dst_shapes_and_strides +
                                            src_nd), // src strides
            this_src_offset                          // modified by reference
        );
        const Ty *in = src_p + this_src_offset;

        py::ssize_t this_dst_offset(0);
        CIndexer_vector<py::ssize_t> dst_indxr(dst_nd);
        py::ssize_t shifted_wiid =
            (static_cast<py::ssize_t>(wiid.get(0)) + offset) % size;
        shifted_wiid = (shifted_wiid >= 0) ? shifted_wiid : shifted_wiid + size;
        dst_indxr.get_displacement<const py::ssize_t *, const py::ssize_t *>(
            shifted_wiid,
            const_cast<const py::ssize_t *>(src_dst_shapes_and_strides +
                                            2 * src_nd), // dst shape
            const_cast<const py::ssize_t *>(src_dst_shapes_and_strides +
                                            2 * src_nd + dst_nd), // dst strides
            this_dst_offset // modified by reference
        );

        Ty *out = dst_p + this_dst_offset;
        *out = *in;
    }
};

// define function type
typedef sycl::event (*copy_for_reshape_fn_ptr_t)(
    sycl::queue,
    py::ssize_t, // shift
    size_t,      // num_elements
    int,
    int,           // src_nd, dst_nd
    py::ssize_t *, // packed shapes and strides
    char *,        // src_data_ptr
    char *,        // dst_data_ptr
    const std::vector<sycl::event> &);

template <typename Ty>
sycl::event
copy_for_reshape_generic_impl(sycl::queue q,
                              py::ssize_t shift,
                              size_t nelems,
                              int src_nd,
                              int dst_nd,
                              py::ssize_t *packed_shapes_and_strides,
                              char *src_p,
                              char *dst_p,
                              const std::vector<sycl::event> &depends)
{
    sycl::event copy_for_reshape_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.parallel_for<copy_for_reshape_generic_kernel<Ty>>(
            sycl::range<1>(nelems),
            GenericCopyForReshapeFunctor<Ty>(shift, nelems, src_nd, dst_nd,
                                             packed_shapes_and_strides, src_p,
                                             dst_p));
    });

    return copy_for_reshape_ev;
}

template <typename fnT, typename Ty> struct CopyForReshapeGenericFactory
{
    fnT get()
    {
        fnT f = copy_for_reshape_generic_impl<Ty>;
        return f;
    }
};

} // namespace copy_and_cast
} // namespace kernels
} // namespace tensor
} // namespace dpctl
