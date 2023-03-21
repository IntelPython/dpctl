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
#include <CL/sycl.hpp>
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
namespace copy_and_cast
{

namespace py = pybind11;
using namespace dpctl::tensor::offset_utils;

template <typename srcT, typename dstT, typename IndexerT>
class copy_cast_generic_kernel;
template <typename srcT, typename dstT, typename IndexerT>
class copy_cast_from_host_kernel;
// template <typename srcT, typename dstT, typename IndexerT>
// class copy_cast_spec_kernel;
template <typename Ty, typename SrcIndexerT, typename DstIndexerT>
class copy_for_reshape_generic_kernel;

template <typename srcT, typename dstT> class Caster
{
public:
    Caster() = default;
    void operator()(const char *src,
                    std::ptrdiff_t src_offset,
                    char *dst,
                    std::ptrdiff_t dst_offset) const
    {
        using dpctl::tensor::type_utils::convert_impl;

        const srcT *src_ = reinterpret_cast<const srcT *>(src) + src_offset;
        dstT *dst_ = reinterpret_cast<dstT *>(dst) + dst_offset;
        *dst_ = convert_impl<dstT, srcT>(*src_);
    }
};

template <typename CastFnT, typename IndexerT> class GenericCopyFunctor
{
private:
    const char *src_ = nullptr;
    char *dst_ = nullptr;
    IndexerT indexer_;

public:
    GenericCopyFunctor(const char *src_cp, char *dst_cp, IndexerT indexer)
        : src_(src_cp), dst_(dst_cp), indexer_(indexer)
    {
    }

    void operator()(sycl::id<1> wiid) const
    {
        auto offsets = indexer_(static_cast<py::ssize_t>(wiid.get(0)));
        py::ssize_t src_offset = offsets.get_first_offset();
        py::ssize_t dst_offset = offsets.get_second_offset();

        CastFnT fn{};
        fn(src_, src_offset, dst_, dst_offset);
    }
};

/*!
  @defgroup CopyAndCastKernels
 */

/*!
 * @brief Function pointer type for generic array cast and copying function.
 */
typedef sycl::event (*copy_and_cast_generic_fn_ptr_t)(
    sycl::queue,
    size_t,
    int,
    const py::ssize_t *,
    const char *,
    py::ssize_t,
    char *,
    py::ssize_t,
    const std::vector<sycl::event> &,
    const std::vector<sycl::event> &);

/*!
 * @brief Generic function to copy `nelems` elements from `src` usm_ndarray to
 `dst` usm_ndarray while casting from `srcTy` to `dstTy`.

   Both arrays have array dimensionality specied via argument `nd`. The
 `shape_and_strides` is kernel accessible USM array of length `3*nd`, where the
 first `nd` elements encode common shape, second `nd` elements contain strides
 of `src` array, and the trailing `nd` elements contain strides of `dst` array.
 `src_p` and `dst_p` represent pointers into respective arrays, but the start of
 iteration begins at offset of `src_offset` elements for `src` array and at
 offset `dst_offset` elements for `dst` array. Kernel is submitted to sycl queue
 `q` with events `depends` and `additional_depends` as dependencies.

   @param  q       Sycl queue to which the kernel is submitted.
   @param  nelems  Number of elements to cast and copy.
   @param  nd      Array dimensionality, i.e. number of indices needed to
 identify an element of each array.
   @param  shape_and_strides  Kernel accessible USM pointer to packed shape and
 strides.
   @param  src_p   Kernel accessible USM pointer for the source array
   @param  src_offset  Offset to the beginning of iteration in number of
 elements of source array from `src_p`.
   @param  dst_p   Kernel accessible USM pointer for the destination array
   @param  dst_offset  Offset to the beginning of iteration in number of
 elements of destination array from `dst_p`.
   @param  depends  List of events to wait for before starting computations, if
 any.
   @param  additional_depends Additional list of events to wait for before
 starting computations, if any.

   @return  Event to wait on to ensure that computation completes.
   @ingroup CopyAndCastKernels
 */
template <typename dstTy, typename srcTy>
sycl::event
copy_and_cast_generic_impl(sycl::queue q,
                           size_t nelems,
                           int nd,
                           const py::ssize_t *shape_and_strides,
                           const char *src_p,
                           py::ssize_t src_offset,
                           char *dst_p,
                           py::ssize_t dst_offset,
                           const std::vector<sycl::event> &depends,
                           const std::vector<sycl::event> &additional_depends)
{
    dpctl::tensor::type_utils::validate_type_for_device<dstTy>(q);
    dpctl::tensor::type_utils::validate_type_for_device<srcTy>(q);

    sycl::event copy_and_cast_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.depends_on(additional_depends);

        TwoOffsets_StridedIndexer indexer{nd, src_offset, dst_offset,
                                          shape_and_strides};

        cgh.parallel_for<class copy_cast_generic_kernel<
            srcTy, dstTy, TwoOffsets_StridedIndexer>>(
            sycl::range<1>(nelems),
            GenericCopyFunctor<Caster<srcTy, dstTy>, TwoOffsets_StridedIndexer>(
                src_p, dst_p, indexer));
    });

    return copy_and_cast_ev;
}

/*!
 * @brief Factory to get generic function pointer of type `fnT` for given source
 * data type `S` and destination data type `D`.
 * @ingroup CopyAndCastKernels
 */
template <typename fnT, typename D, typename S> struct CopyAndCastGenericFactory
{
    fnT get()
    {
        fnT f = copy_and_cast_generic_impl<D, S>;
        return f;
    }
};

// Specialization of copy_and_cast for 1D arrays

/*!
 * @brief Factory to get function pointer for casting and copying 1D arrays.
 * @ingroup CopyAndCastKernels
 */
typedef sycl::event (*copy_and_cast_1d_fn_ptr_t)(
    sycl::queue,
    size_t,
    const std::array<py::ssize_t, 1>,
    const std::array<py::ssize_t, 1>,
    const std::array<py::ssize_t, 1>,
    const char *,
    py::ssize_t,
    char *,
    py::ssize_t,
    const std::vector<sycl::event> &);

/*!
 * @brief Factory to get function pointer for casting and copying 2D arrays.
 * @ingroup CopyAndCastKernels
 */
typedef sycl::event (*copy_and_cast_2d_fn_ptr_t)(
    sycl::queue,
    size_t,
    const std::array<py::ssize_t, 2>,
    const std::array<py::ssize_t, 2>,
    const std::array<py::ssize_t, 2>,
    const char *,
    py::ssize_t,
    char *,
    py::ssize_t,
    const std::vector<sycl::event> &);

/*!
 * @brief Specialized for given array dimension function to copy `nelems`
 elements from `src` usm_ndarray to `dst` usm_ndarray while casting from `srcTy`
 to `dstTy`.

   Both arrays have array dimensionality known at compile time and specified in
 template parameters `nd`. Arrays' shape and strides are provided as
 `std::array`. `src_p` and `dst_p` represent pointers into respective arrays,
 but the start of iteration begins at offset of `src_offset` elements for `src`
 array and at offset `dst_offset` elements for `dst` array. Kernel is submitted
 to sycl queue `q` with events `depends` as dependencies.

   @param q  The queue where the routine should be executed.
   @param nelems  Number of elements to cast and copy.
   @param shape   Common shape of the arrays.
   @param src_strides Strides of the source array.
   @param dst_strides Strides of the destination array.
   @param src_p  Kernel accessible USM pointer for the source array
   @param src_offset  Offset to the beginning of iteration in number of elements
 of the source array from `src_p`.
   @param dst_p  Kernel accessible USM pointer for the destination array
   @param dst_offset  Offset to the beginning of iteration in number of elements
 of the destination array from `src_p`.
   @param depends  List of events to wait for before starting computations, if
 any.

   @return  Event to wait on to ensure that computation completes.
 * @ingroup CopyAndCastKernels
 */
template <typename dstTy, typename srcTy, int nd>
sycl::event
copy_and_cast_nd_specialized_impl(sycl::queue q,
                                  size_t nelems,
                                  const std::array<py::ssize_t, nd> shape,
                                  const std::array<py::ssize_t, nd> src_strides,
                                  const std::array<py::ssize_t, nd> dst_strides,
                                  const char *src_p,
                                  py::ssize_t src_offset,
                                  char *dst_p,
                                  py::ssize_t dst_offset,
                                  const std::vector<sycl::event> &depends)
{
    dpctl::tensor::type_utils::validate_type_for_device<dstTy>(q);
    dpctl::tensor::type_utils::validate_type_for_device<srcTy>(q);

    sycl::event copy_and_cast_ev = q.submit([&](sycl::handler &cgh) {
        TwoOffsets_StridedIndexerArray<nd> indexer{
            shape, src_strides, dst_strides, src_offset, dst_offset};

        cgh.depends_on(depends);
        cgh.parallel_for<class copy_cast_generic_kernel<
            srcTy, dstTy, TwoOffsets_StridedIndexerArray<nd>>>(
            sycl::range<1>(nelems),
            GenericCopyFunctor<Caster<srcTy, dstTy>,
                               TwoOffsets_StridedIndexerArray<nd>>(src_p, dst_p,
                                                                   indexer));
    });

    return copy_and_cast_ev;
}

/*!
 * @brief Factory to get 1D-specialized function pointer of type `fnT` for given
 * source data type `S` and destination data type `D`.
 * @ingroup CopyAndCastKernels
 */
template <typename fnT, typename D, typename S> struct CopyAndCast1DFactory
{
    fnT get()
    {
        fnT f = copy_and_cast_nd_specialized_impl<D, S, 1>;
        return f;
    }
};

/*!
 * @brief Factory to get 2D-specialized function pointer of type `fnT` for given
 * source data type `S` and destination data type `D`.
 * @ingroup CopyAndCastKernels
 */
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

template <typename CastFnT, typename AccessorT, typename IndexerT>
class GenericCopyFromHostFunctor
{
private:
    AccessorT src_acc_;
    char *dst_ = nullptr;
    IndexerT indexer_;

public:
    GenericCopyFromHostFunctor(AccessorT src_acc,
                               char *dst_cp,
                               IndexerT indexer)
        : src_acc_(src_acc), dst_(dst_cp), indexer_(indexer)
    {
    }

    void operator()(sycl::id<1> wiid) const
    {
        auto offsets = indexer_(static_cast<py::ssize_t>(wiid.get(0)));
        py::ssize_t src_offset = offsets.get_first_offset();
        py::ssize_t dst_offset = offsets.get_second_offset();
        CastFnT fn{};
        fn(src_acc_, src_offset, dst_, dst_offset);
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

/*!
 * @brief Function to copy from NumPy's ndarray with elements of type `srcTy`
 * into usm_ndarray with elements of type `srcTy`.
 *
 * Function to cast and copy elements from numpy.ndarray specified by typeless
 * `host_src_p` and the `src_offset` given in the number of array elements.
 * Arrays' metadata are given in packed USM vector of length `3*nd` whose first
 * `nd` elements contain arrays' shape, next `nd` elements specify source
 * strides in elements (not bytes), and trailing `nd` elements specify
 * destination array strides. Kernel dependencies are given by two vectors of
 * events: `depends` and `additional_depends`. The function execution is
 * complete at the return.
 *
 * @param q  The queue where the routine should be executed.
 * @param nelems Number of elements to cast and copy.
 * @param nd The dimensionality of arrays
 * @param shape_and_strides  Kernel accessible USM pointer to packed shape and
 * strides.
 * @param host_src_p  Host (not USM allocated) pointer associated with the
 * source array.
 * @param src_offset  Offset to the beginning of iteration in number of elements
 * of the source array from `host_src_p`.
 * @param src_min_nelem_offset  Smallest value of offset relative to
 * `host_src_p` in number of elements attained while iterating over elements of
 * the source array.
 * @param src_max_nelem_offset  Largest value of offset relative to `host_src_p`
 * in number of elements attained while iterating over elements of the source
 * array.
 * @param dst_p  USM pointer associated with the destination array.
 * @param dst_offset  Offset to the beginning of iteration in number of elements
 * of the destination array from `dst_p`.
 * @param depends  List of events to wait for before starting computations, if
 * any.
 * @param additional_depends List of additional events to wait for before
 * starting computations, if any.
 *
 * @ingroup CopyAndCastKernels
 */
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

    dpctl::tensor::type_utils::validate_type_for_device<dstTy>(q);
    dpctl::tensor::type_utils::validate_type_for_device<srcTy>(q);

    sycl::buffer<srcTy, 1> npy_buf(
        reinterpret_cast<const srcTy *>(host_src_p) + src_min_nelem_offset,
        sycl::range<1>(nelems_range), {sycl::property::buffer::use_host_ptr{}});

    sycl::event copy_and_cast_from_host_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.depends_on(additional_depends);

        sycl::accessor npy_acc(npy_buf, cgh, sycl::read_only);

        TwoOffsets_StridedIndexer indexer{
            nd, src_offset - src_min_nelem_offset, dst_offset,
            const_cast<const py::ssize_t *>(shape_and_strides)};

        cgh.parallel_for<copy_cast_from_host_kernel<srcTy, dstTy,
                                                    TwoOffsets_StridedIndexer>>(
            sycl::range<1>(nelems),
            GenericCopyFromHostFunctor<
                CasterForAccessor<srcTy, dstTy, decltype(npy_acc)>,
                decltype(npy_acc), TwoOffsets_StridedIndexer>(npy_acc, dst_p,
                                                              indexer));
    });

    // perform explicit synchronization. Implicit synchronization would be
    // performed by sycl::buffer destructor.
    copy_and_cast_from_host_ev.wait_and_throw();

    return;
}

/*!
 * @brief Factory to get function pointer of type `fnT` for given NumPy array
 * source data type `S` and destination data type `D`.
 * @defgroup CopyAndCastKernels
 */
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

template <typename Ty, typename SrcIndexerT, typename DstIndexerT>
class GenericCopyForReshapeFunctor
{
private:
    py::ssize_t offset = 0;
    py::ssize_t size = 1;
    // USM array of size 2*(src_nd + dst_nd)
    //   [ src_shape; src_strides; dst_shape; dst_strides ]
    Ty *src_p = nullptr;
    Ty *dst_p = nullptr;
    SrcIndexerT src_indexer_;
    DstIndexerT dst_indexer_;

public:
    GenericCopyForReshapeFunctor(py::ssize_t shift,
                                 py::ssize_t nelems,
                                 char *src_ptr,
                                 char *dst_ptr,
                                 SrcIndexerT src_indexer,
                                 DstIndexerT dst_indexer)
        : offset(shift), size(nelems), src_p(reinterpret_cast<Ty *>(src_ptr)),
          dst_p(reinterpret_cast<Ty *>(dst_ptr)), src_indexer_(src_indexer),
          dst_indexer_(dst_indexer)
    {
    }

    void operator()(sycl::id<1> wiid) const
    {
        py::ssize_t this_src_offset = src_indexer_(wiid.get(0));
        const Ty *in = src_p + this_src_offset;

        py::ssize_t shifted_wiid =
            (static_cast<py::ssize_t>(wiid.get(0)) + offset) % size;
        shifted_wiid = (shifted_wiid >= 0) ? shifted_wiid : shifted_wiid + size;

        py::ssize_t this_dst_offset = dst_indexer_(shifted_wiid);

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

/*!
 * @brief Function to copy content of array while reshaping.
 *
 * Submits a kernel to perform a copy `dst[unravel_index((i + shift) % nelems ,
 * dst.shape)] = src[unravel_undex(i, src.shape)]`.
 *
 * @param  q      The execution queue where kernel is submitted.
 * @param  shift  The shift in flat indexing.
 * @param  nelems The number of elements to copy
 * @param  src_nd Array dimension of the source array
 * @param  dst_nd Array dimension of the destination array
 * @param  packed_shapes_and_strides Kernel accessible USM array of size
 * `2*src_nd + 2*dst_nd` with contant `[src_shape, src_strides, dst_shape,
 * dst_strides]`.
 * @param  src_p  Typeless USM pointer to the buffer of the source array
 * @param  dst_p  Typeless USM pointer to the buffer of the destination array
 * @param  depends  List of events to wait for before starting computations, if
 * any.
 *
 * @return Event to wait on to ensure that computation completes.
 * @ingroup CopyAndCastKernels
 */
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
    dpctl::tensor::type_utils::validate_type_for_device<Ty>(q);

    sycl::event copy_for_reshape_ev = q.submit([&](sycl::handler &cgh) {
        StridedIndexer src_indexer{
            src_nd, 0,
            const_cast<const py::ssize_t *>(packed_shapes_and_strides)};
        StridedIndexer dst_indexer{
            dst_nd, 0,
            const_cast<const py::ssize_t *>(packed_shapes_and_strides +
                                            (2 * src_nd))};
        cgh.depends_on(depends);
        cgh.parallel_for<copy_for_reshape_generic_kernel<Ty, StridedIndexer,
                                                         StridedIndexer>>(
            sycl::range<1>(nelems),
            GenericCopyForReshapeFunctor<Ty, StridedIndexer, StridedIndexer>(
                shift, nelems, src_p, dst_p, src_indexer, dst_indexer));
    });

    return copy_for_reshape_ev;
}

/*!
 * @brief Factory to get function pointer of type `fnT` for given array data
 * type `Ty`.
 * @ingroup CopyAndCastKernels
 */
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
