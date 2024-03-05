//=== copy_and_cast.hpp - Implementation of copy-and-cast kernels *-C++-*/===//
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
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines kernels for tensor copying and value casting.
//===----------------------------------------------------------------------===//

#pragma once
#include <complex>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "dpctl_tensor_types.hpp"
#include "kernels/alignment.hpp"
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

using namespace dpctl::tensor::offset_utils;

using dpctl::tensor::kernels::alignment_utils::
    disabled_sg_loadstore_wrapper_krn;
using dpctl::tensor::kernels::alignment_utils::is_aligned;
using dpctl::tensor::kernels::alignment_utils::required_alignment;

template <typename srcT, typename dstT, typename IndexerT>
class copy_cast_generic_kernel;

template <typename srcT,
          typename dstT,
          unsigned int vec_sz,
          unsigned int n_vecs>
class copy_cast_contig_kernel;

template <typename srcT, typename dstT, typename IndexerT>
class copy_cast_from_host_kernel;

template <typename srcTy, typename dstTy> class Caster
{
public:
    Caster() = default;
    dstTy operator()(const srcTy &src) const
    {
        using dpctl::tensor::type_utils::convert_impl;
        return convert_impl<dstTy, srcTy>(src);
    }
};

template <typename srcT, typename dstT, typename CastFnT, typename IndexerT>
class GenericCopyFunctor
{
private:
    const srcT *src_ = nullptr;
    dstT *dst_ = nullptr;
    const IndexerT indexer_;

public:
    GenericCopyFunctor(const srcT *src_p, dstT *dst_p, const IndexerT &indexer)
        : src_(src_p), dst_(dst_p), indexer_(indexer)
    {
    }

    void operator()(sycl::id<1> wiid) const
    {
        const auto &offsets = indexer_(static_cast<ssize_t>(wiid.get(0)));
        const ssize_t &src_offset = offsets.get_first_offset();
        const ssize_t &dst_offset = offsets.get_second_offset();

        CastFnT fn{};
        dst_[dst_offset] = fn(src_[src_offset]);
    }
};

/*!
  @defgroup CopyAndCastKernels
 */

/*!
 * @brief Function pointer type for generic array cast and copying function.
 */
typedef sycl::event (*copy_and_cast_generic_fn_ptr_t)(
    sycl::queue &,
    size_t,
    int,
    const ssize_t *,
    const char *,
    ssize_t,
    char *,
    ssize_t,
    const std::vector<sycl::event> &,
    const std::vector<sycl::event> &);

/*!
 * @brief Generic function to copy `nelems` elements from `src` usm_ndarray to
 `dst` usm_ndarray while casting from `srcTy` to `dstTy`.

   Both arrays have array dimensionality specified via argument `nd`. The
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
copy_and_cast_generic_impl(sycl::queue &q,
                           size_t nelems,
                           int nd,
                           const ssize_t *shape_and_strides,
                           const char *src_p,
                           ssize_t src_offset,
                           char *dst_p,
                           ssize_t dst_offset,
                           const std::vector<sycl::event> &depends,
                           const std::vector<sycl::event> &additional_depends)
{
    dpctl::tensor::type_utils::validate_type_for_device<dstTy>(q);
    dpctl::tensor::type_utils::validate_type_for_device<srcTy>(q);

    sycl::event copy_and_cast_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.depends_on(additional_depends);

        const TwoOffsets_StridedIndexer indexer{nd, src_offset, dst_offset,
                                                shape_and_strides};
        const srcTy *src_tp = reinterpret_cast<const srcTy *>(src_p);
        dstTy *dst_tp = reinterpret_cast<dstTy *>(dst_p);

        cgh.parallel_for<class copy_cast_generic_kernel<
            srcTy, dstTy, TwoOffsets_StridedIndexer>>(
            sycl::range<1>(nelems),
            GenericCopyFunctor<srcTy, dstTy, Caster<srcTy, dstTy>,
                               TwoOffsets_StridedIndexer>(src_tp, dst_tp,
                                                          indexer));
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

// Specialization of copy_and_cast for contiguous arrays

template <typename srcT,
          typename dstT,
          typename CastFnT,
          int vec_sz = 4,
          int n_vecs = 2,
          bool enable_sg_loadstore = true>
class ContigCopyFunctor
{
private:
    const size_t nelems;
    const srcT *src_p = nullptr;
    dstT *dst_p = nullptr;

public:
    ContigCopyFunctor(const size_t nelems_, const srcT *src_p_, dstT *dst_p_)
        : nelems(nelems_), src_p(src_p_), dst_p(dst_p_)
    {
    }

    void operator()(sycl::nd_item<1> ndit) const
    {
        CastFnT fn{};

        using dpctl::tensor::type_utils::is_complex;
        if constexpr (!enable_sg_loadstore || is_complex<srcT>::value ||
                      is_complex<dstT>::value)
        {
            std::uint8_t sgSize = ndit.get_sub_group().get_local_range()[0];
            size_t base = ndit.get_global_linear_id();

            base = (base / sgSize) * sgSize * n_vecs * vec_sz + (base % sgSize);
            for (size_t offset = base;
                 offset < std::min(nelems, base + sgSize * (n_vecs * vec_sz));
                 offset += sgSize)
            {
                dst_p[offset] = fn(src_p[offset]);
            }
        }
        else {
            auto sg = ndit.get_sub_group();
            std::uint8_t sgSize = sg.get_local_range()[0];
            std::uint8_t max_sgSize = sg.get_max_local_range()[0];
            size_t base = n_vecs * vec_sz *
                          (ndit.get_group(0) * ndit.get_local_range(0) +
                           sg.get_group_id()[0] * max_sgSize);

            if (base + n_vecs * vec_sz * sgSize < nelems &&
                sgSize == max_sgSize) {
                sycl::vec<srcT, vec_sz> src_vec;
                sycl::vec<dstT, vec_sz> dst_vec;

#pragma unroll
                for (std::uint8_t it = 0; it < n_vecs * vec_sz; it += vec_sz) {
                    auto src_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(
                        &src_p[base + it * sgSize]);
                    auto dst_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(
                        &dst_p[base + it * sgSize]);

                    src_vec = sg.load<vec_sz>(src_multi_ptr);
#pragma unroll
                    for (std::uint8_t k = 0; k < vec_sz; k++) {
                        dst_vec[k] = fn(src_vec[k]);
                    }
                    sg.store<vec_sz>(dst_multi_ptr, dst_vec);
                }
            }
            else {
                for (size_t k = base + sg.get_local_id()[0]; k < nelems;
                     k += sgSize) {
                    dst_p[k] = fn(src_p[k]);
                }
            }
        }
    }
};

/*!
 * @brief Function pointer type for contiguous array cast and copy function.
 */
typedef sycl::event (*copy_and_cast_contig_fn_ptr_t)(
    sycl::queue &,
    size_t,
    const char *,
    char *,
    const std::vector<sycl::event> &);

/*!
 * @brief Function to copy `nelems` elements from contiguous `src` usm_ndarray
 to contiguous `dst` usm_ndarray while casting from `srcTy` to `dstTy`.

   Both arrays have the same number of elements `nelems`.
 `src_cp` and `dst_cp` represent char pointers to the start of respective
 arrays. Kernel is submitted to sycl queue `q` with events `depends` as
 dependencies.

   @param  q       Sycl queue to which the kernel is submitted.
   @param  nelems  Number of elements to cast and copy.
   @param  src_p   Kernel accessible USM pointer for the source array
   @param  dst_p   Kernel accessible USM pointer for the destination array
   @param  depends  List of events to wait for before starting computations, if
 any.

   @return  Event to wait on to ensure that computation completes.
   @ingroup CopyAndCastKernels
 */
template <typename dstTy, typename srcTy>
sycl::event copy_and_cast_contig_impl(sycl::queue &q,
                                      size_t nelems,
                                      const char *src_cp,
                                      char *dst_cp,
                                      const std::vector<sycl::event> &depends)
{
    dpctl::tensor::type_utils::validate_type_for_device<dstTy>(q);
    dpctl::tensor::type_utils::validate_type_for_device<srcTy>(q);

    sycl::event copy_and_cast_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        const srcTy *src_tp = reinterpret_cast<const srcTy *>(src_cp);
        dstTy *dst_tp = reinterpret_cast<dstTy *>(dst_cp);

        size_t lws = 64;
        constexpr unsigned int vec_sz = 4;
        constexpr unsigned int n_vecs = 2;
        const size_t n_groups =
            ((nelems + lws * n_vecs * vec_sz - 1) / (lws * n_vecs * vec_sz));
        const auto gws_range = sycl::range<1>(n_groups * lws);
        const auto lws_range = sycl::range<1>(lws);

        if (is_aligned<required_alignment>(src_cp) &&
            is_aligned<required_alignment>(dst_cp))
        {
            constexpr bool enable_sg_loadstore = true;
            using KernelName =
                copy_cast_contig_kernel<srcTy, dstTy, vec_sz, n_vecs>;

            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(gws_range, lws_range),
                ContigCopyFunctor<srcTy, dstTy, Caster<srcTy, dstTy>, vec_sz,
                                  n_vecs, enable_sg_loadstore>(nelems, src_tp,
                                                               dst_tp));
        }
        else {
            constexpr bool disable_sg_loadstore = false;
            using InnerKernelName =
                copy_cast_contig_kernel<srcTy, dstTy, vec_sz, n_vecs>;
            using KernelName =
                disabled_sg_loadstore_wrapper_krn<InnerKernelName>;

            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(gws_range, lws_range),
                ContigCopyFunctor<srcTy, dstTy, Caster<srcTy, dstTy>, vec_sz,
                                  n_vecs, disable_sg_loadstore>(nelems, src_tp,
                                                                dst_tp));
        }
    });

    return copy_and_cast_ev;
}

/*!
 * @brief Factory to get specialized function pointer for casting and copying
 * contiguous arrays.
 * @ingroup CopyAndCastKernels
 */
template <typename fnT, typename D, typename S> struct CopyAndCastContigFactory
{
    fnT get()
    {
        fnT f = copy_and_cast_contig_impl<D, S>;
        return f;
    }
};

// Specialization of copy_and_cast for 1D arrays

/*!
 * @brief Factory to get function pointer for casting and copying 1D arrays.
 * @ingroup CopyAndCastKernels
 */
typedef sycl::event (*copy_and_cast_1d_fn_ptr_t)(
    sycl::queue &,
    size_t,
    const std::array<ssize_t, 1>,
    const std::array<ssize_t, 1>,
    const std::array<ssize_t, 1>,
    const char *,
    ssize_t,
    char *,
    ssize_t,
    const std::vector<sycl::event> &);

/*!
 * @brief Factory to get function pointer for casting and copying 2D arrays.
 * @ingroup CopyAndCastKernels
 */
typedef sycl::event (*copy_and_cast_2d_fn_ptr_t)(
    sycl::queue &,
    size_t,
    const std::array<ssize_t, 2>,
    const std::array<ssize_t, 2>,
    const std::array<ssize_t, 2>,
    const char *,
    ssize_t,
    char *,
    ssize_t,
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
copy_and_cast_nd_specialized_impl(sycl::queue &q,
                                  size_t nelems,
                                  const std::array<ssize_t, nd> shape,
                                  const std::array<ssize_t, nd> src_strides,
                                  const std::array<ssize_t, nd> dst_strides,
                                  const char *src_p,
                                  ssize_t src_offset,
                                  char *dst_p,
                                  ssize_t dst_offset,
                                  const std::vector<sycl::event> &depends)
{
    dpctl::tensor::type_utils::validate_type_for_device<dstTy>(q);
    dpctl::tensor::type_utils::validate_type_for_device<srcTy>(q);

    sycl::event copy_and_cast_ev = q.submit([&](sycl::handler &cgh) {
        using IndexerT = TwoOffsets_FixedDimStridedIndexer<nd>;
        const IndexerT indexer{shape, src_strides, dst_strides, src_offset,
                               dst_offset};
        const srcTy *src_tp = reinterpret_cast<const srcTy *>(src_p);
        dstTy *dst_tp = reinterpret_cast<dstTy *>(dst_p);

        cgh.depends_on(depends);
        cgh.parallel_for<
            class copy_cast_generic_kernel<srcTy, dstTy, IndexerT>>(
            sycl::range<1>(nelems),
            GenericCopyFunctor<srcTy, dstTy, Caster<srcTy, dstTy>, IndexerT>(
                src_tp, dst_tp, indexer));
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

template <typename AccessorT,
          typename dstTy,
          typename CastFnT,
          typename IndexerT>
class GenericCopyFromHostFunctor
{
private:
    const AccessorT src_acc_;
    dstTy *dst_ = nullptr;
    const IndexerT indexer_;

public:
    GenericCopyFromHostFunctor(const AccessorT &src_acc,
                               dstTy *dst_p,
                               const IndexerT &indexer)
        : src_acc_(src_acc), dst_(dst_p), indexer_(indexer)
    {
    }

    void operator()(sycl::id<1> wiid) const
    {
        const auto &offsets = indexer_(static_cast<ssize_t>(wiid.get(0)));
        const ssize_t &src_offset = offsets.get_first_offset();
        const ssize_t &dst_offset = offsets.get_second_offset();

        CastFnT fn{};
        dst_[dst_offset] = fn(src_acc_[src_offset]);
    }
};

typedef void (*copy_and_cast_from_host_blocking_fn_ptr_t)(
    sycl::queue &,
    size_t,
    int,
    ssize_t *,
    const char *,
    ssize_t,
    ssize_t,
    ssize_t,
    char *,
    ssize_t,
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
    sycl::queue &q,
    size_t nelems,
    int nd,
    ssize_t *shape_and_strides,
    const char *host_src_p,
    ssize_t src_offset,
    ssize_t src_min_nelem_offset,
    ssize_t src_max_nelem_offset,
    char *dst_p,
    ssize_t dst_offset,
    const std::vector<sycl::event> &depends,
    const std::vector<sycl::event> &additional_depends)
{
    ssize_t nelems_range = src_max_nelem_offset - src_min_nelem_offset + 1;

    dpctl::tensor::type_utils::validate_type_for_device<dstTy>(q);
    dpctl::tensor::type_utils::validate_type_for_device<srcTy>(q);

    sycl::buffer<srcTy, 1> npy_buf(
        reinterpret_cast<const srcTy *>(host_src_p) + src_min_nelem_offset,
        sycl::range<1>(nelems_range), {sycl::property::buffer::use_host_ptr{}});

    sycl::event copy_and_cast_from_host_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.depends_on(additional_depends);

        sycl::accessor npy_acc(npy_buf, cgh, sycl::read_only);

        const TwoOffsets_StridedIndexer indexer{
            nd, src_offset - src_min_nelem_offset, dst_offset,
            const_cast<const ssize_t *>(shape_and_strides)};

        dstTy *dst_tp = reinterpret_cast<dstTy *>(dst_p);

        cgh.parallel_for<copy_cast_from_host_kernel<srcTy, dstTy,
                                                    TwoOffsets_StridedIndexer>>(
            sycl::range<1>(nelems),
            GenericCopyFromHostFunctor<decltype(npy_acc), dstTy,
                                       Caster<srcTy, dstTy>,
                                       TwoOffsets_StridedIndexer>(
                npy_acc, dst_tp, indexer));
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
class copy_for_reshape_generic_kernel;

template <typename Ty, typename SrcIndexerT, typename DstIndexerT>
class GenericCopyForReshapeFunctor
{
private:
    const Ty *src_p = nullptr;
    Ty *dst_p = nullptr;
    const SrcIndexerT src_indexer_;
    const DstIndexerT dst_indexer_;

public:
    GenericCopyForReshapeFunctor(const char *src_ptr,
                                 char *dst_ptr,
                                 const SrcIndexerT &src_indexer,
                                 const DstIndexerT &dst_indexer)
        : src_p(reinterpret_cast<const Ty *>(src_ptr)),
          dst_p(reinterpret_cast<Ty *>(dst_ptr)), src_indexer_(src_indexer),
          dst_indexer_(dst_indexer)
    {
    }

    void operator()(sycl::id<1> wiid) const
    {
        const ssize_t src_offset = src_indexer_(wiid.get(0));
        const ssize_t dst_offset = dst_indexer_(wiid.get(0));

        dst_p[dst_offset] = src_p[src_offset];
    }
};

// define function type
typedef sycl::event (*copy_for_reshape_fn_ptr_t)(
    sycl::queue &,
    size_t,       // num_elements
    int,          // src_nd
    int,          // dst_nd
    ssize_t *,    // packed shapes and strides
    const char *, // src_data_ptr
    char *,       // dst_data_ptr
    const std::vector<sycl::event> &);

/*!
 * @brief Function to copy content of array while reshaping.
 *
 * Submits a kernel to perform a copy `dst[unravel_index(i,
 * dst.shape)] = src[unravel_undex(i, src.shape)]`.
 *
 * @param  q      The execution queue where kernel is submitted.
 * @param  nelems The number of elements to copy
 * @param  src_nd Array dimension of the source array
 * @param  dst_nd Array dimension of the destination array
 * @param  packed_shapes_and_strides Kernel accessible USM array of size
 * `2*src_nd + 2*dst_nd` with content `[src_shape, src_strides, dst_shape,
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
copy_for_reshape_generic_impl(sycl::queue &q,
                              size_t nelems,
                              int src_nd,
                              int dst_nd,
                              ssize_t *packed_shapes_and_strides,
                              const char *src_p,
                              char *dst_p,
                              const std::vector<sycl::event> &depends)
{
    dpctl::tensor::type_utils::validate_type_for_device<Ty>(q);

    sycl::event copy_for_reshape_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        // packed_shapes_and_strides:
        //   USM array of size 2*(src_nd + dst_nd)
        //   [ src_shape; src_strides; dst_shape; dst_strides ]

        const ssize_t *src_shape_and_strides =
            const_cast<const ssize_t *>(packed_shapes_and_strides);

        const ssize_t *dst_shape_and_strides = const_cast<const ssize_t *>(
            packed_shapes_and_strides + (2 * src_nd));

        const StridedIndexer src_indexer{src_nd, 0, src_shape_and_strides};
        const StridedIndexer dst_indexer{dst_nd, 0, dst_shape_and_strides};

        using KernelName =
            copy_for_reshape_generic_kernel<Ty, StridedIndexer, StridedIndexer>;

        cgh.parallel_for<KernelName>(
            sycl::range<1>(nelems),
            GenericCopyForReshapeFunctor<Ty, StridedIndexer, StridedIndexer>(
                src_p, dst_p, src_indexer, dst_indexer));
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

// ================== Copying for roll ================== //

/*! @brief Functor to cyclically roll global_id to the left */
struct LeftRolled1DTransformer
{
    LeftRolled1DTransformer(size_t offset, size_t size)
        : offset_(offset), size_(size)
    {
    }

    size_t operator()(size_t gid) const
    {
        const size_t shifted_gid =
            ((gid < offset_) ? gid + size_ - offset_ : gid - offset_);
        return shifted_gid;
    }

private:
    size_t offset_ = 0;
    size_t size_ = 1;
};

/*! @brief Indexer functor to compose indexer and transformer */
template <typename IndexerT, typename TransformerT> struct CompositionIndexer
{
    CompositionIndexer(IndexerT f, TransformerT t) : f_(f), t_(t) {}

    auto operator()(size_t gid) const
    {
        return f_(t_(gid));
    }

private:
    IndexerT f_;
    TransformerT t_;
};

/*! @brief Indexer functor to find offset for nd-shifted indices lifted from
 * iteration id */
struct RolledNDIndexer
{
    RolledNDIndexer(int nd,
                    const ssize_t *shape,
                    const ssize_t *strides,
                    const ssize_t *ndshifts,
                    ssize_t starting_offset)
        : nd_(nd), shape_(shape), strides_(strides), ndshifts_(ndshifts),
          starting_offset_(starting_offset)
    {
    }

    ssize_t operator()(size_t gid) const
    {
        return compute_offset(gid);
    }

private:
    int nd_ = -1;
    const ssize_t *shape_ = nullptr;
    const ssize_t *strides_ = nullptr;
    const ssize_t *ndshifts_ = nullptr;
    ssize_t starting_offset_ = 0;

    ssize_t compute_offset(ssize_t gid) const
    {
        using dpctl::tensor::strides::CIndexer_vector;

        CIndexer_vector _ind(nd_);
        ssize_t relative_offset_(0);
        _ind.get_left_rolled_displacement<const ssize_t *, const ssize_t *>(
            gid,
            shape_,    // shape ptr
            strides_,  // strides ptr
            ndshifts_, // shifts ptr
            relative_offset_);
        return starting_offset_ + relative_offset_;
    }
};

template <typename Ty, typename SrcIndexerT, typename DstIndexerT>
class copy_for_roll_strided_kernel;

template <typename Ty, typename SrcIndexerT, typename DstIndexerT>
class StridedCopyForRollFunctor
{
private:
    const Ty *src_p = nullptr;
    Ty *dst_p = nullptr;
    const SrcIndexerT src_indexer_;
    const DstIndexerT dst_indexer_;

public:
    StridedCopyForRollFunctor(const Ty *src_ptr,
                              Ty *dst_ptr,
                              const SrcIndexerT &src_indexer,
                              const DstIndexerT &dst_indexer)
        : src_p(src_ptr), dst_p(dst_ptr), src_indexer_(src_indexer),
          dst_indexer_(dst_indexer)
    {
    }

    void operator()(sycl::id<1> wiid) const
    {
        const size_t gid = wiid.get(0);

        const ssize_t src_offset = src_indexer_(gid);
        const ssize_t dst_offset = dst_indexer_(gid);

        dst_p[dst_offset] = src_p[src_offset];
    }
};

// define function type
typedef sycl::event (*copy_for_roll_strided_fn_ptr_t)(
    sycl::queue &,
    size_t,          // shift
    size_t,          // num_elements
    int,             // common_nd
    const ssize_t *, // packed shapes and strides
    const char *,    // src_data_ptr
    ssize_t,         // src_offset
    char *,          // dst_data_ptr
    ssize_t,         // dst_offset
    const std::vector<sycl::event> &);

/*!
 * @brief Function to copy content of array with a shift.
 *
 * Submits a kernel to perform a copy `dst[unravel_index((i + shift) % nelems ,
 * dst.shape)] = src[unravel_undex(i, src.shape)]`.
 *
 * @param  q      The execution queue where kernel is submitted.
 * @param  shift  The shift in flat indexing, must be non-negative.
 * @param  nelems The number of elements to copy
 * @param  nd     Array dimensionality of the destination and source arrays
 * @param  packed_shapes_and_strides Kernel accessible USM array
 * of size `3*nd` with content `[common_shape, src_strides, dst_strides]`.
 * @param  src_p  Typeless USM pointer to the buffer of the source array
 * @param  src_offset Displacement of first element of src relative src_p in
 * elements
 * @param  dst_p  Typeless USM pointer to the buffer of the destination array
 * @param  dst_offset Displacement of first element of dst relative dst_p in
 * elements
 * @param  depends  List of events to wait for before starting computations, if
 * any.
 *
 * @return Event to wait on to ensure that computation completes.
 * @ingroup CopyAndCastKernels
 */
template <typename Ty>
sycl::event copy_for_roll_strided_impl(sycl::queue &q,
                                       size_t shift,
                                       size_t nelems,
                                       int nd,
                                       const ssize_t *packed_shapes_and_strides,
                                       const char *src_p,
                                       ssize_t src_offset,
                                       char *dst_p,
                                       ssize_t dst_offset,
                                       const std::vector<sycl::event> &depends)
{
    dpctl::tensor::type_utils::validate_type_for_device<Ty>(q);

    sycl::event copy_for_roll_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        // packed_shapes_and_strides:
        //   USM array of size 3 * nd
        //   [ common_shape; src_strides; dst_strides ]

        const StridedIndexer src_indexer{nd, src_offset,
                                         packed_shapes_and_strides};
        const LeftRolled1DTransformer left_roll_transformer{shift, nelems};

        using CompositeIndexerT =
            CompositionIndexer<StridedIndexer, LeftRolled1DTransformer>;

        const CompositeIndexerT rolled_src_indexer(src_indexer,
                                                   left_roll_transformer);

        UnpackedStridedIndexer dst_indexer{nd, dst_offset,
                                           packed_shapes_and_strides,
                                           packed_shapes_and_strides + 2 * nd};

        using KernelName = copy_for_roll_strided_kernel<Ty, CompositeIndexerT,
                                                        UnpackedStridedIndexer>;

        const Ty *src_tp = reinterpret_cast<const Ty *>(src_p);
        Ty *dst_tp = reinterpret_cast<Ty *>(dst_p);

        cgh.parallel_for<KernelName>(
            sycl::range<1>(nelems),
            StridedCopyForRollFunctor<Ty, CompositeIndexerT,
                                      UnpackedStridedIndexer>(
                src_tp, dst_tp, rolled_src_indexer, dst_indexer));
    });

    return copy_for_roll_ev;
}

// define function type
typedef sycl::event (*copy_for_roll_contig_fn_ptr_t)(
    sycl::queue &,
    size_t,       // shift
    size_t,       // num_elements
    const char *, // src_data_ptr
    ssize_t,      // src_offset
    char *,       // dst_data_ptr
    ssize_t,      // dst_offset
    const std::vector<sycl::event> &);

template <typename Ty> class copy_for_roll_contig_kernel;

/*!
 * @brief Function to copy content of array with a shift.
 *
 * Submits a kernel to perform a copy `dst[unravel_index((i + shift) % nelems ,
 * dst.shape)] = src[unravel_undex(i, src.shape)]`.
 *
 * @param  q      The execution queue where kernel is submitted.
 * @param  shift  The shift in flat indexing, must be non-negative.
 * @param  nelems The number of elements to copy
 * @param  src_p  Typeless USM pointer to the buffer of the source array
 * @param  src_offset Displacement of the start of array src relative src_p in
 * elements
 * @param  dst_p  Typeless USM pointer to the buffer of the destination array
 * @param  dst_offset Displacement of the start of array dst relative dst_p in
 * elements
 * @param  depends  List of events to wait for before starting computations, if
 * any.
 *
 * @return Event to wait on to ensure that computation completes.
 * @ingroup CopyAndCastKernels
 */
template <typename Ty>
sycl::event copy_for_roll_contig_impl(sycl::queue &q,
                                      size_t shift,
                                      size_t nelems,
                                      const char *src_p,
                                      ssize_t src_offset,
                                      char *dst_p,
                                      ssize_t dst_offset,
                                      const std::vector<sycl::event> &depends)
{
    dpctl::tensor::type_utils::validate_type_for_device<Ty>(q);

    sycl::event copy_for_roll_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        constexpr NoOpIndexer src_indexer{};
        const LeftRolled1DTransformer roller{shift, nelems};

        const CompositionIndexer<NoOpIndexer, LeftRolled1DTransformer>
            left_rolled_src_indexer{src_indexer, roller};
        constexpr NoOpIndexer dst_indexer{};

        using KernelName = copy_for_roll_contig_kernel<Ty>;

        const Ty *src_tp = reinterpret_cast<const Ty *>(src_p) + src_offset;
        Ty *dst_tp = reinterpret_cast<Ty *>(dst_p) + dst_offset;

        cgh.parallel_for<KernelName>(
            sycl::range<1>(nelems),
            StridedCopyForRollFunctor<
                Ty, CompositionIndexer<NoOpIndexer, LeftRolled1DTransformer>,
                NoOpIndexer>(src_tp, dst_tp, left_rolled_src_indexer,
                             dst_indexer));
    });

    return copy_for_roll_ev;
}

/*!
 * @brief Factory to get function pointer of type `fnT` for given array data
 * type `Ty`.
 * @ingroup CopyAndCastKernels
 */
template <typename fnT, typename Ty> struct CopyForRollStridedFactory
{
    fnT get()
    {
        fnT f = copy_for_roll_strided_impl<Ty>;
        return f;
    }
};

/*!
 * @brief Factory to get function pointer of type `fnT` for given array data
 * type `Ty`.
 * @ingroup CopyAndCastKernels
 */
template <typename fnT, typename Ty> struct CopyForRollContigFactory
{
    fnT get()
    {
        fnT f = copy_for_roll_contig_impl<Ty>;
        return f;
    }
};

template <typename Ty, typename SrcIndexerT, typename DstIndexerT>
class copy_for_roll_ndshift_strided_kernel;

// define function type
typedef sycl::event (*copy_for_roll_ndshift_strided_fn_ptr_t)(
    sycl::queue &,
    size_t,          // num_elements
    int,             // common_nd
    const ssize_t *, // packed shape, strides, shifts
    const char *,    // src_data_ptr
    ssize_t,         // src_offset
    char *,          // dst_data_ptr
    ssize_t,         // dst_offset
    const std::vector<sycl::event> &);

template <typename Ty>
sycl::event copy_for_roll_ndshift_strided_impl(
    sycl::queue &q,
    size_t nelems,
    int nd,
    const ssize_t *packed_shapes_and_strides_and_shifts,
    const char *src_p,
    ssize_t src_offset,
    char *dst_p,
    ssize_t dst_offset,
    const std::vector<sycl::event> &depends)
{
    dpctl::tensor::type_utils::validate_type_for_device<Ty>(q);

    sycl::event copy_for_roll_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        // packed_shapes_and_strides_and_shifts:
        //   USM array of size 4 * nd
        //   [ common_shape; src_strides; dst_strides; shifts ]

        const ssize_t *shape_ptr = packed_shapes_and_strides_and_shifts;
        const ssize_t *src_strides_ptr =
            packed_shapes_and_strides_and_shifts + nd;
        const ssize_t *dst_strides_ptr =
            packed_shapes_and_strides_and_shifts + 2 * nd;
        const ssize_t *shifts_ptr =
            packed_shapes_and_strides_and_shifts + 3 * nd;

        const RolledNDIndexer src_indexer{nd, shape_ptr, src_strides_ptr,
                                          shifts_ptr, src_offset};

        const UnpackedStridedIndexer dst_indexer{nd, dst_offset, shape_ptr,
                                                 dst_strides_ptr};

        using KernelName = copy_for_roll_strided_kernel<Ty, RolledNDIndexer,
                                                        UnpackedStridedIndexer>;

        const Ty *src_tp = reinterpret_cast<const Ty *>(src_p);
        Ty *dst_tp = reinterpret_cast<Ty *>(dst_p);

        cgh.parallel_for<KernelName>(
            sycl::range<1>(nelems),
            StridedCopyForRollFunctor<Ty, RolledNDIndexer,
                                      UnpackedStridedIndexer>(
                src_tp, dst_tp, src_indexer, dst_indexer));
    });

    return copy_for_roll_ev;
}

/*!
 * @brief Factory to get function pointer of type `fnT` for given array data
 * type `Ty`.
 * @ingroup CopyAndCastKernels
 */
template <typename fnT, typename Ty> struct CopyForRollNDShiftFactory
{
    fnT get()
    {
        fnT f = copy_for_roll_ndshift_strided_impl<Ty>;
        return f;
    }
};

} // namespace copy_and_cast
} // namespace kernels
} // namespace tensor
} // namespace dpctl
