//===-- tensor_py.cpp - Implementation of _tensor_impl module  --*-C++-*-/===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2021 Intel Corporation
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
/// This file defines functions of dpctl.tensor._tensor_impl extensions
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <algorithm>
#include <complex>
#include <cstdint>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <thread>
#include <type_traits>

#include "dpctl4pybind11.hpp"
#include "utils/strided_iters.hpp"
#include "utils/type_dispatch.hpp"

namespace py = pybind11;

template <typename srcT, typename dstT> class copy_cast_generic_kernel;
template <typename srcT, typename dstT> class copy_cast_from_host_kernel;
template <typename srcT, typename dstT, int nd> class copy_cast_spec_kernel;
template <typename Ty> class copy_for_reshape_generic_kernel;
template <typename Ty> class linear_sequence_step_kernel;
template <typename Ty, typename wTy> class linear_sequence_affine_kernel;

static dpctl::tensor::detail::usm_ndarray_types array_types;

namespace
{

template <class T> struct is_complex : std::false_type
{
};
template <class T> struct is_complex<std::complex<T>> : std::true_type
{
};
template <typename dstTy, typename srcTy> dstTy convert_impl(const srcTy &v)
{
    if constexpr (std::is_same<dstTy, srcTy>::value) {
        return v;
    }
    else if constexpr (std::is_same_v<dstTy, bool> && is_complex<srcTy>::value)
    {
        // bool(complex_v) == (complex_v.real() != 0) && (complex_v.imag() !=0)
        return (convert_impl<bool, typename srcTy::value_type>(v.real()) ||
                convert_impl<bool, typename srcTy::value_type>(v.imag()));
    }
    else if constexpr (is_complex<srcTy>::value && !is_complex<dstTy>::value) {
        // real_t(complex_v) == real_t(complex_v.real())
        return convert_impl<dstTy, typename srcTy::value_type>(v.real());
    }
    else if constexpr (!std::is_integral<srcTy>::value &&
                       !std::is_same<dstTy, bool>::value &&
                       std::is_integral<dstTy>::value &&
                       std::is_unsigned<dstTy>::value)
    {
        // first cast to signed variant, the cast to unsigned one
        using signedT = typename std::make_signed<dstTy>::type;
        return static_cast<dstTy>(convert_impl<signedT, srcTy>(v));
    }
    else {
        return static_cast<dstTy>(v);
    }
}

template <typename srcT, typename dstT> class Caster
{
public:
    Caster() = default;
    void operator()(char *src,
                    std::ptrdiff_t src_offset,
                    char *dst,
                    std::ptrdiff_t dst_offset) const
    {
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

namespace _ns = dpctl::tensor::detail;

static copy_and_cast_generic_fn_ptr_t
    copy_and_cast_generic_dispatch_table[_ns::num_types][_ns::num_types];
static copy_and_cast_1d_fn_ptr_t
    copy_and_cast_1d_dispatch_table[_ns::num_types][_ns::num_types];
static copy_and_cast_2d_fn_ptr_t
    copy_and_cast_2d_dispatch_table[_ns::num_types][_ns::num_types];

template <typename fnT, typename D, typename S> struct CopyAndCastGenericFactory
{
    fnT get()
    {
        fnT f = copy_and_cast_generic_impl<D, S>;
        return f;
    }
};

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

std::vector<py::ssize_t> c_contiguous_strides(int nd,
                                              const py::ssize_t *shape,
                                              py::ssize_t element_size = 1)
{
    if (nd > 0) {
        std::vector<py::ssize_t> c_strides(nd, element_size);
        for (int ic = nd - 1; ic > 0;) {
            py::ssize_t next_v = c_strides[ic] * shape[ic];
            c_strides[--ic] = next_v;
        }
        return c_strides;
    }
    else {
        return std::vector<py::ssize_t>();
    }
}

std::vector<py::ssize_t> f_contiguous_strides(int nd,
                                              const py::ssize_t *shape,
                                              py::ssize_t element_size = 1)
{
    if (nd > 0) {
        std::vector<py::ssize_t> f_strides(nd, element_size);
        for (int i = 0; i < nd - 1;) {
            py::ssize_t next_v = f_strides[i] * shape[i];
            f_strides[++i] = next_v;
        }
        return f_strides;
    }
    else {
        return std::vector<py::ssize_t>();
    }
}

using dpctl::utils::keep_args_alive;

void simplify_iteration_space(int &nd,
                              const py::ssize_t *&shape,
                              const py::ssize_t *&src_strides,
                              py::ssize_t src_itemsize,
                              bool is_src_c_contig,
                              bool is_src_f_contig,
                              const py::ssize_t *&dst_strides,
                              py::ssize_t dst_itemsize,
                              bool is_dst_c_contig,
                              bool is_dst_f_contig,
                              std::vector<py::ssize_t> &simplified_shape,
                              std::vector<py::ssize_t> &simplified_src_strides,
                              std::vector<py::ssize_t> &simplified_dst_strides,
                              py::ssize_t &src_offset,
                              py::ssize_t &dst_offset)
{
    if (nd > 1) {
        // Simplify iteration space to reduce dimensionality
        // and improve access pattern
        simplified_shape.reserve(nd);
        for (int i = 0; i < nd; ++i) {
            simplified_shape.push_back(shape[i]);
        }

        simplified_src_strides.reserve(nd);
        simplified_dst_strides.reserve(nd);
        if (src_strides == nullptr) {
            if (is_src_c_contig) {
                simplified_src_strides =
                    c_contiguous_strides(nd, shape, src_itemsize);
            }
            else if (is_src_f_contig) {
                simplified_src_strides =
                    f_contiguous_strides(nd, shape, src_itemsize);
            }
            else {
                throw std::runtime_error(
                    "Source array has null strides "
                    "but has neither C- nor F- contiguous flag set");
            }
        }
        else {
            for (int i = 0; i < nd; ++i) {
                simplified_src_strides.push_back(src_strides[i]);
            }
        }
        if (dst_strides == nullptr) {
            if (is_dst_c_contig) {
                simplified_dst_strides =
                    c_contiguous_strides(nd, shape, dst_itemsize);
            }
            else if (is_dst_f_contig) {
                simplified_dst_strides =
                    f_contiguous_strides(nd, shape, dst_itemsize);
            }
            else {
                throw std::runtime_error(
                    "Destination array has null strides "
                    "but has neither C- nor F- contiguous flag set");
            }
        }
        else {
            for (int i = 0; i < nd; ++i) {
                simplified_dst_strides.push_back(dst_strides[i]);
            }
        }

        assert(simplified_shape.size() == static_cast<size_t>(nd));
        assert(simplified_src_strides.size() == static_cast<size_t>(nd));
        assert(simplified_dst_strides.size() == static_cast<size_t>(nd));
        int contracted_nd = simplify_iteration_two_strides(
            nd, simplified_shape.data(), simplified_src_strides.data(),
            simplified_dst_strides.data(),
            src_offset, // modified by reference
            dst_offset  // modified by reference
        );
        simplified_shape.resize(contracted_nd);
        simplified_src_strides.resize(contracted_nd);
        simplified_dst_strides.resize(contracted_nd);

        nd = contracted_nd;
        shape = const_cast<const py::ssize_t *>(simplified_shape.data());
        src_strides =
            const_cast<const py::ssize_t *>(simplified_src_strides.data());
        dst_strides =
            const_cast<const py::ssize_t *>(simplified_dst_strides.data());
    }
    else if (nd == 1) {
        // Populate vectors
        simplified_shape.reserve(nd);
        simplified_shape.push_back(shape[0]);

        simplified_src_strides.reserve(nd);
        simplified_dst_strides.reserve(nd);

        if (src_strides == nullptr) {
            if (is_src_c_contig) {
                simplified_src_strides.push_back(src_itemsize);
            }
            else if (is_src_f_contig) {
                simplified_src_strides.push_back(src_itemsize);
            }
            else {
                throw std::runtime_error(
                    "Source array has null strides "
                    "but has neither C- nor F- contiguous flag set");
            }
        }
        else {
            simplified_src_strides.push_back(src_strides[0]);
        }
        if (dst_strides == nullptr) {
            if (is_dst_c_contig) {
                simplified_dst_strides.push_back(dst_itemsize);
            }
            else if (is_dst_f_contig) {
                simplified_dst_strides.push_back(dst_itemsize);
            }
            else {
                throw std::runtime_error(
                    "Destination array has null strides "
                    "but has neither C- nor F- contiguous flag set");
            }
        }
        else {
            simplified_dst_strides.push_back(dst_strides[0]);
        }

        assert(simplified_shape.size() == static_cast<size_t>(nd));
        assert(simplified_src_strides.size() == static_cast<size_t>(nd));
        assert(simplified_dst_strides.size() == static_cast<size_t>(nd));
    }
}

sycl::event _populate_packed_shape_strides_for_copycast_kernel(
    sycl::queue exec_q,
    int src_flags,
    int dst_flags,
    py::ssize_t *device_shape_strides, // to be populated
    const std::vector<py::ssize_t> &common_shape,
    const std::vector<py::ssize_t> &src_strides,
    const std::vector<py::ssize_t> &dst_strides)
{
    using shT = std::vector<py::ssize_t>;
    size_t nd = common_shape.size();

    // create host temporary for packed shape and strides managed by shared
    // pointer. Packed vector is concatenation of common_shape, src_stride and
    // std_strides
    std::shared_ptr<shT> shp_host_shape_strides = std::make_shared<shT>(3 * nd);
    std::copy(common_shape.begin(), common_shape.end(),
              shp_host_shape_strides->begin());

    std::copy(src_strides.begin(), src_strides.end(),
              shp_host_shape_strides->begin() + nd);

    std::copy(dst_strides.begin(), dst_strides.end(),
              shp_host_shape_strides->begin() + 2 * nd);

    sycl::event copy_shape_ev = exec_q.copy<py::ssize_t>(
        shp_host_shape_strides->data(), device_shape_strides,
        shp_host_shape_strides->size());

    exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(copy_shape_ev);
        cgh.host_task([shp_host_shape_strides]() {
            // increment shared pointer ref-count to keep it alive
            // till copy operation completes;
        });
    });

    return copy_shape_ev;
}

std::pair<sycl::event, sycl::event>
copy_usm_ndarray_into_usm_ndarray(dpctl::tensor::usm_ndarray src,
                                  dpctl::tensor::usm_ndarray dst,
                                  sycl::queue exec_q,
                                  const std::vector<sycl::event> &depends = {})
{

    // array dimensions must be the same
    int src_nd = src.get_ndim();
    int dst_nd = dst.get_ndim();
    if (src_nd != dst_nd) {
        throw py::value_error("Array dimensions are not the same.");
    }

    // shapes must be the same
    const py::ssize_t *src_shape = src.get_shape_raw();
    const py::ssize_t *dst_shape = dst.get_shape_raw();

    bool shapes_equal(true);
    size_t src_nelems(1);

    for (int i = 0; i < src_nd; ++i) {
        src_nelems *= static_cast<size_t>(src_shape[i]);
        shapes_equal = shapes_equal && (src_shape[i] == dst_shape[i]);
    }
    if (!shapes_equal) {
        throw py::value_error("Array shapes are not the same.");
    }

    if (src_nelems == 0) {
        // nothing to do
        return std::make_pair(sycl::event(), sycl::event());
    }

    auto dst_offsets = dst.get_minmax_offsets();
    // destination must be ample enough to accomodate all elements
    {
        size_t range =
            static_cast<size_t>(dst_offsets.second - dst_offsets.first);
        if (range + 1 < src_nelems) {
            throw py::value_error(
                "Destination array can not accomodate all the "
                "elements of source array.");
        }
    }

    // check compatibility of execution queue and allocation queue
    sycl::queue src_q = src.get_queue();
    sycl::queue dst_q = dst.get_queue();

    if (!dpctl::utils::queues_are_compatible(exec_q, {src_q, dst_q})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    int src_type_id = array_types.typenum_to_lookup_id(src_typenum);
    int dst_type_id = array_types.typenum_to_lookup_id(dst_typenum);

    char *src_data = src.get_data();
    char *dst_data = dst.get_data();

    // check that arrays do not overlap, and concurrent copying is safe.
    auto src_offsets = src.get_minmax_offsets();
    int src_elem_size = src.get_elemsize();
    int dst_elem_size = dst.get_elemsize();

    bool memory_overlap =
        ((dst_data - src_data > src_offsets.second * src_elem_size -
                                    dst_offsets.first * dst_elem_size) &&
         (src_data - dst_data > dst_offsets.second * dst_elem_size -
                                    src_offsets.first * src_elem_size));
    if (memory_overlap) {
        // TODO: could use a temporary, but this is done by the caller
        throw py::value_error("Arrays index overlapping segments of memory");
    }

    int src_flags = src.get_flags();
    int dst_flags = dst.get_flags();

    // check for applicability of special cases:
    //      (same type && (both C-contiguous || both F-contiguous)
    bool both_c_contig = ((src_flags & USM_ARRAY_C_CONTIGUOUS) &&
                          (dst_flags & USM_ARRAY_C_CONTIGUOUS));
    bool both_f_contig = ((src_flags & USM_ARRAY_F_CONTIGUOUS) &&
                          (dst_flags & USM_ARRAY_F_CONTIGUOUS));
    if (both_c_contig || both_f_contig) {
        if (src_type_id == dst_type_id) {

            sycl::event copy_ev =
                exec_q.memcpy(static_cast<void *>(dst_data),
                              static_cast<const void *>(src_data),
                              src_nelems * src_elem_size, depends);

            // make sure src and dst are not GC-ed before copy_ev is complete
            return std::make_pair(
                keep_args_alive(exec_q, {src, dst}, {copy_ev}), copy_ev);
        }
        // With contract_iter2 in place, there is no need to write
        // dedicated kernels for casting between contiguous arrays
    }

    const py::ssize_t *src_strides = src.get_strides_raw();
    const py::ssize_t *dst_strides = dst.get_strides_raw();

    using shT = std::vector<py::ssize_t>;
    shT simplified_shape;
    shT simplified_src_strides;
    shT simplified_dst_strides;
    py::ssize_t src_offset(0);
    py::ssize_t dst_offset(0);

    int nd = src_nd;
    const py::ssize_t *shape = src_shape;

    bool is_src_c_contig = ((src_flags & USM_ARRAY_C_CONTIGUOUS) != 0);
    bool is_src_f_contig = ((src_flags & USM_ARRAY_F_CONTIGUOUS) != 0);

    bool is_dst_c_contig = ((dst_flags & USM_ARRAY_C_CONTIGUOUS) != 0);
    bool is_dst_f_contig = ((dst_flags & USM_ARRAY_F_CONTIGUOUS) != 0);

    constexpr py::ssize_t src_itemsize = 1; // in elements
    constexpr py::ssize_t dst_itemsize = 1; // in elements

    // all args except itemsizes and is_?_contig bools can be modified by
    // reference
    simplify_iteration_space(nd, shape, src_strides, src_itemsize,
                             is_src_c_contig, is_src_f_contig, dst_strides,
                             dst_itemsize, is_dst_c_contig, is_dst_f_contig,
                             simplified_shape, simplified_src_strides,
                             simplified_dst_strides, src_offset, dst_offset);

    if (nd < 3) {
        if (nd == 1) {
            std::array<py::ssize_t, 1> shape_arr = {shape[0]};
            // strides may be null
            std::array<py::ssize_t, 1> src_strides_arr = {
                (src_strides ? src_strides[0] : 1)};
            std::array<py::ssize_t, 1> dst_strides_arr = {
                (dst_strides ? dst_strides[0] : 1)};

            auto fn = copy_and_cast_1d_dispatch_table[dst_type_id][src_type_id];
            sycl::event copy_and_cast_1d_event = fn(
                exec_q, src_nelems, shape_arr, src_strides_arr, dst_strides_arr,
                src_data, src_offset, dst_data, dst_offset, depends);

            return std::make_pair(
                keep_args_alive(exec_q, {src, dst}, {copy_and_cast_1d_event}),
                copy_and_cast_1d_event);
        }
        else if (nd == 2) {
            std::array<py::ssize_t, 2> shape_arr = {shape[0], shape[1]};
            std::array<py::ssize_t, 2> src_strides_arr = {src_strides[0],
                                                          src_strides[1]};
            std::array<py::ssize_t, 2> dst_strides_arr = {dst_strides[0],
                                                          dst_strides[1]};

            auto fn = copy_and_cast_2d_dispatch_table[dst_type_id][src_type_id];
            sycl::event copy_and_cast_2d_event = fn(
                exec_q, src_nelems, shape_arr, src_strides_arr, dst_strides_arr,
                src_data, src_offset, dst_data, dst_offset, depends);

            return std::make_pair(
                keep_args_alive(exec_q, {src, dst}, {copy_and_cast_2d_event}),
                copy_and_cast_2d_event);
        }
        else if (nd == 0) { // case of a scalar
            assert(src_nelems == 1);
            std::array<py::ssize_t, 1> shape_arr = {1};
            std::array<py::ssize_t, 1> src_strides_arr = {1};
            std::array<py::ssize_t, 1> dst_strides_arr = {1};

            auto fn = copy_and_cast_1d_dispatch_table[dst_type_id][src_type_id];
            sycl::event copy_and_cast_0d_event = fn(
                exec_q, src_nelems, shape_arr, src_strides_arr, dst_strides_arr,
                src_data, src_offset, dst_data, dst_offset, depends);

            return std::make_pair(
                keep_args_alive(exec_q, {src, dst}, {copy_and_cast_0d_event}),
                copy_and_cast_0d_event);
        }
    }

    // Generic implementation
    auto copy_and_cast_fn =
        copy_and_cast_generic_dispatch_table[dst_type_id][src_type_id];

    //   If shape/strides are accessed with accessors, buffer destructor
    //   will force syncronization.
    py::ssize_t *shape_strides =
        sycl::malloc_device<py::ssize_t>(3 * nd, exec_q);

    if (shape_strides == nullptr) {
        throw std::runtime_error("Unabled to allocate device memory");
    }

    sycl::event copy_shape_ev =
        _populate_packed_shape_strides_for_copycast_kernel(
            exec_q, src_flags, dst_flags, shape_strides, simplified_shape,
            simplified_src_strides, simplified_dst_strides);

    sycl::event copy_and_cast_generic_ev = copy_and_cast_fn(
        exec_q, src_nelems, nd, shape_strides, src_data, src_offset, dst_data,
        dst_offset, depends, {copy_shape_ev});

    // async free of shape_strides temporary
    auto ctx = exec_q.get_context();
    exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(copy_and_cast_generic_ev);
        cgh.host_task(
            [ctx, shape_strides]() { sycl::free(shape_strides, ctx); });
    });

    return std::make_pair(
        keep_args_alive(exec_q, {src, dst}, {copy_and_cast_generic_ev}),
        copy_and_cast_generic_ev);
}

/* =========================== Copy for reshape ==============================
 */

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

// define static vector
static copy_for_reshape_fn_ptr_t
    copy_for_reshape_generic_dispatch_vector[_ns::num_types];

template <typename fnT, typename Ty> struct CopyForReshapeGenericFactory
{
    fnT get()
    {
        fnT f = copy_for_reshape_generic_impl<Ty>;
        return f;
    }
};

/*
 * Copies src into dst (same data type) of different shapes by using flat
 * iterations.
 *
 * Equivalent to the following loop:
 *
 * for i for range(src.size):
 *     dst[np.multi_index(i, dst.shape)] = src[np.multi_index(i, src.shape)]
 */
std::pair<sycl::event, sycl::event>
copy_usm_ndarray_for_reshape(dpctl::tensor::usm_ndarray src,
                             dpctl::tensor::usm_ndarray dst,
                             py::ssize_t shift,
                             sycl::queue exec_q,
                             const std::vector<sycl::event> &depends = {})
{
    py::ssize_t src_nelems = src.get_size();
    py::ssize_t dst_nelems = dst.get_size();

    // Must have the same number of elements
    if (src_nelems != dst_nelems) {
        throw py::value_error(
            "copy_usm_ndarray_for_reshape requires src and dst to "
            "have the same number of elements.");
    }

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    // typenames must be the same
    if (src_typenum != dst_typenum) {
        throw py::value_error(
            "copy_usm_ndarray_for_reshape requires src and dst to "
            "have the same type.");
    }

    if (src_nelems == 0) {
        return std::make_pair(sycl::event(), sycl::event());
    }

    // destination must be ample enough to accomodate all elements
    {
        auto dst_offsets = dst.get_minmax_offsets();
        py::ssize_t range =
            static_cast<py::ssize_t>(dst_offsets.second - dst_offsets.first);
        if (range + 1 < src_nelems) {
            throw py::value_error(
                "Destination array can not accomodate all the "
                "elements of source array.");
        }
    }

    // check same contexts
    sycl::queue src_q = src.get_queue();
    sycl::queue dst_q = dst.get_queue();

    if (!dpctl::utils::queues_are_compatible(exec_q, {src_q, dst_q})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    if (src_nelems == 1) {
        // handle special case of 1-element array
        int src_elemsize = src.get_elemsize();
        char *src_data = src.get_data();
        char *dst_data = dst.get_data();
        sycl::event copy_ev =
            exec_q.copy<char>(src_data, dst_data, src_elemsize);
        return std::make_pair(keep_args_alive(exec_q, {src, dst}, {copy_ev}),
                              copy_ev);
    }

    // dimensions may be different
    int src_nd = src.get_ndim();
    int dst_nd = dst.get_ndim();

    const py::ssize_t *src_shape = src.get_shape_raw();
    const py::ssize_t *dst_shape = dst.get_shape_raw();

    int type_id = array_types.typenum_to_lookup_id(src_typenum);

    auto fn = copy_for_reshape_generic_dispatch_vector[type_id];

    // packed_shape_strides = [src_shape, src_strides, dst_shape, dst_strides]
    py::ssize_t *packed_shapes_strides =
        sycl::malloc_device<py::ssize_t>(2 * (src_nd + dst_nd), exec_q);

    if (packed_shapes_strides == nullptr) {
        throw std::runtime_error("Unabled to allocate device memory");
    }

    using shT = std::vector<py::ssize_t>;
    std::shared_ptr<shT> packed_host_shapes_strides_shp =
        std::make_shared<shT>(2 * (src_nd + dst_nd));

    std::copy(src_shape, src_shape + src_nd,
              packed_host_shapes_strides_shp->begin());
    std::copy(dst_shape, dst_shape + dst_nd,
              packed_host_shapes_strides_shp->begin() + 2 * src_nd);

    const py::ssize_t *src_strides = src.get_strides_raw();
    if (src_strides == nullptr) {
        int src_flags = src.get_flags();
        if (src_flags & USM_ARRAY_C_CONTIGUOUS) {
            const shT &src_contig_strides =
                c_contiguous_strides(src_nd, src_shape);
            std::copy(src_contig_strides.begin(), src_contig_strides.end(),
                      packed_host_shapes_strides_shp->begin() + src_nd);
        }
        else if (src_flags & USM_ARRAY_F_CONTIGUOUS) {
            const shT &src_contig_strides =
                c_contiguous_strides(src_nd, src_shape);
            std::copy(src_contig_strides.begin(), src_contig_strides.end(),
                      packed_host_shapes_strides_shp->begin() + src_nd);
        }
        else {
            sycl::free(packed_shapes_strides, exec_q);
            throw std::runtime_error(
                "Invalid src array encountered: in copy_for_reshape function");
        }
    }
    else {
        std::copy(src_strides, src_strides + src_nd,
                  packed_host_shapes_strides_shp->begin() + src_nd);
    }

    const py::ssize_t *dst_strides = dst.get_strides_raw();
    if (dst_strides == nullptr) {
        int dst_flags = dst.get_flags();
        if (dst_flags & USM_ARRAY_C_CONTIGUOUS) {
            const shT &dst_contig_strides =
                c_contiguous_strides(dst_nd, dst_shape);
            std::copy(dst_contig_strides.begin(), dst_contig_strides.end(),
                      packed_host_shapes_strides_shp->begin() + 2 * src_nd +
                          dst_nd);
        }
        else if (dst_flags & USM_ARRAY_F_CONTIGUOUS) {
            const shT &dst_contig_strides =
                f_contiguous_strides(dst_nd, dst_shape);
            std::copy(dst_contig_strides.begin(), dst_contig_strides.end(),
                      packed_host_shapes_strides_shp->begin() + 2 * src_nd +
                          dst_nd);
        }
        else {
            sycl::free(packed_shapes_strides, exec_q);
            throw std::runtime_error(
                "Invalid dst array encountered: in copy_for_reshape function");
        }
    }
    else {
        std::copy(dst_strides, dst_strides + dst_nd,
                  packed_host_shapes_strides_shp->begin() + 2 * src_nd +
                      dst_nd);
    }

    // copy packed shapes and strides from host to devices
    sycl::event packed_shape_strides_copy_ev = exec_q.copy<py::ssize_t>(
        packed_host_shapes_strides_shp->data(), packed_shapes_strides,
        packed_host_shapes_strides_shp->size());
    exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(packed_shape_strides_copy_ev);
        cgh.host_task([packed_host_shapes_strides_shp] {
            // Capturing shared pointer ensures that the underlying vector is
            // not destroyed until after its data are copied into packed USM
            // vector
        });
    });

    char *src_data = src.get_data();
    char *dst_data = dst.get_data();

    std::vector<sycl::event> all_deps(depends.size() + 1);
    all_deps.push_back(packed_shape_strides_copy_ev);
    all_deps.insert(std::end(all_deps), std::begin(depends), std::end(depends));

    sycl::event copy_for_reshape_event =
        fn(exec_q, shift, src_nelems, src_nd, dst_nd, packed_shapes_strides,
           src_data, dst_data, all_deps);

    exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(copy_for_reshape_event);
        auto ctx = exec_q.get_context();
        cgh.host_task([packed_shapes_strides, ctx]() {
            sycl::free(packed_shapes_strides, ctx);
        });
    });

    return std::make_pair(
        keep_args_alive(exec_q, {src, dst}, {copy_for_reshape_event}),
        copy_for_reshape_event);
}

/* ============= Copy from numpy.ndarray to usm_ndarray ==================== */

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

static copy_and_cast_from_host_blocking_fn_ptr_t
    copy_and_cast_from_host_blocking_dispatch_table[_ns::num_types]
                                                   [_ns::num_types];

template <typename fnT, typename D, typename S>
struct CopyAndCastFromHostFactory
{
    fnT get()
    {
        fnT f = copy_and_cast_from_host_impl<D, S>;
        return f;
    }
};

void copy_numpy_ndarray_into_usm_ndarray(
    py::array npy_src,
    dpctl::tensor::usm_ndarray dst,
    sycl::queue exec_q,
    const std::vector<sycl::event> &depends = {})
{
    int src_ndim = npy_src.ndim();
    int dst_ndim = dst.get_ndim();

    if (src_ndim != dst_ndim) {
        throw py::value_error("Source ndarray and destination usm_ndarray have "
                              "different array ranks, "
                              "i.e. different number of indices needed to "
                              "address array elements.");
    }

    const py::ssize_t *src_shape = npy_src.shape();
    const py::ssize_t *dst_shape = dst.get_shape_raw();
    bool shapes_equal(true);
    size_t src_nelems(1);
    for (int i = 0; i < src_ndim; ++i) {
        shapes_equal = shapes_equal && (src_shape[i] == dst_shape[i]);
        src_nelems *= static_cast<size_t>(src_shape[i]);
    }

    if (!shapes_equal) {
        throw py::value_error("Source ndarray and destination usm_ndarray have "
                              "difference shapes.");
    }

    if (src_nelems == 0) {
        // nothing to do
        return;
    }

    auto dst_offsets = dst.get_minmax_offsets();
    // destination must be ample enough to accomodate all elements of source
    // array
    {
        size_t range =
            static_cast<size_t>(dst_offsets.second - dst_offsets.first);
        if (range + 1 < src_nelems) {
            throw py::value_error(
                "Destination array can not accomodate all the "
                "elements of source array.");
        }
    }

    sycl::queue dst_q = dst.get_queue();

    if (!dpctl::utils::queues_are_compatible(exec_q, {dst_q})) {
        throw py::value_error("Execution queue is not compatible with the "
                              "allocation queue");
    }

    // here we assume that NumPy's type numbers agree with ours for types
    // supported in both
    int src_typenum =
        py::detail::array_descriptor_proxy(npy_src.dtype().ptr())->type_num;
    int dst_typenum = dst.get_typenum();

    int src_type_id = array_types.typenum_to_lookup_id(src_typenum);
    int dst_type_id = array_types.typenum_to_lookup_id(dst_typenum);

    py::buffer_info src_pybuf = npy_src.request();
    const char *const src_data = static_cast<const char *const>(src_pybuf.ptr);
    char *dst_data = dst.get_data();

    int src_flags = npy_src.flags();
    int dst_flags = dst.get_flags();

    // check for applicability of special cases:
    //      (same type && (both C-contiguous || both F-contiguous)
    bool both_c_contig = ((src_flags & py::array::c_style) &&
                          (dst_flags & USM_ARRAY_C_CONTIGUOUS));
    bool both_f_contig = ((src_flags & py::array::f_style) &&
                          (dst_flags & USM_ARRAY_F_CONTIGUOUS));
    if (both_c_contig || both_f_contig) {
        if (src_type_id == dst_type_id) {
            int src_elem_size = npy_src.itemsize();

            sycl::event copy_ev =
                exec_q.memcpy(static_cast<void *>(dst_data),
                              static_cast<const void *>(src_data),
                              src_nelems * src_elem_size, depends);

            // wait for copy_ev to complete
            copy_ev.wait_and_throw();

            return;
        }
        // With contract_iter2 in place, there is no need to write
        // dedicated kernels for casting between contiguous arrays
    }

    const py::ssize_t *src_strides =
        npy_src.strides(); // N.B.: strides in bytes
    const py::ssize_t *dst_strides =
        dst.get_strides_raw(); // N.B.: strides in elements

    using shT = std::vector<py::ssize_t>;
    shT simplified_shape;
    shT simplified_src_strides;
    shT simplified_dst_strides;
    py::ssize_t src_offset(0);
    py::ssize_t dst_offset(0);

    py::ssize_t src_itemsize = npy_src.itemsize(); // item size in bytes
    constexpr py::ssize_t dst_itemsize = 1;        // item size in elements

    int nd = src_ndim;
    const py::ssize_t *shape = src_shape;

    bool is_src_c_contig = ((src_flags & py::array::c_style) != 0);
    bool is_src_f_contig = ((src_flags & py::array::f_style) != 0);

    bool is_dst_c_contig = ((dst_flags & USM_ARRAY_C_CONTIGUOUS) != 0);
    bool is_dst_f_contig = ((dst_flags & USM_ARRAY_F_CONTIGUOUS) != 0);

    // all args except itemsizes and is_?_contig bools can be modified by
    // reference
    simplify_iteration_space(nd, shape, src_strides, src_itemsize,
                             is_src_c_contig, is_src_f_contig, dst_strides,
                             dst_itemsize, is_dst_c_contig, is_dst_f_contig,
                             simplified_shape, simplified_src_strides,
                             simplified_dst_strides, src_offset, dst_offset);

    assert(simplified_shape.size() == static_cast<size_t>(nd));
    assert(simplified_src_strides.size() == static_cast<size_t>(nd));
    assert(simplified_dst_strides.size() == static_cast<size_t>(nd));

    // handle nd == 0
    if (nd == 0) {
        nd = 1;
        simplified_shape.reserve(nd);
        simplified_shape.push_back(1);

        simplified_src_strides.reserve(nd);
        simplified_src_strides.push_back(src_itemsize);

        simplified_dst_strides.reserve(nd);
        simplified_dst_strides.push_back(dst_itemsize);
    }

    // Minumum and maximum element offsets for source np.ndarray
    py::ssize_t npy_src_min_nelem_offset(0);
    py::ssize_t npy_src_max_nelem_offset(0);
    for (int i = 0; i < nd; ++i) {
        // convert source strides from bytes to elements
        simplified_src_strides[i] = simplified_src_strides[i] / src_itemsize;
        if (simplified_src_strides[i] < 0) {
            npy_src_min_nelem_offset +=
                simplified_src_strides[i] * (simplified_shape[i] - 1);
        }
        else {
            npy_src_max_nelem_offset +=
                simplified_src_strides[i] * (simplified_shape[i] - 1);
        }
    }

    // Create shared pointers with shape and src/dst strides, copy into device
    // memory
    using shT = std::vector<py::ssize_t>;

    // Get implementation function pointer
    auto copy_and_cast_from_host_blocking_fn =
        copy_and_cast_from_host_blocking_dispatch_table[dst_type_id]
                                                       [src_type_id];

    //   If shape/strides are accessed with accessors, buffer destructor
    //   will force syncronization.
    py::ssize_t *shape_strides =
        sycl::malloc_device<py::ssize_t>(3 * nd, exec_q);

    if (shape_strides == nullptr) {
        throw std::runtime_error("Unabled to allocate device memory");
    }

    std::shared_ptr<shT> host_shape_strides_shp = std::make_shared<shT>(3 * nd);
    std::copy(simplified_shape.begin(), simplified_shape.end(),
              host_shape_strides_shp->begin());
    std::copy(simplified_src_strides.begin(), simplified_src_strides.end(),
              host_shape_strides_shp->begin() + nd);
    std::copy(simplified_dst_strides.begin(), simplified_dst_strides.end(),
              host_shape_strides_shp->begin() + 2 * nd);

    sycl::event copy_packed_ev =
        exec_q.copy<py::ssize_t>(host_shape_strides_shp->data(), shape_strides,
                                 host_shape_strides_shp->size());

    copy_and_cast_from_host_blocking_fn(
        exec_q, src_nelems, nd, shape_strides, src_data, src_offset,
        npy_src_min_nelem_offset, npy_src_max_nelem_offset, dst_data,
        dst_offset, depends, {copy_packed_ev});

    sycl::free(shape_strides, exec_q);

    return;
}

/* =========== Unboxing Python scalar =============== */

template <typename T> T unbox_py_scalar(py::object o)
{
    return py::cast<T>(o);
}

template <> sycl::half unbox_py_scalar<sycl::half>(py::object o)
{
    float tmp = py::cast<float>(o);
    return static_cast<sycl::half>(tmp);
}

/* ============= linear-sequence ==================== */

typedef sycl::event (*lin_space_step_fn_ptr_t)(
    sycl::queue,
    size_t, // num_elements
    py::object start,
    py::object step,
    char *, // dst_data_ptr
    const std::vector<sycl::event> &);

static lin_space_step_fn_ptr_t lin_space_step_dispatch_vector[_ns::num_types];

template <typename Ty> class LinearSequenceStepFunctor
{
private:
    Ty *p = nullptr;
    Ty start_v;
    Ty step_v;

public:
    LinearSequenceStepFunctor(char *dst_p, Ty v0, Ty dv)
        : p(reinterpret_cast<Ty *>(dst_p)), start_v(v0), step_v(dv)
    {
    }

    void operator()(sycl::id<1> wiid) const
    {
        auto i = wiid.get(0);
        if constexpr (is_complex<Ty>::value) {
            p[i] = Ty{start_v.real() + i * step_v.real(),
                      start_v.imag() + i * step_v.imag()};
        }
        else {
            p[i] = start_v + i * step_v;
        }
    }
};

template <typename Ty>
sycl::event lin_space_step_impl(sycl::queue exec_q,
                                size_t nelems,
                                py::object start,
                                py::object step,
                                char *array_data,
                                const std::vector<sycl::event> &depends)
{
    Ty start_v;
    Ty step_v;
    try {
        start_v = unbox_py_scalar<Ty>(start);
        step_v = unbox_py_scalar<Ty>(step);
    } catch (const py::error_already_set &e) {
        throw;
    }

    sycl::event lin_space_step_event = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.parallel_for<linear_sequence_step_kernel<Ty>>(
            sycl::range<1>{nelems},
            LinearSequenceStepFunctor<Ty>(array_data, start_v, step_v));
    });

    return lin_space_step_event;
}

template <typename fnT, typename Ty> struct LinSpaceStepFactory
{
    fnT get()
    {
        fnT f = lin_space_step_impl<Ty>;
        return f;
    }
};

typedef sycl::event (*lin_space_affine_fn_ptr_t)(
    sycl::queue,
    size_t, // num_elements
    py::object start,
    py::object end,
    bool include_endpoint,
    char *, // dst_data_ptr
    const std::vector<sycl::event> &);

static lin_space_affine_fn_ptr_t
    lin_space_affine_dispatch_vector[_ns::num_types];

template <typename Ty, typename wTy> class LinearSequenceAffineFunctor
{
private:
    Ty *p = nullptr;
    Ty start_v;
    Ty end_v;
    size_t n;

public:
    LinearSequenceAffineFunctor(char *dst_p, Ty v0, Ty v1, size_t den)
        : p(reinterpret_cast<Ty *>(dst_p)), start_v(v0), end_v(v1),
          n((den == 0) ? 1 : den)
    {
    }

    void operator()(sycl::id<1> wiid) const
    {
        auto i = wiid.get(0);
        wTy wc = wTy(i) / n;
        wTy w = wTy(n - i) / n;
        if constexpr (is_complex<Ty>::value) {
            auto _w = static_cast<typename Ty::value_type>(w);
            auto _wc = static_cast<typename Ty::value_type>(wc);
            auto re_comb = start_v.real() * _w + end_v.real() * _wc;
            auto im_comb = start_v.imag() * _w + end_v.imag() * _wc;
            Ty affine_comb = Ty{re_comb, im_comb};
            p[i] = affine_comb;
        }
        else {
            auto affine_comb = start_v * w + end_v * wc;
            p[i] = convert_impl<Ty, decltype(affine_comb)>(affine_comb);
        }
    }
};

template <typename Ty>
sycl::event lin_space_affine_impl(sycl::queue exec_q,
                                  size_t nelems,
                                  py::object start,
                                  py::object end,
                                  bool include_endpoint,
                                  char *array_data,
                                  const std::vector<sycl::event> &depends)
{
    Ty start_v, end_v;
    try {
        start_v = unbox_py_scalar<Ty>(start);
        end_v = unbox_py_scalar<Ty>(end);
    } catch (const py::error_already_set &e) {
        throw;
    }

    bool device_supports_doubles = exec_q.get_device().has(sycl::aspect::fp64);
    sycl::event lin_space_affine_event = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        if (device_supports_doubles) {
            cgh.parallel_for<linear_sequence_affine_kernel<Ty, double>>(
                sycl::range<1>{nelems},
                LinearSequenceAffineFunctor<Ty, double>(
                    array_data, start_v, end_v,
                    (include_endpoint) ? nelems - 1 : nelems));
        }
        else {
            cgh.parallel_for<linear_sequence_affine_kernel<Ty, float>>(
                sycl::range<1>{nelems},
                LinearSequenceAffineFunctor<Ty, float>(
                    array_data, start_v, end_v,
                    (include_endpoint) ? nelems - 1 : nelems));
        }
    });

    return lin_space_affine_event;
}

template <typename fnT, typename Ty> struct LinSpaceAffineFactory
{
    fnT get()
    {
        fnT f = lin_space_affine_impl<Ty>;
        return f;
    }
};

std::pair<sycl::event, sycl::event>
usm_ndarray_linear_sequence_step(py::object start,
                                 py::object dt,
                                 dpctl::tensor::usm_ndarray dst,
                                 sycl::queue exec_q,
                                 const std::vector<sycl::event> &depends = {})
{
    // dst must be 1D and C-contiguous
    // start, end should be coercible into data type of dst

    if (dst.get_ndim() != 1) {
        throw py::value_error(
            "usm_ndarray_linspace: Expecting 1D array to populate");
    }

    int flags = dst.get_flags();
    if (!(flags & USM_ARRAY_C_CONTIGUOUS)) {
        throw py::value_error(
            "usm_ndarray_linspace: Non-contiguous arrays are not supported");
    }

    sycl::queue dst_q = dst.get_queue();
    if (dst_q != exec_q && dst_q.get_context() != exec_q.get_context()) {
        throw py::value_error(
            "Execution queue context is not the same as allocation context");
    }

    int dst_typenum = dst.get_typenum();
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    py::ssize_t len = dst.get_shape(0);
    if (len == 0) {
        // nothing to do
        return std::make_pair(sycl::event{}, sycl::event{});
    }

    char *dst_data = dst.get_data();
    sycl::event linspace_step_event;

    auto fn = lin_space_step_dispatch_vector[dst_typeid];

    linspace_step_event =
        fn(exec_q, static_cast<size_t>(len), start, dt, dst_data, depends);

    return std::make_pair(keep_args_alive(exec_q, {dst}, {linspace_step_event}),
                          linspace_step_event);
}

std::pair<sycl::event, sycl::event>
usm_ndarray_linear_sequence_affine(py::object start,
                                   py::object end,
                                   dpctl::tensor::usm_ndarray dst,
                                   bool include_endpoint,
                                   sycl::queue exec_q,
                                   const std::vector<sycl::event> &depends = {})
{
    // dst must be 1D and C-contiguous
    // start, end should be coercible into data type of dst

    if (dst.get_ndim() != 1) {
        throw py::value_error(
            "usm_ndarray_linspace: Expecting 1D array to populate");
    }

    int flags = dst.get_flags();
    if (!(flags & USM_ARRAY_C_CONTIGUOUS)) {
        throw py::value_error(
            "usm_ndarray_linspace: Non-contiguous arrays are not supported");
    }

    sycl::queue dst_q = dst.get_queue();
    if (dst_q != exec_q && dst_q.get_context() != exec_q.get_context()) {
        throw py::value_error(
            "Execution queue context is not the same as allocation context");
    }

    int dst_typenum = dst.get_typenum();
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    py::ssize_t len = dst.get_shape(0);
    if (len == 0) {
        // nothing to do
        return std::make_pair(sycl::event{}, sycl::event{});
    }

    char *dst_data = dst.get_data();
    sycl::event linspace_affine_event;

    auto fn = lin_space_affine_dispatch_vector[dst_typeid];

    linspace_affine_event = fn(exec_q, static_cast<size_t>(len), start, end,
                               include_endpoint, dst_data, depends);

    return std::make_pair(
        keep_args_alive(exec_q, {dst}, {linspace_affine_event}),
        linspace_affine_event);
}

/* ================ Full ================== */

typedef sycl::event (*full_contig_fn_ptr_t)(sycl::queue,
                                            size_t,
                                            py::object,
                                            char *,
                                            const std::vector<sycl::event> &);

static full_contig_fn_ptr_t full_contig_dispatch_vector[_ns::num_types];

template <typename dstTy>
sycl::event full_contig_impl(sycl::queue q,
                             size_t nelems,
                             py::object py_value,
                             char *dst_p,
                             const std::vector<sycl::event> &depends)
{
    dstTy fill_v;
    try {
        fill_v = unbox_py_scalar<dstTy>(py_value);
    } catch (const py::error_already_set &e) {
        throw;
    }

    sycl::event fill_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        dstTy *p = reinterpret_cast<dstTy *>(dst_p);
        cgh.fill<dstTy>(p, fill_v, nelems);
    });

    return fill_ev;
}

template <typename fnT, typename Ty> struct FullContigFactory
{
    fnT get()
    {
        fnT f = full_contig_impl<Ty>;
        return f;
    }
};

std::pair<sycl::event, sycl::event>
usm_ndarray_full(py::object py_value,
                 dpctl::tensor::usm_ndarray dst,
                 sycl::queue exec_q,
                 const std::vector<sycl::event> &depends = {})
{
    // start, end should be coercible into data type of dst

    py::ssize_t dst_nelems = dst.get_size();

    if (dst_nelems == 0) {
        // nothing to do
        return std::make_pair(sycl::event(), sycl::event());
    }

    int dst_flags = dst.get_flags();

    sycl::queue dst_q = dst.get_queue();
    if (dst_q != exec_q && dst_q.get_context() != exec_q.get_context()) {
        throw py::value_error(
            "Execution queue context is not the same as allocation context");
    }

    int dst_typenum = dst.get_typenum();
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    char *dst_data = dst.get_data();
    sycl::event full_event;

    if (dst_nelems == 1 || (dst_flags & USM_ARRAY_C_CONTIGUOUS) ||
        (dst_flags & USM_ARRAY_F_CONTIGUOUS))
    {
        auto fn = full_contig_dispatch_vector[dst_typeid];

        sycl::event full_contig_event =
            fn(exec_q, static_cast<size_t>(dst_nelems), py_value, dst_data,
               depends);

        return std::make_pair(
            keep_args_alive(exec_q, {dst}, {full_contig_event}),
            full_contig_event);
    }
    else {
        throw std::runtime_error(
            "Only population of contiguous usm_ndarray objects is supported.");
    }
}

// populate dispatch tables
void init_copy_and_cast_dispatch_tables(void)
{
    using namespace dpctl::tensor::detail;

    DispatchTableBuilder<copy_and_cast_generic_fn_ptr_t,
                         CopyAndCastGenericFactory, num_types>
        dtb_generic;
    dtb_generic.populate_dispatch_table(copy_and_cast_generic_dispatch_table);

    DispatchTableBuilder<copy_and_cast_1d_fn_ptr_t, CopyAndCast1DFactory,
                         num_types>
        dtb_1d;
    dtb_1d.populate_dispatch_table(copy_and_cast_1d_dispatch_table);

    DispatchTableBuilder<copy_and_cast_2d_fn_ptr_t, CopyAndCast2DFactory,
                         num_types>
        dtb_2d;
    dtb_2d.populate_dispatch_table(copy_and_cast_2d_dispatch_table);

    DispatchTableBuilder<copy_and_cast_from_host_blocking_fn_ptr_t,
                         CopyAndCastFromHostFactory, num_types>
        dtb_copy_from_numpy;

    dtb_copy_from_numpy.populate_dispatch_table(
        copy_and_cast_from_host_blocking_dispatch_table);

    return;
}

// populate dispatch vectors
void init_copy_for_reshape_dispatch_vector(void)
{
    using namespace dpctl::tensor::detail;

    DispatchVectorBuilder<copy_for_reshape_fn_ptr_t,
                          CopyForReshapeGenericFactory, num_types>
        dvb;
    dvb.populate_dispatch_vector(copy_for_reshape_generic_dispatch_vector);

    DispatchVectorBuilder<lin_space_step_fn_ptr_t, LinSpaceStepFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(lin_space_step_dispatch_vector);

    DispatchVectorBuilder<lin_space_affine_fn_ptr_t, LinSpaceAffineFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(lin_space_affine_dispatch_vector);

    DispatchVectorBuilder<full_contig_fn_ptr_t, FullContigFactory, num_types>
        dvb3;
    dvb3.populate_dispatch_vector(full_contig_dispatch_vector);

    return;
}

std::string get_default_device_fp_type(sycl::device d)
{
    if (d.has(sycl::aspect::fp64)) {
        return "f8";
    }
    else {
        return "f4";
    }
}

std::string get_default_device_int_type(sycl::device)
{
    return "i8";
}

std::string get_default_device_complex_type(sycl::device d)
{
    if (d.has(sycl::aspect::fp64)) {
        return "c16";
    }
    else {
        return "c8";
    }
}

std::string get_default_device_bool_type(sycl::device)
{
    return "b1";
}

} // namespace

PYBIND11_MODULE(_tensor_impl, m)
{

    init_copy_and_cast_dispatch_tables();
    init_copy_for_reshape_dispatch_vector();
    import_dpctl();

    // populate types constants for type dispatching functions
    array_types = dpctl::tensor::detail::usm_ndarray_types::get();

    m.def(
        "_contract_iter", &contract_iter,
        "Simplifies iteration of array of given shape & stride. Returns "
        "a triple: shape, stride and offset for the new iterator of possible "
        "smaller dimension, which traverses the same elements as the original "
        "iterator, possibly in a different order.");

    m.def("_copy_usm_ndarray_into_usm_ndarray",
          &copy_usm_ndarray_into_usm_ndarray,
          "Copies from usm_ndarray `src` into usm_ndarray `dst` of the same "
          "shape. "
          "Returns a tuple of events: (host_task_event, compute_task_event)",
          py::arg("src"), py::arg("dst"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());

    m.def(
        "_contract_iter2", &contract_iter2,
        "Simplifies iteration over elements of pair of arrays of given shape "
        "with strides stride1 and stride2. Returns "
        "a 5-tuple: shape, stride and offset for the new iterator of possible "
        "smaller dimension for each array, which traverses the same elements "
        "as the original "
        "iterator, possibly in a different order.");

    m.def("_copy_usm_ndarray_for_reshape", &copy_usm_ndarray_for_reshape,
          "Copies from usm_ndarray `src` into usm_ndarray `dst` with the same "
          "number of elements using underlying 'C'-contiguous order for flat "
          "traversal with shift. "
          "Returns a tuple of events: (ht_event, comp_event)",
          py::arg("src"), py::arg("dst"), py::arg("shift"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    m.def("_linspace_step", &usm_ndarray_linear_sequence_step,
          "Fills input 1D contiguous usm_ndarray `dst` with linear sequence "
          "specified by "
          "starting point `start` and step `dt`. "
          "Returns a tuple of events: (ht_event, comp_event)",
          py::arg("start"), py::arg("dt"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    m.def("_linspace_affine", &usm_ndarray_linear_sequence_affine,
          "Fills input 1D contiguous usm_ndarray `dst` with linear sequence "
          "specified by "
          "starting point `start` and end point `end`. "
          "Returns a tuple of events: (ht_event, comp_event)",
          py::arg("start"), py::arg("end"), py::arg("dst"),
          py::arg("include_endpoint"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());

    m.def("_copy_numpy_ndarray_into_usm_ndarray",
          &copy_numpy_ndarray_into_usm_ndarray,
          "Copy fom numpy array `src` into usm_ndarray `dst` synchronously.",
          py::arg("src"), py::arg("dst"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());

    m.def("_full_usm_ndarray", &usm_ndarray_full,
          "Populate usm_ndarray `dst` with given fill_value.",
          py::arg("fill_value"), py::arg("dst"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());

    m.def("default_device_fp_type", [](sycl::queue q) -> std::string {
        return get_default_device_fp_type(q.get_device());
    });
    m.def("default_device_fp_type_device", [](sycl::device dev) -> std::string {
        return get_default_device_fp_type(dev);
    });

    m.def("default_device_int_type", [](sycl::queue q) -> std::string {
        return get_default_device_int_type(q.get_device());
    });
    m.def("default_device_int_type_device",
          [](sycl::device dev) -> std::string {
              return get_default_device_int_type(dev);
          });

    m.def("default_device_bool_type", [](sycl::queue q) -> std::string {
        return get_default_device_bool_type(q.get_device());
    });
    m.def("default_device_bool_type_device",
          [](sycl::device dev) -> std::string {
              return get_default_device_bool_type(dev);
          });

    m.def("default_device_complex_type", [](sycl::queue q) -> std::string {
        return get_default_device_complex_type(q.get_device());
    });
    m.def("default_device_complex_type_device",
          [](sycl::device dev) -> std::string {
              return get_default_device_complex_type(dev);
          });
}
