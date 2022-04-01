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
#include <complex>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <thread>
#include <type_traits>

#include "dpctl4pybind11.hpp"
#include "utils/strided_iters.hpp"
#include "utils/type_dispatch.hpp"

namespace py = pybind11;

template <typename srcT, typename dstT> class copy_cast_generic_kernel;
template <typename srcT, typename dstT, int nd> class copy_cast_spec_kernel;
template <typename Ty> class copy_for_reshape_generic_kernel;

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

    return;
}

std::vector<py::ssize_t> c_contiguous_strides(int nd, const py::ssize_t *shape)
{
    if (nd > 0) {
        std::vector<py::ssize_t> c_strides(nd, 1);
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

std::vector<py::ssize_t> f_contiguous_strides(int nd, const py::ssize_t *shape)
{
    if (nd > 0) {
        std::vector<py::ssize_t> f_strides(nd, 1);
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

template <std::size_t num>
sycl::event keep_args_alive(sycl::queue q,
                            const py::object (&py_objs)[num],
                            const std::vector<sycl::event> &depends = {})
{
    sycl::event host_task_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        std::array<std::shared_ptr<py::handle>, num> shp_arr;
        for (std::size_t i = 0; i < num; ++i) {
            shp_arr[i] = std::make_shared<py::handle>(py_objs[i]);
            shp_arr[i]->inc_ref();
        }
        cgh.host_task([=]() {
            bool guard = (Py_IsInitialized() && !_Py_IsFinalizing());
            if (guard) {
                PyGILState_STATE gstate;
                gstate = PyGILState_Ensure();
                for (std::size_t i = 0; i < num; ++i) {
                    shp_arr[i]->dec_ref();
                }
                PyGILState_Release(gstate);
            }
        });
    });

    return host_task_ev;
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

    // check same contexts
    sycl::queue src_q = src.get_queue();
    sycl::queue dst_q = dst.get_queue();

    sycl::context exec_ctx = exec_q.get_context();
    if (src_q.get_context() != exec_ctx || dst_q.get_context() != exec_ctx) {
        throw py::value_error(
            "Execution queue context is not the same as allocation contexts");
    }

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    int src_type_id = array_types.typenum_to_lookup_id(src_typenum);
    int dst_type_id = array_types.typenum_to_lookup_id(dst_typenum);

    {
        auto type_id_check = [](int id) -> bool {
            return ((id >= 0) && (id < _ns::num_types));
        };
        if (!(type_id_check(src_type_id) && type_id_check(dst_type_id))) {
            throw std::runtime_error("Type dispatching failed.");
        }
    }

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

            sycl::event copy_ev = exec_q.memcpy(
                dst_data, src_data, src_nelems * src_elem_size, depends);

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

    if (src_nd > 1) {
        // Simplify iteration space to reduce dimensionality
        // and improve access pattern
        simplified_shape.reserve(nd);
        simplified_src_strides.reserve(nd);
        simplified_dst_strides.reserve(nd);
        for (int i = 0; i < nd; ++i) {
            simplified_shape.push_back(shape[i]);
        }
        if (src_strides == nullptr) {
            if (src_flags & USM_ARRAY_C_CONTIGUOUS) {
                simplified_src_strides = c_contiguous_strides(nd, shape);
            }
            else if (src_flags & USM_ARRAY_F_CONTIGUOUS) {
                simplified_src_strides = f_contiguous_strides(nd, shape);
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
            if (dst_flags & USM_ARRAY_C_CONTIGUOUS) {
                simplified_dst_strides = c_contiguous_strides(nd, shape);
            }
            else if (dst_flags & USM_ARRAY_F_CONTIGUOUS) {
                simplified_dst_strides = f_contiguous_strides(nd, shape);
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

        assert(simplified_shape.size() == nd);
        assert(simplified_src_strides.size() == nd);
        assert(simplified_dst_strides.size() == nd);
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

    std::shared_ptr<shT> shp_shape = std::make_shared<shT>(simplified_shape);
    std::shared_ptr<shT> shp_src_strides =
        std::make_shared<shT>(simplified_src_strides);
    std::shared_ptr<shT> shp_dst_strides =
        std::make_shared<shT>(simplified_dst_strides);

    // Generic implementation
    auto copy_and_cast_fn =
        copy_and_cast_generic_dispatch_table[dst_type_id][src_type_id];

    //   If shape/strides are accessed with accessors, buffer destructor
    //   will force syncronization.
    py::ssize_t *shape_strides =
        sycl::malloc_device<py::ssize_t>(3 * nd, exec_q);

    // TODO: handle failed allocation

    sycl::event copy_shape_ev =
        exec_q.copy<py::ssize_t>(shp_shape->data(), shape_strides, nd);

    exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(copy_shape_ev);
        cgh.host_task([shp_shape]() {
            // increment shared pointer ref-count to keep it alive
            // till copy operation completes;
        });
    });

    sycl::event copy_src_strides_ev;
    if (src_strides == nullptr) {
        std::shared_ptr<shT> shp_contig_src_strides =
            std::make_shared<shT>((src_flags & USM_ARRAY_C_CONTIGUOUS)
                                      ? c_contiguous_strides(nd, shape)
                                      : f_contiguous_strides(nd, shape));
        copy_src_strides_ev = exec_q.copy<py::ssize_t>(
            shp_contig_src_strides->data(), shape_strides + nd, nd);
        exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(copy_src_strides_ev);
            cgh.host_task([shp_contig_src_strides]() {
                // increment shared pointer ref-count to keep it alive
                // till copy operation completes;
            });
        });
    }
    else {
        copy_src_strides_ev = exec_q.copy<py::ssize_t>(shp_src_strides->data(),
                                                       shape_strides + nd, nd);
        exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(copy_src_strides_ev);
            cgh.host_task([shp_src_strides]() {
                // increment shared pointer ref-count to keep it alive
                // till copy operation completes;
            });
        });
    }

    sycl::event copy_dst_strides_ev;
    if (dst_strides == nullptr) {
        std::shared_ptr<shT> shp_contig_dst_strides =
            std::make_shared<shT>((dst_flags & USM_ARRAY_C_CONTIGUOUS)
                                      ? c_contiguous_strides(nd, shape)
                                      : f_contiguous_strides(nd, shape));
        copy_dst_strides_ev = exec_q.copy<py::ssize_t>(
            shp_contig_dst_strides->data(), shape_strides + 2 * nd, nd);
        exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(copy_dst_strides_ev);
            cgh.host_task([shp_contig_dst_strides]() {
                // increment shared pointer ref-count to keep it alive
                // till copy operation completes;
            });
        });
    }
    else {
        copy_dst_strides_ev = exec_q.copy<py::ssize_t>(
            shp_dst_strides->data(), shape_strides + 2 * nd, nd);
        exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(copy_dst_strides_ev);
            cgh.host_task([shp_dst_strides]() {
                // increment shared pointer ref-count to keep it alive
                // till copy operation completes;
            });
        });
    }

    sycl::event copy_and_cast_generic_ev = copy_and_cast_fn(
        exec_q, src_nelems, nd, shape_strides, src_data, src_offset, dst_data,
        dst_offset, depends,
        {copy_shape_ev, copy_src_strides_ev, copy_dst_strides_ev});

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

// TODO: define function type
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

// TODO: define static vector
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

// TODO: define function to populate the vector
void init_copy_for_reshape_dispatch_vector(void)
{
    using namespace dpctl::tensor::detail;

    DispatchVectorBuilder<copy_for_reshape_fn_ptr_t,
                          CopyForReshapeGenericFactory, num_types>
        dvb;
    dvb.populate_dispatch_vector(copy_for_reshape_generic_dispatch_vector);

    return;
}

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

    sycl::context exec_ctx = exec_q.get_context();
    if (src_q.get_context() != exec_ctx || dst_q.get_context() != exec_ctx) {
        throw py::value_error(
            "Execution queue context is not the same as allocation contexts");
    }

    // dimensions may be different
    int src_nd = src.get_ndim();
    int dst_nd = dst.get_ndim();

    const py::ssize_t *src_shape = src.get_shape_raw();
    const py::ssize_t *dst_shape = dst.get_shape_raw();

    int type_id = array_types.typenum_to_lookup_id(src_typenum);

    auto fn = copy_for_reshape_generic_dispatch_vector[type_id];

    py::ssize_t *packed_shapes_strides =
        sycl::malloc_device<py::ssize_t>(2 * (src_nd + dst_nd), exec_q);

    sycl::event src_shape_copy_ev =
        exec_q.copy<py::ssize_t>(src_shape, packed_shapes_strides, src_nd);
    sycl::event dst_shape_copy_ev = exec_q.copy<py::ssize_t>(
        dst_shape, packed_shapes_strides + 2 * src_nd, dst_nd);

    const py::ssize_t *src_strides = src.get_strides_raw();
    sycl::event src_strides_copy_ev;
    if (src_strides == nullptr) {
        using shT = std::vector<py::ssize_t>;
        int src_flags = src.get_flags();
        std::shared_ptr<shT> contig_src_strides_shp;
        if (src_flags & USM_ARRAY_C_CONTIGUOUS) {
            contig_src_strides_shp =
                std::make_shared<shT>(c_contiguous_strides(src_nd, src_shape));
        }
        else if (src_flags & USM_ARRAY_F_CONTIGUOUS) {
            contig_src_strides_shp =
                std::make_shared<shT>(f_contiguous_strides(src_nd, src_shape));
        }
        else {
            sycl::event::wait({src_shape_copy_ev, dst_shape_copy_ev});
            sycl::free(packed_shapes_strides, exec_q);
            throw std::runtime_error(
                "Invalid src array encountered: in copy_for_reshape function");
        }
        src_strides_copy_ev =
            exec_q.copy<py::ssize_t>(contig_src_strides_shp->data(),
                                     packed_shapes_strides + src_nd, src_nd);
        exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(src_strides_copy_ev);
            cgh.host_task([contig_src_strides_shp]() {
                // Capturing shared pointer ensure it is freed after its data
                // are copied into packed USM vector
            });
        });
    }
    else {
        src_strides_copy_ev = exec_q.copy<py::ssize_t>(
            src_strides, packed_shapes_strides + src_nd, src_nd);
    }

    const py::ssize_t *dst_strides = dst.get_strides_raw();
    sycl::event dst_strides_copy_ev;
    if (dst_strides == nullptr) {
        using shT = std::vector<py::ssize_t>;
        int dst_flags = dst.get_flags();
        std::shared_ptr<shT> contig_dst_strides_shp;
        if (dst_flags & USM_ARRAY_C_CONTIGUOUS) {
            contig_dst_strides_shp =
                std::make_shared<shT>(c_contiguous_strides(dst_nd, dst_shape));
        }
        else if (dst_flags & USM_ARRAY_F_CONTIGUOUS) {
            contig_dst_strides_shp =
                std::make_shared<shT>(f_contiguous_strides(dst_nd, dst_shape));
        }
        else {
            sycl::event::wait(
                {src_shape_copy_ev, dst_shape_copy_ev, src_strides_copy_ev});
            sycl::free(packed_shapes_strides, exec_q);
            throw std::runtime_error(
                "Invalid dst array encountered: in copy_for_reshape function");
        }
        dst_strides_copy_ev = exec_q.copy<py::ssize_t>(
            contig_dst_strides_shp->data(),
            packed_shapes_strides + 2 * src_nd + dst_nd, dst_nd);
        exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(dst_strides_copy_ev);
            cgh.host_task([contig_dst_strides_shp]() {
                // Capturing shared pointer ensure it is freed after its data
                // are copied into packed USM vector
            });
        });
    }
    else {
        dst_strides_copy_ev = exec_q.copy<py::ssize_t>(
            dst_strides, packed_shapes_strides + 2 * src_nd + dst_nd, dst_nd);
    }

    char *src_data = src.get_data();
    char *dst_data = dst.get_data();

    sycl::event copy_for_reshape_event =
        fn(exec_q, shift, src_nelems, src_nd, dst_nd, packed_shapes_strides,
           src_data, dst_data, depends);

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
}
