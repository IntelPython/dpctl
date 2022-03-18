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
        std::vector<py::ssize_t> c_strides(nd);
        c_strides[nd - 1] = py::ssize_t(1);
        for (int i = 1; i < nd; ++i) {
            int ic = nd - i;
            c_strides[ic - 1] = c_strides[ic] * shape[ic];
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
        std::vector<py::ssize_t> f_strides(nd);
        f_strides[0] = py::ssize_t(1);
        for (int i = 0; i < nd - 1; ++i) {
            f_strides[i + 1] = f_strides[i] * shape[i];
        }
        return f_strides;
    }
    else {
        return std::vector<py::ssize_t>();
    }
}

sycl::event keep_args_alive(sycl::queue q,
                            py::object o1,
                            py::object o2,
                            const std::vector<sycl::event> &depends = {})
{
    sycl::event host_task_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        std::shared_ptr<py::handle> shp_1 = std::make_shared<py::handle>(o1);
        std::shared_ptr<py::handle> shp_2 = std::make_shared<py::handle>(o2);
        shp_1->inc_ref();
        shp_2->inc_ref();
        cgh.host_task([=]() {
            bool guard = (Py_IsInitialized() && !_Py_IsFinalizing());
            if (guard) {
                PyGILState_STATE gstate;
                gstate = PyGILState_Ensure();
                shp_1->dec_ref();
                shp_2->dec_ref();
                PyGILState_Release(gstate);
            }
        });
    });

    return host_task_ev;
}

std::pair<sycl::event, sycl::event>
copy_usm_ndarray_into_usm_ndarray(usm_ndarray src,
                                  usm_ndarray dst,
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

    // destination must be ample enough to accomodate all elements
    {
        auto dst_offsets = dst.get_minmax_offsets();
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

    // TODO: check that arrays do not overlap, and concurrent copying is safe.
    bool memory_overlap = false;
    if (memory_overlap) {
        // TODO: could use a temporary
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
            char *src_data = src.get_data();
            char *dst_data = dst.get_data();
            int src_elem_size = src.get_elemsize();
            sycl::event copy_ev = exec_q.memcpy(
                dst_data, src_data, src_nelems * src_elem_size, depends);

            // make sure src and dst are not GC-ed before copy_ev is complete
            return std::make_pair(keep_args_alive(exec_q, src, dst, {copy_ev}),
                                  copy_ev);
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

    char *src_data = src.get_data();
    char *dst_data = dst.get_data();

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
                keep_args_alive(exec_q, src, dst, {copy_and_cast_1d_event}),
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
                keep_args_alive(exec_q, src, dst, {copy_and_cast_2d_event}),
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
                keep_args_alive(exec_q, src, dst, {copy_and_cast_0d_event}),
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
        keep_args_alive(exec_q, src, dst, {copy_and_cast_generic_ev}),
        copy_and_cast_generic_ev);
}

} // namespace

PYBIND11_MODULE(_tensor_impl, m)
{

    init_copy_and_cast_dispatch_tables();
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
          "Copies into usm_ndarray `src` from usm_ndarray `dst`.",
          py::arg("src"), py::arg("dst"),
          py::arg("queue") = py::cast<py::none>(Py_None),
          py::arg("depends") = py::list());

    m.def(
        "_contract_iter2", &contract_iter2,
        "Simplifies iteration over elements of pair of arrays of given shape "
        "with strides stride1 and stride2. Returns "
        "a 5-tuple: shape, stride and offset for the new iterator of possible "
        "smaller dimension for each array, which traverses the same elements "
        "as the original "
        "iterator, possibly in a different order.");
}
