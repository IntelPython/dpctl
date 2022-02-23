//===----- tensor_py.cpp - Implementation of _tensor_impl module  ----*-C++-*-
//===//
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

#include "dpctl4pybind11.hpp"
#include "utils/strided_iters.hpp"
#include <CL/sycl.hpp>
#include <complex>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <thread>
#include <type_traits>

namespace py = pybind11;

namespace
{
enum typenum_t : int
{
    BOOL = 0,
    INT8, // 1
    UINT8,
    INT16,
    UINT16,
    INT32, // 5
    UINT32,
    INT64,
    UINT64,
    HALF,
    FLOAT, // 10
    DOUBLE,
    CFLOAT,
    CDOUBLE, // 13
};
constexpr int num_types = 14; // number of elements in typenum_t

template <typename funcPtrT,
          template <typename fnT, typename D, typename S>
          typename factory,
          int num_types>
class DispatchTableBuilder
{
private:
    template <typename dstTy>
    const std::vector<funcPtrT> row_per_dst_type() const
    {
        std::vector<funcPtrT> per_dstTy = {
            factory<funcPtrT, dstTy, bool>{}.get(),
            factory<funcPtrT, dstTy, int8_t>{}.get(),
            factory<funcPtrT, dstTy, uint8_t>{}.get(),
            factory<funcPtrT, dstTy, int16_t>{}.get(),
            factory<funcPtrT, dstTy, uint16_t>{}.get(),
            factory<funcPtrT, dstTy, int32_t>{}.get(),
            factory<funcPtrT, dstTy, uint32_t>{}.get(),
            factory<funcPtrT, dstTy, int64_t>{}.get(),
            factory<funcPtrT, dstTy, uint64_t>{}.get(),
            factory<funcPtrT, dstTy, sycl::half>{}.get(),
            factory<funcPtrT, dstTy, float>{}.get(),
            factory<funcPtrT, dstTy, double>{}.get(),
            factory<funcPtrT, dstTy, std::complex<float>>{}.get(),
            factory<funcPtrT, dstTy, std::complex<double>>{}.get()};
        return per_dstTy;
    }

public:
    DispatchTableBuilder() = default;
    ~DispatchTableBuilder() = default;

    void populate_dispatch_table(funcPtrT table[][num_types]) const
    {
        const auto map_by_dst_type = {row_per_dst_type<bool>(),
                                      row_per_dst_type<int8_t>(),
                                      row_per_dst_type<uint8_t>(),
                                      row_per_dst_type<int16_t>(),
                                      row_per_dst_type<uint16_t>(),
                                      row_per_dst_type<int32_t>(),
                                      row_per_dst_type<uint32_t>(),
                                      row_per_dst_type<int64_t>(),
                                      row_per_dst_type<uint64_t>(),
                                      row_per_dst_type<sycl::half>(),
                                      row_per_dst_type<float>(),
                                      row_per_dst_type<double>(),
                                      row_per_dst_type<std::complex<float>>(),
                                      row_per_dst_type<std::complex<double>>()};
        int dst_id = 0;
        for (auto &row : map_by_dst_type) {
            int src_id = 0;
            for (auto &fn_ptr : row) {
                table[dst_id][src_id] = fn_ptr;
                ++src_id;
            }
            ++dst_id;
        }
    }
};

// Lookup a type according to its size, and return a value corresponding to the
// NumPy typenum.
template <typename Concrete> constexpr int platform_typeid_lookup()
{
    return -1;
}

template <typename Concrete, typename T, typename... Ts, typename... Ints>
constexpr int platform_typeid_lookup(int I, Ints... Is)
{
    return sizeof(Concrete) == sizeof(T)
               ? I
               : platform_typeid_lookup<Concrete, Ts...>(Is...);
}

// Platform-dependent normalization
int UAR_INT8 = -1;
int UAR_UINT8 = -1;
int UAR_INT16 = -1;
int UAR_UINT16 = -1;
int UAR_INT32 = -1;
int UAR_UINT32 = -1;
int UAR_INT64 = -1;
int UAR_UINT64 = -1;

typenum_t typenum_to_lookup_id(int typenum)
{
    if (typenum == UAR_DOUBLE) {
        return typenum_t::DOUBLE;
    }
    else if (typenum == UAR_INT64) {
        return typenum_t::INT64;
    }
    else if (typenum == UAR_INT32) {
        return typenum_t::INT32;
    }
    else if (typenum == UAR_BOOL) {
        return typenum_t::BOOL;
    }
    else if (typenum == UAR_CDOUBLE) {
        return typenum_t::CDOUBLE;
    }
    else if (typenum == UAR_FLOAT) {
        return typenum_t::FLOAT;
    }
    else if (typenum == UAR_INT16) {
        return typenum_t::INT16;
    }
    else if (typenum == UAR_INT8) {
        return typenum_t::INT8;
    }
    else if (typenum == UAR_UINT64) {
        return typenum_t::UINT64;
    }
    else if (typenum == UAR_UINT32) {
        return typenum_t::UINT32;
    }
    else if (typenum == UAR_UINT16) {
        return typenum_t::UINT16;
    }
    else if (typenum == UAR_UINT8) {
        return typenum_t::UINT8;
    }
    else if (typenum == UAR_CFLOAT) {
        return typenum_t::CFLOAT;
    }
    else if (typenum == UAR_HALF) {
        return typenum_t::HALF;
    }
    else {
        throw std::runtime_error("Unrecogized typenum " +
                                 std::to_string(typenum) + " encountered.");
    }
}

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

public:
    GenericCopyFunctor(char *src_cp,
                       char *dst_cp,
                       py::ssize_t *shape_strides,
                       int nd)
        : src_(src_cp), dst_(dst_cp), shape_strides_(shape_strides), nd_(nd)
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
        fn(src_, src_offset, dst_, dst_offset);
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

public:
    NDSpecializedCopyFunctor(char *src_cp, // USM pointer
                             char *dst_cp, // USM pointer
                             const std::array<py::ssize_t, nd> shape,
                             const std::array<py::ssize_t, nd> src_strides,
                             const std::array<py::ssize_t, nd> dst_strides)
        : src_(src_cp), dst_(dst_cp), indxr(shape), src_strides_(src_strides),
          dst_strides_(dst_strides)
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
        fn(src_, src_offset, dst_, dst_offset);
    }
};

template <typename srcT, typename dstT> class copy_cast_generic_kernel;

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
                src_p + src_offset * sizeof(srcTy),
                dst_p + dst_offset * sizeof(dstTy), shape_and_strides, nd));
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

template <typename srcT, typename dstT, int nd> class copy_cast_spec_kernel;

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
                src_p + src_offset * sizeof(srcTy),
                dst_p + dst_offset * sizeof(dstTy), shape, src_strides,
                dst_strides));
    });

    return copy_and_cast_ev;
}

static copy_and_cast_generic_fn_ptr_t
    copy_and_cast_generic_dispatch_table[num_types][num_types];
static copy_and_cast_1d_fn_ptr_t copy_and_cast_1d_dispatch_table[num_types]
                                                                [num_types];
static copy_and_cast_2d_fn_ptr_t copy_and_cast_2d_dispatch_table[num_types]
                                                                [num_types];

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

} // namespace

namespace
{

using vecT = std::vector<py::ssize_t>;
std::tuple<vecT, vecT, py::size_t> contract_iter(vecT shape, vecT strides)
{
    const size_t dim = shape.size();
    if (dim != strides.size()) {
        throw py::value_error("Shape and strides must be of equal size.");
    }
    vecT out_shape = shape;
    vecT out_strides = strides;
    py::ssize_t disp(0);

    int nd = simplify_iteration_stride(dim, out_shape.data(),
                                       out_strides.data(), disp);
    out_shape.resize(nd);
    out_strides.resize(nd);
    return std::make_tuple(out_shape, out_strides, disp);
}

std::tuple<vecT, vecT, py::size_t, vecT, py::ssize_t>
contract_iter2(vecT shape, vecT strides1, vecT strides2)
{
    const size_t dim = shape.size();
    if (dim != strides1.size() || dim != strides2.size()) {
        throw py::value_error("Shape and strides must be of equal size.");
    }
    vecT out_shape = shape;
    vecT out_strides1 = strides1;
    vecT out_strides2 = strides2;
    py::ssize_t disp1(0);
    py::ssize_t disp2(0);

    int nd = simplify_iteration_two_strides(dim, out_shape.data(),
                                            out_strides1.data(),
                                            out_strides2.data(), disp1, disp2);
    out_shape.resize(nd);
    out_strides1.resize(nd);
    out_strides2.resize(nd);
    return std::make_tuple(out_shape, out_strides1, disp1, out_strides2, disp2);
}

bool usm_ndarray_check_(py::object o)
{
    PyObject *raw_o = o.ptr();
    return ((raw_o != nullptr) &&
            static_cast<bool>(PyObject_TypeCheck(raw_o, &PyUSMArrayType)));
}

int usm_ndarray_ndim_(py::object ar)
{
    PyObject *raw_o = ar.ptr();
    PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

    return UsmNDArray_GetNDim(raw_ar);
}

const py::ssize_t *usm_ndarray_shape_(py::object ar)
{
    PyObject *raw_o = ar.ptr();
    PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

    return UsmNDArray_GetShape(raw_ar);
}

const py::ssize_t *usm_ndarray_strides_(py::object ar)
{
    PyObject *raw_o = ar.ptr();
    PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

    return UsmNDArray_GetStrides(raw_ar);
}

std::pair<py::ssize_t, py::ssize_t> usm_ndarray_minmax_offsets_(py::object ar)
{
    PyObject *raw_o = ar.ptr();
    PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

    int nd = UsmNDArray_GetNDim(raw_ar);
    const py::ssize_t *shape = UsmNDArray_GetShape(raw_ar);
    const py::ssize_t *strides = UsmNDArray_GetStrides(raw_ar);

    py::ssize_t offset_min = 0;
    py::ssize_t offset_max = 0;
    if (strides == nullptr) {
        py::ssize_t stride(1);
        for (int i = 0; i < nd; ++i) {
            offset_max += stride * (shape[i] - 1);
            stride *= shape[i];
        }
    }
    else {
        offset_min = UsmNDArray_GetOffset(raw_ar);
        offset_max = offset_min;
        for (int i = 0; i < nd; ++i) {
            py::ssize_t delta = strides[i] * (shape[i] - 1);
            if (strides[i] > 0) {
                offset_max += delta;
            }
            else {
                offset_min += delta;
            }
        }
    }
    return std::make_pair(offset_min, offset_max);
}

sycl::queue usm_ndarray_get_queue_(py::object ar)
{
    PyObject *raw_o = ar.ptr();
    PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

    DPCTLSyclQueueRef QRef = UsmNDArray_GetQueueRef(raw_ar);
    return *(reinterpret_cast<sycl::queue *>(QRef));
}

int usm_ndarray_get_typenum_(py::object ar)
{
    PyObject *raw_o = ar.ptr();
    PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

    return UsmNDArray_GetTypenum(raw_ar);
}

int usm_ndarray_get_flags_(py::object ar)
{
    PyObject *raw_o = ar.ptr();
    PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

    return UsmNDArray_GetFlags(raw_ar);
}

int usm_ndarray_get_elemsize_(py::object ar)
{
    PyObject *raw_o = ar.ptr();
    PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

    return UsmNDArray_GetElementSize(raw_ar);
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

char *usm_ndarray_get_data_(py::object ar)
{
    PyObject *raw_o = ar.ptr();
    PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

    return UsmNDArray_GetData(raw_ar);
}

sycl::event keep_args_alive(sycl::queue q,
                            py::object o1,
                            py::object o2,
                            const std::vector<sycl::event> &depends = {})
{
    sycl::event host_task_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        o1.inc_ref();
        o2.inc_ref();
        cgh.host_task([=]() {
            if (Py_IsInitialized() && !_Py_IsFinalizing()) {
                PyGILState_STATE gstate;
                gstate = PyGILState_Ensure();
                o1.dec_ref();
                o2.dec_ref();
                PyGILState_Release(gstate);
            }
        });
    });

    return host_task_ev;
}

std::pair<sycl::event, sycl::event>
copy_usm_ndarray_into_usm_ndarray(py::object src,
                                  py::object dst,
                                  sycl::queue exec_q,
                                  const std::vector<sycl::event> &depends = {})
{
    if (!usm_ndarray_check_(src) || !usm_ndarray_check_(dst)) {
        throw py::type_error("Arguments of type usm_ndarray expected");
    }

    // array dimensions must be the same
    int src_nd = usm_ndarray_ndim_(src);
    int dst_nd = usm_ndarray_ndim_(dst);
    if (src_nd != dst_nd) {
        throw py::value_error("Array dimensions are not the same.");
    }

    // shapes must be the same
    const py::ssize_t *src_shape = usm_ndarray_shape_(src);
    const py::ssize_t *dst_shape = usm_ndarray_shape_(dst);
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
        auto dst_offsets = usm_ndarray_minmax_offsets_(dst);
        size_t range =
            static_cast<size_t>(dst_offsets.second - dst_offsets.first);
        if (range + 1 < src_nelems) {
            throw py::value_error(
                "Destination array can not accomodate all the "
                "elements of source array.");
        }
    }

    // check same contexts
    sycl::queue src_q = usm_ndarray_get_queue_(src);
    sycl::queue dst_q = usm_ndarray_get_queue_(dst);

    sycl::context exec_ctx = exec_q.get_context();
    if (src_q.get_context() != exec_ctx || dst_q.get_context() != exec_ctx) {
        throw py::value_error(
            "Execution queue context is not the same as allocation contexts");
    }

    int src_typenum = usm_ndarray_get_typenum_(src);
    int dst_typenum = usm_ndarray_get_typenum_(dst);

    int src_type_id = typenum_to_lookup_id(src_typenum);
    int dst_type_id = typenum_to_lookup_id(dst_typenum);

    {
        auto type_id_check = [](int id) -> bool {
            return ((id >= 0) && (id < num_types));
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

    // TODO: should we check can cast.
    // Currently force-cast
    bool can_cast = true;
    if (!can_cast) {
        throw py::value_error("Can not cast destinary array elements to source "
                              "array element type.");
    }

    int src_flags = usm_ndarray_get_flags_(src);
    int dst_flags = usm_ndarray_get_flags_(dst);

    // check for applicability of special cases:
    //      (same type && (both C-contiguous || both F-contiguous)
    bool both_c_contig = ((src_flags & USM_ARRAY_C_CONTIGUOUS) &&
                          (dst_flags & USM_ARRAY_C_CONTIGUOUS));
    bool both_f_contig = ((src_flags & USM_ARRAY_F_CONTIGUOUS) &&
                          (dst_flags & USM_ARRAY_F_CONTIGUOUS));
    if (both_c_contig || both_f_contig) {
        if (src_type_id == dst_type_id) {
            char *src_data = usm_ndarray_get_data_(src);
            char *dst_data = usm_ndarray_get_data_(dst);
            int src_elem_size = usm_ndarray_get_elemsize_(src);
            sycl::event copy_ev = exec_q.memcpy(
                dst_data, src_data, src_nelems * src_elem_size, depends);

            // make sure src and dst are not GC-ed before copy_ev is complete
            return std::make_pair(keep_args_alive(exec_q, src, dst, {copy_ev}),
                                  copy_ev);
        }
        // With contract_iter2 in place, there is no need to write
        // dedicated kernels for casting between contiguous arrays
    }

    const py::ssize_t *src_strides = usm_ndarray_strides_(src);
    const py::ssize_t *dst_strides = usm_ndarray_strides_(dst);

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

    char *src_data = usm_ndarray_get_data_(src);
    char *dst_data = usm_ndarray_get_data_(dst);

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

    UAR_INT8 = UAR_BYTE;
    UAR_UINT8 = UAR_UBYTE;
    UAR_INT16 = UAR_SHORT;
    UAR_UINT16 = UAR_USHORT;
    UAR_INT32 = platform_typeid_lookup<std::int32_t, long, int, short>(
        UAR_LONG, UAR_INT, UAR_SHORT);
    UAR_UINT32 =
        platform_typeid_lookup<std::uint32_t, unsigned long, unsigned int,
                               unsigned short>(UAR_ULONG, UAR_UINT, UAR_USHORT);
    UAR_INT64 = platform_typeid_lookup<std::int64_t, long, long long, int>(
        UAR_LONG, UAR_LONGLONG, UAR_INT);
    UAR_UINT64 = platform_typeid_lookup<std::uint64_t, unsigned long,
                                        unsigned long long, unsigned int>(
        UAR_ULONG, UAR_ULONGLONG, UAR_UINT);

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
