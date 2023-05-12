//===----------- Implementation of _tensor_impl module  ---------*-C++-*-/===//
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
/// This file defines functions of dpctl.tensor._tensor_impl extensions,
/// specifically functions for elementwise operations.
//===----------------------------------------------------------------------===//

#include "dpctl4pybind11.hpp"
#include <CL/sycl.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "elementwise_functions.hpp"
#include "utils/type_dispatch.hpp"

#include "kernels/elementwise_functions/abs.hpp"
#include "kernels/elementwise_functions/add.hpp"
#include "kernels/elementwise_functions/cos.hpp"
#include "kernels/elementwise_functions/isfinite.hpp"
#include "kernels/elementwise_functions/isinf.hpp"
#include "kernels/elementwise_functions/isnan.hpp"
#include "kernels/elementwise_functions/sqrt.hpp"

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

namespace td_ns = dpctl::tensor::type_dispatch;

py::dtype _dtype_from_typenum(td_ns::typenum_t dst_typenum_t)
{
    switch (dst_typenum_t) {
    case td_ns::typenum_t::BOOL:
        return py::dtype("?");
    case td_ns::typenum_t::INT8:
        return py::dtype("i1");
    case td_ns::typenum_t::UINT8:
        return py::dtype("u1");
    case td_ns::typenum_t::INT16:
        return py::dtype("i2");
    case td_ns::typenum_t::UINT16:
        return py::dtype("u2");
    case td_ns::typenum_t::INT32:
        return py::dtype("i4");
    case td_ns::typenum_t::UINT32:
        return py::dtype("u4");
    case td_ns::typenum_t::INT64:
        return py::dtype("i8");
    case td_ns::typenum_t::UINT64:
        return py::dtype("u8");
    case td_ns::typenum_t::HALF:
        return py::dtype("f2");
    case td_ns::typenum_t::FLOAT:
        return py::dtype("f4");
    case td_ns::typenum_t::DOUBLE:
        return py::dtype("f8");
    case td_ns::typenum_t::CFLOAT:
        return py::dtype("c8");
    case td_ns::typenum_t::CDOUBLE:
        return py::dtype("c16");
    default:
        throw py::value_error("Unrecognized dst_typeid");
    }
}

int _result_typeid(int arg_typeid, const int *fn_output_id)
{
    if (arg_typeid < 0 || arg_typeid >= td_ns::num_types) {
        throw py::value_error("Input typeid " + std::to_string(arg_typeid) +
                              " is outside of expected bounds.");
    }

    return fn_output_id[arg_typeid];
}

// ABS
namespace impl
{

namespace abs_fn_ns = dpctl::tensor::kernels::abs;
using abs_fn_ns::abs_contig_impl_fn_ptr_t;
using abs_fn_ns::abs_strided_impl_fn_ptr_t;

static abs_contig_impl_fn_ptr_t abs_contig_dispatch_vector[td_ns::num_types];
static int abs_output_typeid_vector[td_ns::num_types];
static abs_strided_impl_fn_ptr_t abs_strided_dispatch_vector[td_ns::num_types];

void populate_abs_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = abs_fn_ns;

    using fn_ns::AbsContigFactory;
    DispatchVectorBuilder<abs_contig_impl_fn_ptr_t, AbsContigFactory, num_types>
        dvb1;
    dvb1.populate_dispatch_vector(abs_contig_dispatch_vector);

    using fn_ns::AbsStridedFactory;
    DispatchVectorBuilder<abs_strided_impl_fn_ptr_t, AbsStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(abs_strided_dispatch_vector);

    using fn_ns::AbsTypeMapFactory;
    DispatchVectorBuilder<int, AbsTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(abs_output_typeid_vector);
};

} // namespace impl

// ISFINITE
namespace impl
{
namespace isfinite_fn_ns = dpctl::tensor::kernels::isfinite;
using isfinite_fn_ns::isfinite_contig_impl_fn_ptr_t;
using isfinite_fn_ns::isfinite_strided_impl_fn_ptr_t;

static isfinite_contig_impl_fn_ptr_t
    isfinite_contig_dispatch_vector[td_ns::num_types];
static int isfinite_output_typeid_vector[td_ns::num_types];
static isfinite_strided_impl_fn_ptr_t
    isfinite_strided_dispatch_vector[td_ns::num_types];

void populate_isfinite_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = isfinite_fn_ns;

    using fn_ns::IsFiniteContigFactory;
    DispatchVectorBuilder<isfinite_contig_impl_fn_ptr_t, IsFiniteContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(isfinite_contig_dispatch_vector);

    using fn_ns::IsFiniteStridedFactory;
    DispatchVectorBuilder<isfinite_strided_impl_fn_ptr_t,
                          IsFiniteStridedFactory, num_types>
        dvb2;
    dvb2.populate_dispatch_vector(isfinite_strided_dispatch_vector);

    using fn_ns::IsFiniteTypeMapFactory;
    DispatchVectorBuilder<int, IsFiniteTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(isfinite_output_typeid_vector);
}

} // namespace impl

// ISINF
namespace impl
{
namespace isinf_fn_ns = dpctl::tensor::kernels::isinf;
using isinf_fn_ns::isinf_contig_impl_fn_ptr_t;
using isinf_fn_ns::isinf_strided_impl_fn_ptr_t;

static isinf_contig_impl_fn_ptr_t
    isinf_contig_dispatch_vector[td_ns::num_types];
static int isinf_output_typeid_vector[td_ns::num_types];
static isinf_strided_impl_fn_ptr_t
    isinf_strided_dispatch_vector[td_ns::num_types];

void populate_isinf_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = isinf_fn_ns;

    using fn_ns::IsInfContigFactory;
    DispatchVectorBuilder<isinf_contig_impl_fn_ptr_t, IsInfContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(isinf_contig_dispatch_vector);

    using fn_ns::IsInfStridedFactory;
    DispatchVectorBuilder<isinf_strided_impl_fn_ptr_t, IsInfStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(isinf_strided_dispatch_vector);

    using fn_ns::IsInfTypeMapFactory;
    DispatchVectorBuilder<int, IsInfTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(isinf_output_typeid_vector);
}

} // namespace impl

// ISNAN
namespace impl
{
namespace isnan_fn_ns = dpctl::tensor::kernels::isnan;
using isnan_fn_ns::isnan_contig_impl_fn_ptr_t;
using isnan_fn_ns::isnan_strided_impl_fn_ptr_t;

static isnan_contig_impl_fn_ptr_t
    isnan_contig_dispatch_vector[td_ns::num_types];
static int isnan_output_typeid_vector[td_ns::num_types];
static isnan_strided_impl_fn_ptr_t
    isnan_strided_dispatch_vector[td_ns::num_types];

void populate_isnan_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = isnan_fn_ns;

    using fn_ns::IsNanContigFactory;
    DispatchVectorBuilder<isnan_contig_impl_fn_ptr_t, IsNanContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(isnan_contig_dispatch_vector);

    using fn_ns::IsNanStridedFactory;
    DispatchVectorBuilder<isnan_strided_impl_fn_ptr_t, IsNanStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(isnan_strided_dispatch_vector);

    using fn_ns::IsNanTypeMapFactory;
    DispatchVectorBuilder<int, IsNanTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(isnan_output_typeid_vector);
}

} // namespace impl

// COS
namespace impl
{

namespace cos_fn_ns = dpctl::tensor::kernels::cos;
using cos_fn_ns::cos_contig_impl_fn_ptr_t;
using cos_fn_ns::cos_strided_impl_fn_ptr_t;

static cos_contig_impl_fn_ptr_t cos_contig_dispatch_vector[td_ns::num_types];
static int cos_output_typeid_vector[td_ns::num_types];
static cos_strided_impl_fn_ptr_t cos_strided_dispatch_vector[td_ns::num_types];

void populate_cos_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = cos_fn_ns;

    using fn_ns::CosContigFactory;
    DispatchVectorBuilder<cos_contig_impl_fn_ptr_t, CosContigFactory, num_types>
        dvb1;
    dvb1.populate_dispatch_vector(cos_contig_dispatch_vector);

    using fn_ns::CosStridedFactory;
    DispatchVectorBuilder<cos_strided_impl_fn_ptr_t, CosStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(cos_strided_dispatch_vector);

    using fn_ns::CosTypeMapFactory;
    DispatchVectorBuilder<int, CosTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(cos_output_typeid_vector);
}

} // namespace impl

// ADD

namespace impl
{
namespace fn_ns = dpctl::tensor::kernels::add;

using fn_ns::add_contig_impl_fn_ptr_t;
using fn_ns::add_contig_matrix_contig_row_broadcast_impl_fn_ptr_t;
using fn_ns::add_strided_impl_fn_ptr_t;

static add_contig_impl_fn_ptr_t add_contig_dispatch_table[td_ns::num_types]
                                                         [td_ns::num_types];
static int add_output_id_table[td_ns::num_types][td_ns::num_types];

static add_strided_impl_fn_ptr_t add_strided_dispatch_table[td_ns::num_types]
                                                           [td_ns::num_types];

static add_contig_matrix_contig_row_broadcast_impl_fn_ptr_t
    add_contig_matrix_contig_row_broadcast_dispatch_table[td_ns::num_types]
                                                         [td_ns::num_types];

void populate_add_dispatch_tables(void)
{
    using namespace td_ns;

    using fn_ns::AddContigFactory;
    DispatchTableBuilder<add_contig_impl_fn_ptr_t, AddContigFactory, num_types>
        dtb1;
    dtb1.populate_dispatch_table(add_contig_dispatch_table);

    using fn_ns::AddStridedFactory;
    DispatchTableBuilder<add_strided_impl_fn_ptr_t, AddStridedFactory,
                         num_types>
        dtb2;
    dtb2.populate_dispatch_table(add_strided_dispatch_table);

    using fn_ns::AddTypeMapFactory;
    DispatchTableBuilder<int, AddTypeMapFactory, num_types> dtb3;
    dtb3.populate_dispatch_table(add_output_id_table);

    using fn_ns::AddContigMatrixContigRowBroadcastFactory;
    DispatchTableBuilder<add_contig_matrix_contig_row_broadcast_impl_fn_ptr_t,
                         AddContigMatrixContigRowBroadcastFactory, num_types>
        dtb4;
    dtb4.populate_dispatch_table(
        add_contig_matrix_contig_row_broadcast_dispatch_table);
};

} // namespace impl

// SQRT
namespace impl
{

namespace sqrt_fn_ns = dpctl::tensor::kernels::sqrt;
using sqrt_fn_ns::sqrt_contig_impl_fn_ptr_t;
using sqrt_fn_ns::sqrt_strided_impl_fn_ptr_t;

static sqrt_contig_impl_fn_ptr_t sqrt_contig_dispatch_vector[td_ns::num_types];
static int sqrt_output_typeid_vector[td_ns::num_types];
static sqrt_strided_impl_fn_ptr_t
    sqrt_strided_dispatch_vector[td_ns::num_types];

void populate_sqrt_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = sqrt_fn_ns;

    using fn_ns::SqrtContigFactory;
    DispatchVectorBuilder<sqrt_contig_impl_fn_ptr_t, SqrtContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(sqrt_contig_dispatch_vector);

    using fn_ns::SqrtStridedFactory;
    DispatchVectorBuilder<sqrt_strided_impl_fn_ptr_t, SqrtStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(sqrt_strided_dispatch_vector);

    using fn_ns::SqrtTypeMapFactory;
    DispatchVectorBuilder<int, SqrtTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(sqrt_output_typeid_vector);
}

} // namespace impl

namespace py = pybind11;

void init_elementwise_functions(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;

    // U01: ==== ABS   (x)
    {
        impl::populate_abs_dispatch_vectors();
        using impl::abs_contig_dispatch_vector;
        using impl::abs_output_typeid_vector;
        using impl::abs_strided_dispatch_vector;

        auto abs_pyapi = [&](arrayT src, arrayT dst, sycl::queue exec_q,
                             const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, abs_output_typeid_vector,
                abs_contig_dispatch_vector, abs_strided_dispatch_vector);
        };
        m.def("_abs", abs_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto abs_result_type_pyapi = [&](py::dtype dtype) {
            return py_unary_ufunc_result_type(dtype, abs_output_typeid_vector);
        };
        m.def("_abs_result_type", abs_result_type_pyapi);
    }

    // U02: ==== ACOS   (x)
    // FIXME:

    // U03: ===== ACOSH (x)
    // FIXME:

    // B01: ===== ADD   (x1, x2)
    {
        impl::populate_add_dispatch_tables();
        using impl::add_contig_dispatch_table;
        using impl::add_contig_matrix_contig_row_broadcast_dispatch_table;
        using impl::add_output_id_table;
        using impl::add_strided_dispatch_table;

        auto add_pyapi = [&](dpctl::tensor::usm_ndarray src1,
                             dpctl::tensor::usm_ndarray src2,
                             dpctl::tensor::usm_ndarray dst, sycl::queue exec_q,
                             const std::vector<sycl::event> &depends = {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends, add_output_id_table,
                // function pointers to handle operation on contiguous arrays
                // (pointers may be nullptr)
                add_contig_dispatch_table,
                // function pointers to handle operation on strided arrays (most
                // general case)
                add_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                add_contig_matrix_contig_row_broadcast_dispatch_table);
        };
        auto add_result_type_pyapi = [&](py::dtype dtype1, py::dtype dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               add_output_id_table);
        };
        m.def("_add", add_pyapi, "", py::arg("src1"), py::arg("src2"),
              py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_add_result_type", add_result_type_pyapi, "");
    }

    // U04: ===== ASIN  (x)
    // FIXME:

    // U05: ===== ASINH (x)
    // FIXME:

    // U06: ===== ATAN  (x)
    // FIXME:

    // B02: ===== ATAN2 (x1, x2)
    // FIXME:

    // U07: ===== ATANH (x)
    // FIXME:

    // B03: ===== BITWISE_AND           (x1, x2)
    // FIXME:

    // B04: ===== BITWISE_LEFT_SHIFT    (x1, x2)
    // FIXME:

    // U08: ===== BITWISE_INVERT        (x)
    // FIXME:

    // B05: ===== BITWISE_OR            (x1, x2)
    // FIXME:

    // B06: ===== BITWISE_RIGHT_SHIFT   (x1, x2)
    // FIXME:

    // B07: ===== BITWISE_XOR           (x1, x2)
    // FIXME:

    // U09: ==== CEIL          (x)
    // FIXME:

    // U10: ==== CONJ          (x)
    // FIXME:

    // U11: ==== COS           (x)
    {
        impl::populate_cos_dispatch_vectors();
        using impl::cos_contig_dispatch_vector;
        using impl::cos_output_typeid_vector;
        using impl::cos_strided_dispatch_vector;

        auto cos_pyapi = [&](arrayT src, arrayT dst, sycl::queue exec_q,
                             const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, cos_output_typeid_vector,
                cos_contig_dispatch_vector, cos_strided_dispatch_vector);
        };
        m.def("_cos", cos_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto cos_result_type_pyapi = [&](py::dtype dtype) {
            return py_unary_ufunc_result_type(dtype, cos_output_typeid_vector);
        };
        m.def("_cos_result_type", cos_result_type_pyapi);
    }

    // U12: ==== COSH          (x)
    // FIXME:

    // B08: ==== DIVIDE        (x1, x2)
    // FIXME:

    // B09: ==== EQUAL         (x1, x2)
    // FIXME:

    // U13: ==== EXP           (x)
    // FIXME:

    // U14: ==== EXPM1         (x)
    // FIXME:

    // U15: ==== FLOOR         (x)
    // FIXME:

    // B10: ==== FLOOR_DIVIDE  (x1, x2)
    // FIXME:

    // B11: ==== GREATER       (x1, x2)
    // FIXME:

    // B12: ==== GREATER_EQUAL (x1, x2)
    // FIXME:

    // U16: ==== IMAG        (x)
    // FIXME:

    // U17: ==== ISFINITE    (x)
    {
        impl::populate_isfinite_dispatch_vectors();

        using impl::isfinite_contig_dispatch_vector;
        using impl::isfinite_output_typeid_vector;
        using impl::isfinite_strided_dispatch_vector;
        auto isfinite_pyapi =
            [&](dpctl::tensor::usm_ndarray src, dpctl::tensor::usm_ndarray dst,
                sycl::queue exec_q,
                const std::vector<sycl::event> &depends = {}) {
                return py_unary_ufunc(src, dst, exec_q, depends,
                                      isfinite_output_typeid_vector,
                                      isfinite_contig_dispatch_vector,
                                      isfinite_strided_dispatch_vector);
            };
        auto isfinite_result_type_pyapi = [&](py::dtype dtype) {
            return py_unary_ufunc_result_type(dtype,
                                              isfinite_output_typeid_vector);
        };
        m.def("_isfinite", isfinite_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());
        m.def("_isfinite_result_type", isfinite_result_type_pyapi, "");
    }

    // U18: ==== ISINF       (x)
    {
        impl::populate_isinf_dispatch_vectors();

        using impl::isinf_contig_dispatch_vector;
        using impl::isinf_output_typeid_vector;
        using impl::isinf_strided_dispatch_vector;
        auto isinf_pyapi = [&](dpctl::tensor::usm_ndarray src,
                               dpctl::tensor::usm_ndarray dst,
                               sycl::queue exec_q,
                               const std::vector<sycl::event> &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, isinf_output_typeid_vector,
                isinf_contig_dispatch_vector, isinf_strided_dispatch_vector);
        };
        auto isinf_result_type_pyapi = [&](py::dtype dtype) {
            return py_unary_ufunc_result_type(dtype,
                                              isinf_output_typeid_vector);
        };
        m.def("_isinf", isinf_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());
        m.def("_isinf_result_type", isinf_result_type_pyapi, "");
    }

    // U19: ==== ISNAN       (x)
    {
        impl::populate_isnan_dispatch_vectors();

        using impl::isnan_contig_dispatch_vector;
        using impl::isnan_output_typeid_vector;
        using impl::isnan_strided_dispatch_vector;
        auto isnan_pyapi = [&](dpctl::tensor::usm_ndarray src,
                               dpctl::tensor::usm_ndarray dst,
                               sycl::queue exec_q,
                               const std::vector<sycl::event> &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, isnan_output_typeid_vector,
                isnan_contig_dispatch_vector, isnan_strided_dispatch_vector);
        };
        auto isnan_result_type_pyapi = [&](py::dtype dtype) {
            return py_unary_ufunc_result_type(dtype,
                                              isnan_output_typeid_vector);
        };
        m.def("_isnan", isnan_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());
        m.def("_isnan_result_type", isnan_result_type_pyapi, "");
    }

    // B13: ==== LESS        (x1, x2)
    // FIXME:

    // B14: ==== LESS_EQUAL  (x1, x2)
    // FIXME:

    // U20: ==== LOG         (x)
    // FIXME:

    // U21: ==== LOG1P       (x)
    // FIXME:

    // U22: ==== LOG2        (x)
    // FIXME:

    // U23: ==== LOG10       (x)
    // FIXME:

    // B15: ==== LOGADDEXP   (x1, x2)
    // FIXME:

    // B16: ==== LOGICAL_AND (x1, x2)
    // FIXME:

    // U24: ==== LOGICAL_NOT (x)
    // FIXME:

    // B17: ==== LOGICAL_OR  (x1, x2)
    // FIXME:

    // B18: ==== LOGICAL_XOR (x1, x2)
    // FIXME:

    // B19: ==== MULTIPLY    (x1, x2)
    // FIXME:

    // U25: ==== NEGATIVE    (x)
    // FIXME:

    // B20: ==== NOT_EQUAL   (x1, x2)
    // FIXME:

    // U26: ==== POSITIVE    (x)
    // FIXME:

    // B21: ==== POW         (x1, x2)
    // FIXME:

    // U27: ==== REAL        (x)
    // FIXME:

    // B22: ==== REMAINDER   (x1, x2)
    // FIXME:

    // U28: ==== ROUND       (x)
    // FIXME:

    // U29: ==== SIGN        (x)
    // FIXME:

    // U30: ==== SIN         (x)
    // FIXME:

    // U31: ==== SINH        (x)
    // FIXME:

    // U32: ==== SQUARE      (x)
    // FIXME:

    // U33: ==== SQRT        (x)
    {
        impl::populate_sqrt_dispatch_vectors();
        using impl::sqrt_contig_dispatch_vector;
        using impl::sqrt_output_typeid_vector;
        using impl::sqrt_strided_dispatch_vector;

        auto sqrt_pyapi = [&](arrayT src, arrayT dst, sycl::queue exec_q,
                              const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, sqrt_output_typeid_vector,
                sqrt_contig_dispatch_vector, sqrt_strided_dispatch_vector);
        };
        m.def("_sqrt", sqrt_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto sqrt_result_type_pyapi = [&](py::dtype dtype) {
            return py_unary_ufunc_result_type(dtype, sqrt_output_typeid_vector);
        };
        m.def("_sqrt_result_type", sqrt_result_type_pyapi);
    }

    // B23: ==== SUBTRACT    (x1, x2)
    // FIXME:

    // U34: ==== TAN         (x)
    // FIXME:

    // U35: ==== TANH        (x)
    // FIXME:

    // U36: ==== TRUNC       (x)
    // FIXME:
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
