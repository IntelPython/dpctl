//===----------- Implementation of _tensor_impl module  ---------*-C++-*-/===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2023 Intel Corporation
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
#include <utility>

#include "elementwise_functions.hpp"
#include "elementwise_functions4.hpp"
#include "utils/type_dispatch.hpp"

#include "kernels/elementwise_functions/cbrt.hpp"
#include "kernels/elementwise_functions/ceil.hpp"
#include "kernels/elementwise_functions/conj.hpp"
#include "kernels/elementwise_functions/copysign.hpp"
#include "kernels/elementwise_functions/cos.hpp"
#include "kernels/elementwise_functions/cosh.hpp"
#include "kernels/elementwise_functions/equal.hpp"
#include "kernels/elementwise_functions/exp.hpp"
#include "kernels/elementwise_functions/exp2.hpp"
#include "kernels/elementwise_functions/expm1.hpp"
#include "kernels/elementwise_functions/floor.hpp"
#include "kernels/elementwise_functions/floor_divide.hpp"
#include "kernels/elementwise_functions/greater.hpp"
#include "kernels/elementwise_functions/greater_equal.hpp"
#include "kernels/elementwise_functions/hypot.hpp"
#include "kernels/elementwise_functions/imag.hpp"
#include "kernels/elementwise_functions/isfinite.hpp"
#include "kernels/elementwise_functions/isinf.hpp"
#include "kernels/elementwise_functions/isnan.hpp"
#include "kernels/elementwise_functions/true_divide.hpp"

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

namespace td_ns = dpctl::tensor::type_dispatch;

namespace ew_cmn_ns = dpctl::tensor::kernels::elementwise_common;
using ew_cmn_ns::binary_contig_impl_fn_ptr_t;
using ew_cmn_ns::binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t;
using ew_cmn_ns::binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t;
using ew_cmn_ns::binary_strided_impl_fn_ptr_t;
using ew_cmn_ns::unary_contig_impl_fn_ptr_t;
using ew_cmn_ns::unary_strided_impl_fn_ptr_t;

using ew_cmn_ns::binary_inplace_contig_impl_fn_ptr_t;
using ew_cmn_ns::binary_inplace_row_matrix_broadcast_impl_fn_ptr_t;
using ew_cmn_ns::binary_inplace_strided_impl_fn_ptr_t;

// U09: ==== CEIL          (x)
namespace impl
{

namespace ceil_fn_ns = dpctl::tensor::kernels::ceil;

static unary_contig_impl_fn_ptr_t ceil_contig_dispatch_vector[td_ns::num_types];
static int ceil_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    ceil_strided_dispatch_vector[td_ns::num_types];

void populate_ceil_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = ceil_fn_ns;

    using fn_ns::CeilContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, CeilContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(ceil_contig_dispatch_vector);

    using fn_ns::CeilStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, CeilStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(ceil_strided_dispatch_vector);

    using fn_ns::CeilTypeMapFactory;
    DispatchVectorBuilder<int, CeilTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(ceil_output_typeid_vector);
}

} // namespace impl

// U10: ==== CONJ          (x)
namespace impl
{

namespace conj_fn_ns = dpctl::tensor::kernels::conj;

static unary_contig_impl_fn_ptr_t conj_contig_dispatch_vector[td_ns::num_types];
static int conj_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    conj_strided_dispatch_vector[td_ns::num_types];

void populate_conj_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = conj_fn_ns;

    using fn_ns::ConjContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, ConjContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(conj_contig_dispatch_vector);

    using fn_ns::ConjStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, ConjStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(conj_strided_dispatch_vector);

    using fn_ns::ConjTypeMapFactory;
    DispatchVectorBuilder<int, ConjTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(conj_output_typeid_vector);
}
} // namespace impl

// U11: ==== COS           (x)
namespace impl
{

namespace cos_fn_ns = dpctl::tensor::kernels::cos;

static unary_contig_impl_fn_ptr_t cos_contig_dispatch_vector[td_ns::num_types];
static int cos_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    cos_strided_dispatch_vector[td_ns::num_types];

void populate_cos_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = cos_fn_ns;

    using fn_ns::CosContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, CosContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(cos_contig_dispatch_vector);

    using fn_ns::CosStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, CosStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(cos_strided_dispatch_vector);

    using fn_ns::CosTypeMapFactory;
    DispatchVectorBuilder<int, CosTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(cos_output_typeid_vector);
}

} // namespace impl

// U12: ==== COSH          (x)
namespace impl
{

namespace cosh_fn_ns = dpctl::tensor::kernels::cosh;

static unary_contig_impl_fn_ptr_t cosh_contig_dispatch_vector[td_ns::num_types];
static int cosh_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    cosh_strided_dispatch_vector[td_ns::num_types];

void populate_cosh_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = cosh_fn_ns;

    using fn_ns::CoshContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, CoshContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(cosh_contig_dispatch_vector);

    using fn_ns::CoshStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, CoshStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(cosh_strided_dispatch_vector);

    using fn_ns::CoshTypeMapFactory;
    DispatchVectorBuilder<int, CoshTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(cosh_output_typeid_vector);
}

} // namespace impl

// B08: ==== DIVIDE        (x1, x2)
namespace impl
{
namespace true_divide_fn_ns = dpctl::tensor::kernels::true_divide;

static binary_contig_impl_fn_ptr_t
    true_divide_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static int true_divide_output_id_table[td_ns::num_types][td_ns::num_types];
static int true_divide_inplace_output_id_table[td_ns::num_types]
                                              [td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    true_divide_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

// divide(matrix, row)
static binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t
    true_divide_contig_matrix_contig_row_broadcast_dispatch_table
        [td_ns::num_types][td_ns::num_types];

// divide(row, matrix)
static binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t
    true_divide_contig_row_contig_matrix_broadcast_dispatch_table
        [td_ns::num_types][td_ns::num_types];

static binary_inplace_contig_impl_fn_ptr_t
    true_divide_inplace_contig_dispatch_table[td_ns::num_types]
                                             [td_ns::num_types];
static binary_inplace_strided_impl_fn_ptr_t
    true_divide_inplace_strided_dispatch_table[td_ns::num_types]
                                              [td_ns::num_types];
static binary_inplace_row_matrix_broadcast_impl_fn_ptr_t
    true_divide_inplace_row_matrix_dispatch_table[td_ns::num_types]
                                                 [td_ns::num_types];

void populate_true_divide_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = true_divide_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::TrueDivideTypeMapFactory;
    DispatchTableBuilder<int, TrueDivideTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(true_divide_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::TrueDivideStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t, TrueDivideStridedFactory,
                         num_types>
        dtb2;
    dtb2.populate_dispatch_table(true_divide_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::TrueDivideContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, TrueDivideContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(true_divide_contig_dispatch_table);

    // function pointers for operation on contiguous matrix, contiguous row
    // with contiguous matrix output
    using fn_ns::TrueDivideContigMatrixContigRowBroadcastFactory;
    DispatchTableBuilder<
        binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t,
        TrueDivideContigMatrixContigRowBroadcastFactory, num_types>
        dtb4;
    dtb4.populate_dispatch_table(
        true_divide_contig_matrix_contig_row_broadcast_dispatch_table);

    // function pointers for operation on contiguous row, contiguous matrix
    // with contiguous matrix output
    using fn_ns::TrueDivideContigRowContigMatrixBroadcastFactory;
    DispatchTableBuilder<
        binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t,
        TrueDivideContigRowContigMatrixBroadcastFactory, num_types>
        dtb5;
    dtb5.populate_dispatch_table(
        true_divide_contig_row_contig_matrix_broadcast_dispatch_table);

    // which input types are supported, and what is the type of the result
    using fn_ns::TrueDivideInplaceTypeMapFactory;
    DispatchTableBuilder<int, TrueDivideInplaceTypeMapFactory, num_types> dtb6;
    dtb6.populate_dispatch_table(true_divide_inplace_output_id_table);

    // function pointers for inplace operation on general strided arrays
    using fn_ns::TrueDivideInplaceStridedFactory;
    DispatchTableBuilder<binary_inplace_strided_impl_fn_ptr_t,
                         TrueDivideInplaceStridedFactory, num_types>
        dtb7;
    dtb7.populate_dispatch_table(true_divide_inplace_strided_dispatch_table);

    // function pointers for inplace operation on contiguous inputs and output
    using fn_ns::TrueDivideInplaceContigFactory;
    DispatchTableBuilder<binary_inplace_contig_impl_fn_ptr_t,
                         TrueDivideInplaceContigFactory, num_types>
        dtb8;
    dtb8.populate_dispatch_table(true_divide_inplace_contig_dispatch_table);

    // function pointers for inplace operation on contiguous matrix
    // and contiguous row
    using fn_ns::TrueDivideInplaceRowMatrixBroadcastFactory;
    DispatchTableBuilder<binary_inplace_row_matrix_broadcast_impl_fn_ptr_t,
                         TrueDivideInplaceRowMatrixBroadcastFactory, num_types>
        dtb9;
    dtb9.populate_dispatch_table(true_divide_inplace_row_matrix_dispatch_table);
};

} // namespace impl

// B09: ==== EQUAL         (x1, x2)
namespace impl
{
namespace equal_fn_ns = dpctl::tensor::kernels::equal;

static binary_contig_impl_fn_ptr_t
    equal_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static int equal_output_id_table[td_ns::num_types][td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    equal_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

void populate_equal_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = equal_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::EqualTypeMapFactory;
    DispatchTableBuilder<int, EqualTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(equal_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::EqualStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t, EqualStridedFactory,
                         num_types>
        dtb2;
    dtb2.populate_dispatch_table(equal_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::EqualContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, EqualContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(equal_contig_dispatch_table);
};
} // namespace impl

// U13: ==== EXP           (x)
namespace impl
{

namespace exp_fn_ns = dpctl::tensor::kernels::exp;

static unary_contig_impl_fn_ptr_t exp_contig_dispatch_vector[td_ns::num_types];
static int exp_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    exp_strided_dispatch_vector[td_ns::num_types];

void populate_exp_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = exp_fn_ns;

    using fn_ns::ExpContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, ExpContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(exp_contig_dispatch_vector);

    using fn_ns::ExpStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, ExpStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(exp_strided_dispatch_vector);

    using fn_ns::ExpTypeMapFactory;
    DispatchVectorBuilder<int, ExpTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(exp_output_typeid_vector);
}

} // namespace impl

// U14: ==== EXPM1         (x)
namespace impl
{

namespace expm1_fn_ns = dpctl::tensor::kernels::expm1;

static unary_contig_impl_fn_ptr_t
    expm1_contig_dispatch_vector[td_ns::num_types];
static int expm1_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    expm1_strided_dispatch_vector[td_ns::num_types];

void populate_expm1_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = expm1_fn_ns;

    using fn_ns::Expm1ContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, Expm1ContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(expm1_contig_dispatch_vector);

    using fn_ns::Expm1StridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, Expm1StridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(expm1_strided_dispatch_vector);

    using fn_ns::Expm1TypeMapFactory;
    DispatchVectorBuilder<int, Expm1TypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(expm1_output_typeid_vector);
}

} // namespace impl

// U15: ==== FLOOR         (x)
namespace impl
{

namespace floor_fn_ns = dpctl::tensor::kernels::floor;

static unary_contig_impl_fn_ptr_t
    floor_contig_dispatch_vector[td_ns::num_types];
static int floor_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    floor_strided_dispatch_vector[td_ns::num_types];

void populate_floor_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = floor_fn_ns;

    using fn_ns::FloorContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, FloorContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(floor_contig_dispatch_vector);

    using fn_ns::FloorStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, FloorStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(floor_strided_dispatch_vector);

    using fn_ns::FloorTypeMapFactory;
    DispatchVectorBuilder<int, FloorTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(floor_output_typeid_vector);
}

} // namespace impl

// B10: ==== FLOOR_DIVIDE  (x1, x2)
namespace impl
{
namespace floor_divide_fn_ns = dpctl::tensor::kernels::floor_divide;

static binary_contig_impl_fn_ptr_t
    floor_divide_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static int floor_divide_output_id_table[td_ns::num_types][td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    floor_divide_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

static binary_inplace_contig_impl_fn_ptr_t
    floor_divide_inplace_contig_dispatch_table[td_ns::num_types]
                                              [td_ns::num_types];
static binary_inplace_strided_impl_fn_ptr_t
    floor_divide_inplace_strided_dispatch_table[td_ns::num_types]
                                               [td_ns::num_types];

void populate_floor_divide_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = floor_divide_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::FloorDivideTypeMapFactory;
    DispatchTableBuilder<int, FloorDivideTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(floor_divide_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::FloorDivideStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t,
                         FloorDivideStridedFactory, num_types>
        dtb2;
    dtb2.populate_dispatch_table(floor_divide_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::FloorDivideContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, FloorDivideContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(floor_divide_contig_dispatch_table);

    // function pointers for inplace operation on general strided arrays
    using fn_ns::FloorDivideInplaceStridedFactory;
    DispatchTableBuilder<binary_inplace_strided_impl_fn_ptr_t,
                         FloorDivideInplaceStridedFactory, num_types>
        dtb4;
    dtb4.populate_dispatch_table(floor_divide_inplace_strided_dispatch_table);

    // function pointers for inplace operation on contiguous inputs and output
    using fn_ns::FloorDivideInplaceContigFactory;
    DispatchTableBuilder<binary_inplace_contig_impl_fn_ptr_t,
                         FloorDivideInplaceContigFactory, num_types>
        dtb5;
    dtb5.populate_dispatch_table(floor_divide_inplace_contig_dispatch_table);
};

} // namespace impl

// B11: ==== GREATER       (x1, x2)
namespace impl
{
namespace greater_fn_ns = dpctl::tensor::kernels::greater;

static binary_contig_impl_fn_ptr_t
    greater_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static int greater_output_id_table[td_ns::num_types][td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    greater_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

void populate_greater_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = greater_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::GreaterTypeMapFactory;
    DispatchTableBuilder<int, GreaterTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(greater_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::GreaterStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t, GreaterStridedFactory,
                         num_types>
        dtb2;
    dtb2.populate_dispatch_table(greater_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::GreaterContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, GreaterContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(greater_contig_dispatch_table);
};
} // namespace impl

// B12: ==== GREATER_EQUAL (x1, x2)
namespace impl
{
namespace greater_equal_fn_ns = dpctl::tensor::kernels::greater_equal;

static binary_contig_impl_fn_ptr_t
    greater_equal_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static int greater_equal_output_id_table[td_ns::num_types][td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    greater_equal_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

void populate_greater_equal_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = greater_equal_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::GreaterEqualTypeMapFactory;
    DispatchTableBuilder<int, GreaterEqualTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(greater_equal_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::GreaterEqualStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t,
                         GreaterEqualStridedFactory, num_types>
        dtb2;
    dtb2.populate_dispatch_table(greater_equal_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::GreaterEqualContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, GreaterEqualContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(greater_equal_contig_dispatch_table);
};
} // namespace impl

// U16: ==== IMAG        (x)
namespace impl
{

namespace imag_fn_ns = dpctl::tensor::kernels::imag;

static unary_contig_impl_fn_ptr_t imag_contig_dispatch_vector[td_ns::num_types];
static int imag_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    imag_strided_dispatch_vector[td_ns::num_types];

void populate_imag_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = imag_fn_ns;

    using fn_ns::ImagContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, ImagContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(imag_contig_dispatch_vector);

    using fn_ns::ImagStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, ImagStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(imag_strided_dispatch_vector);

    using fn_ns::ImagTypeMapFactory;
    DispatchVectorBuilder<int, ImagTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(imag_output_typeid_vector);
}
} // namespace impl

// U17: ==== ISFINITE    (x)
namespace impl
{
namespace isfinite_fn_ns = dpctl::tensor::kernels::isfinite;

static unary_contig_impl_fn_ptr_t
    isfinite_contig_dispatch_vector[td_ns::num_types];
static int isfinite_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    isfinite_strided_dispatch_vector[td_ns::num_types];

void populate_isfinite_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = isfinite_fn_ns;

    using fn_ns::IsFiniteContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, IsFiniteContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(isfinite_contig_dispatch_vector);

    using fn_ns::IsFiniteStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, IsFiniteStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(isfinite_strided_dispatch_vector);

    using fn_ns::IsFiniteTypeMapFactory;
    DispatchVectorBuilder<int, IsFiniteTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(isfinite_output_typeid_vector);
}

} // namespace impl

// U18: ==== ISINF       (x)
namespace impl
{
namespace isinf_fn_ns = dpctl::tensor::kernels::isinf;

static unary_contig_impl_fn_ptr_t
    isinf_contig_dispatch_vector[td_ns::num_types];
static int isinf_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    isinf_strided_dispatch_vector[td_ns::num_types];

void populate_isinf_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = isinf_fn_ns;

    using fn_ns::IsInfContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, IsInfContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(isinf_contig_dispatch_vector);

    using fn_ns::IsInfStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, IsInfStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(isinf_strided_dispatch_vector);

    using fn_ns::IsInfTypeMapFactory;
    DispatchVectorBuilder<int, IsInfTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(isinf_output_typeid_vector);
}

} // namespace impl

// U19: ==== ISNAN       (x)
namespace impl
{
namespace isnan_fn_ns = dpctl::tensor::kernels::isnan;

static unary_contig_impl_fn_ptr_t
    isnan_contig_dispatch_vector[td_ns::num_types];
static int isnan_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    isnan_strided_dispatch_vector[td_ns::num_types];

void populate_isnan_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = isnan_fn_ns;

    using fn_ns::IsNanContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, IsNanContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(isnan_contig_dispatch_vector);

    using fn_ns::IsNanStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, IsNanStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(isnan_strided_dispatch_vector);

    using fn_ns::IsNanTypeMapFactory;
    DispatchVectorBuilder<int, IsNanTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(isnan_output_typeid_vector);
}

} // namespace impl

// B24:  ==== HYPOT    (x1, x2)
namespace impl
{
namespace hypot_fn_ns = dpctl::tensor::kernels::hypot;

static binary_contig_impl_fn_ptr_t
    hypot_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static int hypot_output_id_table[td_ns::num_types][td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    hypot_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

void populate_hypot_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = hypot_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::HypotTypeMapFactory;
    DispatchTableBuilder<int, HypotTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(hypot_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::HypotStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t, HypotStridedFactory,
                         num_types>
        dtb2;
    dtb2.populate_dispatch_table(hypot_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::HypotContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, HypotContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(hypot_contig_dispatch_table);
};

} // namespace impl

// U37: ==== CBRT        (x)
namespace impl
{

namespace cbrt_fn_ns = dpctl::tensor::kernels::cbrt;

static unary_contig_impl_fn_ptr_t cbrt_contig_dispatch_vector[td_ns::num_types];
static int cbrt_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    cbrt_strided_dispatch_vector[td_ns::num_types];

void populate_cbrt_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = cbrt_fn_ns;

    using fn_ns::CbrtContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, CbrtContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(cbrt_contig_dispatch_vector);

    using fn_ns::CbrtStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, CbrtStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(cbrt_strided_dispatch_vector);

    using fn_ns::CbrtTypeMapFactory;
    DispatchVectorBuilder<int, CbrtTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(cbrt_output_typeid_vector);
}

} // namespace impl

// B24:  ==== COPYSIGN    (x1, x2)
namespace impl
{
namespace copysign_fn_ns = dpctl::tensor::kernels::copysign;

static binary_contig_impl_fn_ptr_t
    copysign_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static int copysign_output_id_table[td_ns::num_types][td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    copysign_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

void populate_copysign_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = copysign_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::CopysignTypeMapFactory;
    DispatchTableBuilder<int, CopysignTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(copysign_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::CopysignStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t, CopysignStridedFactory,
                         num_types>
        dtb2;
    dtb2.populate_dispatch_table(copysign_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::CopysignContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, CopysignContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(copysign_contig_dispatch_table);
};

} // namespace impl

// U38: ==== EXP2           (x)
namespace impl
{

namespace exp2_fn_ns = dpctl::tensor::kernels::exp2;

static unary_contig_impl_fn_ptr_t exp2_contig_dispatch_vector[td_ns::num_types];
static int exp2_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    exp2_strided_dispatch_vector[td_ns::num_types];

void populate_exp2_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = exp2_fn_ns;

    using fn_ns::Exp2ContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, Exp2ContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(exp2_contig_dispatch_vector);

    using fn_ns::Exp2StridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, Exp2StridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(exp2_strided_dispatch_vector);

    using fn_ns::Exp2TypeMapFactory;
    DispatchVectorBuilder<int, Exp2TypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(exp2_output_typeid_vector);
}

} // namespace impl

// ==========================================================================================
// //

namespace py = pybind11;

void init_elementwise_functions4(py::module_ m)
{

    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;

    // U09: ==== CEIL          (x)
    {
        impl::populate_ceil_dispatch_vectors();
        using impl::ceil_contig_dispatch_vector;
        using impl::ceil_output_typeid_vector;
        using impl::ceil_strided_dispatch_vector;

        auto ceil_pyapi = [&](const arrayT &src, const arrayT &dst,
                              sycl::queue &exec_q,
                              const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, ceil_output_typeid_vector,
                ceil_contig_dispatch_vector, ceil_strided_dispatch_vector);
        };
        m.def("_ceil", ceil_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto ceil_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype, ceil_output_typeid_vector);
        };
        m.def("_ceil_result_type", ceil_result_type_pyapi);
    }

    // U10: ==== CONJ          (x)
    {
        impl::populate_conj_dispatch_vectors();
        using impl::conj_contig_dispatch_vector;
        using impl::conj_output_typeid_vector;
        using impl::conj_strided_dispatch_vector;

        auto conj_pyapi = [&](const arrayT &src, const arrayT &dst,
                              sycl::queue &exec_q,
                              const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, conj_output_typeid_vector,
                conj_contig_dispatch_vector, conj_strided_dispatch_vector);
        };
        m.def("_conj", conj_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto conj_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype, conj_output_typeid_vector);
        };
        m.def("_conj_result_type", conj_result_type_pyapi);
    }

    // U11: ==== COS           (x)
    {
        impl::populate_cos_dispatch_vectors();
        using impl::cos_contig_dispatch_vector;
        using impl::cos_output_typeid_vector;
        using impl::cos_strided_dispatch_vector;

        auto cos_pyapi = [&](const arrayT &src, const arrayT &dst,
                             sycl::queue &exec_q,
                             const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, cos_output_typeid_vector,
                cos_contig_dispatch_vector, cos_strided_dispatch_vector);
        };
        m.def("_cos", cos_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto cos_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype, cos_output_typeid_vector);
        };
        m.def("_cos_result_type", cos_result_type_pyapi);
    }

    // U12: ==== COSH          (x)
    {
        impl::populate_cosh_dispatch_vectors();
        using impl::cosh_contig_dispatch_vector;
        using impl::cosh_output_typeid_vector;
        using impl::cosh_strided_dispatch_vector;

        auto cosh_pyapi = [&](const arrayT &src, const arrayT &dst,
                              sycl::queue &exec_q,
                              const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, cosh_output_typeid_vector,
                cosh_contig_dispatch_vector, cosh_strided_dispatch_vector);
        };
        m.def("_cosh", cosh_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto cosh_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype, cosh_output_typeid_vector);
        };
        m.def("_cosh_result_type", cosh_result_type_pyapi);
    }

    // B08: ==== DIVIDE        (x1, x2)
    {
        impl::populate_true_divide_dispatch_tables();
        using impl::true_divide_contig_dispatch_table;
        using impl::
            true_divide_contig_matrix_contig_row_broadcast_dispatch_table;
        using impl::
            true_divide_contig_row_contig_matrix_broadcast_dispatch_table;
        using impl::true_divide_output_id_table;
        using impl::true_divide_strided_dispatch_table;

        auto divide_pyapi = [&](const dpctl::tensor::usm_ndarray &src1,
                                const dpctl::tensor::usm_ndarray &src2,
                                const dpctl::tensor::usm_ndarray &dst,
                                sycl::queue &exec_q,
                                const std::vector<sycl::event> &depends = {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends, true_divide_output_id_table,
                // function pointers to handle operation on contiguous arrays
                // (pointers may be nullptr)
                true_divide_contig_dispatch_table,
                // function pointers to handle operation on strided arrays (most
                // general case)
                true_divide_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                true_divide_contig_matrix_contig_row_broadcast_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                true_divide_contig_row_contig_matrix_broadcast_dispatch_table);
        };
        auto divide_result_type_pyapi = [&](const py::dtype &dtype1,
                                            const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               true_divide_output_id_table);
        };
        m.def("_divide", divide_pyapi, "", py::arg("src1"), py::arg("src2"),
              py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_divide_result_type", divide_result_type_pyapi, "");

        using impl::true_divide_inplace_contig_dispatch_table;
        using impl::true_divide_inplace_output_id_table;
        using impl::true_divide_inplace_row_matrix_dispatch_table;
        using impl::true_divide_inplace_strided_dispatch_table;

        auto divide_inplace_pyapi =
            [&](const dpctl::tensor::usm_ndarray &src,
                const dpctl::tensor::usm_ndarray &dst, sycl::queue &exec_q,
                const std::vector<sycl::event> &depends = {}) {
                return py_binary_inplace_ufunc(
                    src, dst, exec_q, depends,
                    true_divide_inplace_output_id_table,
                    // function pointers to handle inplace operation on
                    // contiguous arrays (pointers may be nullptr)
                    true_divide_inplace_contig_dispatch_table,
                    // function pointers to handle inplace operation on strided
                    // arrays (most general case)
                    true_divide_inplace_strided_dispatch_table,
                    // function pointers to handle inplace operation on
                    // c-contig matrix with c-contig row with broadcasting
                    // (may be nullptr)
                    true_divide_inplace_row_matrix_dispatch_table);
            };
        m.def("_divide_inplace", divide_inplace_pyapi, "", py::arg("lhs"),
              py::arg("rhs"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
    }

    // B09: ==== EQUAL         (x1, x2)
    {
        impl::populate_equal_dispatch_tables();
        using impl::equal_contig_dispatch_table;
        using impl::equal_output_id_table;
        using impl::equal_strided_dispatch_table;

        auto equal_pyapi = [&](const dpctl::tensor::usm_ndarray &src1,
                               const dpctl::tensor::usm_ndarray &src2,
                               const dpctl::tensor::usm_ndarray &dst,
                               sycl::queue &exec_q,
                               const std::vector<sycl::event> &depends = {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends, equal_output_id_table,
                // function pointers to handle operation on contiguous arrays
                // (pointers may be nullptr)
                equal_contig_dispatch_table,
                // function pointers to handle operation on strided arrays (most
                // general case)
                equal_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t>{},
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t>{});
        };
        auto equal_result_type_pyapi = [&](const py::dtype &dtype1,
                                           const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               equal_output_id_table);
        };
        m.def("_equal", equal_pyapi, "", py::arg("src1"), py::arg("src2"),
              py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_equal_result_type", equal_result_type_pyapi, "");
    }

    // U13: ==== EXP           (x)
    {
        impl::populate_exp_dispatch_vectors();
        using impl::exp_contig_dispatch_vector;
        using impl::exp_output_typeid_vector;
        using impl::exp_strided_dispatch_vector;

        auto exp_pyapi = [&](const arrayT &src, const arrayT &dst,
                             sycl::queue &exec_q,
                             const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, exp_output_typeid_vector,
                exp_contig_dispatch_vector, exp_strided_dispatch_vector);
        };
        m.def("_exp", exp_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto exp_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype, exp_output_typeid_vector);
        };
        m.def("_exp_result_type", exp_result_type_pyapi);
    }

    // U14: ==== EXPM1         (x)
    {
        impl::populate_expm1_dispatch_vectors();
        using impl::expm1_contig_dispatch_vector;
        using impl::expm1_output_typeid_vector;
        using impl::expm1_strided_dispatch_vector;

        auto expm1_pyapi = [&](const arrayT &src, const arrayT &dst,
                               sycl::queue &exec_q,
                               const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, expm1_output_typeid_vector,
                expm1_contig_dispatch_vector, expm1_strided_dispatch_vector);
        };
        m.def("_expm1", expm1_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto expm1_result_type_pyapi = [&](const py::dtype dtype) {
            return py_unary_ufunc_result_type(dtype,
                                              expm1_output_typeid_vector);
        };
        m.def("_expm1_result_type", expm1_result_type_pyapi);
    }

    // U15: ==== FLOOR         (x)
    {
        impl::populate_floor_dispatch_vectors();
        using impl::floor_contig_dispatch_vector;
        using impl::floor_output_typeid_vector;
        using impl::floor_strided_dispatch_vector;

        auto floor_pyapi = [&](const arrayT &src, const arrayT &dst,
                               sycl::queue &exec_q,
                               const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, floor_output_typeid_vector,
                floor_contig_dispatch_vector, floor_strided_dispatch_vector);
        };
        m.def("_floor", floor_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto floor_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype,
                                              floor_output_typeid_vector);
        };
        m.def("_floor_result_type", floor_result_type_pyapi);
    }

    // B10: ==== FLOOR_DIVIDE  (x1, x2)
    {
        impl::populate_floor_divide_dispatch_tables();
        using impl::floor_divide_contig_dispatch_table;
        using impl::floor_divide_output_id_table;
        using impl::floor_divide_strided_dispatch_table;

        auto floor_divide_pyapi = [&](const dpctl::tensor::usm_ndarray &src1,
                                      const dpctl::tensor::usm_ndarray &src2,
                                      const dpctl::tensor::usm_ndarray &dst,
                                      sycl::queue &exec_q,
                                      const std::vector<sycl::event> &depends =
                                          {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends, floor_divide_output_id_table,
                // function pointers to handle operation on contiguous arrays
                // (pointers may be nullptr)
                floor_divide_contig_dispatch_table,
                // function pointers to handle operation on strided arrays (most
                // general case)
                floor_divide_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t>{},
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t>{});
        };
        auto floor_divide_result_type_pyapi = [&](const py::dtype &dtype1,
                                                  const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               floor_divide_output_id_table);
        };
        m.def("_floor_divide", floor_divide_pyapi, "", py::arg("src1"),
              py::arg("src2"), py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_floor_divide_result_type", floor_divide_result_type_pyapi, "");

        using impl::floor_divide_inplace_contig_dispatch_table;
        using impl::floor_divide_inplace_strided_dispatch_table;

        auto floor_divide_inplace_pyapi =
            [&](const dpctl::tensor::usm_ndarray &src,
                const dpctl::tensor::usm_ndarray &dst, sycl::queue &exec_q,
                const std::vector<sycl::event> &depends = {}) {
                return py_binary_inplace_ufunc(
                    src, dst, exec_q, depends, floor_divide_output_id_table,
                    // function pointers to handle inplace operation on
                    // contiguous arrays (pointers may be nullptr)
                    floor_divide_inplace_contig_dispatch_table,
                    // function pointers to handle inplace operation on strided
                    // arrays (most general case)
                    floor_divide_inplace_strided_dispatch_table,
                    // function pointers to handle inplace operation on
                    // c-contig matrix with c-contig row with broadcasting
                    // (may be nullptr)
                    td_ns::NullPtrTable<
                        binary_inplace_row_matrix_broadcast_impl_fn_ptr_t>{});
            };
        m.def("_floor_divide_inplace", floor_divide_inplace_pyapi, "",
              py::arg("lhs"), py::arg("rhs"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
    }

    // B11: ==== GREATER       (x1, x2)
    {
        impl::populate_greater_dispatch_tables();
        using impl::greater_contig_dispatch_table;
        using impl::greater_output_id_table;
        using impl::greater_strided_dispatch_table;

        auto greater_pyapi = [&](const dpctl::tensor::usm_ndarray &src1,
                                 const dpctl::tensor::usm_ndarray &src2,
                                 const dpctl::tensor::usm_ndarray &dst,
                                 sycl::queue &exec_q,
                                 const std::vector<sycl::event> &depends = {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends, greater_output_id_table,
                // function pointers to handle operation on contiguous arrays
                // (pointers may be nullptr)
                greater_contig_dispatch_table,
                // function pointers to handle operation on strided arrays (most
                // general case)
                greater_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t>{},
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t>{});
        };
        auto greater_result_type_pyapi = [&](const py::dtype &dtype1,
                                             const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               greater_output_id_table);
        };
        m.def("_greater", greater_pyapi, "", py::arg("src1"), py::arg("src2"),
              py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_greater_result_type", greater_result_type_pyapi, "");
    }

    // B12: ==== GREATER_EQUAL (x1, x2)
    {
        impl::populate_greater_equal_dispatch_tables();
        using impl::greater_equal_contig_dispatch_table;
        using impl::greater_equal_output_id_table;
        using impl::greater_equal_strided_dispatch_table;

        auto greater_equal_pyapi = [&](const dpctl::tensor::usm_ndarray &src1,
                                       const dpctl::tensor::usm_ndarray &src2,
                                       const dpctl::tensor::usm_ndarray &dst,
                                       sycl::queue &exec_q,
                                       const std::vector<sycl::event> &depends =
                                           {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends, greater_equal_output_id_table,
                // function pointers to handle operation on contiguous arrays
                // (pointers may be nullptr)
                greater_equal_contig_dispatch_table,
                // function pointers to handle operation on strided arrays (most
                // general case)
                greater_equal_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t>{},
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t>{});
        };
        auto greater_equal_result_type_pyapi = [&](const py::dtype &dtype1,
                                                   const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               greater_equal_output_id_table);
        };
        m.def("_greater_equal", greater_equal_pyapi, "", py::arg("src1"),
              py::arg("src2"), py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_greater_equal_result_type", greater_equal_result_type_pyapi,
              "");
    }

    // U16: ==== IMAG        (x)
    {
        impl::populate_imag_dispatch_vectors();
        using impl::imag_contig_dispatch_vector;
        using impl::imag_output_typeid_vector;
        using impl::imag_strided_dispatch_vector;

        auto imag_pyapi = [&](const arrayT &src, const arrayT &dst,
                              sycl::queue &exec_q,
                              const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, imag_output_typeid_vector,
                imag_contig_dispatch_vector, imag_strided_dispatch_vector);
        };
        m.def("_imag", imag_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto imag_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype, imag_output_typeid_vector);
        };
        m.def("_imag_result_type", imag_result_type_pyapi);
    }

    // U17: ==== ISFINITE    (x)
    {
        impl::populate_isfinite_dispatch_vectors();

        using impl::isfinite_contig_dispatch_vector;
        using impl::isfinite_output_typeid_vector;
        using impl::isfinite_strided_dispatch_vector;
        auto isfinite_pyapi =
            [&](const dpctl::tensor::usm_ndarray &src,
                const dpctl::tensor::usm_ndarray &dst, sycl::queue &exec_q,
                const std::vector<sycl::event> &depends = {}) {
                return py_unary_ufunc(src, dst, exec_q, depends,
                                      isfinite_output_typeid_vector,
                                      isfinite_contig_dispatch_vector,
                                      isfinite_strided_dispatch_vector);
            };
        auto isfinite_result_type_pyapi = [&](const py::dtype &dtype) {
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
        auto isinf_pyapi = [&](const dpctl::tensor::usm_ndarray &src,
                               const dpctl::tensor::usm_ndarray &dst,
                               sycl::queue &exec_q,
                               const std::vector<sycl::event> &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, isinf_output_typeid_vector,
                isinf_contig_dispatch_vector, isinf_strided_dispatch_vector);
        };
        auto isinf_result_type_pyapi = [&](const py::dtype &dtype) {
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
        auto isnan_pyapi = [&](const dpctl::tensor::usm_ndarray &src,
                               const dpctl::tensor::usm_ndarray &dst,
                               sycl::queue &exec_q,
                               const std::vector<sycl::event> &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, isnan_output_typeid_vector,
                isnan_contig_dispatch_vector, isnan_strided_dispatch_vector);
        };
        auto isnan_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype,
                                              isnan_output_typeid_vector);
        };
        m.def("_isnan", isnan_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());
        m.def("_isnan_result_type", isnan_result_type_pyapi, "");
    }

    // B24: ==== HYPOT       (x1, x2)
    {
        impl::populate_hypot_dispatch_tables();
        using impl::hypot_contig_dispatch_table;
        using impl::hypot_output_id_table;
        using impl::hypot_strided_dispatch_table;

        auto hypot_pyapi = [&](const dpctl::tensor::usm_ndarray &src1,
                               const dpctl::tensor::usm_ndarray &src2,
                               const dpctl::tensor::usm_ndarray &dst,
                               sycl::queue &exec_q,
                               const std::vector<sycl::event> &depends = {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends, hypot_output_id_table,
                // function pointers to handle operation on contiguous arrays
                // (pointers may be nullptr)
                hypot_contig_dispatch_table,
                // function pointers to handle operation on strided arrays (most
                // general case)
                hypot_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t>{},
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t>{});
        };
        auto hypot_result_type_pyapi = [&](const py::dtype &dtype1,
                                           const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               hypot_output_id_table);
        };
        m.def("_hypot", hypot_pyapi, "", py::arg("src1"), py::arg("src2"),
              py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_hypot_result_type", hypot_result_type_pyapi, "");
    }

    // U37: ==== CBRT        (x)
    {
        impl::populate_cbrt_dispatch_vectors();
        using impl::cbrt_contig_dispatch_vector;
        using impl::cbrt_output_typeid_vector;
        using impl::cbrt_strided_dispatch_vector;

        auto cbrt_pyapi = [&](const arrayT &src, const arrayT &dst,
                              sycl::queue &exec_q,
                              const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, cbrt_output_typeid_vector,
                cbrt_contig_dispatch_vector, cbrt_strided_dispatch_vector);
        };
        m.def("_cbrt", cbrt_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto cbrt_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype, cbrt_output_typeid_vector);
        };
        m.def("_cbrt_result_type", cbrt_result_type_pyapi);
    }

    // B25: ==== COPYSIGN       (x1, x2)
    {
        impl::populate_copysign_dispatch_tables();
        using impl::copysign_contig_dispatch_table;
        using impl::copysign_output_id_table;
        using impl::copysign_strided_dispatch_table;

        auto copysign_pyapi = [&](const dpctl::tensor::usm_ndarray &src1,
                                  const dpctl::tensor::usm_ndarray &src2,
                                  const dpctl::tensor::usm_ndarray &dst,
                                  sycl::queue &exec_q,
                                  const std::vector<sycl::event> &depends =
                                      {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends, copysign_output_id_table,
                // function pointers to handle operation on contiguous arrays
                // (pointers may be nullptr)
                copysign_contig_dispatch_table,
                // function pointers to handle operation on strided arrays (most
                // general case)
                copysign_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t>{},
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t>{});
        };
        auto copysign_result_type_pyapi = [&](const py::dtype &dtype1,
                                              const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               copysign_output_id_table);
        };
        m.def("_copysign", copysign_pyapi, "", py::arg("src1"), py::arg("src2"),
              py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_copysign_result_type", copysign_result_type_pyapi, "");
    }

    // U38: ==== EXP2      (x)
    {
        impl::populate_exp2_dispatch_vectors();
        using impl::exp2_contig_dispatch_vector;
        using impl::exp2_output_typeid_vector;
        using impl::exp2_strided_dispatch_vector;

        auto exp2_pyapi = [&](const arrayT &src, const arrayT &dst,
                              sycl::queue &exec_q,
                              const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, exp2_output_typeid_vector,
                exp2_contig_dispatch_vector, exp2_strided_dispatch_vector);
        };
        m.def("_exp2", exp2_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto exp2_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype, exp2_output_typeid_vector);
        };
        m.def("_exp2_result_type", exp2_result_type_pyapi);
    }
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
