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
#include "elementwise_functions3.hpp"
#include "utils/type_dispatch.hpp"

#include "kernels/elementwise_functions/not_equal.hpp"
#include "kernels/elementwise_functions/positive.hpp"
#include "kernels/elementwise_functions/pow.hpp"
#include "kernels/elementwise_functions/proj.hpp"
#include "kernels/elementwise_functions/real.hpp"
#include "kernels/elementwise_functions/remainder.hpp"
#include "kernels/elementwise_functions/round.hpp"
#include "kernels/elementwise_functions/rsqrt.hpp"
#include "kernels/elementwise_functions/sign.hpp"
#include "kernels/elementwise_functions/signbit.hpp"
#include "kernels/elementwise_functions/sin.hpp"
#include "kernels/elementwise_functions/sinh.hpp"
#include "kernels/elementwise_functions/sqrt.hpp"
#include "kernels/elementwise_functions/square.hpp"
#include "kernels/elementwise_functions/subtract.hpp"
#include "kernels/elementwise_functions/tan.hpp"
#include "kernels/elementwise_functions/tanh.hpp"
#include "kernels/elementwise_functions/trunc.hpp"

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

// B20: ==== NOT_EQUAL   (x1, x2)
namespace impl
{
namespace not_equal_fn_ns = dpctl::tensor::kernels::not_equal;

static binary_contig_impl_fn_ptr_t
    not_equal_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static int not_equal_output_id_table[td_ns::num_types][td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    not_equal_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

void populate_not_equal_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = not_equal_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::NotEqualTypeMapFactory;
    DispatchTableBuilder<int, NotEqualTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(not_equal_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::NotEqualStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t, NotEqualStridedFactory,
                         num_types>
        dtb2;
    dtb2.populate_dispatch_table(not_equal_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::NotEqualContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, NotEqualContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(not_equal_contig_dispatch_table);
};
} // namespace impl

// U26: ==== POSITIVE    (x)
namespace impl
{

namespace positive_fn_ns = dpctl::tensor::kernels::positive;

static unary_contig_impl_fn_ptr_t
    positive_contig_dispatch_vector[td_ns::num_types];
static int positive_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    positive_strided_dispatch_vector[td_ns::num_types];

void populate_positive_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = positive_fn_ns;

    using fn_ns::PositiveContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, PositiveContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(positive_contig_dispatch_vector);

    using fn_ns::PositiveStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, PositiveStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(positive_strided_dispatch_vector);

    using fn_ns::PositiveTypeMapFactory;
    DispatchVectorBuilder<int, PositiveTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(positive_output_typeid_vector);
}

} // namespace impl

// B21: ==== POW         (x1, x2)
namespace impl
{

namespace pow_fn_ns = dpctl::tensor::kernels::pow;

static binary_contig_impl_fn_ptr_t pow_contig_dispatch_table[td_ns::num_types]
                                                            [td_ns::num_types];
static int pow_output_id_table[td_ns::num_types][td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    pow_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

static binary_inplace_contig_impl_fn_ptr_t
    pow_inplace_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static binary_inplace_strided_impl_fn_ptr_t
    pow_inplace_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

void populate_pow_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = pow_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::PowTypeMapFactory;
    DispatchTableBuilder<int, PowTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(pow_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::PowStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t, PowStridedFactory,
                         num_types>
        dtb2;
    dtb2.populate_dispatch_table(pow_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::PowContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, PowContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(pow_contig_dispatch_table);

    // function pointers for inplace operation on general strided arrays
    using fn_ns::PowInplaceStridedFactory;
    DispatchTableBuilder<binary_inplace_strided_impl_fn_ptr_t,
                         PowInplaceStridedFactory, num_types>
        dtb4;
    dtb4.populate_dispatch_table(pow_inplace_strided_dispatch_table);

    // function pointers for inplace operation on contiguous inputs and output
    using fn_ns::PowInplaceContigFactory;
    DispatchTableBuilder<binary_inplace_contig_impl_fn_ptr_t,
                         PowInplaceContigFactory, num_types>
        dtb5;
    dtb5.populate_dispatch_table(pow_inplace_contig_dispatch_table);
};

} // namespace impl

// U??: ==== PROJ        (x)
namespace impl
{

namespace proj_fn_ns = dpctl::tensor::kernels::proj;

static unary_contig_impl_fn_ptr_t proj_contig_dispatch_vector[td_ns::num_types];
static int proj_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    proj_strided_dispatch_vector[td_ns::num_types];

void populate_proj_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = proj_fn_ns;

    using fn_ns::ProjContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, ProjContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(proj_contig_dispatch_vector);

    using fn_ns::ProjStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, ProjStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(proj_strided_dispatch_vector);

    using fn_ns::ProjTypeMapFactory;
    DispatchVectorBuilder<int, ProjTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(proj_output_typeid_vector);
}
} // namespace impl

// U27: ==== REAL        (x)
namespace impl
{

namespace real_fn_ns = dpctl::tensor::kernels::real;

static unary_contig_impl_fn_ptr_t real_contig_dispatch_vector[td_ns::num_types];
static int real_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    real_strided_dispatch_vector[td_ns::num_types];

void populate_real_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = real_fn_ns;

    using fn_ns::RealContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, RealContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(real_contig_dispatch_vector);

    using fn_ns::RealStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, RealStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(real_strided_dispatch_vector);

    using fn_ns::RealTypeMapFactory;
    DispatchVectorBuilder<int, RealTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(real_output_typeid_vector);
}
} // namespace impl

// B22: ==== REMAINDER   (x1, x2)
namespace impl
{

namespace remainder_fn_ns = dpctl::tensor::kernels::remainder;

static binary_contig_impl_fn_ptr_t
    remainder_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static int remainder_output_id_table[td_ns::num_types][td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    remainder_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

static binary_inplace_contig_impl_fn_ptr_t
    remainder_inplace_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static binary_inplace_strided_impl_fn_ptr_t
    remainder_inplace_strided_dispatch_table[td_ns::num_types]
                                            [td_ns::num_types];

void populate_remainder_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = remainder_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::RemainderTypeMapFactory;
    DispatchTableBuilder<int, RemainderTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(remainder_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::RemainderStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t, RemainderStridedFactory,
                         num_types>
        dtb2;
    dtb2.populate_dispatch_table(remainder_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::RemainderContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, RemainderContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(remainder_contig_dispatch_table);

    // function pointers for inplace operation on general strided arrays
    using fn_ns::RemainderInplaceStridedFactory;
    DispatchTableBuilder<binary_inplace_strided_impl_fn_ptr_t,
                         RemainderInplaceStridedFactory, num_types>
        dtb4;
    dtb4.populate_dispatch_table(remainder_inplace_strided_dispatch_table);

    // function pointers for inplace operation on contiguous inputs and output
    using fn_ns::RemainderInplaceContigFactory;
    DispatchTableBuilder<binary_inplace_contig_impl_fn_ptr_t,
                         RemainderInplaceContigFactory, num_types>
        dtb5;
    dtb5.populate_dispatch_table(remainder_inplace_contig_dispatch_table);
}

} // namespace impl

// U28: ==== ROUND       (x)
namespace impl
{

namespace round_fn_ns = dpctl::tensor::kernels::round;

static unary_contig_impl_fn_ptr_t
    round_contig_dispatch_vector[td_ns::num_types];
static int round_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    round_strided_dispatch_vector[td_ns::num_types];

void populate_round_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = round_fn_ns;

    using fn_ns::RoundContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, RoundContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(round_contig_dispatch_vector);

    using fn_ns::RoundStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, RoundStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(round_strided_dispatch_vector);

    using fn_ns::RoundTypeMapFactory;
    DispatchVectorBuilder<int, RoundTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(round_output_typeid_vector);
}

} // namespace impl

// U29: ==== SIGN        (x)
namespace impl
{

namespace sign_fn_ns = dpctl::tensor::kernels::sign;

static unary_contig_impl_fn_ptr_t sign_contig_dispatch_vector[td_ns::num_types];
static int sign_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    sign_strided_dispatch_vector[td_ns::num_types];

void populate_sign_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = sign_fn_ns;

    using fn_ns::SignContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, SignContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(sign_contig_dispatch_vector);

    using fn_ns::SignStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, SignStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(sign_strided_dispatch_vector);

    using fn_ns::SignTypeMapFactory;
    DispatchVectorBuilder<int, SignTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(sign_output_typeid_vector);
}

} // namespace impl

// ==== SIGNBIT        (x)
namespace impl
{

namespace signbit_fn_ns = dpctl::tensor::kernels::signbit;

static unary_contig_impl_fn_ptr_t
    signbit_contig_dispatch_vector[td_ns::num_types];
static int signbit_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    signbit_strided_dispatch_vector[td_ns::num_types];

void populate_signbit_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = signbit_fn_ns;

    using fn_ns::SignbitContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, SignbitContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(signbit_contig_dispatch_vector);

    using fn_ns::SignbitStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, SignbitStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(signbit_strided_dispatch_vector);

    using fn_ns::SignbitTypeMapFactory;
    DispatchVectorBuilder<int, SignbitTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(signbit_output_typeid_vector);
}

} // namespace impl

// U30: ==== SIN         (x)
namespace impl
{

namespace sin_fn_ns = dpctl::tensor::kernels::sin;

static unary_contig_impl_fn_ptr_t sin_contig_dispatch_vector[td_ns::num_types];
static int sin_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    sin_strided_dispatch_vector[td_ns::num_types];

void populate_sin_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = sin_fn_ns;

    using fn_ns::SinContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, SinContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(sin_contig_dispatch_vector);

    using fn_ns::SinStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, SinStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(sin_strided_dispatch_vector);

    using fn_ns::SinTypeMapFactory;
    DispatchVectorBuilder<int, SinTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(sin_output_typeid_vector);
}

} // namespace impl

// U31: ==== SINH        (x)
namespace impl
{

namespace sinh_fn_ns = dpctl::tensor::kernels::sinh;

static unary_contig_impl_fn_ptr_t sinh_contig_dispatch_vector[td_ns::num_types];
static int sinh_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    sinh_strided_dispatch_vector[td_ns::num_types];

void populate_sinh_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = sinh_fn_ns;

    using fn_ns::SinhContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, SinhContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(sinh_contig_dispatch_vector);

    using fn_ns::SinhStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, SinhStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(sinh_strided_dispatch_vector);

    using fn_ns::SinhTypeMapFactory;
    DispatchVectorBuilder<int, SinhTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(sinh_output_typeid_vector);
}

} // namespace impl

// U32: ==== SQUARE      (x)
namespace impl
{

namespace square_fn_ns = dpctl::tensor::kernels::square;

static unary_contig_impl_fn_ptr_t
    square_contig_dispatch_vector[td_ns::num_types];
static int square_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    square_strided_dispatch_vector[td_ns::num_types];

void populate_square_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = square_fn_ns;

    using fn_ns::SquareContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, SquareContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(square_contig_dispatch_vector);

    using fn_ns::SquareStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, SquareStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(square_strided_dispatch_vector);

    using fn_ns::SquareTypeMapFactory;
    DispatchVectorBuilder<int, SquareTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(square_output_typeid_vector);
}

} // namespace impl

// U33: ==== SQRT        (x)
namespace impl
{

namespace sqrt_fn_ns = dpctl::tensor::kernels::sqrt;

static unary_contig_impl_fn_ptr_t sqrt_contig_dispatch_vector[td_ns::num_types];
static int sqrt_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    sqrt_strided_dispatch_vector[td_ns::num_types];

void populate_sqrt_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = sqrt_fn_ns;

    using fn_ns::SqrtContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, SqrtContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(sqrt_contig_dispatch_vector);

    using fn_ns::SqrtStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, SqrtStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(sqrt_strided_dispatch_vector);

    using fn_ns::SqrtTypeMapFactory;
    DispatchVectorBuilder<int, SqrtTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(sqrt_output_typeid_vector);
}

} // namespace impl

// B23: ==== SUBTRACT    (x1, x2)
namespace impl
{
namespace subtract_fn_ns = dpctl::tensor::kernels::subtract;

static binary_contig_impl_fn_ptr_t
    subtract_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static int subtract_output_id_table[td_ns::num_types][td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    subtract_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

// sub(matrix, row)
static binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t
    subtract_contig_matrix_contig_row_broadcast_dispatch_table
        [td_ns::num_types][td_ns::num_types];

// sub(row, matrix)
static binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t
    subtract_contig_row_contig_matrix_broadcast_dispatch_table
        [td_ns::num_types][td_ns::num_types];

static binary_inplace_contig_impl_fn_ptr_t
    subtract_inplace_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static binary_inplace_strided_impl_fn_ptr_t
    subtract_inplace_strided_dispatch_table[td_ns::num_types][td_ns::num_types];
static binary_inplace_row_matrix_broadcast_impl_fn_ptr_t
    subtract_inplace_row_matrix_dispatch_table[td_ns::num_types]
                                              [td_ns::num_types];

void populate_subtract_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = subtract_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::SubtractTypeMapFactory;
    DispatchTableBuilder<int, SubtractTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(subtract_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::SubtractStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t, SubtractStridedFactory,
                         num_types>
        dtb2;
    dtb2.populate_dispatch_table(subtract_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::SubtractContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, SubtractContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(subtract_contig_dispatch_table);

    // function pointers for operation on contiguous matrix, contiguous row
    // with contiguous matrix output
    using fn_ns::SubtractContigMatrixContigRowBroadcastFactory;
    DispatchTableBuilder<
        binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t,
        SubtractContigMatrixContigRowBroadcastFactory, num_types>
        dtb4;
    dtb4.populate_dispatch_table(
        subtract_contig_matrix_contig_row_broadcast_dispatch_table);

    // function pointers for operation on contiguous row, contiguous matrix
    // with contiguous matrix output
    using fn_ns::SubtractContigRowContigMatrixBroadcastFactory;
    DispatchTableBuilder<
        binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t,
        SubtractContigRowContigMatrixBroadcastFactory, num_types>
        dtb5;
    dtb5.populate_dispatch_table(
        subtract_contig_row_contig_matrix_broadcast_dispatch_table);

    // function pointers for inplace operation on general strided arrays
    using fn_ns::SubtractInplaceStridedFactory;
    DispatchTableBuilder<binary_inplace_strided_impl_fn_ptr_t,
                         SubtractInplaceStridedFactory, num_types>
        dtb6;
    dtb6.populate_dispatch_table(subtract_inplace_strided_dispatch_table);

    // function pointers for inplace operation on contiguous inputs and output
    using fn_ns::SubtractInplaceContigFactory;
    DispatchTableBuilder<binary_inplace_contig_impl_fn_ptr_t,
                         SubtractInplaceContigFactory, num_types>
        dtb7;
    dtb7.populate_dispatch_table(subtract_inplace_contig_dispatch_table);

    // function pointers for inplace operation on contiguous matrix
    // and contiguous row
    using fn_ns::SubtractInplaceRowMatrixBroadcastFactory;
    DispatchTableBuilder<binary_inplace_row_matrix_broadcast_impl_fn_ptr_t,
                         SubtractInplaceRowMatrixBroadcastFactory, num_types>
        dtb8;
    dtb8.populate_dispatch_table(subtract_inplace_row_matrix_dispatch_table);
};

} // namespace impl

// U34: ==== TAN         (x)
namespace impl
{

namespace tan_fn_ns = dpctl::tensor::kernels::tan;

static unary_contig_impl_fn_ptr_t tan_contig_dispatch_vector[td_ns::num_types];
static int tan_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    tan_strided_dispatch_vector[td_ns::num_types];

void populate_tan_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = tan_fn_ns;

    using fn_ns::TanContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, TanContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(tan_contig_dispatch_vector);

    using fn_ns::TanStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, TanStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(tan_strided_dispatch_vector);

    using fn_ns::TanTypeMapFactory;
    DispatchVectorBuilder<int, TanTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(tan_output_typeid_vector);
}

} // namespace impl

// U35: ==== TANH        (x)
namespace impl
{

namespace tanh_fn_ns = dpctl::tensor::kernels::tanh;

static unary_contig_impl_fn_ptr_t tanh_contig_dispatch_vector[td_ns::num_types];
static int tanh_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    tanh_strided_dispatch_vector[td_ns::num_types];

void populate_tanh_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = tanh_fn_ns;

    using fn_ns::TanhContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, TanhContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(tanh_contig_dispatch_vector);

    using fn_ns::TanhStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, TanhStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(tanh_strided_dispatch_vector);

    using fn_ns::TanhTypeMapFactory;
    DispatchVectorBuilder<int, TanhTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(tanh_output_typeid_vector);
}

} // namespace impl

// U36: ==== TRUNC       (x)
namespace impl
{

namespace trunc_fn_ns = dpctl::tensor::kernels::trunc;

static unary_contig_impl_fn_ptr_t
    trunc_contig_dispatch_vector[td_ns::num_types];
static int trunc_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    trunc_strided_dispatch_vector[td_ns::num_types];

void populate_trunc_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = trunc_fn_ns;

    using fn_ns::TruncContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, TruncContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(trunc_contig_dispatch_vector);

    using fn_ns::TruncStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, TruncStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(trunc_strided_dispatch_vector);

    using fn_ns::TruncTypeMapFactory;
    DispatchVectorBuilder<int, TruncTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(trunc_output_typeid_vector);
}

} // namespace impl

// U39: ==== RSQRT        (x)
namespace impl
{

namespace rsqrt_fn_ns = dpctl::tensor::kernels::rsqrt;

static unary_contig_impl_fn_ptr_t
    rsqrt_contig_dispatch_vector[td_ns::num_types];
static int rsqrt_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    rsqrt_strided_dispatch_vector[td_ns::num_types];

void populate_rsqrt_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = rsqrt_fn_ns;

    using fn_ns::RsqrtContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, RsqrtContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(rsqrt_contig_dispatch_vector);

    using fn_ns::RsqrtStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, RsqrtStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(rsqrt_strided_dispatch_vector);

    using fn_ns::RsqrtTypeMapFactory;
    DispatchVectorBuilder<int, RsqrtTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(rsqrt_output_typeid_vector);
}

} // namespace impl

// ==========================================================================================
// //

namespace py = pybind11;

void init_elementwise_functions3(py::module_ m)
{

    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;

    // B20: ==== NOT_EQUAL   (x1, x2)
    {
        impl::populate_not_equal_dispatch_tables();
        using impl::not_equal_contig_dispatch_table;
        using impl::not_equal_output_id_table;
        using impl::not_equal_strided_dispatch_table;

        auto not_equal_pyapi = [&](const dpctl::tensor::usm_ndarray &src1,
                                   const dpctl::tensor::usm_ndarray &src2,
                                   const dpctl::tensor::usm_ndarray &dst,
                                   sycl::queue &exec_q,
                                   const std::vector<sycl::event> &depends =
                                       {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends, not_equal_output_id_table,
                // function pointers to handle operation on contiguous arrays
                // (pointers may be nullptr)
                not_equal_contig_dispatch_table,
                // function pointers to handle operation on strided arrays (most
                // general case)
                not_equal_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t>{},
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t>{});
        };
        auto not_equal_result_type_pyapi = [&](const py::dtype &dtype1,
                                               const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               not_equal_output_id_table);
        };
        m.def("_not_equal", not_equal_pyapi, "", py::arg("src1"),
              py::arg("src2"), py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_not_equal_result_type", not_equal_result_type_pyapi, "");
    }

    // U26: ==== POSITIVE    (x)
    {
        impl::populate_positive_dispatch_vectors();
        using impl::positive_contig_dispatch_vector;
        using impl::positive_output_typeid_vector;
        using impl::positive_strided_dispatch_vector;

        auto positive_pyapi = [&](const arrayT &src, const arrayT &dst,
                                  sycl::queue &exec_q,
                                  const event_vecT &depends = {}) {
            return py_unary_ufunc(src, dst, exec_q, depends,
                                  positive_output_typeid_vector,
                                  positive_contig_dispatch_vector,
                                  positive_strided_dispatch_vector);
        };
        m.def("_positive", positive_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto positive_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype,
                                              positive_output_typeid_vector);
        };
        m.def("_positive_result_type", positive_result_type_pyapi);
    }

    // B21: ==== POW         (x1, x2)
    {
        impl::populate_pow_dispatch_tables();
        using impl::pow_contig_dispatch_table;
        using impl::pow_output_id_table;
        using impl::pow_strided_dispatch_table;

        auto pow_pyapi = [&](const dpctl::tensor::usm_ndarray &src1,
                             const dpctl::tensor::usm_ndarray &src2,
                             const dpctl::tensor::usm_ndarray &dst,
                             sycl::queue &exec_q,
                             const std::vector<sycl::event> &depends = {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends, pow_output_id_table,
                // function pointers to handle operation on contiguous arrays
                // (pointers may be nullptr)
                pow_contig_dispatch_table,
                // function pointers to handle operation on strided arrays (most
                // general case)
                pow_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t>{},
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t>{});
        };
        auto pow_result_type_pyapi = [&](const py::dtype &dtype1,
                                         const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               pow_output_id_table);
        };
        m.def("_pow", pow_pyapi, "", py::arg("src1"), py::arg("src2"),
              py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_pow_result_type", pow_result_type_pyapi, "");

        using impl::pow_inplace_contig_dispatch_table;
        using impl::pow_inplace_strided_dispatch_table;

        auto pow_inplace_pyapi =
            [&](const dpctl::tensor::usm_ndarray &src,
                const dpctl::tensor::usm_ndarray &dst, sycl::queue &exec_q,
                const std::vector<sycl::event> &depends = {}) {
                return py_binary_inplace_ufunc(
                    src, dst, exec_q, depends, pow_output_id_table,
                    // function pointers to handle inplace operation on
                    // contiguous arrays (pointers may be nullptr)
                    pow_inplace_contig_dispatch_table,
                    // function pointers to handle inplace operation on strided
                    // arrays (most general case)
                    pow_inplace_strided_dispatch_table,
                    // function pointers to handle inplace operation on
                    // c-contig matrix with c-contig row with broadcasting
                    // (may be nullptr)
                    td_ns::NullPtrTable<
                        binary_inplace_row_matrix_broadcast_impl_fn_ptr_t>{});
            };
        m.def("_pow_inplace", pow_inplace_pyapi, "", py::arg("lhs"),
              py::arg("rhs"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
    }

    // U??: ==== PROJ        (x)
    {
        impl::populate_proj_dispatch_vectors();
        using impl::proj_contig_dispatch_vector;
        using impl::proj_output_typeid_vector;
        using impl::proj_strided_dispatch_vector;

        auto proj_pyapi = [&](const arrayT &src, const arrayT &dst,
                              sycl::queue &exec_q,
                              const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, proj_output_typeid_vector,
                proj_contig_dispatch_vector, proj_strided_dispatch_vector);
        };
        m.def("_proj", proj_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto proj_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype, proj_output_typeid_vector);
        };
        m.def("_proj_result_type", proj_result_type_pyapi);
    }

    // U27: ==== REAL        (x)
    {
        impl::populate_real_dispatch_vectors();
        using impl::real_contig_dispatch_vector;
        using impl::real_output_typeid_vector;
        using impl::real_strided_dispatch_vector;

        auto real_pyapi = [&](const arrayT &src, const arrayT &dst,
                              sycl::queue &exec_q,
                              const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, real_output_typeid_vector,
                real_contig_dispatch_vector, real_strided_dispatch_vector);
        };
        m.def("_real", real_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto real_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype, real_output_typeid_vector);
        };
        m.def("_real_result_type", real_result_type_pyapi);
    }

    // B22: ==== REMAINDER   (x1, x2)
    {
        impl::populate_remainder_dispatch_tables();
        using impl::remainder_contig_dispatch_table;
        using impl::remainder_output_id_table;
        using impl::remainder_strided_dispatch_table;

        auto remainder_pyapi = [&](const dpctl::tensor::usm_ndarray &src1,
                                   const dpctl::tensor::usm_ndarray &src2,
                                   const dpctl::tensor::usm_ndarray &dst,
                                   sycl::queue &exec_q,
                                   const std::vector<sycl::event> &depends =
                                       {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends, remainder_output_id_table,
                // function pointers to handle operation on contiguous arrays
                // (pointers may be nullptr)
                remainder_contig_dispatch_table,
                // function pointers to handle operation on strided arrays (most
                // general case)
                remainder_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t>{},
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t>{});
        };
        auto remainder_result_type_pyapi = [&](const py::dtype &dtype1,
                                               const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               remainder_output_id_table);
        };
        m.def("_remainder", remainder_pyapi, "", py::arg("src1"),
              py::arg("src2"), py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_remainder_result_type", remainder_result_type_pyapi, "");

        using impl::remainder_inplace_contig_dispatch_table;
        using impl::remainder_inplace_strided_dispatch_table;

        auto remainder_inplace_pyapi =
            [&](const dpctl::tensor::usm_ndarray &src,
                const dpctl::tensor::usm_ndarray &dst, sycl::queue &exec_q,
                const std::vector<sycl::event> &depends = {}) {
                return py_binary_inplace_ufunc(
                    src, dst, exec_q, depends, remainder_output_id_table,
                    // function pointers to handle inplace operation on
                    // contiguous arrays (pointers may be nullptr)
                    remainder_inplace_contig_dispatch_table,
                    // function pointers to handle inplace operation on strided
                    // arrays (most general case)
                    remainder_inplace_strided_dispatch_table,
                    // function pointers to handle inplace operation on
                    // c-contig matrix with c-contig row with broadcasting
                    // (may be nullptr)
                    td_ns::NullPtrTable<
                        binary_inplace_row_matrix_broadcast_impl_fn_ptr_t>{});
            };
        m.def("_remainder_inplace", remainder_inplace_pyapi, "", py::arg("lhs"),
              py::arg("rhs"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
    }

    // U28: ==== ROUND       (x)
    {
        impl::populate_round_dispatch_vectors();
        using impl::round_contig_dispatch_vector;
        using impl::round_output_typeid_vector;
        using impl::round_strided_dispatch_vector;

        auto round_pyapi = [&](const arrayT &src, const arrayT &dst,
                               sycl::queue &exec_q,
                               const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, round_output_typeid_vector,
                round_contig_dispatch_vector, round_strided_dispatch_vector);
        };
        m.def("_round", round_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto round_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype,
                                              round_output_typeid_vector);
        };
        m.def("_round_result_type", round_result_type_pyapi);
    }

    // U29: ==== SIGN        (x)
    {
        impl::populate_sign_dispatch_vectors();
        using impl::sign_contig_dispatch_vector;
        using impl::sign_output_typeid_vector;
        using impl::sign_strided_dispatch_vector;

        auto sign_pyapi = [&](const arrayT &src, const arrayT &dst,
                              sycl::queue &exec_q,
                              const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, sign_output_typeid_vector,
                sign_contig_dispatch_vector, sign_strided_dispatch_vector);
        };
        m.def("_sign", sign_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto sign_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype, sign_output_typeid_vector);
        };
        m.def("_sign_result_type", sign_result_type_pyapi);
    }

    // ==== SIGNBIT        (x)
    {
        impl::populate_signbit_dispatch_vectors();
        using impl::signbit_contig_dispatch_vector;
        using impl::signbit_output_typeid_vector;
        using impl::signbit_strided_dispatch_vector;

        auto signbit_pyapi = [&](const arrayT &src, const arrayT &dst,
                                 sycl::queue &exec_q,
                                 const event_vecT &depends = {}) {
            return py_unary_ufunc(src, dst, exec_q, depends,
                                  signbit_output_typeid_vector,
                                  signbit_contig_dispatch_vector,
                                  signbit_strided_dispatch_vector);
        };
        m.def("_signbit", signbit_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto signbit_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype,
                                              signbit_output_typeid_vector);
        };
        m.def("_signbit_result_type", signbit_result_type_pyapi);
    }

    // U30: ==== SIN         (x)
    {
        impl::populate_sin_dispatch_vectors();
        using impl::sin_contig_dispatch_vector;
        using impl::sin_output_typeid_vector;
        using impl::sin_strided_dispatch_vector;

        auto sin_pyapi = [&](const arrayT &src, const arrayT &dst,
                             sycl::queue &exec_q,
                             const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, sin_output_typeid_vector,
                sin_contig_dispatch_vector, sin_strided_dispatch_vector);
        };
        m.def("_sin", sin_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto sin_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype, sin_output_typeid_vector);
        };
        m.def("_sin_result_type", sin_result_type_pyapi);
    }
    // U31: ==== SINH        (x)
    {
        impl::populate_sinh_dispatch_vectors();
        using impl::sinh_contig_dispatch_vector;
        using impl::sinh_output_typeid_vector;
        using impl::sinh_strided_dispatch_vector;

        auto sinh_pyapi = [&](const arrayT &src, const arrayT &dst,
                              sycl::queue &exec_q,
                              const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, sinh_output_typeid_vector,
                sinh_contig_dispatch_vector, sinh_strided_dispatch_vector);
        };
        m.def("_sinh", sinh_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto sinh_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype, sinh_output_typeid_vector);
        };
        m.def("_sinh_result_type", sinh_result_type_pyapi);
    }

    // U32: ==== SQUARE      (x)
    {
        impl::populate_square_dispatch_vectors();
        using impl::square_contig_dispatch_vector;
        using impl::square_output_typeid_vector;
        using impl::square_strided_dispatch_vector;

        auto square_pyapi = [&](const arrayT &src, const arrayT &dst,
                                sycl::queue &exec_q,
                                const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, square_output_typeid_vector,
                square_contig_dispatch_vector, square_strided_dispatch_vector);
        };
        m.def("_square", square_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto square_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype,
                                              square_output_typeid_vector);
        };
        m.def("_square_result_type", square_result_type_pyapi);
    }

    // U33: ==== SQRT        (x)
    {
        impl::populate_sqrt_dispatch_vectors();
        using impl::sqrt_contig_dispatch_vector;
        using impl::sqrt_output_typeid_vector;
        using impl::sqrt_strided_dispatch_vector;

        auto sqrt_pyapi = [&](const arrayT &src, const arrayT &dst,
                              sycl::queue &exec_q,
                              const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, sqrt_output_typeid_vector,
                sqrt_contig_dispatch_vector, sqrt_strided_dispatch_vector);
        };
        m.def("_sqrt", sqrt_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto sqrt_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype, sqrt_output_typeid_vector);
        };
        m.def("_sqrt_result_type", sqrt_result_type_pyapi);
    }

    // B23: ==== SUBTRACT    (x1, x2)
    {
        impl::populate_subtract_dispatch_tables();
        using impl::subtract_contig_dispatch_table;
        using impl::subtract_contig_matrix_contig_row_broadcast_dispatch_table;
        using impl::subtract_contig_row_contig_matrix_broadcast_dispatch_table;
        using impl::subtract_output_id_table;
        using impl::subtract_strided_dispatch_table;

        auto subtract_pyapi =
            [&](const dpctl::tensor::usm_ndarray &src1,
                const dpctl::tensor::usm_ndarray &src2,
                const dpctl::tensor::usm_ndarray &dst, sycl::queue &exec_q,
                const std::vector<sycl::event> &depends = {}) {
                return py_binary_ufunc(
                    src1, src2, dst, exec_q, depends, subtract_output_id_table,
                    // function pointers to handle operation on contiguous
                    // arrays (pointers may be nullptr)
                    subtract_contig_dispatch_table,
                    // function pointers to handle operation on strided arrays
                    // (most general case)
                    subtract_strided_dispatch_table,
                    // function pointers to handle operation of c-contig matrix
                    // and c-contig row with broadcasting (may be nullptr)
                    subtract_contig_matrix_contig_row_broadcast_dispatch_table,
                    // function pointers to handle operation of c-contig matrix
                    // and c-contig row with broadcasting (may be nullptr)
                    subtract_contig_row_contig_matrix_broadcast_dispatch_table);
            };
        auto subtract_result_type_pyapi = [&](const py::dtype &dtype1,
                                              const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               subtract_output_id_table);
        };
        m.def("_subtract", subtract_pyapi, "", py::arg("src1"), py::arg("src2"),
              py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_subtract_result_type", subtract_result_type_pyapi, "");

        using impl::subtract_inplace_contig_dispatch_table;
        using impl::subtract_inplace_row_matrix_dispatch_table;
        using impl::subtract_inplace_strided_dispatch_table;

        auto subtract_inplace_pyapi =
            [&](const dpctl::tensor::usm_ndarray &src,
                const dpctl::tensor::usm_ndarray &dst, sycl::queue &exec_q,
                const std::vector<sycl::event> &depends = {}) {
                return py_binary_inplace_ufunc(
                    src, dst, exec_q, depends, subtract_output_id_table,
                    // function pointers to handle inplace operation on
                    // contiguous arrays (pointers may be nullptr)
                    subtract_inplace_contig_dispatch_table,
                    // function pointers to handle inplace operation on strided
                    // arrays (most general case)
                    subtract_inplace_strided_dispatch_table,
                    // function pointers to handle inplace operation on
                    // c-contig matrix with c-contig row with broadcasting
                    // (may be nullptr)
                    subtract_inplace_row_matrix_dispatch_table);
            };
        m.def("_subtract_inplace", subtract_inplace_pyapi, "", py::arg("lhs"),
              py::arg("rhs"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
    }

    // U34: ==== TAN         (x)
    {
        impl::populate_tan_dispatch_vectors();
        using impl::tan_contig_dispatch_vector;
        using impl::tan_output_typeid_vector;
        using impl::tan_strided_dispatch_vector;

        auto tan_pyapi = [&](const arrayT &src, const arrayT &dst,
                             sycl::queue &exec_q,
                             const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, tan_output_typeid_vector,
                tan_contig_dispatch_vector, tan_strided_dispatch_vector);
        };
        m.def("_tan", tan_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto tan_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype, tan_output_typeid_vector);
        };
        m.def("_tan_result_type", tan_result_type_pyapi);
    }

    // U35: ==== TANH        (x)
    {
        impl::populate_tanh_dispatch_vectors();
        using impl::tanh_contig_dispatch_vector;
        using impl::tanh_output_typeid_vector;
        using impl::tanh_strided_dispatch_vector;

        auto tanh_pyapi = [&](const arrayT &src, const arrayT &dst,
                              sycl::queue &exec_q,
                              const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, tanh_output_typeid_vector,
                tanh_contig_dispatch_vector, tanh_strided_dispatch_vector);
        };
        m.def("_tanh", tanh_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto tanh_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype, tanh_output_typeid_vector);
        };
        m.def("_tanh_result_type", tanh_result_type_pyapi);
    }

    // U36: ==== TRUNC       (x)
    {
        impl::populate_trunc_dispatch_vectors();
        using impl::trunc_contig_dispatch_vector;
        using impl::trunc_output_typeid_vector;
        using impl::trunc_strided_dispatch_vector;

        auto trunc_pyapi = [&](const arrayT &src, const arrayT &dst,
                               sycl::queue &exec_q,
                               const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, trunc_output_typeid_vector,
                trunc_contig_dispatch_vector, trunc_strided_dispatch_vector);
        };
        m.def("_trunc", trunc_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto trunc_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype,
                                              trunc_output_typeid_vector);
        };
        m.def("_trunc_result_type", trunc_result_type_pyapi);
    }

    // U39: ==== RSQRT        (x)
    {
        impl::populate_rsqrt_dispatch_vectors();
        using impl::rsqrt_contig_dispatch_vector;
        using impl::rsqrt_output_typeid_vector;
        using impl::rsqrt_strided_dispatch_vector;

        auto rsqrt_pyapi = [&](const arrayT &src, const arrayT &dst,
                               sycl::queue &exec_q,
                               const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, rsqrt_output_typeid_vector,
                rsqrt_contig_dispatch_vector, rsqrt_strided_dispatch_vector);
        };
        m.def("_rsqrt", rsqrt_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto rsqrt_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype,
                                              rsqrt_output_typeid_vector);
        };
        m.def("_rsqrt_result_type", rsqrt_result_type_pyapi);
    }
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
