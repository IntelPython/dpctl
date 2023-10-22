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
#include "elementwise_functions2.hpp"
#include "utils/type_dispatch.hpp"

#include "kernels/elementwise_functions/abs.hpp"
#include "kernels/elementwise_functions/acos.hpp"
#include "kernels/elementwise_functions/acosh.hpp"
#include "kernels/elementwise_functions/add.hpp"
#include "kernels/elementwise_functions/asin.hpp"
#include "kernels/elementwise_functions/asinh.hpp"
#include "kernels/elementwise_functions/atan.hpp"
#include "kernels/elementwise_functions/atan2.hpp"
#include "kernels/elementwise_functions/atanh.hpp"
#include "kernels/elementwise_functions/bitwise_and.hpp"
#include "kernels/elementwise_functions/bitwise_invert.hpp"
#include "kernels/elementwise_functions/bitwise_left_shift.hpp"
#include "kernels/elementwise_functions/bitwise_or.hpp"
#include "kernels/elementwise_functions/bitwise_right_shift.hpp"
#include "kernels/elementwise_functions/bitwise_xor.hpp"

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

// U01: ==== ABS   (x)
namespace impl
{

namespace abs_fn_ns = dpctl::tensor::kernels::abs;

static unary_contig_impl_fn_ptr_t abs_contig_dispatch_vector[td_ns::num_types];
static int abs_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    abs_strided_dispatch_vector[td_ns::num_types];

void populate_abs_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = abs_fn_ns;

    using fn_ns::AbsContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, AbsContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(abs_contig_dispatch_vector);

    using fn_ns::AbsStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, AbsStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(abs_strided_dispatch_vector);

    using fn_ns::AbsTypeMapFactory;
    DispatchVectorBuilder<int, AbsTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(abs_output_typeid_vector);
};

} // namespace impl

// U02: ==== ACOS   (x)
namespace impl
{

namespace acos_fn_ns = dpctl::tensor::kernels::acos;

static unary_contig_impl_fn_ptr_t acos_contig_dispatch_vector[td_ns::num_types];
static int acos_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    acos_strided_dispatch_vector[td_ns::num_types];

void populate_acos_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = acos_fn_ns;

    using fn_ns::AcosContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, AcosContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(acos_contig_dispatch_vector);

    using fn_ns::AcosStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, AcosStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(acos_strided_dispatch_vector);

    using fn_ns::AcosTypeMapFactory;
    DispatchVectorBuilder<int, AcosTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(acos_output_typeid_vector);
}

} // namespace impl

// U03: ===== ACOSH (x)
namespace impl
{

namespace acosh_fn_ns = dpctl::tensor::kernels::acosh;

static unary_contig_impl_fn_ptr_t
    acosh_contig_dispatch_vector[td_ns::num_types];
static int acosh_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    acosh_strided_dispatch_vector[td_ns::num_types];

void populate_acosh_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = acosh_fn_ns;

    using fn_ns::AcoshContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, AcoshContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(acosh_contig_dispatch_vector);

    using fn_ns::AcoshStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, AcoshStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(acosh_strided_dispatch_vector);

    using fn_ns::AcoshTypeMapFactory;
    DispatchVectorBuilder<int, AcoshTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(acosh_output_typeid_vector);
}

} // namespace impl

// B01: ===== ADD   (x1, x2)
namespace impl
{
namespace add_fn_ns = dpctl::tensor::kernels::add;

static binary_contig_impl_fn_ptr_t add_contig_dispatch_table[td_ns::num_types]
                                                            [td_ns::num_types];
static int add_output_id_table[td_ns::num_types][td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    add_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

// add(matrix, row)
static binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t
    add_contig_matrix_contig_row_broadcast_dispatch_table[td_ns::num_types]
                                                         [td_ns::num_types];

// add(row, matrix)
static binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t
    add_contig_row_contig_matrix_broadcast_dispatch_table[td_ns::num_types]
                                                         [td_ns::num_types];

static binary_inplace_contig_impl_fn_ptr_t
    add_inplace_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static binary_inplace_strided_impl_fn_ptr_t
    add_inplace_strided_dispatch_table[td_ns::num_types][td_ns::num_types];
static binary_inplace_row_matrix_broadcast_impl_fn_ptr_t
    add_inplace_row_matrix_dispatch_table[td_ns::num_types][td_ns::num_types];

void populate_add_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = add_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::AddTypeMapFactory;
    DispatchTableBuilder<int, AddTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(add_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::AddStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t, AddStridedFactory,
                         num_types>
        dtb2;
    dtb2.populate_dispatch_table(add_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::AddContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, AddContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(add_contig_dispatch_table);

    // function pointers for operation on contiguous matrix, contiguous row
    // with contiguous matrix output
    using fn_ns::AddContigMatrixContigRowBroadcastFactory;
    DispatchTableBuilder<
        binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t,
        AddContigMatrixContigRowBroadcastFactory, num_types>
        dtb4;
    dtb4.populate_dispatch_table(
        add_contig_matrix_contig_row_broadcast_dispatch_table);

    // function pointers for operation on contiguous row, contiguous matrix
    // with contiguous matrix output
    using fn_ns::AddContigRowContigMatrixBroadcastFactory;
    DispatchTableBuilder<
        binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t,
        AddContigRowContigMatrixBroadcastFactory, num_types>
        dtb5;
    dtb5.populate_dispatch_table(
        add_contig_row_contig_matrix_broadcast_dispatch_table);

    // function pointers for inplace operation on general strided arrays
    using fn_ns::AddInplaceStridedFactory;
    DispatchTableBuilder<binary_inplace_strided_impl_fn_ptr_t,
                         AddInplaceStridedFactory, num_types>
        dtb6;
    dtb6.populate_dispatch_table(add_inplace_strided_dispatch_table);

    // function pointers for inplace operation on contiguous inputs and output
    using fn_ns::AddInplaceContigFactory;
    DispatchTableBuilder<binary_inplace_contig_impl_fn_ptr_t,
                         AddInplaceContigFactory, num_types>
        dtb7;
    dtb7.populate_dispatch_table(add_inplace_contig_dispatch_table);

    // function pointers for inplace operation on contiguous matrix
    // and contiguous row
    using fn_ns::AddInplaceRowMatrixBroadcastFactory;
    DispatchTableBuilder<binary_inplace_row_matrix_broadcast_impl_fn_ptr_t,
                         AddInplaceRowMatrixBroadcastFactory, num_types>
        dtb8;
    dtb8.populate_dispatch_table(add_inplace_row_matrix_dispatch_table);
};

} // namespace impl

// U04: ===== ASIN  (x)
namespace impl
{

namespace asin_fn_ns = dpctl::tensor::kernels::asin;

static unary_contig_impl_fn_ptr_t asin_contig_dispatch_vector[td_ns::num_types];
static int asin_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    asin_strided_dispatch_vector[td_ns::num_types];

void populate_asin_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = asin_fn_ns;

    using fn_ns::AsinContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, AsinContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(asin_contig_dispatch_vector);

    using fn_ns::AsinStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, AsinStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(asin_strided_dispatch_vector);

    using fn_ns::AsinTypeMapFactory;
    DispatchVectorBuilder<int, AsinTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(asin_output_typeid_vector);
}

} // namespace impl

// U05: ===== ASINH (x)
namespace impl
{

namespace asinh_fn_ns = dpctl::tensor::kernels::asinh;

static unary_contig_impl_fn_ptr_t
    asinh_contig_dispatch_vector[td_ns::num_types];
static int asinh_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    asinh_strided_dispatch_vector[td_ns::num_types];

void populate_asinh_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = asinh_fn_ns;

    using fn_ns::AsinhContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, AsinhContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(asinh_contig_dispatch_vector);

    using fn_ns::AsinhStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, AsinhStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(asinh_strided_dispatch_vector);

    using fn_ns::AsinhTypeMapFactory;
    DispatchVectorBuilder<int, AsinhTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(asinh_output_typeid_vector);
}

} // namespace impl

// U06: ===== ATAN  (x)
namespace impl
{

namespace atan_fn_ns = dpctl::tensor::kernels::atan;

static unary_contig_impl_fn_ptr_t atan_contig_dispatch_vector[td_ns::num_types];
static int atan_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    atan_strided_dispatch_vector[td_ns::num_types];

void populate_atan_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = atan_fn_ns;

    using fn_ns::AtanContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, AtanContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(atan_contig_dispatch_vector);

    using fn_ns::AtanStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, AtanStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(atan_strided_dispatch_vector);

    using fn_ns::AtanTypeMapFactory;
    DispatchVectorBuilder<int, AtanTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(atan_output_typeid_vector);
}

} // namespace impl

// B02: ===== ATAN2 (x1, x2)
namespace impl
{
namespace atan2_fn_ns = dpctl::tensor::kernels::atan2;

static binary_contig_impl_fn_ptr_t
    atan2_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static int atan2_output_id_table[td_ns::num_types][td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    atan2_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

void populate_atan2_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = atan2_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::Atan2TypeMapFactory;
    DispatchTableBuilder<int, Atan2TypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(atan2_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::Atan2StridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t, Atan2StridedFactory,
                         num_types>
        dtb2;
    dtb2.populate_dispatch_table(atan2_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::Atan2ContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, Atan2ContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(atan2_contig_dispatch_table);
};

} // namespace impl

// U07: ===== ATANH (x)
namespace impl
{

namespace atanh_fn_ns = dpctl::tensor::kernels::atanh;

static unary_contig_impl_fn_ptr_t
    atanh_contig_dispatch_vector[td_ns::num_types];
static int atanh_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    atanh_strided_dispatch_vector[td_ns::num_types];

void populate_atanh_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = atanh_fn_ns;

    using fn_ns::AtanhContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, AtanhContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(atanh_contig_dispatch_vector);

    using fn_ns::AtanhStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, AtanhStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(atanh_strided_dispatch_vector);

    using fn_ns::AtanhTypeMapFactory;
    DispatchVectorBuilder<int, AtanhTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(atanh_output_typeid_vector);
}

} // namespace impl

// B03: ===== BITWISE_AND           (x1, x2)
namespace impl
{
namespace bitwise_and_fn_ns = dpctl::tensor::kernels::bitwise_and;

static binary_contig_impl_fn_ptr_t
    bitwise_and_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static int bitwise_and_output_id_table[td_ns::num_types][td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    bitwise_and_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

static binary_inplace_contig_impl_fn_ptr_t
    bitwise_and_inplace_contig_dispatch_table[td_ns::num_types]
                                             [td_ns::num_types];
static binary_inplace_strided_impl_fn_ptr_t
    bitwise_and_inplace_strided_dispatch_table[td_ns::num_types]
                                              [td_ns::num_types];

void populate_bitwise_and_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = bitwise_and_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::BitwiseAndTypeMapFactory;
    DispatchTableBuilder<int, BitwiseAndTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(bitwise_and_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::BitwiseAndStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t, BitwiseAndStridedFactory,
                         num_types>
        dtb2;
    dtb2.populate_dispatch_table(bitwise_and_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::BitwiseAndContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, BitwiseAndContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(bitwise_and_contig_dispatch_table);

    // function pointers for inplace operation on general strided arrays
    using fn_ns::BitwiseAndInplaceStridedFactory;
    DispatchTableBuilder<binary_inplace_strided_impl_fn_ptr_t,
                         BitwiseAndInplaceStridedFactory, num_types>
        dtb4;
    dtb4.populate_dispatch_table(bitwise_and_inplace_strided_dispatch_table);

    // function pointers for inplace operation on contiguous inputs and output
    using fn_ns::BitwiseAndInplaceContigFactory;
    DispatchTableBuilder<binary_inplace_contig_impl_fn_ptr_t,
                         BitwiseAndInplaceContigFactory, num_types>
        dtb5;
    dtb5.populate_dispatch_table(bitwise_and_inplace_contig_dispatch_table);
};

} // namespace impl

// B04: ===== BITWISE_LEFT_SHIFT    (x1, x2)
namespace impl
{
namespace bitwise_left_shift_fn_ns = dpctl::tensor::kernels::bitwise_left_shift;

static binary_contig_impl_fn_ptr_t
    bitwise_left_shift_contig_dispatch_table[td_ns::num_types]
                                            [td_ns::num_types];
static int bitwise_left_shift_output_id_table[td_ns::num_types]
                                             [td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    bitwise_left_shift_strided_dispatch_table[td_ns::num_types]
                                             [td_ns::num_types];

static binary_inplace_contig_impl_fn_ptr_t
    bitwise_left_shift_inplace_contig_dispatch_table[td_ns::num_types]
                                                    [td_ns::num_types];
static binary_inplace_strided_impl_fn_ptr_t
    bitwise_left_shift_inplace_strided_dispatch_table[td_ns::num_types]
                                                     [td_ns::num_types];

void populate_bitwise_left_shift_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = bitwise_left_shift_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::BitwiseLeftShiftTypeMapFactory;
    DispatchTableBuilder<int, BitwiseLeftShiftTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(bitwise_left_shift_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::BitwiseLeftShiftStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t,
                         BitwiseLeftShiftStridedFactory, num_types>
        dtb2;
    dtb2.populate_dispatch_table(bitwise_left_shift_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::BitwiseLeftShiftContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t,
                         BitwiseLeftShiftContigFactory, num_types>
        dtb3;
    dtb3.populate_dispatch_table(bitwise_left_shift_contig_dispatch_table);

    // function pointers for inplace operation on general strided arrays
    using fn_ns::BitwiseLeftShiftInplaceStridedFactory;
    DispatchTableBuilder<binary_inplace_strided_impl_fn_ptr_t,
                         BitwiseLeftShiftInplaceStridedFactory, num_types>
        dtb4;
    dtb4.populate_dispatch_table(
        bitwise_left_shift_inplace_strided_dispatch_table);

    // function pointers for inplace operation on contiguous inputs and output
    using fn_ns::BitwiseLeftShiftInplaceContigFactory;
    DispatchTableBuilder<binary_inplace_contig_impl_fn_ptr_t,
                         BitwiseLeftShiftInplaceContigFactory, num_types>
        dtb5;
    dtb5.populate_dispatch_table(
        bitwise_left_shift_inplace_contig_dispatch_table);
};

} // namespace impl

// U08: ===== BITWISE_INVERT        (x)
namespace impl
{

namespace bitwise_invert_fn_ns = dpctl::tensor::kernels::bitwise_invert;

static unary_contig_impl_fn_ptr_t
    bitwise_invert_contig_dispatch_vector[td_ns::num_types];
static int bitwise_invert_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    bitwise_invert_strided_dispatch_vector[td_ns::num_types];

void populate_bitwise_invert_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = bitwise_invert_fn_ns;

    using fn_ns::BitwiseInvertContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t,
                          BitwiseInvertContigFactory, num_types>
        dvb1;
    dvb1.populate_dispatch_vector(bitwise_invert_contig_dispatch_vector);

    using fn_ns::BitwiseInvertStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t,
                          BitwiseInvertStridedFactory, num_types>
        dvb2;
    dvb2.populate_dispatch_vector(bitwise_invert_strided_dispatch_vector);

    using fn_ns::BitwiseInvertTypeMapFactory;
    DispatchVectorBuilder<int, BitwiseInvertTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(bitwise_invert_output_typeid_vector);
};

} // namespace impl

// B05: ===== BITWISE_OR            (x1, x2)
namespace impl
{
namespace bitwise_or_fn_ns = dpctl::tensor::kernels::bitwise_or;

static binary_contig_impl_fn_ptr_t
    bitwise_or_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static int bitwise_or_output_id_table[td_ns::num_types][td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    bitwise_or_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

static binary_inplace_contig_impl_fn_ptr_t
    bitwise_or_inplace_contig_dispatch_table[td_ns::num_types]
                                            [td_ns::num_types];
static binary_inplace_strided_impl_fn_ptr_t
    bitwise_or_inplace_strided_dispatch_table[td_ns::num_types]
                                             [td_ns::num_types];

void populate_bitwise_or_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = bitwise_or_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::BitwiseOrTypeMapFactory;
    DispatchTableBuilder<int, BitwiseOrTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(bitwise_or_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::BitwiseOrStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t, BitwiseOrStridedFactory,
                         num_types>
        dtb2;
    dtb2.populate_dispatch_table(bitwise_or_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::BitwiseOrContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, BitwiseOrContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(bitwise_or_contig_dispatch_table);

    // function pointers for inplace operation on general strided arrays
    using fn_ns::BitwiseOrInplaceStridedFactory;
    DispatchTableBuilder<binary_inplace_strided_impl_fn_ptr_t,
                         BitwiseOrInplaceStridedFactory, num_types>
        dtb4;
    dtb4.populate_dispatch_table(bitwise_or_inplace_strided_dispatch_table);

    // function pointers for inplace operation on contiguous inputs and output
    using fn_ns::BitwiseOrInplaceContigFactory;
    DispatchTableBuilder<binary_inplace_contig_impl_fn_ptr_t,
                         BitwiseOrInplaceContigFactory, num_types>
        dtb5;
    dtb5.populate_dispatch_table(bitwise_or_inplace_contig_dispatch_table);
};
} // namespace impl

// B06: ===== BITWISE_RIGHT_SHIFT   (x1, x2)
namespace impl
{
namespace bitwise_right_shift_fn_ns =
    dpctl::tensor::kernels::bitwise_right_shift;

static binary_contig_impl_fn_ptr_t
    bitwise_right_shift_contig_dispatch_table[td_ns::num_types]
                                             [td_ns::num_types];
static int bitwise_right_shift_output_id_table[td_ns::num_types]
                                              [td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    bitwise_right_shift_strided_dispatch_table[td_ns::num_types]
                                              [td_ns::num_types];

static binary_inplace_contig_impl_fn_ptr_t
    bitwise_right_shift_inplace_contig_dispatch_table[td_ns::num_types]
                                                     [td_ns::num_types];
static binary_inplace_strided_impl_fn_ptr_t
    bitwise_right_shift_inplace_strided_dispatch_table[td_ns::num_types]
                                                      [td_ns::num_types];

void populate_bitwise_right_shift_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = bitwise_right_shift_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::BitwiseRightShiftTypeMapFactory;
    DispatchTableBuilder<int, BitwiseRightShiftTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(bitwise_right_shift_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::BitwiseRightShiftStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t,
                         BitwiseRightShiftStridedFactory, num_types>
        dtb2;
    dtb2.populate_dispatch_table(bitwise_right_shift_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::BitwiseRightShiftContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t,
                         BitwiseRightShiftContigFactory, num_types>
        dtb3;
    dtb3.populate_dispatch_table(bitwise_right_shift_contig_dispatch_table);

    // function pointers for inplace operation on general strided arrays
    using fn_ns::BitwiseRightShiftInplaceStridedFactory;
    DispatchTableBuilder<binary_inplace_strided_impl_fn_ptr_t,
                         BitwiseRightShiftInplaceStridedFactory, num_types>
        dtb4;
    dtb4.populate_dispatch_table(
        bitwise_right_shift_inplace_strided_dispatch_table);

    // function pointers for inplace operation on contiguous inputs and output
    using fn_ns::BitwiseRightShiftInplaceContigFactory;
    DispatchTableBuilder<binary_inplace_contig_impl_fn_ptr_t,
                         BitwiseRightShiftInplaceContigFactory, num_types>
        dtb5;
    dtb5.populate_dispatch_table(
        bitwise_right_shift_inplace_contig_dispatch_table);
};

} // namespace impl

// B07: ===== BITWISE_XOR           (x1, x2)
namespace impl
{
namespace bitwise_xor_fn_ns = dpctl::tensor::kernels::bitwise_xor;

static binary_contig_impl_fn_ptr_t
    bitwise_xor_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static int bitwise_xor_output_id_table[td_ns::num_types][td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    bitwise_xor_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

static binary_inplace_contig_impl_fn_ptr_t
    bitwise_xor_inplace_contig_dispatch_table[td_ns::num_types]
                                             [td_ns::num_types];
static binary_inplace_strided_impl_fn_ptr_t
    bitwise_xor_inplace_strided_dispatch_table[td_ns::num_types]
                                              [td_ns::num_types];

void populate_bitwise_xor_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = bitwise_xor_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::BitwiseXorTypeMapFactory;
    DispatchTableBuilder<int, BitwiseXorTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(bitwise_xor_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::BitwiseXorStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t, BitwiseXorStridedFactory,
                         num_types>
        dtb2;
    dtb2.populate_dispatch_table(bitwise_xor_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::BitwiseXorContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, BitwiseXorContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(bitwise_xor_contig_dispatch_table);

    // function pointers for inplace operation on general strided arrays
    using fn_ns::BitwiseXorInplaceStridedFactory;
    DispatchTableBuilder<binary_inplace_strided_impl_fn_ptr_t,
                         BitwiseXorInplaceStridedFactory, num_types>
        dtb4;
    dtb4.populate_dispatch_table(bitwise_xor_inplace_strided_dispatch_table);

    // function pointers for inplace operation on contiguous inputs and output
    using fn_ns::BitwiseXorInplaceContigFactory;
    DispatchTableBuilder<binary_inplace_contig_impl_fn_ptr_t,
                         BitwiseXorInplaceContigFactory, num_types>
        dtb5;
    dtb5.populate_dispatch_table(bitwise_xor_inplace_contig_dispatch_table);
};
} // namespace impl

// ==========================================================================================
// //

namespace py = pybind11;

void init_elementwise_functions2(py::module_ m)
{

    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;

    // U01: ==== ABS   (x)
    {
        impl::populate_abs_dispatch_vectors();
        using impl::abs_contig_dispatch_vector;
        using impl::abs_output_typeid_vector;
        using impl::abs_strided_dispatch_vector;

        auto abs_pyapi = [&](const arrayT &src, const arrayT &dst,
                             sycl::queue &exec_q,
                             const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, abs_output_typeid_vector,
                abs_contig_dispatch_vector, abs_strided_dispatch_vector);
        };
        m.def("_abs", abs_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto abs_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype, abs_output_typeid_vector);
        };
        m.def("_abs_result_type", abs_result_type_pyapi);
    }

    // U02: ==== ACOS   (x)
    {
        impl::populate_acos_dispatch_vectors();
        using impl::acos_contig_dispatch_vector;
        using impl::acos_output_typeid_vector;
        using impl::acos_strided_dispatch_vector;

        auto acos_pyapi = [&](const arrayT &src, const arrayT &dst,
                              sycl::queue &exec_q,
                              const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, acos_output_typeid_vector,
                acos_contig_dispatch_vector, acos_strided_dispatch_vector);
        };
        m.def("_acos", acos_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto acos_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype, acos_output_typeid_vector);
        };
        m.def("_acos_result_type", acos_result_type_pyapi);
    }

    // U03: ===== ACOSH (x)
    {
        impl::populate_acosh_dispatch_vectors();
        using impl::acosh_contig_dispatch_vector;
        using impl::acosh_output_typeid_vector;
        using impl::acosh_strided_dispatch_vector;

        auto acosh_pyapi = [&](const arrayT &src, const arrayT &dst,
                               sycl::queue &exec_q,
                               const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, acosh_output_typeid_vector,
                acosh_contig_dispatch_vector, acosh_strided_dispatch_vector);
        };
        m.def("_acosh", acosh_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto acosh_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype,
                                              acosh_output_typeid_vector);
        };
        m.def("_acosh_result_type", acosh_result_type_pyapi);
    }

    // B01: ===== ADD   (x1, x2)
    {
        impl::populate_add_dispatch_tables();
        using impl::add_contig_dispatch_table;
        using impl::add_contig_matrix_contig_row_broadcast_dispatch_table;
        using impl::add_contig_row_contig_matrix_broadcast_dispatch_table;
        using impl::add_output_id_table;
        using impl::add_strided_dispatch_table;

        auto add_pyapi = [&](const dpctl::tensor::usm_ndarray &src1,
                             const dpctl::tensor::usm_ndarray &src2,
                             const dpctl::tensor::usm_ndarray &dst,
                             sycl::queue &exec_q,
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
                add_contig_matrix_contig_row_broadcast_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                add_contig_row_contig_matrix_broadcast_dispatch_table);
        };
        auto add_result_type_pyapi = [&](const py::dtype &dtype1,
                                         const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               add_output_id_table);
        };
        m.def("_add", add_pyapi, "", py::arg("src1"), py::arg("src2"),
              py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_add_result_type", add_result_type_pyapi, "");

        using impl::add_inplace_contig_dispatch_table;
        using impl::add_inplace_row_matrix_dispatch_table;
        using impl::add_inplace_strided_dispatch_table;

        auto add_inplace_pyapi =
            [&](const dpctl::tensor::usm_ndarray &src,
                const dpctl::tensor::usm_ndarray &dst, sycl::queue &exec_q,
                const std::vector<sycl::event> &depends = {}) {
                return py_binary_inplace_ufunc(
                    src, dst, exec_q, depends, add_output_id_table,
                    // function pointers to handle inplace operation on
                    // contiguous arrays (pointers may be nullptr)
                    add_inplace_contig_dispatch_table,
                    // function pointers to handle inplace operation on strided
                    // arrays (most general case)
                    add_inplace_strided_dispatch_table,
                    // function pointers to handle inplace operation on
                    // c-contig matrix with c-contig row with broadcasting
                    // (may be nullptr)
                    add_inplace_row_matrix_dispatch_table);
            };
        m.def("_add_inplace", add_inplace_pyapi, "", py::arg("lhs"),
              py::arg("rhs"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
    }

    // U04: ===== ASIN  (x)
    {
        impl::populate_asin_dispatch_vectors();
        using impl::asin_contig_dispatch_vector;
        using impl::asin_output_typeid_vector;
        using impl::asin_strided_dispatch_vector;

        auto asin_pyapi = [&](const arrayT &src, const arrayT &dst,
                              sycl::queue &exec_q,
                              const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, asin_output_typeid_vector,
                asin_contig_dispatch_vector, asin_strided_dispatch_vector);
        };
        m.def("_asin", asin_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto asin_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype, asin_output_typeid_vector);
        };
        m.def("_asin_result_type", asin_result_type_pyapi);
    }

    // U05: ===== ASINH (x)
    {
        impl::populate_asinh_dispatch_vectors();
        using impl::asinh_contig_dispatch_vector;
        using impl::asinh_output_typeid_vector;
        using impl::asinh_strided_dispatch_vector;

        auto asinh_pyapi = [&](const arrayT &src, const arrayT &dst,
                               sycl::queue &exec_q,
                               const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, asinh_output_typeid_vector,
                asinh_contig_dispatch_vector, asinh_strided_dispatch_vector);
        };
        m.def("_asinh", asinh_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto asinh_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype,
                                              asinh_output_typeid_vector);
        };
        m.def("_asinh_result_type", asinh_result_type_pyapi);
    }

    // U06: ===== ATAN  (x)
    {
        impl::populate_atan_dispatch_vectors();
        using impl::atan_contig_dispatch_vector;
        using impl::atan_output_typeid_vector;
        using impl::atan_strided_dispatch_vector;

        auto atan_pyapi = [&](arrayT src, arrayT dst, sycl::queue &exec_q,
                              const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, atan_output_typeid_vector,
                atan_contig_dispatch_vector, atan_strided_dispatch_vector);
        };
        m.def("_atan", atan_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto atan_result_type_pyapi = [&](py::dtype dtype) {
            return py_unary_ufunc_result_type(dtype, atan_output_typeid_vector);
        };
        m.def("_atan_result_type", atan_result_type_pyapi);
    }

    // B02: ===== ATAN2 (x1, x2)
    {
        impl::populate_atan2_dispatch_tables();
        using impl::atan2_contig_dispatch_table;
        using impl::atan2_output_id_table;
        using impl::atan2_strided_dispatch_table;

        auto atan2_pyapi = [&](const dpctl::tensor::usm_ndarray &src1,
                               const dpctl::tensor::usm_ndarray &src2,
                               const dpctl::tensor::usm_ndarray &dst,
                               sycl::queue &exec_q,
                               const std::vector<sycl::event> &depends = {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends, atan2_output_id_table,
                // function pointers to handle operation on contiguous arrays
                // (pointers may be nullptr)
                atan2_contig_dispatch_table,
                // function pointers to handle operation on strided arrays (most
                // general case)
                atan2_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t>{},
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t>{});
        };
        auto atan2_result_type_pyapi = [&](const py::dtype &dtype1,
                                           const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               atan2_output_id_table);
        };
        m.def("_atan2", atan2_pyapi, "", py::arg("src1"), py::arg("src2"),
              py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_atan2_result_type", atan2_result_type_pyapi, "");
    }

    // U07: ===== ATANH (x)
    {
        impl::populate_atanh_dispatch_vectors();
        using impl::atanh_contig_dispatch_vector;
        using impl::atanh_output_typeid_vector;
        using impl::atanh_strided_dispatch_vector;

        auto atanh_pyapi = [&](const arrayT &src, const arrayT &dst,
                               sycl::queue &exec_q,
                               const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, atanh_output_typeid_vector,
                atanh_contig_dispatch_vector, atanh_strided_dispatch_vector);
        };
        m.def("_atanh", atanh_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto atanh_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype,
                                              atanh_output_typeid_vector);
        };
        m.def("_atanh_result_type", atanh_result_type_pyapi);
    }

    // B03: ===== BITWISE_AND           (x1, x2)
    {
        impl::populate_bitwise_and_dispatch_tables();
        using impl::bitwise_and_contig_dispatch_table;
        using impl::bitwise_and_output_id_table;
        using impl::bitwise_and_strided_dispatch_table;

        auto bitwise_and_pyapi = [&](const dpctl::tensor::usm_ndarray &src1,
                                     const dpctl::tensor::usm_ndarray &src2,
                                     const dpctl::tensor::usm_ndarray &dst,
                                     sycl::queue &exec_q,
                                     const std::vector<sycl::event> &depends =
                                         {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends, bitwise_and_output_id_table,
                // function pointers to handle operation on contiguous arrays
                // (pointers may be nullptr)
                bitwise_and_contig_dispatch_table,
                // function pointers to handle operation on strided arrays (most
                // general case)
                bitwise_and_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t>{},
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t>{});
        };
        auto bitwise_and_result_type_pyapi = [&](const py::dtype &dtype1,
                                                 const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               bitwise_and_output_id_table);
        };
        m.def("_bitwise_and", bitwise_and_pyapi, "", py::arg("src1"),
              py::arg("src2"), py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_bitwise_and_result_type", bitwise_and_result_type_pyapi, "");

        using impl::bitwise_and_inplace_contig_dispatch_table;
        using impl::bitwise_and_inplace_strided_dispatch_table;

        auto bitwise_and_inplace_pyapi =
            [&](const dpctl::tensor::usm_ndarray &src,
                const dpctl::tensor::usm_ndarray &dst, sycl::queue &exec_q,
                const std::vector<sycl::event> &depends = {}) {
                return py_binary_inplace_ufunc(
                    src, dst, exec_q, depends, bitwise_and_output_id_table,
                    // function pointers to handle inplace operation on
                    // contiguous arrays (pointers may be nullptr)
                    bitwise_and_inplace_contig_dispatch_table,
                    // function pointers to handle inplace operation on strided
                    // arrays (most general case)
                    bitwise_and_inplace_strided_dispatch_table,
                    // function pointers to handle inplace operation on
                    // c-contig matrix with c-contig row with broadcasting
                    // (may be nullptr)
                    td_ns::NullPtrTable<
                        binary_inplace_row_matrix_broadcast_impl_fn_ptr_t>{});
            };
        m.def("_bitwise_and_inplace", bitwise_and_inplace_pyapi, "",
              py::arg("lhs"), py::arg("rhs"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
    }

    // B04: ===== BITWISE_LEFT_SHIFT    (x1, x2)
    {
        impl::populate_bitwise_left_shift_dispatch_tables();
        using impl::bitwise_left_shift_contig_dispatch_table;
        using impl::bitwise_left_shift_output_id_table;
        using impl::bitwise_left_shift_strided_dispatch_table;

        auto bitwise_left_shift_pyapi = [&](const dpctl::tensor::usm_ndarray
                                                &src1,
                                            const dpctl::tensor::usm_ndarray
                                                &src2,
                                            const dpctl::tensor::usm_ndarray
                                                &dst,
                                            sycl::queue &exec_q,
                                            const std::vector<sycl::event>
                                                &depends = {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends,
                bitwise_left_shift_output_id_table,
                // function pointers to handle operation on contiguous arrays
                // (pointers may be nullptr)
                bitwise_left_shift_contig_dispatch_table,
                // function pointers to handle operation on strided arrays (most
                // general case)
                bitwise_left_shift_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t>{},
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t>{});
        };
        auto bitwise_left_shift_result_type_pyapi =
            [&](const py::dtype &dtype1, const py::dtype &dtype2) {
                return py_binary_ufunc_result_type(
                    dtype1, dtype2, bitwise_left_shift_output_id_table);
            };
        m.def("_bitwise_left_shift", bitwise_left_shift_pyapi, "",
              py::arg("src1"), py::arg("src2"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());
        m.def("_bitwise_left_shift_result_type",
              bitwise_left_shift_result_type_pyapi, "");

        using impl::bitwise_left_shift_inplace_contig_dispatch_table;
        using impl::bitwise_left_shift_inplace_strided_dispatch_table;

        auto bitwise_left_shift_inplace_pyapi =
            [&](const dpctl::tensor::usm_ndarray &src,
                const dpctl::tensor::usm_ndarray &dst, sycl::queue &exec_q,
                const std::vector<sycl::event> &depends = {}) {
                return py_binary_inplace_ufunc(
                    src, dst, exec_q, depends,
                    bitwise_left_shift_output_id_table,
                    // function pointers to handle inplace operation on
                    // contiguous arrays (pointers may be nullptr)
                    bitwise_left_shift_inplace_contig_dispatch_table,
                    // function pointers to handle inplace operation on strided
                    // arrays (most general case)
                    bitwise_left_shift_inplace_strided_dispatch_table,
                    // function pointers to handle inplace operation on
                    // c-contig matrix with c-contig row with broadcasting
                    // (may be nullptr)
                    td_ns::NullPtrTable<
                        binary_inplace_row_matrix_broadcast_impl_fn_ptr_t>{});
            };
        m.def("_bitwise_left_shift_inplace", bitwise_left_shift_inplace_pyapi,
              "", py::arg("lhs"), py::arg("rhs"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
    }

    // U08: ===== BITWISE_INVERT        (x)
    {
        impl::populate_bitwise_invert_dispatch_vectors();
        using impl::bitwise_invert_contig_dispatch_vector;
        using impl::bitwise_invert_output_typeid_vector;
        using impl::bitwise_invert_strided_dispatch_vector;

        auto bitwise_invert_pyapi = [&](const arrayT &src, const arrayT &dst,
                                        sycl::queue &exec_q,
                                        const event_vecT &depends = {}) {
            return py_unary_ufunc(src, dst, exec_q, depends,
                                  bitwise_invert_output_typeid_vector,
                                  bitwise_invert_contig_dispatch_vector,
                                  bitwise_invert_strided_dispatch_vector);
        };
        m.def("_bitwise_invert", bitwise_invert_pyapi, "", py::arg("src"),
              py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());

        auto bitwise_invert_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(
                dtype, bitwise_invert_output_typeid_vector);
        };
        m.def("_bitwise_invert_result_type", bitwise_invert_result_type_pyapi);
    }

    // B05: ===== BITWISE_OR            (x1, x2)
    {
        impl::populate_bitwise_or_dispatch_tables();
        using impl::bitwise_or_contig_dispatch_table;
        using impl::bitwise_or_output_id_table;
        using impl::bitwise_or_strided_dispatch_table;

        auto bitwise_or_pyapi = [&](const dpctl::tensor::usm_ndarray &src1,
                                    const dpctl::tensor::usm_ndarray &src2,
                                    const dpctl::tensor::usm_ndarray &dst,
                                    sycl::queue &exec_q,
                                    const std::vector<sycl::event> &depends =
                                        {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends, bitwise_or_output_id_table,
                // function pointers to handle operation on contiguous arrays
                // (pointers may be nullptr)
                bitwise_or_contig_dispatch_table,
                // function pointers to handle operation on strided arrays (most
                // general case)
                bitwise_or_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t>{},
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t>{});
        };
        auto bitwise_or_result_type_pyapi = [&](const py::dtype &dtype1,
                                                const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               bitwise_or_output_id_table);
        };
        m.def("_bitwise_or", bitwise_or_pyapi, "", py::arg("src1"),
              py::arg("src2"), py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_bitwise_or_result_type", bitwise_or_result_type_pyapi, "");

        using impl::bitwise_or_inplace_contig_dispatch_table;
        using impl::bitwise_or_inplace_strided_dispatch_table;

        auto bitwise_or_inplace_pyapi =
            [&](const dpctl::tensor::usm_ndarray &src,
                const dpctl::tensor::usm_ndarray &dst, sycl::queue &exec_q,
                const std::vector<sycl::event> &depends = {}) {
                return py_binary_inplace_ufunc(
                    src, dst, exec_q, depends, bitwise_or_output_id_table,
                    // function pointers to handle inplace operation on
                    // contiguous arrays (pointers may be nullptr)
                    bitwise_or_inplace_contig_dispatch_table,
                    // function pointers to handle inplace operation on strided
                    // arrays (most general case)
                    bitwise_or_inplace_strided_dispatch_table,
                    // function pointers to handle inplace operation on
                    // c-contig matrix with c-contig row with broadcasting
                    // (may be nullptr)
                    td_ns::NullPtrTable<
                        binary_inplace_row_matrix_broadcast_impl_fn_ptr_t>{});
            };
        m.def("_bitwise_or_inplace", bitwise_or_inplace_pyapi, "",
              py::arg("lhs"), py::arg("rhs"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
    }

    // B06: ===== BITWISE_RIGHT_SHIFT   (x1, x2)
    {
        impl::populate_bitwise_right_shift_dispatch_tables();
        using impl::bitwise_right_shift_contig_dispatch_table;
        using impl::bitwise_right_shift_output_id_table;
        using impl::bitwise_right_shift_strided_dispatch_table;

        auto bitwise_right_shift_pyapi = [&](const dpctl::tensor::usm_ndarray
                                                 &src1,
                                             const dpctl::tensor::usm_ndarray
                                                 &src2,
                                             const dpctl::tensor::usm_ndarray
                                                 &dst,
                                             sycl::queue &exec_q,
                                             const std::vector<sycl::event>
                                                 &depends = {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends,
                bitwise_right_shift_output_id_table,
                // function pointers to handle operation on contiguous arrays
                // (pointers may be nullptr)
                bitwise_right_shift_contig_dispatch_table,
                // function pointers to handle operation on strided arrays (most
                // general case)
                bitwise_right_shift_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t>{},
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t>{});
        };
        auto bitwise_right_shift_result_type_pyapi =
            [&](const py::dtype &dtype1, const py::dtype &dtype2) {
                return py_binary_ufunc_result_type(
                    dtype1, dtype2, bitwise_right_shift_output_id_table);
            };
        m.def("_bitwise_right_shift", bitwise_right_shift_pyapi, "",
              py::arg("src1"), py::arg("src2"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());
        m.def("_bitwise_right_shift_result_type",
              bitwise_right_shift_result_type_pyapi, "");

        using impl::bitwise_right_shift_inplace_contig_dispatch_table;
        using impl::bitwise_right_shift_inplace_strided_dispatch_table;

        auto bitwise_right_shift_inplace_pyapi =
            [&](const dpctl::tensor::usm_ndarray &src,
                const dpctl::tensor::usm_ndarray &dst, sycl::queue &exec_q,
                const std::vector<sycl::event> &depends = {}) {
                return py_binary_inplace_ufunc(
                    src, dst, exec_q, depends,
                    bitwise_right_shift_output_id_table,
                    // function pointers to handle inplace operation on
                    // contiguous arrays (pointers may be nullptr)
                    bitwise_right_shift_inplace_contig_dispatch_table,
                    // function pointers to handle inplace operation on strided
                    // arrays (most general case)
                    bitwise_right_shift_inplace_strided_dispatch_table,
                    // function pointers to handle inplace operation on
                    // c-contig matrix with c-contig row with broadcasting
                    // (may be nullptr)
                    td_ns::NullPtrTable<
                        binary_inplace_row_matrix_broadcast_impl_fn_ptr_t>{});
            };
        m.def("_bitwise_right_shift_inplace", bitwise_right_shift_inplace_pyapi,
              "", py::arg("lhs"), py::arg("rhs"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
    }

    // B07: ===== BITWISE_XOR           (x1, x2)
    {
        impl::populate_bitwise_xor_dispatch_tables();
        using impl::bitwise_xor_contig_dispatch_table;
        using impl::bitwise_xor_output_id_table;
        using impl::bitwise_xor_strided_dispatch_table;

        auto bitwise_xor_pyapi = [&](const dpctl::tensor::usm_ndarray &src1,
                                     const dpctl::tensor::usm_ndarray &src2,
                                     const dpctl::tensor::usm_ndarray &dst,
                                     sycl::queue &exec_q,
                                     const std::vector<sycl::event> &depends =
                                         {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends, bitwise_xor_output_id_table,
                // function pointers to handle operation on contiguous arrays
                // (pointers may be nullptr)
                bitwise_xor_contig_dispatch_table,
                // function pointers to handle operation on strided arrays (most
                // general case)
                bitwise_xor_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t>{},
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t>{});
        };
        auto bitwise_xor_result_type_pyapi = [&](const py::dtype &dtype1,
                                                 const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               bitwise_xor_output_id_table);
        };
        m.def("_bitwise_xor", bitwise_xor_pyapi, "", py::arg("src1"),
              py::arg("src2"), py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_bitwise_xor_result_type", bitwise_xor_result_type_pyapi, "");

        using impl::bitwise_xor_inplace_contig_dispatch_table;
        using impl::bitwise_xor_inplace_strided_dispatch_table;

        auto bitwise_xor_inplace_pyapi =
            [&](const dpctl::tensor::usm_ndarray &src,
                const dpctl::tensor::usm_ndarray &dst, sycl::queue &exec_q,
                const std::vector<sycl::event> &depends = {}) {
                return py_binary_inplace_ufunc(
                    src, dst, exec_q, depends, bitwise_xor_output_id_table,
                    // function pointers to handle inplace operation on
                    // contiguous arrays (pointers may be nullptr)
                    bitwise_xor_inplace_contig_dispatch_table,
                    // function pointers to handle inplace operation on strided
                    // arrays (most general case)
                    bitwise_xor_inplace_strided_dispatch_table,
                    // function pointers to handle inplace operation on
                    // c-contig matrix with c-contig row with broadcasting
                    // (may be nullptr)
                    td_ns::NullPtrTable<
                        binary_inplace_row_matrix_broadcast_impl_fn_ptr_t>{});
            };
        m.def("_bitwise_xor_inplace", bitwise_xor_inplace_pyapi, "",
              py::arg("lhs"), py::arg("rhs"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
    }
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
