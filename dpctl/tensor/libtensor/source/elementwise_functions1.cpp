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
#include "elementwise_functions1.hpp"
#include "utils/type_dispatch.hpp"

#include "kernels/elementwise_functions/less.hpp"
#include "kernels/elementwise_functions/less_equal.hpp"
#include "kernels/elementwise_functions/log.hpp"
#include "kernels/elementwise_functions/log10.hpp"
#include "kernels/elementwise_functions/log1p.hpp"
#include "kernels/elementwise_functions/log2.hpp"
#include "kernels/elementwise_functions/logaddexp.hpp"
#include "kernels/elementwise_functions/logical_and.hpp"
#include "kernels/elementwise_functions/logical_not.hpp"
#include "kernels/elementwise_functions/logical_or.hpp"
#include "kernels/elementwise_functions/logical_xor.hpp"
#include "kernels/elementwise_functions/maximum.hpp"
#include "kernels/elementwise_functions/minimum.hpp"
#include "kernels/elementwise_functions/multiply.hpp"
#include "kernels/elementwise_functions/negative.hpp"

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

// B13: ==== LESS        (x1, x2)
namespace impl
{
namespace less_fn_ns = dpctl::tensor::kernels::less;

static binary_contig_impl_fn_ptr_t less_contig_dispatch_table[td_ns::num_types]
                                                             [td_ns::num_types];
static int less_output_id_table[td_ns::num_types][td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    less_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

void populate_less_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = less_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::LessTypeMapFactory;
    DispatchTableBuilder<int, LessTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(less_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::LessStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t, LessStridedFactory,
                         num_types>
        dtb2;
    dtb2.populate_dispatch_table(less_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::LessContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, LessContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(less_contig_dispatch_table);
};
} // namespace impl

// B14: ==== LESS_EQUAL  (x1, x2)
namespace impl
{
namespace less_equal_fn_ns = dpctl::tensor::kernels::less_equal;

static binary_contig_impl_fn_ptr_t
    less_equal_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static int less_equal_output_id_table[td_ns::num_types][td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    less_equal_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

void populate_less_equal_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = less_equal_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::LessEqualTypeMapFactory;
    DispatchTableBuilder<int, LessEqualTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(less_equal_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::LessEqualStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t, LessEqualStridedFactory,
                         num_types>
        dtb2;
    dtb2.populate_dispatch_table(less_equal_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::LessEqualContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, LessEqualContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(less_equal_contig_dispatch_table);
};
} // namespace impl

// U20: ==== LOG         (x)
namespace impl
{

namespace log_fn_ns = dpctl::tensor::kernels::log;

static unary_contig_impl_fn_ptr_t log_contig_dispatch_vector[td_ns::num_types];
static int log_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    log_strided_dispatch_vector[td_ns::num_types];

void populate_log_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = log_fn_ns;

    using fn_ns::LogContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, LogContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(log_contig_dispatch_vector);

    using fn_ns::LogStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, LogStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(log_strided_dispatch_vector);

    using fn_ns::LogTypeMapFactory;
    DispatchVectorBuilder<int, LogTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(log_output_typeid_vector);
}

} // namespace impl

// U21: ==== LOG1P       (x)
namespace impl
{

namespace log1p_fn_ns = dpctl::tensor::kernels::log1p;

static unary_contig_impl_fn_ptr_t
    log1p_contig_dispatch_vector[td_ns::num_types];
static int log1p_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    log1p_strided_dispatch_vector[td_ns::num_types];

void populate_log1p_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = log1p_fn_ns;

    using fn_ns::Log1pContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, Log1pContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(log1p_contig_dispatch_vector);

    using fn_ns::Log1pStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, Log1pStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(log1p_strided_dispatch_vector);

    using fn_ns::Log1pTypeMapFactory;
    DispatchVectorBuilder<int, Log1pTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(log1p_output_typeid_vector);
}

} // namespace impl

// U22: ==== LOG2        (x)
namespace impl
{

namespace log2_fn_ns = dpctl::tensor::kernels::log2;

static unary_contig_impl_fn_ptr_t log2_contig_dispatch_vector[td_ns::num_types];
static int log2_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    log2_strided_dispatch_vector[td_ns::num_types];

void populate_log2_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = log2_fn_ns;

    using fn_ns::Log2ContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, Log2ContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(log2_contig_dispatch_vector);

    using fn_ns::Log2StridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, Log2StridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(log2_strided_dispatch_vector);

    using fn_ns::Log2TypeMapFactory;
    DispatchVectorBuilder<int, Log2TypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(log2_output_typeid_vector);
};

} // namespace impl

// U23: ==== LOG10       (x)
namespace impl
{

namespace log10_fn_ns = dpctl::tensor::kernels::log10;

static unary_contig_impl_fn_ptr_t
    log10_contig_dispatch_vector[td_ns::num_types];
static int log10_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    log10_strided_dispatch_vector[td_ns::num_types];

void populate_log10_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = log10_fn_ns;

    using fn_ns::Log10ContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, Log10ContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(log10_contig_dispatch_vector);

    using fn_ns::Log10StridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, Log10StridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(log10_strided_dispatch_vector);

    using fn_ns::Log10TypeMapFactory;
    DispatchVectorBuilder<int, Log10TypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(log10_output_typeid_vector);
};

} // namespace impl

// B15: ==== LOGADDEXP   (x1, x2)
namespace impl
{
namespace logaddexp_fn_ns = dpctl::tensor::kernels::logaddexp;

static binary_contig_impl_fn_ptr_t
    logaddexp_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static int logaddexp_output_id_table[td_ns::num_types][td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    logaddexp_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

void populate_logaddexp_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = logaddexp_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::LogAddExpTypeMapFactory;
    DispatchTableBuilder<int, LogAddExpTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(logaddexp_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::LogAddExpStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t, LogAddExpStridedFactory,
                         num_types>
        dtb2;
    dtb2.populate_dispatch_table(logaddexp_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::LogAddExpContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, LogAddExpContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(logaddexp_contig_dispatch_table);
};
} // namespace impl

// B16: ==== LOGICAL_AND (x1, x2)
namespace impl
{
namespace logical_and_fn_ns = dpctl::tensor::kernels::logical_and;

static binary_contig_impl_fn_ptr_t
    logical_and_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static int logical_and_output_id_table[td_ns::num_types][td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    logical_and_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

void populate_logical_and_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = logical_and_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::LogicalAndTypeMapFactory;
    DispatchTableBuilder<int, LogicalAndTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(logical_and_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::LogicalAndStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t, LogicalAndStridedFactory,
                         num_types>
        dtb2;
    dtb2.populate_dispatch_table(logical_and_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::LogicalAndContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, LogicalAndContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(logical_and_contig_dispatch_table);
};
} // namespace impl

// U24: ==== LOGICAL_NOT (x)
namespace impl
{
namespace logical_not_fn_ns = dpctl::tensor::kernels::logical_not;

static unary_contig_impl_fn_ptr_t
    logical_not_contig_dispatch_vector[td_ns::num_types];
static int logical_not_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    logical_not_strided_dispatch_vector[td_ns::num_types];

void populate_logical_not_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = logical_not_fn_ns;

    using fn_ns::LogicalNotContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, LogicalNotContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(logical_not_contig_dispatch_vector);

    using fn_ns::LogicalNotStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, LogicalNotStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(logical_not_strided_dispatch_vector);

    using fn_ns::LogicalNotTypeMapFactory;
    DispatchVectorBuilder<int, LogicalNotTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(logical_not_output_typeid_vector);
};
} // namespace impl

// B17: ==== LOGICAL_OR  (x1, x2)
namespace impl
{
namespace logical_or_fn_ns = dpctl::tensor::kernels::logical_or;

static binary_contig_impl_fn_ptr_t
    logical_or_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static int logical_or_output_id_table[td_ns::num_types][td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    logical_or_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

void populate_logical_or_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = logical_or_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::LogicalOrTypeMapFactory;
    DispatchTableBuilder<int, LogicalOrTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(logical_or_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::LogicalOrStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t, LogicalOrStridedFactory,
                         num_types>
        dtb2;
    dtb2.populate_dispatch_table(logical_or_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::LogicalOrContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, LogicalOrContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(logical_or_contig_dispatch_table);
};
} // namespace impl

// B18: ==== LOGICAL_XOR (x1, x2)
namespace impl
{
namespace logical_xor_fn_ns = dpctl::tensor::kernels::logical_xor;

static binary_contig_impl_fn_ptr_t
    logical_xor_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static int logical_xor_output_id_table[td_ns::num_types][td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    logical_xor_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

void populate_logical_xor_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = logical_xor_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::LogicalXorTypeMapFactory;
    DispatchTableBuilder<int, LogicalXorTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(logical_xor_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::LogicalXorStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t, LogicalXorStridedFactory,
                         num_types>
        dtb2;
    dtb2.populate_dispatch_table(logical_xor_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::LogicalXorContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, LogicalXorContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(logical_xor_contig_dispatch_table);
};
} // namespace impl

// B??: ==== MAXIMUM    (x1, x2)
namespace impl
{

namespace maximum_fn_ns = dpctl::tensor::kernels::maximum;

static binary_contig_impl_fn_ptr_t
    maximum_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static int maximum_output_id_table[td_ns::num_types][td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    maximum_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

void populate_maximum_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = maximum_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::MaximumTypeMapFactory;
    DispatchTableBuilder<int, MaximumTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(maximum_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::MaximumStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t, MaximumStridedFactory,
                         num_types>
        dtb2;
    dtb2.populate_dispatch_table(maximum_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::MaximumContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, MaximumContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(maximum_contig_dispatch_table);
};

} // namespace impl

// B??: ==== MINIMUM    (x1, x2)
namespace impl
{

namespace minimum_fn_ns = dpctl::tensor::kernels::minimum;

static binary_contig_impl_fn_ptr_t
    minimum_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static int minimum_output_id_table[td_ns::num_types][td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    minimum_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

void populate_minimum_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = minimum_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::MinimumTypeMapFactory;
    DispatchTableBuilder<int, MinimumTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(minimum_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::MinimumStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t, MinimumStridedFactory,
                         num_types>
        dtb2;
    dtb2.populate_dispatch_table(minimum_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::MinimumContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, MinimumContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(minimum_contig_dispatch_table);
};

} // namespace impl

// B19: ==== MULTIPLY    (x1, x2)
namespace impl
{

namespace multiply_fn_ns = dpctl::tensor::kernels::multiply;

static binary_contig_impl_fn_ptr_t
    multiply_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static int multiply_output_id_table[td_ns::num_types][td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    multiply_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

// mul(matrix, row)
static binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t
    multiply_contig_matrix_contig_row_broadcast_dispatch_table
        [td_ns::num_types][td_ns::num_types];

// mul(row, matrix)
static binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t
    multiply_contig_row_contig_matrix_broadcast_dispatch_table
        [td_ns::num_types][td_ns::num_types];

static binary_inplace_contig_impl_fn_ptr_t
    multiply_inplace_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static binary_inplace_strided_impl_fn_ptr_t
    multiply_inplace_strided_dispatch_table[td_ns::num_types][td_ns::num_types];
static binary_inplace_row_matrix_broadcast_impl_fn_ptr_t
    multiply_inplace_row_matrix_dispatch_table[td_ns::num_types]
                                              [td_ns::num_types];

void populate_multiply_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = multiply_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::MultiplyTypeMapFactory;
    DispatchTableBuilder<int, MultiplyTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(multiply_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::MultiplyStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t, MultiplyStridedFactory,
                         num_types>
        dtb2;
    dtb2.populate_dispatch_table(multiply_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::MultiplyContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, MultiplyContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(multiply_contig_dispatch_table);

    // function pointers for operation on contiguous matrix, contiguous row
    // with contiguous matrix output
    using fn_ns::MultiplyContigMatrixContigRowBroadcastFactory;
    DispatchTableBuilder<
        binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t,
        MultiplyContigMatrixContigRowBroadcastFactory, num_types>
        dtb4;
    dtb4.populate_dispatch_table(
        multiply_contig_matrix_contig_row_broadcast_dispatch_table);

    // function pointers for operation on contiguous row, contiguous matrix
    // with contiguous matrix output
    using fn_ns::MultiplyContigRowContigMatrixBroadcastFactory;
    DispatchTableBuilder<
        binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t,
        MultiplyContigRowContigMatrixBroadcastFactory, num_types>
        dtb5;
    dtb5.populate_dispatch_table(
        multiply_contig_row_contig_matrix_broadcast_dispatch_table);

    // function pointers for inplace operation on general strided arrays
    using fn_ns::MultiplyInplaceStridedFactory;
    DispatchTableBuilder<binary_inplace_strided_impl_fn_ptr_t,
                         MultiplyInplaceStridedFactory, num_types>
        dtb6;
    dtb6.populate_dispatch_table(multiply_inplace_strided_dispatch_table);

    // function pointers for inplace operation on contiguous inputs and output
    using fn_ns::MultiplyInplaceContigFactory;
    DispatchTableBuilder<binary_inplace_contig_impl_fn_ptr_t,
                         MultiplyInplaceContigFactory, num_types>
        dtb7;
    dtb7.populate_dispatch_table(multiply_inplace_contig_dispatch_table);

    // function pointers for inplace operation on contiguous matrix
    // and contiguous row
    using fn_ns::MultiplyInplaceRowMatrixBroadcastFactory;
    DispatchTableBuilder<binary_inplace_row_matrix_broadcast_impl_fn_ptr_t,
                         MultiplyInplaceRowMatrixBroadcastFactory, num_types>
        dtb8;
    dtb8.populate_dispatch_table(multiply_inplace_row_matrix_dispatch_table);
};

} // namespace impl

// U25: ==== NEGATIVE    (x)
namespace impl
{

namespace negative_fn_ns = dpctl::tensor::kernels::negative;

static unary_contig_impl_fn_ptr_t
    negative_contig_dispatch_vector[td_ns::num_types];
static int negative_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    negative_strided_dispatch_vector[td_ns::num_types];

void populate_negative_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = negative_fn_ns;

    using fn_ns::NegativeContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, NegativeContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(negative_contig_dispatch_vector);

    using fn_ns::NegativeStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, NegativeStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(negative_strided_dispatch_vector);

    using fn_ns::NegativeTypeMapFactory;
    DispatchVectorBuilder<int, NegativeTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(negative_output_typeid_vector);
}

} // namespace impl

// ========================================================================= //

namespace py = pybind11;

void init_elementwise_functions1(py::module_ m)
{

    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;

    // B13: ==== LESS        (x1, x2)
    {
        impl::populate_less_dispatch_tables();
        using impl::less_contig_dispatch_table;
        using impl::less_output_id_table;
        using impl::less_strided_dispatch_table;

        auto less_pyapi = [&](const dpctl::tensor::usm_ndarray &src1,
                              const dpctl::tensor::usm_ndarray &src2,
                              const dpctl::tensor::usm_ndarray &dst,
                              sycl::queue &exec_q,
                              const std::vector<sycl::event> &depends = {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends, less_output_id_table,
                // function pointers to handle operation on contiguous arrays
                // (pointers may be nullptr)
                less_contig_dispatch_table,
                // function pointers to handle operation on strided arrays (most
                // general case)
                less_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t>{},
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t>{});
        };
        auto less_result_type_pyapi = [&](const py::dtype &dtype1,
                                          const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               less_output_id_table);
        };
        m.def("_less", less_pyapi, "", py::arg("src1"), py::arg("src2"),
              py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_less_result_type", less_result_type_pyapi, "");
    }

    // B14: ==== LESS_EQUAL  (x1, x2)
    {
        impl::populate_less_equal_dispatch_tables();
        using impl::less_equal_contig_dispatch_table;
        using impl::less_equal_output_id_table;
        using impl::less_equal_strided_dispatch_table;

        auto less_equal_pyapi = [&](const dpctl::tensor::usm_ndarray &src1,
                                    const dpctl::tensor::usm_ndarray &src2,
                                    const dpctl::tensor::usm_ndarray &dst,
                                    sycl::queue &exec_q,
                                    const std::vector<sycl::event> &depends =
                                        {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends, less_equal_output_id_table,
                // function pointers to handle operation on contiguous arrays
                // (pointers may be nullptr)
                less_equal_contig_dispatch_table,
                // function pointers to handle operation on strided arrays (most
                // general case)
                less_equal_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t>{},
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t>{});
        };
        auto less_equal_result_type_pyapi = [&](const py::dtype &dtype1,
                                                const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               less_equal_output_id_table);
        };
        m.def("_less_equal", less_equal_pyapi, "", py::arg("src1"),
              py::arg("src2"), py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_less_equal_result_type", less_equal_result_type_pyapi, "");
    }

    // U20: ==== LOG         (x)
    {
        impl::populate_log_dispatch_vectors();
        using impl::log_contig_dispatch_vector;
        using impl::log_output_typeid_vector;
        using impl::log_strided_dispatch_vector;

        auto log_pyapi = [&](const arrayT &src, const arrayT &dst,
                             sycl::queue &exec_q,
                             const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, log_output_typeid_vector,
                log_contig_dispatch_vector, log_strided_dispatch_vector);
        };
        m.def("_log", log_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto log_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype, log_output_typeid_vector);
        };
        m.def("_log_result_type", log_result_type_pyapi);
    }

    // U21: ==== LOG1P       (x)
    {
        impl::populate_log1p_dispatch_vectors();
        using impl::log1p_contig_dispatch_vector;
        using impl::log1p_output_typeid_vector;
        using impl::log1p_strided_dispatch_vector;

        auto log1p_pyapi = [&](const arrayT &src, const arrayT &dst,
                               sycl::queue &exec_q,
                               const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, log1p_output_typeid_vector,
                log1p_contig_dispatch_vector, log1p_strided_dispatch_vector);
        };
        m.def("_log1p", log1p_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto log1p_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype,
                                              log1p_output_typeid_vector);
        };
        m.def("_log1p_result_type", log1p_result_type_pyapi);
    }

    // U22: ==== LOG2        (x)
    {
        impl::populate_log2_dispatch_vectors();

        using impl::log2_contig_dispatch_vector;
        using impl::log2_output_typeid_vector;
        using impl::log2_strided_dispatch_vector;
        auto log2_pyapi = [&](const dpctl::tensor::usm_ndarray &src,
                              const dpctl::tensor::usm_ndarray &dst,
                              sycl::queue &exec_q,
                              const std::vector<sycl::event> &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, log2_output_typeid_vector,
                log2_contig_dispatch_vector, log2_strided_dispatch_vector);
        };
        auto log2_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype, log2_output_typeid_vector);
        };
        m.def("_log2", log2_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());
        m.def("_log2_result_type", log2_result_type_pyapi, "");
    }

    // U23: ==== LOG10       (x)
    {
        impl::populate_log10_dispatch_vectors();

        using impl::log10_contig_dispatch_vector;
        using impl::log10_output_typeid_vector;
        using impl::log10_strided_dispatch_vector;
        auto log10_pyapi = [&](const dpctl::tensor::usm_ndarray &src,
                               const dpctl::tensor::usm_ndarray &dst,
                               sycl::queue &exec_q,
                               const std::vector<sycl::event> &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, log10_output_typeid_vector,
                log10_contig_dispatch_vector, log10_strided_dispatch_vector);
        };
        auto log10_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype,
                                              log10_output_typeid_vector);
        };
        m.def("_log10", log10_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());
        m.def("_log10_result_type", log10_result_type_pyapi, "");
    }

    // B15: ==== LOGADDEXP   (x1, x2)
    {
        impl::populate_logaddexp_dispatch_tables();
        using impl::logaddexp_contig_dispatch_table;
        using impl::logaddexp_output_id_table;
        using impl::logaddexp_strided_dispatch_table;

        auto logaddexp_pyapi = [&](const dpctl::tensor::usm_ndarray &src1,
                                   const dpctl::tensor::usm_ndarray &src2,
                                   const dpctl::tensor::usm_ndarray &dst,
                                   sycl::queue &exec_q,
                                   const std::vector<sycl::event> &depends =
                                       {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends, logaddexp_output_id_table,
                // function pointers to handle operation on contiguous arrays
                // (pointers may be nullptr)
                logaddexp_contig_dispatch_table,
                // function pointers to handle operation on strided arrays (most
                // general case)
                logaddexp_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t>{},
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t>{});
        };
        auto logaddexp_result_type_pyapi = [&](const py::dtype &dtype1,
                                               const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               logaddexp_output_id_table);
        };
        m.def("_logaddexp", logaddexp_pyapi, "", py::arg("src1"),
              py::arg("src2"), py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_logaddexp_result_type", logaddexp_result_type_pyapi, "");
    }

    // B16: ==== LOGICAL_AND (x1, x2)
    {
        impl::populate_logical_and_dispatch_tables();
        using impl::logical_and_contig_dispatch_table;
        using impl::logical_and_output_id_table;
        using impl::logical_and_strided_dispatch_table;

        auto logical_and_pyapi = [&](const dpctl::tensor::usm_ndarray &src1,
                                     const dpctl::tensor::usm_ndarray &src2,
                                     const dpctl::tensor::usm_ndarray &dst,
                                     sycl::queue &exec_q,
                                     const std::vector<sycl::event> &depends =
                                         {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends, logical_and_output_id_table,
                // function pointers to handle operation on contiguous arrays
                // (pointers may be nullptr)
                logical_and_contig_dispatch_table,
                // function pointers to handle operation on strided arrays (most
                // general case)
                logical_and_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t>{},
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t>{});
        };
        auto logical_and_result_type_pyapi = [&](const py::dtype &dtype1,
                                                 const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               logical_and_output_id_table);
        };
        m.def("_logical_and", logical_and_pyapi, "", py::arg("src1"),
              py::arg("src2"), py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_logical_and_result_type", logical_and_result_type_pyapi, "");
    }

    // U24: ==== LOGICAL_NOT (x)
    {
        impl::populate_logical_not_dispatch_vectors();
        using impl::logical_not_contig_dispatch_vector;
        using impl::logical_not_output_typeid_vector;
        using impl::logical_not_strided_dispatch_vector;

        auto logical_not_pyapi = [&](const arrayT &src, arrayT dst,
                                     sycl::queue &exec_q,
                                     const event_vecT &depends = {}) {
            return py_unary_ufunc(src, dst, exec_q, depends,
                                  logical_not_output_typeid_vector,
                                  logical_not_contig_dispatch_vector,
                                  logical_not_strided_dispatch_vector);
        };
        m.def("_logical_not", logical_not_pyapi, "", py::arg("src"),
              py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());

        auto logical_not_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype,
                                              logical_not_output_typeid_vector);
        };
        m.def("_logical_not_result_type", logical_not_result_type_pyapi);
    }

    // B17: ==== LOGICAL_OR  (x1, x2)
    {
        impl::populate_logical_or_dispatch_tables();
        using impl::logical_or_contig_dispatch_table;
        using impl::logical_or_output_id_table;
        using impl::logical_or_strided_dispatch_table;

        auto logical_or_pyapi = [&](const dpctl::tensor::usm_ndarray &src1,
                                    const dpctl::tensor::usm_ndarray &src2,
                                    const dpctl::tensor::usm_ndarray &dst,
                                    sycl::queue &exec_q,
                                    const std::vector<sycl::event> &depends =
                                        {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends, logical_or_output_id_table,
                // function pointers to handle operation on contiguous arrays
                // (pointers may be nullptr)
                logical_or_contig_dispatch_table,
                // function pointers to handle operation on strided arrays (most
                // general case)
                logical_or_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t>{},
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t>{});
        };
        auto logical_or_result_type_pyapi = [&](const py::dtype &dtype1,
                                                const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               logical_or_output_id_table);
        };
        m.def("_logical_or", logical_or_pyapi, "", py::arg("src1"),
              py::arg("src2"), py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_logical_or_result_type", logical_or_result_type_pyapi, "");
    }

    // B18: ==== LOGICAL_XOR (x1, x2)
    {
        impl::populate_logical_xor_dispatch_tables();
        using impl::logical_xor_contig_dispatch_table;
        using impl::logical_xor_output_id_table;
        using impl::logical_xor_strided_dispatch_table;

        auto logical_xor_pyapi = [&](const dpctl::tensor::usm_ndarray &src1,
                                     const dpctl::tensor::usm_ndarray &src2,
                                     const dpctl::tensor::usm_ndarray &dst,
                                     sycl::queue &exec_q,
                                     const std::vector<sycl::event> &depends =
                                         {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends, logical_xor_output_id_table,
                // function pointers to handle operation on contiguous arrays
                // (pointers may be nullptr)
                logical_xor_contig_dispatch_table,
                // function pointers to handle operation on strided arrays (most
                // general case)
                logical_xor_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t>{},
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t>{});
        };
        auto logical_xor_result_type_pyapi = [&](const py::dtype &dtype1,
                                                 const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               logical_xor_output_id_table);
        };
        m.def("_logical_xor", logical_xor_pyapi, "", py::arg("src1"),
              py::arg("src2"), py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_logical_xor_result_type", logical_xor_result_type_pyapi, "");
    }

    // B??: ==== MAXIMUM    (x1, x2)
    {
        impl::populate_maximum_dispatch_tables();
        using impl::maximum_contig_dispatch_table;
        using impl::maximum_output_id_table;
        using impl::maximum_strided_dispatch_table;

        auto maximum_pyapi = [&](const dpctl::tensor::usm_ndarray &src1,
                                 const dpctl::tensor::usm_ndarray &src2,
                                 const dpctl::tensor::usm_ndarray &dst,
                                 sycl::queue &exec_q,
                                 const std::vector<sycl::event> &depends = {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends, maximum_output_id_table,
                // function pointers to handle operation on contiguous
                // arrays (pointers may be nullptr)
                maximum_contig_dispatch_table,
                // function pointers to handle operation on strided arrays
                // (most general case)
                maximum_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix
                // and c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t>{},
                // function pointers to handle operation of c-contig matrix
                // and c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t>{});
        };
        auto maximum_result_type_pyapi = [&](const py::dtype &dtype1,
                                             const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               maximum_output_id_table);
        };
        m.def("_maximum", maximum_pyapi, "", py::arg("src1"), py::arg("src2"),
              py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_maximum_result_type", maximum_result_type_pyapi, "");
    }

    // B??: ==== MINIMUM    (x1, x2)
    {
        impl::populate_minimum_dispatch_tables();
        using impl::minimum_contig_dispatch_table;
        using impl::minimum_output_id_table;
        using impl::minimum_strided_dispatch_table;

        auto minimum_pyapi = [&](const dpctl::tensor::usm_ndarray &src1,
                                 const dpctl::tensor::usm_ndarray &src2,
                                 const dpctl::tensor::usm_ndarray &dst,
                                 sycl::queue &exec_q,
                                 const std::vector<sycl::event> &depends = {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends, minimum_output_id_table,
                // function pointers to handle operation on contiguous
                // arrays (pointers may be nullptr)
                minimum_contig_dispatch_table,
                // function pointers to handle operation on strided arrays
                // (most general case)
                minimum_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix
                // and c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t>{},
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t>{});
        };
        auto minimum_result_type_pyapi = [&](const py::dtype &dtype1,
                                             const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               minimum_output_id_table);
        };
        m.def("_minimum", minimum_pyapi, "", py::arg("src1"), py::arg("src2"),
              py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_minimum_result_type", minimum_result_type_pyapi, "");
    }

    // B19: ==== MULTIPLY    (x1, x2)
    {
        impl::populate_multiply_dispatch_tables();
        using impl::multiply_contig_dispatch_table;
        using impl::multiply_contig_matrix_contig_row_broadcast_dispatch_table;
        using impl::multiply_contig_row_contig_matrix_broadcast_dispatch_table;
        using impl::multiply_output_id_table;
        using impl::multiply_strided_dispatch_table;

        auto multiply_pyapi =
            [&](const dpctl::tensor::usm_ndarray &src1,
                const dpctl::tensor::usm_ndarray &src2,
                const dpctl::tensor::usm_ndarray &dst, sycl::queue &exec_q,
                const std::vector<sycl::event> &depends = {}) {
                return py_binary_ufunc(
                    src1, src2, dst, exec_q, depends, multiply_output_id_table,
                    // function pointers to handle operation on contiguous
                    // arrays (pointers may be nullptr)
                    multiply_contig_dispatch_table,
                    // function pointers to handle operation on strided arrays
                    // (most general case)
                    multiply_strided_dispatch_table,
                    // function pointers to handle operation of c-contig matrix
                    // and c-contig row with broadcasting (may be nullptr)
                    multiply_contig_matrix_contig_row_broadcast_dispatch_table,
                    // function pointers to handle operation of c-contig matrix
                    // and c-contig row with broadcasting (may be nullptr)
                    multiply_contig_row_contig_matrix_broadcast_dispatch_table);
            };
        auto multiply_result_type_pyapi = [&](const py::dtype &dtype1,
                                              const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               multiply_output_id_table);
        };
        m.def("_multiply", multiply_pyapi, "", py::arg("src1"), py::arg("src2"),
              py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_multiply_result_type", multiply_result_type_pyapi, "");

        using impl::multiply_inplace_contig_dispatch_table;
        using impl::multiply_inplace_row_matrix_dispatch_table;
        using impl::multiply_inplace_strided_dispatch_table;

        auto multiply_inplace_pyapi =
            [&](const dpctl::tensor::usm_ndarray &src,
                const dpctl::tensor::usm_ndarray &dst, sycl::queue &exec_q,
                const std::vector<sycl::event> &depends = {}) {
                return py_binary_inplace_ufunc(
                    src, dst, exec_q, depends, multiply_output_id_table,
                    // function pointers to handle inplace operation on
                    // contiguous arrays (pointers may be nullptr)
                    multiply_inplace_contig_dispatch_table,
                    // function pointers to handle inplace operation on strided
                    // arrays (most general case)
                    multiply_inplace_strided_dispatch_table,
                    // function pointers to handle inplace operation on
                    // c-contig matrix with c-contig row with broadcasting
                    // (may be nullptr)
                    multiply_inplace_row_matrix_dispatch_table);
            };
        m.def("_multiply_inplace", multiply_inplace_pyapi, "", py::arg("lhs"),
              py::arg("rhs"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
    }

    // U25: ==== NEGATIVE    (x)
    {
        impl::populate_negative_dispatch_vectors();
        using impl::negative_contig_dispatch_vector;
        using impl::negative_output_typeid_vector;
        using impl::negative_strided_dispatch_vector;

        auto negative_pyapi = [&](const arrayT &src, const arrayT &dst,
                                  sycl::queue &exec_q,
                                  const event_vecT &depends = {}) {
            return py_unary_ufunc(src, dst, exec_q, depends,
                                  negative_output_typeid_vector,
                                  negative_contig_dispatch_vector,
                                  negative_strided_dispatch_vector);
        };
        m.def("_negative", negative_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto negative_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype,
                                              negative_output_typeid_vector);
        };
        m.def("_negative_result_type", negative_result_type_pyapi);
    }
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
