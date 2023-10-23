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

#include <pybind11/pybind11.h>

#include "abs.hpp"
#include "acos.hpp"
#include "acosh.hpp"
#include "add.hpp"
#include "asin.hpp"
#include "asinh.hpp"
#include "atan.hpp"
#include "atan2.hpp"
#include "atanh.hpp"
#include "bitwise_and.hpp"
#include "bitwise_invert.hpp"
#include "bitwise_left_shift.hpp"
#include "bitwise_or.hpp"
#include "bitwise_right_shift.hpp"
#include "bitwise_xor.hpp"
#include "cbrt.hpp"
#include "ceil.hpp"
#include "conj.hpp"
#include "copysign.hpp"
#include "cos.hpp"
#include "cosh.hpp"
#include "equal.hpp"
#include "exp.hpp"
#include "exp2.hpp"
#include "expm1.hpp"
#include "floor.hpp"
#include "floor_divide.hpp"
#include "greater.hpp"
#include "greater_equal.hpp"
#include "hypot.hpp"
#include "imag.hpp"
#include "isfinite.hpp"
#include "isinf.hpp"
#include "isnan.hpp"
#include "less.hpp"
#include "less_equal.hpp"
#include "log.hpp"
#include "log10.hpp"
#include "log1p.hpp"
#include "log2.hpp"
#include "logaddexp.hpp"
#include "logical_and.hpp"
#include "logical_not.hpp"
#include "logical_or.hpp"
#include "logical_xor.hpp"
#include "maximum.hpp"
#include "minimum.hpp"
#include "multiply.hpp"
#include "negative.hpp"
#include "not_equal.hpp"
#include "positive.hpp"
#include "pow.hpp"
#include "proj.hpp"
#include "real.hpp"
#include "remainder.hpp"
#include "round.hpp"
#include "rsqrt.hpp"
#include "sign.hpp"
#include "signbit.hpp"
#include "sin.hpp"
#include "sinh.hpp"
#include "sqrt.hpp"
#include "square.hpp"
#include "subtract.hpp"
#include "tan.hpp"
#include "tanh.hpp"
#include "true_divide.hpp"
#include "trunc.hpp"

namespace py = pybind11;

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

void init_elementwise_functions(py::module_ m)
{
    using dpctl::tensor::py_internal::init_abs;
    init_abs(m);
    using dpctl::tensor::py_internal::init_acos;
    init_acos(m);
    using dpctl::tensor::py_internal::init_acosh;
    init_acosh(m);
    using dpctl::tensor::py_internal::init_add;
    init_add(m);
    using dpctl::tensor::py_internal::init_asin;
    init_asin(m);
    using dpctl::tensor::py_internal::init_asinh;
    init_asinh(m);
    using dpctl::tensor::py_internal::init_atan;
    init_atan(m);
    using dpctl::tensor::py_internal::init_atan2;
    init_atan2(m);
    using dpctl::tensor::py_internal::init_atanh;
    init_atanh(m);
    using dpctl::tensor::py_internal::init_bitwise_and;
    init_bitwise_and(m);
    using dpctl::tensor::py_internal::init_bitwise_invert;
    init_bitwise_invert(m);
    using dpctl::tensor::py_internal::init_bitwise_left_shift;
    init_bitwise_left_shift(m);
    using dpctl::tensor::py_internal::init_bitwise_or;
    init_bitwise_or(m);
    using dpctl::tensor::py_internal::init_bitwise_right_shift;
    init_bitwise_right_shift(m);
    using dpctl::tensor::py_internal::init_bitwise_xor;
    init_bitwise_xor(m);
    using dpctl::tensor::py_internal::init_cbrt;
    init_cbrt(m);
    using dpctl::tensor::py_internal::init_ceil;
    init_ceil(m);
    using dpctl::tensor::py_internal::init_conj;
    init_conj(m);
    using dpctl::tensor::py_internal::init_copysign;
    init_copysign(m);
    using dpctl::tensor::py_internal::init_cos;
    init_cos(m);
    using dpctl::tensor::py_internal::init_cosh;
    init_cosh(m);
    using dpctl::tensor::py_internal::init_equal;
    init_equal(m);
    using dpctl::tensor::py_internal::init_exp;
    init_exp(m);
    using dpctl::tensor::py_internal::init_exp2;
    init_exp2(m);
    using dpctl::tensor::py_internal::init_expm1;
    init_expm1(m);
    using dpctl::tensor::py_internal::init_floor;
    init_floor(m);
    using dpctl::tensor::py_internal::init_floor_divide;
    init_floor_divide(m);
    using dpctl::tensor::py_internal::init_greater;
    init_greater(m);
    using dpctl::tensor::py_internal::init_greater_equal;
    init_greater_equal(m);
    using dpctl::tensor::py_internal::init_hypot;
    init_hypot(m);
    using dpctl::tensor::py_internal::init_imag;
    init_imag(m);
    using dpctl::tensor::py_internal::init_isfinite;
    init_isfinite(m);
    using dpctl::tensor::py_internal::init_isinf;
    init_isinf(m);
    using dpctl::tensor::py_internal::init_isnan;
    init_isnan(m);
    using dpctl::tensor::py_internal::init_less;
    init_less(m);
    using dpctl::tensor::py_internal::init_less_equal;
    init_less_equal(m);
    using dpctl::tensor::py_internal::init_log;
    init_log(m);
    using dpctl::tensor::py_internal::init_log10;
    init_log10(m);
    using dpctl::tensor::py_internal::init_log1p;
    init_log1p(m);
    using dpctl::tensor::py_internal::init_log2;
    init_log2(m);
    using dpctl::tensor::py_internal::init_logaddexp;
    init_logaddexp(m);
    using dpctl::tensor::py_internal::init_logical_and;
    init_logical_and(m);
    using dpctl::tensor::py_internal::init_logical_not;
    init_logical_not(m);
    using dpctl::tensor::py_internal::init_logical_or;
    init_logical_or(m);
    using dpctl::tensor::py_internal::init_logical_xor;
    init_logical_xor(m);
    using dpctl::tensor::py_internal::init_maximum;
    init_maximum(m);
    using dpctl::tensor::py_internal::init_minimum;
    init_minimum(m);
    using dpctl::tensor::py_internal::init_multiply;
    init_multiply(m);
    using dpctl::tensor::py_internal::init_negative;
    init_negative(m);
    using dpctl::tensor::py_internal::init_not_equal;
    init_not_equal(m);
    using dpctl::tensor::py_internal::init_positive;
    init_positive(m);
    using dpctl::tensor::py_internal::init_pow;
    init_pow(m);
    using dpctl::tensor::py_internal::init_proj;
    init_proj(m);
    using dpctl::tensor::py_internal::init_real;
    init_real(m);
    using dpctl::tensor::py_internal::init_remainder;
    init_remainder(m);
    using dpctl::tensor::py_internal::init_round;
    init_round(m);
    using dpctl::tensor::py_internal::init_rsqrt;
    init_rsqrt(m);
    using dpctl::tensor::py_internal::init_sign;
    init_sign(m);
    using dpctl::tensor::py_internal::init_signbit;
    init_signbit(m);
    using dpctl::tensor::py_internal::init_sin;
    init_sin(m);
    using dpctl::tensor::py_internal::init_sinh;
    init_sinh(m);
    using dpctl::tensor::py_internal::init_sqrt;
    init_sqrt(m);
    using dpctl::tensor::py_internal::init_square;
    init_square(m);
    using dpctl::tensor::py_internal::init_subtract;
    init_subtract(m);
    using dpctl::tensor::py_internal::init_tan;
    init_tan(m);
    using dpctl::tensor::py_internal::init_tanh;
    init_tanh(m);
    using dpctl::tensor::py_internal::init_divide;
    init_divide(m);
    using dpctl::tensor::py_internal::init_trunc;
    init_trunc(m);
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
