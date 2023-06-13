#                       Data Parallel Control (dpctl)
#
#  Copyright 2020-2023 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import dpctl.tensor._tensor_impl as ti

from ._elementwise_common import BinaryElementwiseFunc, UnaryElementwiseFunc

# U01: ==== ABS    (x)
_abs_docstring_ = """
abs(x, out=None, order='K')

Calculates the absolute value for each element `x_i` of input array `x`.

Args:
    x (usm_ndarray):
        Input array, expected to have numeric data type.
    out ({None, usm_ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    usm_narray:
        An array containing the element-wise absolute values.
        For complex input, the absolute value is its magnitude. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

abs = UnaryElementwiseFunc("abs", ti._abs_result_type, ti._abs, _abs_docstring_)

# U02: ==== ACOS   (x)
# FIXME: implement U02

# U03: ===== ACOSH (x)
# FIXME: implement U03

# B01: ===== ADD   (x1, x2)

_add_docstring_ = """
add(x1, x2, out=None, order='K')

Calculates the sum for each element `x1_i` of the input array `x1` with
the respective element `x2_i` of the input array `x2`.

Args:
    x1 (usm_ndarray):
        First input array, expected to have numeric data type.
    x2 (usm_ndarray):
        Second input array, also expected to have numeric data type.
    out ({None, usm_ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    usm_narray:
        An array containing the element-wise sums. The data type of the
        returned array is determined by the Type Promotion Rules.
"""
add = BinaryElementwiseFunc(
    "add", ti._add_result_type, ti._add, _add_docstring_
)

# U04: ===== ASIN  (x)
# FIXME: implement U04

# U05: ===== ASINH (x)
# FIXME: implement U05

# U06: ===== ATAN  (x)
# FIXME: implement U06

# B02: ===== ATAN2 (x1, x2)
# FIXME: implemetn B02

# U07: ===== ATANH (x)
# FIXME: implemetn U07

# B03: ===== BITWISE_AND           (x1, x2)
# FIXME: implemetn B03

# B04: ===== BITWISE_LEFT_SHIFT    (x1, x2)
# FIXME: implement B04

# U08: ===== BITWISE_INVERT        (x)
# FIXME: implement U08

# B05: ===== BITWISE_OR            (x1, x2)
# FIXME: implement B05

# B06: ===== BITWISE_RIGHT_SHIFT   (x1, x2)
# FIXME: implement B06

# B07: ===== BITWISE_XOR           (x1, x2)
# FIXME: implement B07

# U09: ==== CEIL          (x)
# FIXME: implement U09

# U10: ==== CONJ          (x)
_conj_docstring = """
conj(x, out=None, order='K')

Computes conjugate of each element `x_i` for input array `x`.

Args:
    x (usm_ndarray):
        Input array, expected to have numeric data type.
    out ({None, usm_ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    usm_narray:
        An array containing the element-wise conjugate values. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

conj = UnaryElementwiseFunc(
    "conj", ti._conj_result_type, ti._conj, _conj_docstring
)

# U11: ==== COS           (x)
_cos_docstring = """
cos(x, out=None, order='K')

Computes cosine for each element `x_i` for input array `x`.

Args:
    x (usm_ndarray):
        Input array, expected to have numeric data type.
    out ({None, usm_ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    usm_narray:
        An array containing the element-wise cosine. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

cos = UnaryElementwiseFunc("cos", ti._cos_result_type, ti._cos, _cos_docstring)

# U12: ==== COSH          (x)
# FIXME: implement U12

# B08: ==== DIVIDE        (x1, x2)
_divide_docstring_ = """
divide(x1, x2, out=None, order='K')

Calculates the ratio for each element `x1_i` of the input array `x1` with
the respective element `x2_i` of the input array `x2`.

Args:
    x1 (usm_ndarray):
        First input array, expected to have numeric data type.
    x2 (usm_ndarray):
        Second input array, also expected to have numeric data type.
    out ({None, usm_ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    usm_narray:
        An array containing the result of element-wise division. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

divide = BinaryElementwiseFunc(
    "divide", ti._divide_result_type, ti._divide, _divide_docstring_
)

# B09: ==== EQUAL         (x1, x2)
_equal_docstring_ = """
equal(x1, x2, out=None, order='K')

Calculates equality test results for each element `x1_i` of the input array `x1`
with the respective element `x2_i` of the input array `x2`.

Args:
    x1 (usm_ndarray):
        First input array, expected to have numeric data type.
    x2 (usm_ndarray):
        Second input array, also expected to have numeric data type.
    out ({None, usm_ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    usm_narray:
        An array containing the result of element-wise equality comparison.
        The data type of the returned array is determined by the
        Type Promotion Rules.
"""

equal = BinaryElementwiseFunc(
    "equal", ti._equal_result_type, ti._equal, _equal_docstring_
)

# U13: ==== EXP           (x)
_exp_docstring = """
exp(x, out=None, order='K')

Computes exponential for each element `x_i` of input array `x`.

Args:
    x (usm_ndarray):
        Input array, expected to have numeric data type.
    out ({None, usm_ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    usm_narray:
        An array containing the element-wise exponential of x.
        The data type of the returned array is determined by
        the Type Promotion Rules.
"""

exp = UnaryElementwiseFunc("exp", ti._exp_result_type, ti._exp, _exp_docstring)

# U14: ==== EXPM1         (x)
_expm1_docstring = """
expm1(x, out=None, order='K')
Computes an approximation of exp(x)-1 element-wise.
Args:
    x (usm_ndarray):
        Input array, expected to have numeric data type.
    out (usm_ndarray):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    usm_ndarray:
        An array containing the element-wise exp(x)-1 values.
"""

expm1 = UnaryElementwiseFunc(
    "expm1", ti._expm1_result_type, ti._expm1, _expm1_docstring
)

# U15: ==== FLOOR         (x)
# FIXME: implement U15

# B10: ==== FLOOR_DIVIDE  (x1, x2)
_floor_divide_docstring_ = """
floor_divide(x1, x2, out=None, order='K')

Calculates the ratio for each element `x1_i` of the input array `x1` with
the respective element `x2_i` of the input array `x2` to the greatest
integer-value number that is not greater than the division result.

Args:
    x1 (usm_ndarray):
        First input array, expected to have numeric data type.
    x2 (usm_ndarray):
        Second input array, also expected to have numeric data type.
Returns:
    usm_narray:
        an array containing the result of element-wise  floor division.
        The data type of the returned array is determined by the Type
        Promotion Rules.
"""

floor_divide = BinaryElementwiseFunc(
    "floor_divide",
    ti._floor_divide_result_type,
    ti._floor_divide,
    _floor_divide_docstring_,
)

# B11: ==== GREATER       (x1, x2)
# FIXME: implement B11

# B12: ==== GREATER_EQUAL (x1, x2)
# FIXME: implement B12

# U16: ==== IMAG        (x)
_imag_docstring = """
imag(x, out=None, order='K')

Computes imaginary part of each element `x_i` for input array `x`.

Args:
    x (usm_ndarray):
        Input array, expected to have numeric data type.
    out ({None, usm_ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    usm_narray:
        An array containing the element-wise imaginary component of input.
        The data type of the returned array is determined
        by the Type Promotion Rules.
"""

imag = UnaryElementwiseFunc(
    "imag", ti._imag_result_type, ti._imag, _imag_docstring
)

# U17: ==== ISFINITE    (x)
_isfinite_docstring_ = """
isfinite(x, out=None, order='K')

Checks if each element of input array is a finite number.

Args:
    x (usm_ndarray):
        Input array, expected to have numeric data type.
    out ({None, usm_ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    usm_narray:
        An array which is True where `x` is not positive infinity,
        negative infinity, or NaN, False otherwise.
        The data type of the returned array is boolean.
"""

isfinite = UnaryElementwiseFunc(
    "isfinite", ti._isfinite_result_type, ti._isfinite, _isfinite_docstring_
)

# U18: ==== ISINF       (x)
_isinf_docstring_ = """
isinf(x, out=None, order='K')

Checks if each element of input array is an infinity.

Args:
    x (usm_ndarray):
        Input array, expected to have numeric data type.
    out ({None, usm_ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    usm_narray:
        An array which is True where `x` is positive or negative infinity,
        False otherwise. The data type of the returned array is boolean.
"""

isinf = UnaryElementwiseFunc(
    "isinf", ti._isinf_result_type, ti._isinf, _isinf_docstring_
)

# U19: ==== ISNAN       (x)
_isnan_docstring_ = """
isnan(x, out=None, order='K')

Checks if each element of an input array is a NaN.

Args:
    x (usm_ndarray):
        Input array, expected to have numeric data type.
    out ({None, usm_ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    usm_narray:
        An array which is True where x is NaN, False otherwise.
        The data type of the returned array is boolean.
"""

isnan = UnaryElementwiseFunc(
    "isnan", ti._isnan_result_type, ti._isnan, _isnan_docstring_
)

# B13: ==== LESS        (x1, x2)
# FIXME: implement B13

# B14: ==== LESS_EQUAL  (x1, x2)
_less_equal_docstring_ = """
less_equal(x1, x2, out=None, order='K')
Computes the less-than or equal-to test results for each element `x1_i` of
the input array `x1` the respective element `x2_i` of the input array `x2`.
Args:
    x1 (usm_ndarray):
        First input array, expected to have numeric data type.
    x2 (usm_ndarray):
        Second input array, also expected to have numeric data type.
    out ({None, usm_ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    usm_narray:
        An array containing the result of element-wise less-than or equal-to
        comparison.
        The data type of the returned array is determined by the
        Type Promotion Rules.
"""

less_equal = BinaryElementwiseFunc(
    "less_equal",
    ti._less_equal_result_type,
    ti._less_equal,
    _less_equal_docstring_,
)

# U20: ==== LOG         (x)
_log_docstring = """
log(x, out=None, order='K')
Computes the natural logarithm element-wise.
Args:
    x (usm_ndarray):
        Input array, expected to have numeric data type.
    out (usm_ndarray):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    usm_ndarray:
        An array containing the element-wise natural logarithm values.
"""

log = UnaryElementwiseFunc("log", ti._log_result_type, ti._log, _log_docstring)

# U21: ==== LOG1P       (x)
_log1p_docstring = """
log1p(x, out=None, order='K')
Computes an approximation of log(1+x) element-wise.
Args:
    x (usm_ndarray):
        Input array, expected to have numeric data type.
    out (usm_ndarray):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    usm_ndarray:
        An array containing the element-wise log(1+x) values.
"""

log1p = UnaryElementwiseFunc(
    "log1p", ti._log1p_result_type, ti._log1p, _log1p_docstring
)

# U22: ==== LOG2        (x)
# FIXME: implement U22

# U23: ==== LOG10       (x)
# FIXME: implement U23

# B15: ==== LOGADDEXP   (x1, x2)
# FIXME: implement B15

# B16: ==== LOGICAL_AND (x1, x2)
# FIXME: implement B16

# U24: ==== LOGICAL_NOT (x)
# FIXME: implement U24

# B17: ==== LOGICAL_OR  (x1, x2)
# FIXME: implement B17

# B18: ==== LOGICAL_XOR (x1, x2)
# FIXME: implement B18

# B19: ==== MULTIPLY    (x1, x2)
_multiply_docstring_ = """
multiply(x1, x2, out=None, order='K')

Calculates the product for each element `x1_i` of the input array `x1`
with the respective element `x2_i` of the input array `x2`.

Args:
    x1 (usm_ndarray):
        First input array, expected to have numeric data type.
    x2 (usm_ndarray):
        Second input array, also expected to have numeric data type.
    out ({None, usm_ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    usm_narray:
        An array containing the element-wise products. The data type of
        the returned array is determined by the Type Promotion Rules.
"""
multiply = BinaryElementwiseFunc(
    "multiply", ti._multiply_result_type, ti._multiply, _multiply_docstring_
)

# U25: ==== NEGATIVE    (x)
# FIXME: implement U25

# B20: ==== NOT_EQUAL   (x1, x2)
_not_equal_docstring_ = """
not_equal(x1, x2, out=None, order='K')

Calculates inequality test results for each element `x1_i` of the
input array `x1` with the respective element `x2_i` of the input array `x2`.

Args:
    x1 (usm_ndarray):
        First input array, expected to have numeric data type.
    x2 (usm_ndarray):
        Second input array, also expected to have numeric data type.
    out ({None, usm_ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    usm_narray:
        an array containing the result of element-wise inequality comparison.
        The data type of the returned array is determined by the
        Type Promotion Rules.
"""

not_equal = BinaryElementwiseFunc(
    "not_equal", ti._not_equal_result_type, ti._not_equal, _not_equal_docstring_
)

# U26: ==== POSITIVE    (x)
# FIXME: implement U26

# B21: ==== POW         (x1, x2)
# FIXME: implement B21

# U??: ==== PROJ        (x)
_proj_docstring = """
proj(x, out=None, order='K')

Computes projection of each element `x_i` for input array `x`.

Args:
    x (usm_ndarray):
        Input array, expected to have numeric data type.
    out ({None, usm_ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    usm_narray:
        An array containing the element-wise projection. The data
        type of the returned array is determined by the Type Promotion Rules.
"""

proj = UnaryElementwiseFunc(
    "proj", ti._proj_result_type, ti._proj, _proj_docstring
)

# U27: ==== REAL        (x)
_real_docstring = """
real(x, out=None, order='K')

Computes real part of each element `x_i` for input array `x`.

Args:
    x (usm_ndarray):
        Input array, expected to have numeric data type.
    out ({None, usm_ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    usm_narray:
        An array containing the element-wise real component of input. The data
        type of the returned array is determined by the Type Promotion Rules.
"""

real = UnaryElementwiseFunc(
    "real", ti._real_result_type, ti._real, _real_docstring
)

# B22: ==== REMAINDER   (x1, x2)
# FIXME: implement B22

# U28: ==== ROUND       (x)
# FIXME: implement U28

# U29: ==== SIGN        (x)
# FIXME: implement U29

# U30: ==== SIN         (x)
_sin_docstring = """
sin(x, out=None, order='K')

Computes sine for each element `x_i` of input array `x`.

Args:
    x (usm_ndarray):
        Input array, expected to have numeric data type.
    out ({None, usm_ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    usm_narray:
        An array containing the element-wise sine. The data type of the
        returned array is determined by the Type Promotion Rules.
"""

sin = UnaryElementwiseFunc("sin", ti._sin_result_type, ti._sin, _sin_docstring)

# U31: ==== SINH        (x)
# FIXME: implement U31

# U32: ==== SQUARE      (x)
# FIXME: implement U32

# U33: ==== SQRT        (x)
_sqrt_docstring_ = """
sqrt(x, out=None, order='K')

Computes positive square-root for each element `x_i` for input array `x`.

Args:
    x (usm_ndarray):
        Input array, expected to have numeric data type.
    out ({None, usm_ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    usm_narray:
        An array containing the element-wise positive square-root.
        The data type of the returned array is determined by
        the Type Promotion Rules.
"""

sqrt = UnaryElementwiseFunc(
    "sqrt", ti._sqrt_result_type, ti._sqrt, _sqrt_docstring_
)

# B23: ==== SUBTRACT    (x1, x2)
_subtract_docstring_ = """
subtract(x1, x2, out=None, order='K')

Calculates the difference bewteen each element `x1_i` of the input
array `x1` and the respective element `x2_i` of the input array `x2`.

Args:
    x1 (usm_ndarray):
        First input array, expected to have numeric data type.
    x2 (usm_ndarray):
        Second input array, also expected to have numeric data type.
    out ({None, usm_ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    usm_narray:
        An array containing the element-wise differences. The data type
        of the returned array is determined by the Type Promotion Rules.
"""
subtract = BinaryElementwiseFunc(
    "subtract", ti._subtract_result_type, ti._subtract, _subtract_docstring_
)


# U34: ==== TAN         (x)
# FIXME: implement U34

# U35: ==== TANH        (x)
# FIXME: implement U35

# U36: ==== TRUNC       (x)
# FIXME: implement U36
