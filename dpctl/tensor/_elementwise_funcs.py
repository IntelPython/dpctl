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
Calculate the absolute value element-wise.
"""

abs = UnaryElementwiseFunc("abs", ti._abs_result_type, ti._abs, _abs_docstring_)

# U02: ==== ACOS   (x)
# FIXME: implement U02

# U03: ===== ACOSH (x)
# FIXME: implement U03

# B01: ===== ADD   (x1, x2)

_add_docstring_ = """
add(x1, x2, order='K')

Calculates the sum for each element `x1_i` of the input array `x1` with
the respective element `x2_i` of the input array `x2`.

Args:
    x1 (usm_ndarray):
        First input array, expected to have numeric data type.
    x2 (usm_ndarray):
        Second input array, also expected to have numeric data type.
Returns:
    usm_narray:
        an array containing the element-wise sums. The data type of the
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
# FIXME: implement U10

# U11: ==== COS           (x)
_cos_docstring = """
cos(x, order='K')

Computes cosine for each element `x_i` for input array `x`.
"""

cos = UnaryElementwiseFunc("cos", ti._cos_result_type, ti._cos, _cos_docstring)

# U12: ==== COSH          (x)
# FIXME: implement U12

# B08: ==== DIVIDE        (x1, x2)
_divide_docstring_ = """
divide(x1, x2, order='K')

Calculates the ratio for each element `x1_i` of the input array `x1` with
the respective element `x2_i` of the input array `x2`.

Args:
    x1 (usm_ndarray):
        First input array, expected to have numeric data type.
    x2 (usm_ndarray):
        Second input array, also expected to have numeric data type.
Returns:
    usm_narray:
        an array containing the result of element-wise division. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

divide = BinaryElementwiseFunc(
    "divide", ti._divide_result_type, ti._divide, _divide_docstring_
)

# B09: ==== EQUAL         (x1, x2)
_equal_docstring_ = """
equal(x1, x2, order='K')

Calculates equality test results for each element `x1_i` of the input array `x1`
with the respective element `x2_i` of the input array `x2`.

Args:
    x1 (usm_ndarray):
        First input array, expected to have numeric data type.
    x2 (usm_ndarray):
        Second input array, also expected to have numeric data type.
Returns:
    usm_narray:
        an array containing the result of element-wise equality comparison.
        The data type of the returned array is determined by the
        Type Promotion Rules.
"""

equal = BinaryElementwiseFunc(
    "equal", ti._equal_result_type, ti._equal, _equal_docstring_
)

# U13: ==== EXP           (x)
# FIXME: implement U13

# U14: ==== EXPM1         (x)
# FIXME: implement U14

# U15: ==== FLOOR         (x)
# FIXME: implement U15

# B10: ==== FLOOR_DIVIDE  (x1, x2)
# FIXME: implement B10

# B11: ==== GREATER       (x1, x2)
# FIXME: implement B11

# B12: ==== GREATER_EQUAL (x1, x2)
# FIXME: implement B12

# U16: ==== IMAG        (x)
# FIXME: implement U16

# U17: ==== ISFINITE    (x)
_isfinite_docstring_ = """
Computes if every element of input array is a finite number.
"""

isfinite = UnaryElementwiseFunc(
    "isfinite", ti._isfinite_result_type, ti._isfinite, _isfinite_docstring_
)

# U18: ==== ISINF       (x)
_isinf_docstring_ = """
Computes if every element of input array is an infinity.
"""

isinf = UnaryElementwiseFunc(
    "isinf", ti._isinf_result_type, ti._isinf, _isinf_docstring_
)

# U19: ==== ISNAN       (x)
_isnan_docstring_ = """
Computes if every element of input array is a NaN.
"""

isnan = UnaryElementwiseFunc(
    "isnan", ti._isnan_result_type, ti._isnan, _isnan_docstring_
)

# B13: ==== LESS        (x1, x2)
# FIXME: implement B13

# B14: ==== LESS_EQUAL  (x1, x2)
# FIXME: implement B14

# U20: ==== LOG         (x)
# FIXME: implement U20

# U21: ==== LOG1P       (x)
# FIXME: implement U21

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
# FIXME: implement B19

# U25: ==== NEGATIVE    (x)
# FIXME: implement U25

# B20: ==== NOT_EQUAL   (x1, x2)
# FIXME: implement B20

# U26: ==== POSITIVE    (x)
# FIXME: implement U26

# B21: ==== POW         (x1, x2)
# FIXME: implement B21

# U27: ==== REAL        (x)
# FIXME: implement U27

# B22: ==== REMAINDER   (x1, x2)
# FIXME: implement B22

# U28: ==== ROUND       (x)
# FIXME: implement U28

# U29: ==== SIGN        (x)
# FIXME: implement U29

# U30: ==== SIN         (x)
# FIXME: implement U30

# U31: ==== SINH        (x)
# FIXME: implement U31

# U32: ==== SQUARE      (x)
# FIXME: implement U32

# U33: ==== SQRT        (x)
_sqrt_docstring_ = """
Computes sqrt for each element `x_i` for input array `x`.
"""

sqrt = UnaryElementwiseFunc(
    "sqrt", ti._sqrt_result_type, ti._sqrt, _sqrt_docstring_
)

# B23: ==== SUBTRACT    (x1, x2)
# FIXME: implement B23

# U34: ==== TAN         (x)
# FIXME: implement U34

# U35: ==== TANH        (x)
# FIXME: implement U35

# U36: ==== TRUNC       (x)
# FIXME: implement U36
