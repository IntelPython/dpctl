import dpctl.tensor._tensor_impl as ti

from ._elementwise_common import BinaryElementwiseFunc, UnaryElementwiseFunc

# ABS
_abs_docstring_ = """
Calculate the absolute value element-wise.
"""

abs = UnaryElementwiseFunc("abs", ti._abs_result_type, ti._abs, _abs_docstring_)

# ADD

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


# COS

_cos_docstring = """
cos(x, order='K')

Computes cosine for each element `x_i` for input array `x`.
"""

cos = UnaryElementwiseFunc("cos", ti._cos_result_type, ti._cos, _cos_docstring)

# ISFINITE

_isfinite_docstring_ = """
Computes if every element of input array is a finite number.
"""

isfinite = UnaryElementwiseFunc(
    "isfinite", ti._isfinite_result_type, ti._isfinite, _isfinite_docstring_
)

# ISNAN

_isnan_docstring_ = """
Computes if every element of input array is a NaN.
"""

isnan = UnaryElementwiseFunc(
    "isnan", ti._isnan_result_type, ti._isnan, _isnan_docstring_
)

# ISINF

_isinf_docstring_ = """
Computes if every element of input array is an infinity.
"""

isinf = UnaryElementwiseFunc(
    "isinf", ti._isinf_result_type, ti._isinf, _isinf_docstring_
)

# SQRT

_sqrt_docstring_ = """
Computes sqrt for each element `x_i` for input array `x`.
"""

sqrt = UnaryElementwiseFunc(
    "sqrt", ti._sqrt_result_type, ti._sqrt, _sqrt_docstring_
)
