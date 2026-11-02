#                      Data Parallel Control (dpctl)
#
# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data type handling shared by the dpctl functions taking a data type."""

import struct

import numpy as np

# kinds of NumPy data types whose values are not plain bytes
_NON_POD_KINDS = frozenset("OSUMmT")

# extended precision types, which devices do not support
_EXTENDED_PRECISION_NUMS = frozenset(
    np.dtype(t).num
    for t in (np.longdouble, np.clongdouble)
    if np.finfo(t).nmant > np.finfo(np.float64).nmant
)

# struct formats of the data types whose values are range checked when packed
_STRUCT_FORMATS = {
    ("b", 1): "?",
    ("i", 1): "b",
    ("i", 2): "h",
    ("i", 4): "i",
    ("i", 8): "q",
    ("u", 1): "B",
    ("u", 2): "H",
    ("u", 4): "I",
    ("u", 8): "Q",
    ("f", 2): "e",
    ("f", 4): "f",
    ("f", 8): "d",
    ("c", 8): "ff",
    ("c", 16): "dd",
}


def is_supported_dtype(dt):
    """Returns whether values of :class:`numpy.dtype` ``dt`` can be used on a
    device."""
    return not (
        dt.hasobject
        or dt.kind in _NON_POD_KINDS
        or dt.itemsize == 0
        or dt.num in _EXTENDED_PRECISION_NUMS
    )


def as_dtype(dtype):
    """Returns ``numpy.dtype(dtype)``.

    Raises:
        ValueError: if ``dtype`` is not a supported data type.
    """
    try:
        dt = np.dtype(dtype)
    except TypeError:
        raise ValueError(f"Unrecognized dtype '{dtype}'") from None
    if not is_supported_dtype(dt):
        raise ValueError(f"Unsupported dtype '{dt}'")
    return dt


def as_scalar(value):
    """Converts a Python ``bool``, ``int``, ``float`` or ``complex`` to a
    ``numpy.bool``, ``numpy.int64``, ``numpy.float64`` or ``numpy.complex128``
    respectively. Other values are returned unchanged.

    Raises:
        OverflowError: if an ``int`` does not fit in ``numpy.int64``.
    """
    if isinstance(value, np.generic):
        return value
    if isinstance(value, bool):
        return np.bool_(value)
    if isinstance(value, int):
        return np.int64(value)
    if isinstance(value, float):
        return np.float64(value)
    if isinstance(value, complex):
        return np.complex128(value)
    return value


def pack_value(value, dt):
    """Returns the bytes of ``value`` converted to :class:`numpy.dtype` ``dt``.

    Raises:
        ValueError: if ``value`` can not be represented in ``dt``.
    """
    fmt = _STRUCT_FORMATS.get((dt.kind, dt.itemsize))
    try:
        if fmt is None:
            return np.asarray(value, dtype=dt).tobytes()
        if dt.kind == "c":
            cval = complex(value)
            return struct.pack("=" + fmt, cval.real, cval.imag)
        return struct.pack("=" + fmt, value)
    except (struct.error, OverflowError, TypeError, ValueError) as e:
        raise ValueError(
            f"Value {value!r} cannot be represented as dtype '{dt}': {e}"
        ) from None
