#                      Data Parallel Control (dpctl)
#
# Copyright 2020 Intel Corporation
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

"""Scalar types and dtype descriptors for dpctl kernel arguments."""

import builtins

import numpy as np

__all__ = [
    "dtype",
    "DpctlScalar",
    "bool",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]

_DPCTL_INT8_T = 0
_DPCTL_UINT8_T = 1
_DPCTL_INT16_T = 2
_DPCTL_UINT16_T = 3
_DPCTL_INT32_T = 4
_DPCTL_UINT32_T = 5
_DPCTL_INT64_T = 6
_DPCTL_UINT64_T = 7
_DPCTL_FLOAT32_T = 8
_DPCTL_FLOAT64_T = 9
_DPCTL_FLOAT16_T = 14
_DPCTL_COMPLEX64_T = 15
_DPCTL_COMPLEX128_T = 16


class DpctlScalar:
    """Base class for dpctl scalar kernel argument types."""

    _cast = None
    _arg_type_id = None

    __slots__ = ("_value",)

    def __init__(self, value):
        self._value = self._cast(value)

    @property
    def value(self):
        """The underlying numpy scalar value."""
        return self._value

    @property
    def dtype(self):
        """The numpy dtype of this scalar."""
        return self._value.dtype

    def __repr__(self):
        return f"dpctl.{type(self).__name__}({self._value})"


class bool(DpctlScalar):
    """Boolean scalar."""

    _cast = np.int8
    _arg_type_id = _DPCTL_INT8_T

    def __init__(self, value):
        super().__init__(builtins.bool(value))


class int8(DpctlScalar):
    """Signed 8-bit integer scalar."""

    _cast = np.int8
    _arg_type_id = _DPCTL_INT8_T


class uint8(DpctlScalar):
    """Unsigned 8-bit integer scalar."""

    _cast = np.uint8
    _arg_type_id = _DPCTL_UINT8_T


class int16(DpctlScalar):
    """Signed 16-bit integer scalar."""

    _cast = np.int16
    _arg_type_id = _DPCTL_INT16_T


class uint16(DpctlScalar):
    """Unsigned 16-bit integer scalar."""

    _cast = np.uint16
    _arg_type_id = _DPCTL_UINT16_T


class int32(DpctlScalar):
    """Signed 32-bit integer scalar."""

    _cast = np.int32
    _arg_type_id = _DPCTL_INT32_T


class uint32(DpctlScalar):
    """Unsigned 32-bit integer scalar."""

    _cast = np.uint32
    _arg_type_id = _DPCTL_UINT32_T


class int64(DpctlScalar):
    """Signed 64-bit integer scalar."""

    _cast = np.int64
    _arg_type_id = _DPCTL_INT64_T


class uint64(DpctlScalar):
    """Unsigned 64-bit integer scalar."""

    _cast = np.uint64
    _arg_type_id = _DPCTL_UINT64_T


class float16(DpctlScalar):
    """IEEE 754 half-precision (16-bit) floating-point scalar."""

    _cast = np.float16
    _arg_type_id = _DPCTL_FLOAT16_T


class float32(DpctlScalar):
    """IEEE 754 single-precision (32-bit) floating-point scalar."""

    _cast = np.float32
    _arg_type_id = _DPCTL_FLOAT32_T


class float64(DpctlScalar):
    """IEEE 754 double-precision (64-bit) floating-point scalar."""

    _cast = np.float64
    _arg_type_id = _DPCTL_FLOAT64_T


class complex64(DpctlScalar):
    """Complex type with two float32 components (64 bits total)."""

    _cast = np.complex64
    _arg_type_id = _DPCTL_COMPLEX64_T


class complex128(DpctlScalar):
    """Complex type with two float64 components (128 bits total)."""

    _cast = np.complex128
    _arg_type_id = _DPCTL_COMPLEX128_T


# dtype metadata
_scalar_type_info = {
    bool: (1, "i", "i1"),
    int8: (1, "i", "i1"),
    uint8: (1, "u", "u1"),
    int16: (2, "i", "i2"),
    uint16: (2, "u", "u2"),
    int32: (4, "i", "i4"),
    uint32: (4, "u", "u4"),
    int64: (8, "i", "i8"),
    uint64: (8, "u", "u8"),
    float16: (2, "f", "f2"),
    float32: (4, "f", "f4"),
    float64: (8, "f", "f8"),
    complex64: (8, "c", "c8"),
    complex128: (16, "c", "c16"),
}

_name_to_scalar_type = {
    "bool": bool,
    "int8": int8,
    "uint8": uint8,
    "int16": int16,
    "uint16": uint16,
    "int32": int32,
    "uint32": uint32,
    "int64": int64,
    "uint64": uint64,
    "float16": float16,
    "float32": float32,
    "float64": float64,
    "complex64": complex64,
    "complex128": complex128,
}

_numpy_dtype_num_to_scalar_type = {
    np.dtype(np.bool_).num: bool,
    np.dtype(np.int8).num: int8,
    np.dtype(np.uint8).num: uint8,
    np.dtype(np.int16).num: int16,
    np.dtype(np.uint16).num: uint16,
    np.dtype(np.int32).num: int32,
    np.dtype(np.uint32).num: uint32,
    np.dtype(np.int64).num: int64,
    np.dtype(np.uint64).num: uint64,
    np.dtype(np.longlong).num: int64,
    np.dtype(np.ulonglong).num: uint64,
    np.dtype(np.float16).num: float16,
    np.dtype(np.float32).num: float32,
    np.dtype(np.float64).num: float64,
    np.dtype(np.complex64).num: complex64,
    np.dtype(np.complex128).num: complex128,
}


def _resolve_numpy_arg(arg):
    try:
        np_dt = np.dtype(arg)
    except TypeError:
        raise TypeError(f"Cannot create a dpctl.dtype from {arg!r}")
    if np_dt.byteorder in ("<", ">"):
        raise TypeError("Only native byte order is supported.")
    if np_dt.num in _numpy_dtype_num_to_scalar_type:
        return _numpy_dtype_num_to_scalar_type[np_dt.num]
    raise TypeError(f"Cannot create a dpctl.dtype from {arg!r}")


class dtype:
    """
    Dpctl dtype descriptor class.

    Analogous to :class:`numpy.dtype`, a :class:`dpctl.dtype` identifies a
    data type and carries static metadata.

    A ``dpctl.dtype`` can be constructed from:

    - A dpctl scalar class
    - A string name
    - A NumPy dtype or type
    - Another :class:`dpctl.dtype`

    Non-native byte orders are rejected.
    """

    __slots__ = (
        "_name",
        "_itemsize",
        "_kind",
        "_str",
        "_arg_type_id",
        "_scalar_type",
    )

    def __init__(self, arg):
        if isinstance(arg, dtype):
            self._name = arg._name
            self._itemsize = arg._itemsize
            self._kind = arg._kind
            self._str = arg._str
            self._arg_type_id = arg._arg_type_id
            self._scalar_type = arg._scalar_type
            return

        if isinstance(arg, type) and issubclass(arg, DpctlScalar):
            scalar_cls = arg
        elif isinstance(arg, str) and arg in _name_to_scalar_type:
            scalar_cls = _name_to_scalar_type[arg]
        elif isinstance(arg, str):
            scalar_cls = _resolve_numpy_arg(arg)
        else:
            scalar_cls = _resolve_numpy_arg(arg)

        self._scalar_type = scalar_cls
        self._name = scalar_cls.__name__
        self._arg_type_id = scalar_cls._arg_type_id
        info = _scalar_type_info[scalar_cls]
        self._itemsize = info[0]
        self._kind = info[1]
        self._str = info[2]

    @property
    def name(self):
        """The short name of this dtype"""
        return self._name

    @property
    def itemsize(self):
        """Element size in bytes."""
        return self._itemsize

    @property
    def kind(self):
        """Single-character kind code"""
        return self._kind

    @property
    def str(self):
        """Short dtype string"""
        return self._str

    @property
    def scalar_type(self):
        """The dpctl scalar class for this dtype"""
        return self._scalar_type

    @property
    def arg_type_id(self):
        """The internal DPCTLKernelArgType enum value"""
        return self._arg_type_id

    def __repr__(self):
        return f"dpctl.dtype('{self._name}')"

    def __eq__(self, other):
        if isinstance(other, dtype):
            return self._arg_type_id == other._arg_type_id
        return NotImplemented

    def __hash__(self):
        return hash(self._arg_type_id)

    def __call__(self, value):
        """Construct a scalar of this dtype"""
        return self._scalar_type(value)
