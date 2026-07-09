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

"""Defines unit test cases for the dpctl scalar types and dtype class"""

import numpy as np
import pytest

import dpctl
from dpctl._dtypes import DpctlScalar, dtype


@pytest.mark.parametrize(
    "dpctl_type,value,expected_dtype",
    [
        (dpctl.bool, True, np.dtype(np.int8)),
        (dpctl.bool, False, np.dtype(np.int8)),
        (dpctl.int8, -3, np.dtype(np.int8)),
        (dpctl.uint8, 200, np.dtype(np.uint8)),
        (dpctl.int16, -1000, np.dtype(np.int16)),
        (dpctl.uint16, 60000, np.dtype(np.uint16)),
        (dpctl.int32, -100000, np.dtype(np.int32)),
        (dpctl.uint32, 3000000000, np.dtype(np.uint32)),
        (dpctl.int64, -(2**40), np.dtype(np.int64)),
        (dpctl.uint64, 2**40, np.dtype(np.uint64)),
        (dpctl.float16, 3.14, np.dtype(np.float16)),
        (dpctl.float32, 2.718, np.dtype(np.float32)),
        (dpctl.float64, 1.41421356, np.dtype(np.float64)),
        (dpctl.complex64, 1 + 2j, np.dtype(np.complex64)),
        (dpctl.complex128, 3 + 4j, np.dtype(np.complex128)),
    ],
)
def test_scalar_creation(dpctl_type, value, expected_dtype):
    scalar = dpctl_type(value)
    assert isinstance(scalar, DpctlScalar)
    assert scalar.dtype == expected_dtype


@pytest.mark.parametrize(
    "dpctl_type,value",
    [
        (dpctl.int32, 42),
        (dpctl.float64, 3.14),
        (dpctl.complex128, 1 + 2j),
    ],
)
def test_scalar_value(dpctl_type, value):
    scalar = dpctl_type(value)
    assert scalar.value == dpctl_type._cast(value)


def test_bool_conversion():
    t = dpctl.bool(1)
    f = dpctl.bool(0)
    assert t.value == np.int8(1)
    assert f.value == np.int8(0)
    truthy = dpctl.bool([1, 2, 3])
    assert truthy.value == np.int8(1)


def test_scalar_repr():
    s = dpctl.int32(5)
    assert repr(s) == "dpctl.int32(5)"
    s = dpctl.float16(1.5)
    assert repr(s) == "dpctl.float16(1.5)"


@pytest.mark.parametrize(
    "dpctl_type",
    [
        dpctl.bool,
        dpctl.int8,
        dpctl.uint8,
        dpctl.int16,
        dpctl.uint16,
        dpctl.int32,
        dpctl.uint32,
        dpctl.int64,
        dpctl.uint64,
        dpctl.float16,
        dpctl.float32,
        dpctl.float64,
        dpctl.complex64,
        dpctl.complex128,
    ],
)
def test_scalar_has_arg_type_id(dpctl_type):
    assert dpctl_type._arg_type_id is not None
    assert isinstance(dpctl_type._arg_type_id, int)


@pytest.mark.parametrize(
    "dpctl_type",
    [
        dpctl.bool,
        dpctl.int8,
        dpctl.uint8,
        dpctl.int16,
        dpctl.uint16,
        dpctl.int32,
        dpctl.uint32,
        dpctl.int64,
        dpctl.uint64,
        dpctl.float16,
        dpctl.float32,
        dpctl.float64,
        dpctl.complex64,
        dpctl.complex128,
    ],
)
def test_scalar_numpy_buffer_protocol(dpctl_type):
    """Verify that the underlying numpy scalar supports the buffer protocol."""
    scalar = dpctl_type(1)
    mv = memoryview(scalar.value)
    assert mv.nbytes == scalar.value.dtype.itemsize


@pytest.mark.parametrize(
    "scalar_cls,expected_name,expected_itemsize,expected_kind",
    [
        (dpctl.bool, "bool", 1, "i"),
        (dpctl.int8, "int8", 1, "i"),
        (dpctl.uint8, "uint8", 1, "u"),
        (dpctl.int16, "int16", 2, "i"),
        (dpctl.uint16, "uint16", 2, "u"),
        (dpctl.int32, "int32", 4, "i"),
        (dpctl.uint32, "uint32", 4, "u"),
        (dpctl.int64, "int64", 8, "i"),
        (dpctl.uint64, "uint64", 8, "u"),
        (dpctl.float16, "float16", 2, "f"),
        (dpctl.float32, "float32", 4, "f"),
        (dpctl.float64, "float64", 8, "f"),
        (dpctl.complex64, "complex64", 8, "c"),
        (dpctl.complex128, "complex128", 16, "c"),
    ],
)
def test_dtype_from_scalar_class(
    scalar_cls, expected_name, expected_itemsize, expected_kind
):
    dt = dtype(scalar_cls)
    assert dt.name == expected_name
    assert dt.itemsize == expected_itemsize
    assert dt.kind == expected_kind
    assert dt.scalar_type is scalar_cls


@pytest.mark.parametrize(
    "name",
    [
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
    ],
)
def test_dtype_from_string(name):
    dt = dtype(name)
    assert dt.name == name
    assert dt.itemsize > 0


@pytest.mark.parametrize(
    "np_type,expected_name",
    [
        (np.int8, "int8"),
        (np.uint8, "uint8"),
        (np.int16, "int16"),
        (np.int32, "int32"),
        (np.float16, "float16"),
        (np.float32, "float32"),
        (np.float64, "float64"),
        (np.complex64, "complex64"),
        (np.complex128, "complex128"),
    ],
)
def test_dtype_from_numpy(np_type, expected_name):
    dt = dtype(np_type)
    assert dt.name == expected_name

    dt2 = dtype(np.dtype(np_type))
    assert dt == dt2


def test_dtype_from_dtype():
    dt1 = dtype(dpctl.float32)
    dt2 = dtype(dt1)
    assert dt1 == dt2
    assert dt1.name == dt2.name


def test_dtype_equality_and_hash():
    dt1 = dtype(dpctl.int32)
    dt2 = dtype("int32")
    dt3 = dtype(np.int32)
    assert dt1 == dt2
    assert dt2 == dt3
    assert hash(dt1) == hash(dt2) == hash(dt3)

    dt4 = dtype(dpctl.float32)
    assert dt1 != dt4


def test_dtype_repr():
    dt = dtype(dpctl.float32)
    assert repr(dt) == "dpctl.dtype('float32')"


def test_dtype_callable():
    dt = dtype("float32")
    scalar = dt(3.14)
    assert isinstance(scalar, DpctlScalar)
    assert isinstance(scalar, dpctl.float32)


def test_dtype_invalid():
    with pytest.raises(TypeError):
        dtype("not_a_type")

    with pytest.raises(TypeError):
        dtype(object)


def test_dtype_str_property():
    assert dtype(dpctl.int32).str == "i4"
    assert dtype(dpctl.float16).str == "f2"
    assert dtype(dpctl.complex128).str == "c16"


def _non_native_prefix():
    """Return the byte-order prefix that is non-native on this system."""
    import sys

    return ">" if sys.byteorder == "little" else "<"


@pytest.mark.parametrize(
    "suffix",
    ["f4", "i4", "u8", "i2"],
)
def test_dtype_rejects_non_native_endianness_string(suffix):
    arg = _non_native_prefix() + suffix
    with pytest.raises(TypeError):
        dtype(arg)


@pytest.mark.parametrize(
    "suffix",
    ["f4", "i2"],
)
def test_dtype_rejects_non_native_endianness_numpy(suffix):
    np_dt = np.dtype(_non_native_prefix() + suffix)
    with pytest.raises(TypeError):
        dtype(np_dt)


@pytest.mark.parametrize(
    "arg",
    [
        "=f4",
        "|i1",
        "=i4",
        "=c16",
    ],
)
def test_dtype_accepts_native_endianness(arg):
    dt = dtype(arg)
    assert dt.itemsize > 0
