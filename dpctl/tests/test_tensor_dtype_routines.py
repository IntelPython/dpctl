#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2024 Intel Corporation
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


import pytest

import dpctl
import dpctl.tensor as dpt

list_dtypes = [
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]


dtype_categories = {
    "bool": ["bool"],
    "signed integer": ["int8", "int16", "int32", "int64"],
    "unsigned integer": ["uint8", "uint16", "uint32", "uint64"],
    "integral": [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    ],
    "real floating": ["float16", "float32", "float64"],
    "complex floating": ["complex64", "complex128"],
    "numeric": [d for d in list_dtypes if d != "bool"],
}


@pytest.mark.parametrize("kind_str", dtype_categories.keys())
@pytest.mark.parametrize("dtype_str", list_dtypes)
def test_isdtype_kind_str(dtype_str, kind_str):
    dt = dpt.dtype(dtype_str)
    is_in_kind = dpt.isdtype(dt, kind_str)
    expected = dtype_str in dtype_categories[kind_str]
    assert is_in_kind == expected


@pytest.mark.parametrize("dtype_str", list_dtypes)
def test_isdtype_kind_tuple(dtype_str):
    dt = dpt.dtype(dtype_str)
    if dtype_str.startswith("bool"):
        assert dpt.isdtype(dt, ("real floating", "bool"))
        assert not dpt.isdtype(
            dt, ("integral", "real floating", "complex floating")
        )
    elif dtype_str.startswith("int"):
        assert dpt.isdtype(dt, ("real floating", "signed integer"))
        assert not dpt.isdtype(
            dt, ("bool", "unsigned integer", "real floating")
        )
    elif dtype_str.startswith("uint"):
        assert dpt.isdtype(dt, ("bool", "unsigned integer"))
        assert not dpt.isdtype(dt, ("real floating", "complex floating"))
    elif dtype_str.startswith("float"):
        assert dpt.isdtype(dt, ("complex floating", "real floating"))
        assert not dpt.isdtype(dt, ("integral", "complex floating", "bool"))
    else:
        assert dpt.isdtype(dt, ("integral", "complex floating"))
        assert not dpt.isdtype(dt, ("bool", "integral", "real floating"))


@pytest.mark.parametrize("dtype_str", list_dtypes)
def test_isdtype_kind_tuple_dtypes(dtype_str):
    dt = dpt.dtype(dtype_str)
    if dtype_str.startswith("bool"):
        assert dpt.isdtype(dt, (dpt.int32, dpt.bool))
        assert not dpt.isdtype(dt, (dpt.int16, dpt.uint32, dpt.float64))

    elif dtype_str.startswith("int"):
        assert dpt.isdtype(dt, (dpt.int8, dpt.int16, dpt.int32, dpt.int64))
        assert not dpt.isdtype(dt, (dpt.bool, dpt.float32, dpt.complex64))

    elif dtype_str.startswith("uint"):
        assert dpt.isdtype(dt, (dpt.uint8, dpt.uint16, dpt.uint32, dpt.uint64))
        assert not dpt.isdtype(dt, (dpt.bool, dpt.int32, dpt.float32))

    elif dtype_str.startswith("float"):
        assert dpt.isdtype(dt, (dpt.float16, dpt.float32, dpt.float64))
        assert not dpt.isdtype(dt, (dpt.bool, dpt.complex64, dpt.int8))

    else:
        assert dpt.isdtype(dt, (dpt.complex64, dpt.complex128))
        assert not dpt.isdtype(dt, (dpt.bool, dpt.uint64, dpt.int8))


@pytest.mark.parametrize(
    "kind",
    [
        [dpt.int32, dpt.bool],
        "f4",
        float,
        123,
        "complex",
    ],
)
def test_isdtype_invalid_kind(kind):
    with pytest.raises((TypeError, ValueError)):
        dpt.isdtype(dpt.int32, kind)


def test_finfo_array():
    try:
        x = dpt.empty(tuple(), dtype="f4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("Default-selected SYCL device unavailable")
    o = dpt.finfo(x)
    assert o.dtype == dpt.float32


def test_iinfo_array():
    try:
        x = dpt.empty(tuple(), dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("Default-selected SYCL device unavailable")
    o = dpt.iinfo(x)
    assert o.dtype == dpt.int32


def test_iinfo_validation():
    with pytest.raises(ValueError):
        dpt.iinfo("O")


def test_finfo_validation():
    with pytest.raises(ValueError):
        dpt.iinfo("O")
