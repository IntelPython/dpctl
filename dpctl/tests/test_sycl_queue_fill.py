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

"""Defines unit test cases for the SyclQueue.fill."""

import numpy as np
import pytest

import dpctl
import dpctl.memory

# Maps a dtype string to the NumPy dtype used to build the expected result.
_dtype_to_np = {
    "i1": np.int8,
    "u1": np.uint8,
    "i2": np.int16,
    "u2": np.uint16,
    "i4": np.int32,
    "u4": np.uint32,
    "f4": np.float32,
    "i8": np.int64,
    "u8": np.uint64,
    "f8": np.float64,
    "c8": np.complex64,
    "c16": np.complex128,
}

# A representative fill value per dtype, chosen to exercise all bytes and,
# for signed types, the sign bit.
_fill_values = {
    "i1": -7,
    "u1": 0xAB,
    "i2": -1234,
    "u2": 0xABCD,
    "i4": -123456,
    "u4": 0x0BADC0DE,
    "f4": 3.5,
    "i8": -1234567890123,
    "u8": 0x0123456789ABCDEF,
    "f8": 2.718281828459045,
    "c8": complex(1.5, -2.5),
    "c16": complex(3.14159, -2.71828),
}


def _read_back(mem, nbytes):
    """Return the first ``nbytes`` bytes of a USM allocation as ``bytes``."""
    if isinstance(mem, dpctl.memory.MemoryUSMDevice):
        result = bytearray(nbytes)
        mem.copy_to_host(result)
        return bytes(result)
    return memoryview(mem)[:nbytes].tobytes()


@pytest.mark.parametrize("dtype", list(_dtype_to_np))
@pytest.mark.parametrize(
    "usm_type",
    [
        lambda n, q: dpctl.memory.MemoryUSMShared(n, queue=q),
        lambda n, q: dpctl.memory.MemoryUSMHost(n, queue=q),
        lambda n, q: dpctl.memory.MemoryUSMDevice(n, queue=q),
    ],
    ids=["shared", "host", "device"],
)
def test_fill_with_dtype_valid(dtype, usm_type):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")

    np_dt = _dtype_to_np[dtype]
    value = _fill_values[dtype]
    num_elements = 16
    element_size = np.dtype(np_dt).itemsize
    nbytes = num_elements * element_size

    mem = usm_type(nbytes, q)
    q.fill(mem, value, num_elements, dtype=dtype)

    expected = np.full(num_elements, value, dtype=np_dt).tobytes()
    assert _read_back(mem, nbytes) == expected


def test_fill_default_dtype_is_bytewise():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")

    nbytes = 64
    mem = dpctl.memory.MemoryUSMShared(nbytes, queue=q)

    q.fill(mem, 0, nbytes)
    assert memoryview(mem).tobytes() == b"\x00" * nbytes

    q.fill(mem, 0xFF, nbytes)
    assert memoryview(mem).tobytes() == b"\xff" * nbytes


def test_fill_signed_negative_value():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")

    num_elements = 8
    mem = dpctl.memory.MemoryUSMShared(num_elements, queue=q)

    q.fill(mem, -1, num_elements, dtype="i1")
    assert memoryview(mem).tobytes() == b"\xff" * num_elements


@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_fill_complex_from_real_scalar(dtype):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")

    np_dt = _dtype_to_np[dtype]
    num_elements = 8
    nbytes = num_elements * np.dtype(np_dt).itemsize

    mem = dpctl.memory.MemoryUSMShared(nbytes, queue=q)
    q.fill(mem, 2.0, num_elements, dtype=dtype)

    expected = np.full(num_elements, complex(2.0, 0.0), dtype=np_dt).tobytes()
    assert memoryview(mem)[:nbytes].tobytes() == expected


@pytest.mark.parametrize(
    "dtype,element_size",
    [
        ("i2", 2),
        ("i4", 4),
        ("f8", 8),
        ("u8", 8),
    ],
)
def test_fill_count_is_in_elements(dtype, element_size):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")

    num_elements = 8
    nbytes = num_elements * element_size

    mem = dpctl.memory.MemoryUSMShared(nbytes, queue=q)

    # Seed the whole allocation with a known sentinel to verify the untouched
    # tail is left intact.
    mv = memoryview(mem)
    for i in range(nbytes):
        mv[i] = 0xAA

    # Filling half the elements writes exactly half the bytes.
    q.fill(mem, 1, num_elements // 2, dtype=dtype)

    half_bytes = (num_elements // 2) * element_size
    expected_head = np.full(
        num_elements // 2, 1, dtype=_dtype_to_np[dtype]
    ).tobytes()
    assert mv[:half_bytes].tobytes() == expected_head
    assert mv[half_bytes:nbytes].tobytes() == b"\xaa" * (nbytes - half_bytes)


def test_fill_with_invalid_dtype():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")

    mem = dpctl.memory.MemoryUSMShared(64, queue=q)

    for bad_dtype in ["i3", "f16", "u16", "x4", "42", "float", ""]:
        with pytest.raises(ValueError) as cm:
            q.fill(mem, 0, 8, dtype=bad_dtype)
        assert (
            "dtype" in str(cm.value).lower()
            or "unrecognized" in str(cm.value).lower()
        )


@pytest.mark.parametrize(
    "dtype,value",
    [
        ("u1", 256),
        ("u1", -1),
        ("i1", 128),
        ("u2", 1 << 16),
        ("i4", 1.5),
        ("u8", 1 << 64),
    ],
)
def test_fill_value_out_of_range(dtype, value):
    """fill raises ValueError when value cannot be represented as dtype."""
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")

    mem = dpctl.memory.MemoryUSMShared(64, queue=q)

    with pytest.raises(ValueError):
        q.fill(mem, value, 8, dtype=dtype)


def test_fill_type_error():
    """fill raises TypeError when ``dest`` is not a USM allocation."""
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")

    with pytest.raises(TypeError) as cm:
        q.fill(None, 0, 4)
    assert "_Memory" in str(cm.value)

    with pytest.raises(TypeError) as cm:
        q.fill(bytearray(16), 0, 4)
    assert "_Memory" in str(cm.value)
