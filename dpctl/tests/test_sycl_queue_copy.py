#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2025 Intel Corporation
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

"""Defines unit test cases for the SyclQueue.copy."""

import numpy as np
import pytest

import dpctl
import dpctl.memory


def _create_memory(q):
    nbytes = 1024
    mobj = dpctl.memory.MemoryUSMShared(nbytes, queue=q)
    return mobj


def test_copy_host_to_host():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")

    src_buf = b"abcdefghijklmnopqrstuvwxyz"
    dst_buf = bytearray(len(src_buf))

    q.copy(dst_buf, src_buf, len(src_buf))

    assert dst_buf == src_buf


def test_copy_async():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")

    src_buf = b"abcdefghijklmnopqrstuvwxyz"
    n = len(src_buf)
    dst_buf = bytearray(n)
    dst_buf2 = bytearray(n)

    e = q.copy_async(dst_buf, src_buf, n)
    e2 = q.copy_async(dst_buf2, src_buf, n, [e])

    e.wait()
    e2.wait()
    assert dst_buf == src_buf
    assert dst_buf2 == src_buf


def test_copy_type_error():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")
    mobj = _create_memory(q)

    with pytest.raises(TypeError) as cm:
        q.copy(None, mobj, 3)
    assert "_Memory" in str(cm.value)

    with pytest.raises(TypeError) as cm:
        q.copy(mobj, None, 3)
    assert "_Memory" in str(cm.value)


@pytest.mark.parametrize(
    "dtype,element_size",
    [
        ("i1", 1),
        ("u1", 1),
        ("i2", 2),
        ("u2", 2),
        ("i4", 4),
        ("u4", 4),
        ("f4", 4),
        ("i8", 8),
        ("u8", 8),
        ("f8", 8),
    ],
)
@pytest.mark.parametrize(
    "usm_type",
    [
        lambda n, q: dpctl.memory.MemoryUSMShared(n, queue=q),
        lambda n, q: dpctl.memory.MemoryUSMHost(n, queue=q),
        lambda n, q: dpctl.memory.MemoryUSMDevice(n, queue=q),
    ],
    ids=["shared", "host", "device"],
)
def test_copy_with_dtype_valid(dtype, element_size, usm_type):
    """Test copy with valid dtype parameter."""
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")

    num_elements = 16
    nbytes = num_elements * element_size

    src = usm_type(nbytes, q)
    dst = usm_type(nbytes, q)

    # For device memory, use host buffer for init and verification
    if isinstance(src, dpctl.memory.MemoryUSMDevice):
        # Initialize via host buffer
        host_buf = bytearray(i % 256 for i in range(nbytes))
        src.copy_from_host(host_buf)

        # Copy device to device, count given in elements of dtype
        q.copy(dst, src, num_elements, dtype=dtype)

        # Verify via host buffer
        result_buf = bytearray(nbytes)
        dst.copy_to_host(result_buf)
        assert result_buf == host_buf
    else:
        # Shared/Host memory can be accessed directly
        src_mv = memoryview(src)
        for i in range(nbytes):
            src_mv[i] = i % 256

        # Copy, count given in elements of dtype
        q.copy(dst, src, num_elements, dtype=dtype)

        # Verify
        dst_mv = memoryview(dst)
        assert dst_mv[:nbytes].tobytes() == src_mv[:nbytes].tobytes()


def test_copy_async_with_dtype_valid():
    """Test copy_async with valid dtype parameter."""
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")

    dtype = "f8"
    element_size = 8
    num_elements = 10
    nbytes = num_elements * element_size

    src = dpctl.memory.MemoryUSMShared(nbytes, queue=q)
    dst = dpctl.memory.MemoryUSMShared(nbytes, queue=q)

    # Initialize source
    src_mv = memoryview(src)
    for i in range(nbytes):
        src_mv[i] = i % 256

    # Async copy with dtype, count given in elements
    e = q.copy_async(dst, src, num_elements, dtype=dtype)
    e.wait()

    # Verify
    dst_mv = memoryview(dst)
    assert dst_mv[:nbytes].tobytes() == src_mv[:nbytes].tobytes()


@pytest.mark.parametrize(
    "dtype,element_size",
    [
        ("i2", 2),
        ("i4", 4),
        ("f8", 8),
        ("u8", 8),
    ],
)
def test_copy_count_is_in_elements(dtype, element_size):
    """``count`` is interpreted as a number of elements of ``dtype``."""
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")

    num_elements = 4
    nbytes = num_elements * element_size

    src = dpctl.memory.MemoryUSMShared(nbytes, queue=q)
    dst = dpctl.memory.MemoryUSMShared(nbytes, queue=q)

    # USM allocations are uninitialized, so seed dst with a known sentinel
    # to verify the untouched half is left intact.
    src_mv = memoryview(src)
    dst_mv = memoryview(dst)
    for i in range(nbytes):
        src_mv[i] = i % 256
        dst_mv[i] = 0xAA

    # Copying half the elements transfers exactly half the bytes.
    q.copy(dst, src, num_elements // 2, dtype=dtype)

    half_bytes = (num_elements // 2) * element_size
    assert dst_mv[:half_bytes].tobytes() == src_mv[:half_bytes].tobytes()
    assert dst_mv[half_bytes:nbytes].tobytes() == b"\xaa" * (
        nbytes - half_bytes
    )


def test_copy_with_invalid_dtype():
    """Test that copy raises ValueError for unrecognized dtype."""
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")

    nbytes = 64
    src = dpctl.memory.MemoryUSMShared(nbytes, queue=q)
    dst = dpctl.memory.MemoryUSMShared(nbytes, queue=q)

    invalid_dtypes = ["i3", "f16", "u16", "x4", "42", "float", ""]

    for bad_dtype in invalid_dtypes:
        with pytest.raises(ValueError) as cm:
            q.copy(dst, src, nbytes, dtype=bad_dtype)
        assert (
            "dtype" in str(cm.value).lower()
            or "unrecognized" in str(cm.value).lower()
        )


def test_copy_with_dtype_host_buffers():
    """Test typed copy with host buffers (numpy arrays)."""
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")

    dtype = "f4"
    num_elements = 20

    src = np.arange(num_elements, dtype=np.float32)
    dst = np.zeros(num_elements, dtype=np.float32)

    q.copy(dst, src, num_elements, dtype=dtype)

    assert np.array_equal(dst, src)


def test_copy_with_dtype_mixed_sources():
    """Test typed copy with mixed USM and host buffers."""
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")

    dtype = "i8"
    num_elements = 10
    nbytes = num_elements * 8

    # Host to USM
    src_host = np.arange(num_elements, dtype=np.int64)
    dst_usm = dpctl.memory.MemoryUSMShared(nbytes, queue=q)

    q.copy(dst_usm, src_host, num_elements, dtype=dtype)

    # USM to host
    dst_host = np.zeros(num_elements, dtype=np.int64)
    q.copy(dst_host, dst_usm, num_elements, dtype=dtype)

    assert np.array_equal(dst_host, src_host)


def test_copy_without_dtype_backward_compat():
    """Test that copy without dtype parameter works (backward compatibility)."""
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")

    nbytes = 32
    src = dpctl.memory.MemoryUSMShared(nbytes, queue=q)
    dst = dpctl.memory.MemoryUSMShared(nbytes, queue=q)

    src_mv = memoryview(src)
    for i in range(nbytes):
        src_mv[i] = i % 256

    # Should work without dtype parameter (existing behavior)
    q.copy(dst, src, nbytes)

    dst_mv = memoryview(dst)
    assert dst_mv[:nbytes].tobytes() == src_mv[:nbytes].tobytes()
