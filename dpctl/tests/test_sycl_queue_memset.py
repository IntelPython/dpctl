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

"""Defines unit test cases for the SyclQueue.memset."""

import pytest

import dpctl
import dpctl.memory


def _create_memory(q, nbytes=1024):
    return dpctl.memory.MemoryUSMShared(nbytes, queue=q)


_MEMORY_CLASSES = [
    dpctl.memory.MemoryUSMShared,
    dpctl.memory.MemoryUSMHost,
    dpctl.memory.MemoryUSMDevice,
]


def _read_back(q, mobj, nbytes):
    """Copy USM memory to host and return it as bytes (works for device)."""
    host = bytearray(nbytes)
    q.memcpy(host, mobj, nbytes)
    return bytes(host)


@pytest.mark.parametrize("mem_cls", _MEMORY_CLASSES)
def test_memset_fills_whole_allocation(mem_cls):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")
    nbytes = 256
    mobj = mem_cls(nbytes, queue=q)

    q.memset(mobj, 0xAB)

    assert _read_back(q, mobj, nbytes) == b"\xab" * nbytes


def test_memset_zero_count_fills_whole_allocation():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")
    nbytes = 64
    mobj = _create_memory(q, nbytes)

    q.memset(mobj, 0x01, 0)

    assert bytes(memoryview(mobj)) == b"\x01" * nbytes


def test_memset_partial_count():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")
    nbytes = 16
    mobj = _create_memory(q, nbytes)

    # zero-out first, then fill only the leading 4 bytes
    q.memset(mobj, 0x00)
    q.memset(mobj, 0x7F, 4)

    assert bytes(memoryview(mobj)) == b"\x7f" * 4 + b"\x00" * (nbytes - 4)


def test_memset_count_clamped_to_allocation():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")
    nbytes = 8
    mobj = _create_memory(q, nbytes)

    # requesting more bytes than allocated must not overrun; it is clamped
    q.memset(mobj, 0x02, 4 * nbytes)

    assert bytes(memoryview(mobj)) == b"\x02" * nbytes


def test_memset_zero_value():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")
    nbytes = 32
    mobj = _create_memory(q, nbytes)

    q.memset(mobj, 0xFF)
    q.memset(mobj, 0)

    assert bytes(memoryview(mobj)) == b"\x00" * nbytes


def test_memset_type_error():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")

    with pytest.raises(TypeError) as cm:
        q.memset(None, 1)
    assert "_Memory" in str(cm.value)


@pytest.mark.parametrize("mem_cls", _MEMORY_CLASSES)
def test_memset_async(mem_cls):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")
    nbytes = 64
    mobj = mem_cls(nbytes, queue=q)

    e = q.memset_async(mobj, 0xAB)
    assert isinstance(e, dpctl.SyclEvent)
    e.wait()

    assert _read_back(q, mobj, nbytes) == b"\xab" * nbytes


def test_memset_async_with_dependent_events():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")
    nbytes = 64
    mobj = _create_memory(q, nbytes)

    e1 = q.memset_async(mobj, 0x01)
    e2 = q.memset_async(mobj, 0x02, nbytes, [e1])
    e2.wait()

    assert bytes(memoryview(mobj)) == b"\x02" * nbytes


def test_memset_async_partial_count():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")
    nbytes = 16
    mobj = _create_memory(q, nbytes)

    q.memset(mobj, 0x00)
    e = q.memset_async(mobj, 0x7F, 4)
    e.wait()

    assert bytes(memoryview(mobj)) == b"\x7f" * 4 + b"\x00" * (nbytes - 4)


def test_memset_async_type_error():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")
    mobj = _create_memory(q)

    with pytest.raises(TypeError) as cm:
        q.memset_async(None, 1)
    assert "_Memory" in str(cm.value)

    with pytest.raises(TypeError):
        q.memset_async(mobj, 1, 0, [None])


@pytest.mark.parametrize(
    "val, expected",
    [
        (0xAB, 0xAB),
        (0, 0x00),
        (255, 0xFF),
        (256, 0x00),
        (-1, 0xFF),
        (300, 0x2C),
    ],
)
def test_memset_value_truncated_to_byte(val, expected):
    # ``val`` is used as a single byte, so values wrap modulo 256
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")
    nbytes = 8
    mobj = _create_memory(q, nbytes)

    q.memset(mobj, val)

    assert bytes(memoryview(mobj)) == bytes([expected]) * nbytes
