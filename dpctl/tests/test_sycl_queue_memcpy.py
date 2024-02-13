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

"""Defines unit test cases for the SyclQueue.memcpy.
"""

import pytest

import dpctl
import dpctl.memory


def _create_memory(q):
    nbytes = 1024
    mobj = dpctl.memory.MemoryUSMShared(nbytes, queue=q)
    return mobj


def test_memcpy_copy_usm_to_usm():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")
    mobj1 = _create_memory(q)
    mobj2 = _create_memory(q)

    mv1 = memoryview(mobj1)
    mv2 = memoryview(mobj2)

    mv1[:3] = b"123"

    q.memcpy(mobj2, mobj1, 3)

    assert mv2[:3] == b"123"


def test_memcpy_copy_host_to_usm():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")
    usm_obj = _create_memory(q)

    canary = bytearray(b"123456789")
    host_obj = memoryview(canary)

    q.memcpy(usm_obj, host_obj, len(canary))

    mv2 = memoryview(usm_obj)

    assert mv2[: len(canary)] == canary


def test_memcpy_copy_usm_to_host():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")
    usm_obj = _create_memory(q)
    mv2 = memoryview(usm_obj)

    n = 9
    for id in range(n):
        mv2[id] = ord("a") + id

    host_obj = bytearray(b" " * n)

    q.memcpy(host_obj, usm_obj, n)

    assert host_obj == b"abcdefghi"


def test_memcpy_copy_host_to_host():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")

    src_buf = b"abcdefghijklmnopqrstuvwxyz"
    dst_buf = bytearray(len(src_buf))

    q.memcpy(dst_buf, src_buf, len(src_buf))

    assert dst_buf == src_buf


def test_memcpy_async():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")

    src_buf = b"abcdefghijklmnopqrstuvwxyz"
    n = len(src_buf)
    dst_buf = bytearray(n)
    dst_buf2 = bytearray(n)

    e = q.memcpy_async(dst_buf, src_buf, n)
    e2 = q.memcpy_async(dst_buf2, src_buf, n, [e])

    e.wait()
    e2.wait()
    assert dst_buf == src_buf
    assert dst_buf2 == src_buf


def test_memcpy_type_error():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")
    mobj = _create_memory(q)

    with pytest.raises(TypeError) as cm:
        q.memcpy(None, mobj, 3)
    assert "_Memory" in str(cm.value)

    with pytest.raises(TypeError) as cm:
        q.memcpy(mobj, None, 3)
    assert "_Memory" in str(cm.value)
