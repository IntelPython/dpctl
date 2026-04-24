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

import pytest

import dpctl
import dpctl.memory


def _create_memory(q):
    nbytes = 1024
    mobj = dpctl.memory.MemoryUSMShared(nbytes, queue=q)
    return mobj


def test_copy_copy_host_to_host():
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
