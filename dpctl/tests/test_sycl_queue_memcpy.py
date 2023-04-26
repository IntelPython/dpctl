#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2022 Intel Corporation
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

    assert mv2[:3], b"123"


def test_memcpy_type_error():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default constructor for SyclQueue failed")
    mobj = _create_memory(q)

    with pytest.raises(TypeError) as cm:
        q.memcpy(None, mobj, 3)
    assert "`dest`" in str(cm.value)

    with pytest.raises(TypeError) as cm:
        q.memcpy(mobj, None, 3)
    assert "`src`" in str(cm.value)
