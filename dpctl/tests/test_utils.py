#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2021 Intel Corporation
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

""" Defines unit test cases for utility functions.
"""

import pytest

import dpctl
import dpctl.utils


def test_get_execution_queue_input_validation():
    with pytest.raises(TypeError):
        dpctl.utils.get_execution_queue(dict())


def test_get_execution_queue():
    try:
        q = dpctl.SyclQueue()
        q2 = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be create for default device")

    exec_q = dpctl.utils.get_execution_queue(())
    assert exec_q is None

    exec_q = dpctl.utils.get_execution_queue([q])
    assert exec_q is q

    exec_q = dpctl.utils.get_execution_queue([q, q, q, q])
    assert exec_q is q

    exec_q = dpctl.utils.get_execution_queue((q, q, None, q))
    assert exec_q is None

    exec_q = dpctl.utils.get_execution_queue(
        (
            q,
            q2,
            q,
        )
    )
    assert exec_q is q


def test_get_execution_queue_nonequiv():
    try:
        q = dpctl.SyclQueue("cpu")
        d1, d2 = q.sycl_device.create_sub_devices(partition=[1, 1])
        ctx = dpctl.SyclContext([q.sycl_device, d1, d2])
        q1 = dpctl.SyclQueue(ctx, d1)
        q2 = dpctl.SyclQueue(ctx, d2)
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be create for default device")

    exec_q = dpctl.utils.get_execution_queue((q, q1, q2))
    assert exec_q is None
