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

import time

import numpy as np
import pytest

import dpctl
from dpctl.utils import SequentialOrderManager


@pytest.fixture
def profiling_queue():
    try:
        q = dpctl.SyclQueue(property="enable_profiling")
    except dpctl.SyclQueueCreationError:
        pytest.skip(
            "Could not created profiling queue " "for default-selected device"
        )
    return q


@pytest.mark.parametrize("device_timer", [None, "queue_barrier"])
def test_sycl_timer_queue_barrier(profiling_queue, device_timer):
    timer = dpctl.SyclTimer(
        host_timer=time.perf_counter, device_timer=device_timer, time_scale=1e3
    )
    x = np.linspace(0, 1, num=10**6)
    res = np.empty_like(x)

    with timer(profiling_queue):
        # round-trip through USM device memory into new NumPy array
        x_usm = dpctl.memory.MemoryUSMDevice(x.nbytes, queue=profiling_queue)
        e1 = profiling_queue.memcpy_async(
            dest=x_usm,
            src=x,
            count=x.nbytes,
        )
        e2 = profiling_queue.memcpy_async(
            dest=res,
            src=x_usm,
            count=res.nbytes,
            dEvents=[e1],
        )

    e2.wait()
    host_dt, device_dt = timer.dt

    assert np.all(res == x)
    assert host_dt > 0
    assert device_dt > 0


def test_sycl_timer_order_manager(profiling_queue):
    q = profiling_queue
    timer = dpctl.SyclTimer(
        host_timer=time.perf_counter,
        device_timer="order_manager",
        time_scale=1e3,
    )

    om = SequentialOrderManager[q]

    x = np.linspace(0, 1, num=10**6)
    res = np.empty_like(x)

    with timer(q):
        x_usm = dpctl.memory.MemoryUSMDevice(x.nbytes, queue=q)
        e1 = q.memcpy_async(
            dest=x_usm,
            src=x,
            count=x.nbytes,
            dEvents=om.submitted_events,
        )
        ht1 = q._submit_keep_args_alive((x_usm, x), [e1])
        om.add_event_pair(ht1, e1)
        e2 = q.memcpy_async(
            dest=res,
            src=x_usm,
            count=res.nbytes,
            dEvents=om.submitted_events,
        )
        ht2 = q._submit_keep_args_alive((res, x_usm), [e2])
        om.add_event_pair(ht2, e2)

    e2.wait()
    ht2.wait()
    host_dt, device_dt = timer.dt

    assert np.all(res == x)
    assert host_dt > 0
    assert device_dt > 0


def test_sycl_timer_accumulation(profiling_queue):
    q = profiling_queue

    timer = dpctl.SyclTimer(
        host_timer=time.perf_counter,
        device_timer="order_manager",
        time_scale=1e3,
    )

    om = SequentialOrderManager[q]

    x = np.linspace(0, 1, num=10**6)
    res = np.empty_like(x)
    x_usm = dpctl.memory.MemoryUSMDevice(x.nbytes, queue=q)

    # repeat round-trip several times to exercise timer accumulation
    for _ in range(8):
        with timer(q):
            depends = om.submitted_events
            e1 = q.memcpy_async(
                dest=x_usm,
                src=x,
                count=x.nbytes,
                dEvents=depends,
            )
            ht1 = q._submit_keep_args_alive((x_usm, x), [e1])
            om.add_event_pair(ht1, e1)
            e2 = q.memcpy_async(
                dest=res,
                src=x_usm,
                count=res.nbytes,
                dEvents=[e1],
            )
            ht2 = q._submit_keep_args_alive((res, x_usm), [e2])
            om.add_event_pair(ht2, e2)
    e2.wait()
    ht2.wait()
    assert np.all(res == x)

    dev_dt = timer.dt.device_dt
    assert dev_dt > 0


def test_sycl_timer_validation():
    with pytest.raises(ValueError):
        dpctl.SyclTimer(device_timer="invalid")

    timer = dpctl.SyclTimer()
    mock_queue = Ellipsis

    with pytest.raises(TypeError):
        timer(mock_queue)
