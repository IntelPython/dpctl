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

import pytest

import dpctl
import dpctl.tensor as dpt


@pytest.fixture
def profiling_queue():
    try:
        q = dpctl.SyclQueue(property="enable_profiling")
    except dpctl.SyclQueueCreationError:
        pytest.skip(
            "Could not created profiling queue " "for default-selected device"
        )
    return q


@pytest.mark.parametrize(
    "device_timer", [None, "queue_barrier", "order_manager"]
)
def test_sycl_timer_queue_barrier(profiling_queue, device_timer):
    dev = dpt.Device.create_device(profiling_queue)

    timer = dpctl.SyclTimer(
        host_timer=time.perf_counter, device_timer=device_timer, time_scale=1e3
    )

    with timer(dev.sycl_queue):
        x = dpt.linspace(0, 1, num=10**6, device=dev)
        y = 3.0 - dpt.square(x - 0.5)
        z = dpt.sort(y)
        res1 = z[-1]
        res2 = dpt.max(y)

    host_dt, device_dt = timer.dt

    assert dpt.all(res1 == res2)
    assert host_dt > 0
    assert device_dt > 0


def test_sycl_timer_accumulation(profiling_queue):
    q = profiling_queue

    timer = dpctl.SyclTimer(
        host_timer=time.perf_counter,
        device_timer="order_manager",
        time_scale=1e3,
    )

    # initial condition
    x = dpt.linspace(0, 1, num=10**6, sycl_queue=q)

    aitkens_data = [
        x,
    ]

    # 16 iterations of Aitken's accelerated Newton's method
    # x <- x - f(x)/f'(x) for f(x) = x - cos(x)
    for _ in range(16):
        # only time Newton step
        with timer(q):
            s = dpt.sin(x)
            x = (dpt.cos(x) + x * s) / (1 + s)
        aitkens_data.append(x)
        aitkens_data = aitkens_data[-3:]
        if len(aitkens_data) == 3:
            # apply Aitkens acceleration
            d1 = aitkens_data[-1] - aitkens_data[-2]
            d2 = aitkens_data[-2] - aitkens_data[-3]
            if not dpt.any(d1 == d2):
                x = aitkens_data[-1] - dpt.square(d1) / (d1 - d2)

    # Total time for 16 iterations
    dev_dt = timer.dt.device_dt
    assert dev_dt > 0

    # check convergence
    assert dpt.max(x) - dpt.min(x) < 1e-5


def test_sycl_timer_validation():
    with pytest.raises(ValueError):
        dpctl.SyclTimer(device_timer="invalid")

    timer = dpctl.SyclTimer()
    mock_queue = Ellipsis

    with pytest.raises(TypeError):
        timer(mock_queue)
