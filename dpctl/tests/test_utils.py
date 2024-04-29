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
    assert exec_q is None
    q_c = dpctl.SyclQueue(q._get_capsule())
    assert q == q_c
    exec_q = dpctl.utils.get_execution_queue(
        (
            q,
            q_c,
            q,
        )
    )
    assert exec_q == q


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


def test_get_coerced_usm_type():
    _t = ["device", "shared", "host"]

    for i1 in range(len(_t)):
        for i2 in range(len(_t)):
            assert (
                dpctl.utils.get_coerced_usm_type([_t[i1], _t[i2]])
                == _t[min(i1, i2)]
            )

    assert dpctl.utils.get_coerced_usm_type([]) is None
    with pytest.raises(TypeError):
        dpctl.utils.get_coerced_usm_type(dict())


def validate_usm_type_arg():
    _t = ["device", "shared", "host"]

    for i in range(len(_t)):
        dpctl.utils.validate_usm_type(_t[i])
        dpctl.utils.validate_usm_type(_t[i], allow_none=False)
    dpctl.utils.validate_usm_type(None, allow_none=True)
    with pytest.raises(TypeError):
        dpctl.utils.validate_usm_type(dict(), allow_none=True)
    with pytest.raises(TypeError):
        dpctl.utils.validate_usm_type(dict(), allow_none=False)
    with pytest.raises(ValueError):
        dpctl.utils.validate_usm_type("inv", allow_none=True)
    with pytest.raises(ValueError):
        dpctl.utils.validate_usm_type("inv", allow_none=False)


@pytest.mark.filterwarnings("ignore:.*:RuntimeWarning")
def test_onetrace_enabled():
    import os

    v_name = "PTI_ENABLE_COLLECTION"
    v_v = os.getenv(v_name, None)
    with dpctl.utils.onetrace_enabled():
        assert os.getenv(v_name, None) == "1"
    assert os.getenv(v_name, None) == v_v


def test_intel_device_info():
    try:
        d = dpctl.select_default_device()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("Default device could not be created")
    descr = dpctl.utils.intel_device_info(d)
    assert isinstance(descr, dict)
    assert ("device_id" in descr) or not descr
    allowed_names = [
        "device_id",
        "gpu_slices",
        "gpu_eu_count",
        "gpu_eu_simd_width",
        "gpu_hw_threads_per_eu",
        "gpu_subslices_per_slice",
        "gpu_eu_count_per_subslice",
        "max_mem_bandwidth",
        "free_memory",
        "memory_clock_rate",
        "memory_bus_width",
    ]
    for descriptor_name in descr.keys():
        test = descriptor_name in allowed_names
        err_msg = f"Key '{descriptor_name}' is not recognized"
        assert test, err_msg
