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

""" Defines unit test cases for the SyclContxt class.
"""

import pytest

import dpctl

list_of_valid_filter_selectors = [
    "opencl",
    "opencl:gpu",
    "opencl:cpu",
    "opencl:gpu:0",
    "gpu",
    "cpu",
    "level_zero",
    "level_zero:gpu",
    "opencl:cpu:0",
    "level_zero:gpu:0",
    "gpu:0",
    "gpu:1",
    "1",
]


@pytest.fixture(params=list_of_valid_filter_selectors)
def valid_filter(request):
    return request.param


def test_ctxt_creation_from_filter(valid_filter):
    """
    Test SyclContext creation using a filter selector string.
    """
    try:
        dpctl.SyclContext(valid_filter)
    except ValueError:
        pytest.skip("Failed to create context with supported filter")


def test_address_of():
    """
    Test if the address_of method returns an int value.
    """
    ctx = dpctl.SyclContext()
    assert ctx.addressof_ref() is not None
    assert isinstance(ctx.addressof_ref(), int)


def test_context_not_equals():
    """
    Test if context equality fails when devices in different contexts are
    compared.
    """
    try:
        ctx_gpu = dpctl.SyclContext("gpu")
    except ValueError:
        pytest.skip()
    try:
        ctx_cpu = dpctl.SyclContext("cpu")
    except ValueError:
        pytest.skip()
    assert ctx_cpu != ctx_gpu
    assert hash(ctx_cpu) != hash(ctx_gpu)


def test_context_not_equals2():
    """
    Test if comparing a SyclContext object to some random Python object is
    correctly handled and returns False.
    """
    ctx = dpctl.SyclContext()
    assert ctx != "some context"


def test_context_equals():
    try:
        ctx1 = dpctl.SyclContext("gpu")
        ctx0 = dpctl.SyclContext("gpu")
    except ValueError:
        pytest.skip()
    assert ctx0 == ctx1
    assert hash(ctx0) == hash(ctx1)


def test_name():
    """
    Test if a __name__ method is defined for SyclContext.
    """
    ctx = dpctl.SyclContext()
    assert ctx.__name__ == "SyclContext"


def test_repr():
    """
    Test if a __repr__ method is defined for SyclContext.
    """
    ctx = dpctl.SyclContext()
    assert ctx.__repr__ is not None


def test_context_can_be_used_in_queue(valid_filter):
    try:
        ctx = dpctl.SyclContext(valid_filter)
    except ValueError:
        pytest.skip()
    devs = ctx.get_devices()
    assert len(devs) == ctx.device_count
    for d in devs:
        dpctl.SyclQueue(ctx, d)


def test_context_can_be_used_in_queue2(valid_filter):
    try:
        d = dpctl.SyclDevice(valid_filter)
    except ValueError:
        pytest.skip()
    if d.default_selector_score < 0:
        # skip test for devices rejected by default selector
        pytest.skip()
    ctx = dpctl.SyclContext(d)
    dpctl.SyclQueue(ctx, d)


def test_context_multi_device():
    try:
        d = dpctl.SyclDevice("cpu")
    except ValueError:
        pytest.skip()
    if d.default_selector_score < 0:
        pytest.skip()
    n = d.max_compute_units
    n1 = n // 2
    n2 = n - n1
    if n1 == 0 or n2 == 0:
        pytest.skip()
    d1, d2 = d.create_sub_devices(partition=(n1, n2))
    ctx = dpctl.SyclContext((d1, d2))
    assert ctx.device_count == 2
    q1 = dpctl.SyclQueue(ctx, d1)
    q2 = dpctl.SyclQueue(ctx, d2)
    import dpctl.memory as dpmem

    shmem_1 = dpmem.MemoryUSMShared(256, queue=q1)
    shmem_2 = dpmem.MemoryUSMDevice(256, queue=q2)
    shmem_2.copy_from_device(shmem_1)
    # create context for single sub-device
    ctx1 = dpctl.SyclContext(d1)
    q1 = dpctl.SyclQueue(ctx1, d1)
    shmem_1 = dpmem.MemoryUSMShared(256, queue=q1)
    cap = ctx1._get_capsule()
    del ctx1
    ctx2 = dpctl.SyclContext(cap)
    q2 = dpctl.SyclQueue(ctx2, d1)
    shmem_2 = dpmem.MemoryUSMDevice(256, queue=q2)
    shmem_2.copy_from_device(shmem_1)


def test_hashing_of_context():
    """
    Test that a :class:`dpctl.SyclContext` object can be used
    as a dictionary key.

    """
    ctx_dict = {dpctl.SyclContext(): "default_context"}
    assert ctx_dict


def test_context_repr():
    ctx = dpctl.SyclContext()
    assert type(ctx.__repr__()) is str
