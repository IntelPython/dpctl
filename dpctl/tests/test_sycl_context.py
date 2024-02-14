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

""" Defines unit test cases for the :class:`dpctl.SyclContext` class.
"""

import pytest
from helper import create_invalid_capsule

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
    except dpctl.SyclContextCreationError:
        pytest.skip("Failed to create context with supported filter")


def test_address_of():
    """
    Test if the address_of method returns an int value.
    """
    try:
        ctx = dpctl.SyclContext()
    except dpctl.SyclContextCreationError:
        pytest.skip("Failed to create context using default constructor")
    assert ctx.addressof_ref() is not None
    assert isinstance(ctx.addressof_ref(), int)


def test_context_not_equals():
    """
    Test if context equality fails when devices in different contexts are
    compared.
    """
    try:
        ctx_gpu = dpctl.SyclContext("gpu")
    except dpctl.SyclContextCreationError:
        pytest.skip()
    try:
        ctx_cpu = dpctl.SyclContext("cpu")
    except dpctl.SyclContextCreationError:
        pytest.skip()
    assert ctx_cpu != ctx_gpu
    assert hash(ctx_cpu) != hash(ctx_gpu)


def test_context_not_equals2():
    """
    Test if comparing a SyclContext object to some random Python object is
    correctly handled and returns False.
    """
    try:
        ctx = dpctl.SyclContext()
    except dpctl.SyclContextCreationError:
        pytest.skip("Failed to create context using default constructor")
    assert ctx != "some context"


def test_context_equals():
    try:
        ctx1 = dpctl.SyclContext("gpu")
        ctx0 = dpctl.SyclContext("gpu")
    except dpctl.SyclContextCreationError:
        pytest.skip()
    assert ctx0 == ctx1
    assert hash(ctx0) == hash(ctx1)


def test_name():
    """
    Test if a __name__ method is defined for SyclContext.
    """
    try:
        ctx = dpctl.SyclContext()
    except dpctl.SyclContextCreationError:
        pytest.skip("Failed to create context using default constructor")
    assert ctx.__name__ == "SyclContext"


def test_repr():
    """
    Test if a __repr__ method is defined for SyclContext.
    """
    try:
        ctx = dpctl.SyclContext()
    except dpctl.SyclContextCreationError:
        pytest.skip("Failed to create context using default constructor")
    assert ctx.__repr__ is not None


def test_context_can_be_used_in_queue(valid_filter):
    try:
        ctx = dpctl.SyclContext(valid_filter)
    except dpctl.SyclContextCreationError:
        pytest.skip()
    devs = ctx.get_devices()
    assert len(devs) == ctx.device_count
    for d in devs:
        dpctl.SyclQueue(ctx, d)


def test_context_can_be_used_in_queue2(valid_filter):
    try:
        d = dpctl.SyclDevice(valid_filter)
    except dpctl.SyclDeviceCreationError:
        pytest.skip()
    if d.default_selector_score < 0:
        # skip test for devices rejected by default selector
        pytest.skip()
    ctx = dpctl.SyclContext(d)
    dpctl.SyclQueue(ctx, d)


def test_context_multi_device():
    try:
        d = dpctl.SyclDevice("cpu")
    except dpctl.SyclDeviceCreationError:
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
    assert type(repr(ctx)) is str
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
    cap2 = ctx1._get_capsule()
    del ctx1
    del cap2  # exercise deleter of non-renamed capsule
    ctx2 = dpctl.SyclContext(cap)
    q2 = dpctl.SyclQueue(ctx2, d1)
    shmem_2 = dpmem.MemoryUSMDevice(256, queue=q2)
    shmem_2.copy_from_device(shmem_1)


def test_hashing_of_context():
    """
    Test that a :class:`dpctl.SyclContext` object can be used
    as a dictionary key.

    """
    try:
        ctx = dpctl.SyclContext()
    except dpctl.SyclContextCreationError:
        pytest.skip("Failed to create context using default constructor")
    ctx_dict = {ctx: "default_context"}
    assert ctx_dict


def test_context_repr():
    try:
        ctx = dpctl.SyclContext()
    except dpctl.SyclContextCreationError:
        pytest.skip("Failed to create context using default constructor")
    assert type(ctx.__repr__()) is str


def test_cpython_api_SyclContext_GetContextRef():
    import ctypes
    import sys

    try:
        ctx = dpctl.SyclContext()
    except dpctl.SyclContextCreationError:
        pytest.skip("Failed to create context using default constructor")
    mod = sys.modules[ctx.__class__.__module__]
    # get capsule storign SyclContext_GetContextRef function ptr
    ctx_ref_fn_cap = mod.__pyx_capi__["SyclContext_GetContextRef"]
    # construct Python callable to invoke "SyclContext_GetContextRef"
    cap_ptr_fn = ctypes.pythonapi.PyCapsule_GetPointer
    cap_ptr_fn.restype = ctypes.c_void_p
    cap_ptr_fn.argtypes = [ctypes.py_object, ctypes.c_char_p]
    ctx_ref_fn_ptr = cap_ptr_fn(
        ctx_ref_fn_cap, b"DPCTLSyclContextRef (struct PySyclContextObject *)"
    )
    callable_maker = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object)
    get_context_ref_fn = callable_maker(ctx_ref_fn_ptr)

    r2 = ctx.addressof_ref()
    r1 = get_context_ref_fn(ctx)
    assert r1 == r2


def test_cpython_api_SyclContext_Make():
    import ctypes
    import sys

    try:
        ctx = dpctl.SyclContext()
    except dpctl.SyclContextCreationError:
        pytest.skip("Failed to create context using default constructor")
    mod = sys.modules[ctx.__class__.__module__]
    # get capsule storign SyclContext_Make function ptr
    make_ctx_fn_cap = mod.__pyx_capi__["SyclContext_Make"]
    # construct Python callable to invoke "SyclContext_Make"
    cap_ptr_fn = ctypes.pythonapi.PyCapsule_GetPointer
    cap_ptr_fn.restype = ctypes.c_void_p
    cap_ptr_fn.argtypes = [ctypes.py_object, ctypes.c_char_p]
    make_ctx_fn_ptr = cap_ptr_fn(
        make_ctx_fn_cap, b"struct PySyclContextObject *(DPCTLSyclContextRef)"
    )
    callable_maker = ctypes.PYFUNCTYPE(ctypes.py_object, ctypes.c_void_p)
    make_ctx_fn = callable_maker(make_ctx_fn_ptr)

    ctx2 = make_ctx_fn(ctx.addressof_ref())
    assert ctx == ctx2


def test_invalid_capsule():
    cap = create_invalid_capsule()
    with pytest.raises(ValueError):
        dpctl.SyclContext(cap)


def test_multi_device_different_platforms():
    devs = dpctl.get_devices()  # all devices
    if len(devs) > 1 and len(set(d.sycl_platform for d in devs)) > 1:
        with pytest.raises(dpctl.SyclContextCreationError):
            dpctl.SyclContext(devs)
    else:
        pytest.skip("Insufficient amount of available devices for this test")
