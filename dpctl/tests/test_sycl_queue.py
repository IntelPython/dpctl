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

""" Defines unit test cases for the SyclQueue class.
"""

import pytest

import dpctl

list_of_standard_selectors = [
    dpctl.select_accelerator_device,
    dpctl.select_cpu_device,
    dpctl.select_default_device,
    dpctl.select_gpu_device,
    dpctl.select_host_device,
]

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

list_of_invalid_filter_selectors = [
    "-1",
    "opencl:gpu:-1",
    "level_zero:cpu:0",
    "abc",
]


# Unit test cases that will be run for every device
def check_get_max_compute_units(device):
    max_compute_units = device.max_compute_units
    assert max_compute_units > 0


def check_get_max_work_item_dims(device):
    max_work_item_dims = device.max_work_item_dims
    assert max_work_item_dims > 0


def check_get_max_work_item_sizes(device):
    max_work_item_sizes = device.max_work_item_sizes
    for size in max_work_item_sizes:
        assert size is not None


def check_get_max_work_group_size(device):
    max_work_group_size = device.max_work_group_size
    # Special case for FPGA simulator
    if device.is_accelerator:
        assert max_work_group_size >= 0
    else:
        assert max_work_group_size > 0


def check_get_max_num_sub_groups(device):
    max_num_sub_groups = device.max_num_sub_groups
    # Special case for FPGA simulator
    if device.is_accelerator or device.is_host:
        assert max_num_sub_groups >= 0
    else:
        assert max_num_sub_groups > 0


def check_has_aspect_host(device):
    try:
        device.has_aspect_host
    except Exception:
        pytest.fail("has_aspect_host call failed")


def check_has_aspect_cpu(device):
    try:
        device.has_aspect_cpu
    except Exception:
        pytest.fail("has_aspect_cpu call failed")


def check_has_aspect_gpu(device):
    try:
        device.has_aspect_gpu
    except Exception:
        pytest.fail("has_aspect_gpu call failed")


def check_has_aspect_accelerator(device):
    try:
        device.has_aspect_accelerator
    except Exception:
        pytest.fail("has_aspect_accelerator call failed")


def check_has_aspect_custom(device):
    try:
        device.has_aspect_custom
    except Exception:
        pytest.fail("has_aspect_custom call failed")


def check_has_aspect_fp16(device):
    try:
        device.has_aspect_fp16
    except Exception:
        pytest.fail("has_aspect_fp16 call failed")


def check_has_aspect_fp64(device):
    try:
        device.has_aspect_fp64
    except Exception:
        pytest.fail("has_aspect_fp64 call failed")


def check_has_aspect_int64_base_atomics(device):
    try:
        device.has_aspect_int64_base_atomics
    except Exception:
        pytest.fail("has_aspect_int64_base_atomics call failed")


def check_has_aspect_int64_extended_atomics(device):
    try:
        device.has_aspect_int64_extended_atomics
    except Exception:
        pytest.fail("has_aspect_int64_extended_atomics call failed")


def check_has_aspect_image(device):
    try:
        device.has_aspect_image
    except Exception:
        pytest.fail("has_aspect_image call failed")


def check_has_aspect_online_compiler(device):
    try:
        device.has_aspect_online_compiler
    except Exception:
        pytest.fail("has_aspect_online_compiler call failed")


def check_has_aspect_online_linker(device):
    try:
        device.has_aspect_online_linker
    except Exception:
        pytest.fail("has_aspect_online_linker call failed")


def check_has_aspect_queue_profiling(device):
    try:
        device.has_aspect_queue_profiling
    except Exception:
        pytest.fail("has_aspect_queue_profiling call failed")


def check_has_aspect_usm_device_allocations(device):
    try:
        device.has_aspect_usm_device_allocations
    except Exception:
        pytest.fail("has_aspect_usm_device_allocations call failed")


def check_has_aspect_usm_host_allocations(device):
    try:
        device.has_aspect_usm_host_allocations
    except Exception:
        pytest.fail("has_aspect_usm_host_allocations call failed")


def check_has_aspect_usm_shared_allocations(device):
    try:
        device.has_aspect_usm_shared_allocations
    except Exception:
        pytest.fail("has_aspect_usm_shared_allocations call failed")


def check_has_aspect_usm_restricted_shared_allocations(device):
    try:
        device.has_aspect_usm_restricted_shared_allocations
    except Exception:
        pytest.fail("has_aspect_usm_restricted_shared_allocations call failed")


def check_has_aspect_usm_system_allocator(device):
    try:
        device.has_aspect_usm_system_allocator
    except Exception:
        pytest.fail("has_aspect_usm_system_allocator call failed")


def check_is_accelerator(device):
    try:
        device.is_accelerator
    except Exception:
        pytest.fail("is_accelerator call failed")


def check_is_cpu(device):
    try:
        device.is_cpu
    except Exception:
        pytest.fail("is_cpu call failed")


def check_is_gpu(device):
    try:
        device.is_gpu
    except Exception:
        pytest.fail("is_gpu call failed")


def check_is_host(device):
    try:
        device.is_host
    except Exception:
        pytest.fail("is_hostcall failed")


list_of_checks = [
    check_get_max_compute_units,
    check_get_max_work_item_dims,
    check_get_max_work_item_sizes,
    check_get_max_work_group_size,
    check_get_max_num_sub_groups,
    check_is_accelerator,
    check_is_cpu,
    check_is_gpu,
    check_is_host,
    check_has_aspect_host,
    check_has_aspect_cpu,
    check_has_aspect_gpu,
    check_has_aspect_accelerator,
    check_has_aspect_custom,
    check_has_aspect_fp16,
    check_has_aspect_fp64,
    check_has_aspect_int64_base_atomics,
    check_has_aspect_int64_extended_atomics,
    check_has_aspect_image,
    check_has_aspect_online_compiler,
    check_has_aspect_online_linker,
    check_has_aspect_queue_profiling,
    check_has_aspect_usm_device_allocations,
    check_has_aspect_usm_host_allocations,
    check_has_aspect_usm_shared_allocations,
    check_has_aspect_usm_restricted_shared_allocations,
    check_has_aspect_usm_system_allocator,
]


@pytest.fixture(params=list_of_valid_filter_selectors)
def valid_filter(request):
    return request.param


@pytest.fixture(params=list_of_invalid_filter_selectors)
def invalid_filter(request):
    return request.param


@pytest.fixture(params=list_of_standard_selectors)
def device_selector(request):
    return request.param


@pytest.fixture(params=list_of_checks)
def check(request):
    return request.param


def test_standard_selectors(device_selector, check):
    """
    Tests if the standard SYCL device_selectors are able to select a device.
    """
    try:
        device = device_selector()
        if device.default_selector_score < 0:
            pytest.skip()
        q = dpctl.SyclQueue(device)
        check(q.get_sycl_device())
    except ValueError:
        pytest.skip()


def test_current_device(check):
    """
    Test is the device for the current queue is valid.
    """
    try:
        q = dpctl.get_current_queue()
    except Exception:
        pytest.fail("Encountered an exception inside get_current_queue().")
    device = q.get_sycl_device()
    check(device)


def test_valid_filter_selectors(valid_filter, check):
    """
    Tests if we can create a SyclDevice using a supported filter selector
    string.
    """
    device = None
    try:
        q = dpctl.SyclQueue(valid_filter)
        device = q.get_sycl_device()
        assert q.is_in_order is False
        q2 = dpctl.SyclQueue(valid_filter, property="in_order")
        # assert device == q2.get_sycl_device()
        assert q2.is_in_order is True
    except dpctl.SyclQueueCreationError:
        pytest.skip("Failed to create device with supported filter")
    check(device)


def test_invalid_filter_selectors(invalid_filter):
    """
    An invalid filter string should always be caught and a
    SyclQueueCreationError raised.
    """
    with pytest.raises(dpctl.SyclQueueCreationError):
        dpctl.SyclQueue(invalid_filter)


def test_context_not_equals():
    try:
        gpuQ = dpctl.SyclQueue("gpu")
    except dpctl.SyclQueueCreationError:
        pytest.skip()
    ctx_gpu = gpuQ.get_sycl_context()
    try:
        cpuQ = dpctl.SyclQueue("cpu")
    except dpctl.SyclQueueCreationError:
        pytest.skip()
    ctx_cpu = cpuQ.get_sycl_context()
    assert ctx_cpu != ctx_gpu
    assert hash(ctx_cpu) != hash(ctx_gpu)


def test_context_equals():
    try:
        gpuQ1 = dpctl.SyclQueue("gpu")
        gpuQ0 = dpctl.SyclQueue("gpu")
    except dpctl.SyclQueueCreationError:
        pytest.skip()
    ctx0 = gpuQ0.get_sycl_context()
    ctx1 = gpuQ1.get_sycl_context()
    assert ctx0 == ctx1
    assert hash(ctx0) == hash(ctx1)


def test_has_enable_profiling():
    try:
        q = dpctl.SyclQueue(property="enable_profiling")
    except dpctl.SyclQueueCreationError:
        pytest.skip()
    assert q.has_enable_profiling


def test_hashing_of_queue():
    """
    Test that a :class:`dpctl.SyclQueue` object can be used as
    a dictionary key.

    """
    queue_dict = {dpctl.SyclQueue(): "default_queue"}
    assert queue_dict


def test_channeling_device_properties():
    try:
        q = dpctl.SyclQueue()
        dev = q.sycl_device
    except dpctl.SyclQueueCreationError:
        pytest.fail("Failed to create device from default selector")
    import io
    from contextlib import redirect_stdout

    f1 = io.StringIO()
    with redirect_stdout(f1):
        q.print_device_info()  # should execute without raising
    f2 = io.StringIO()
    with redirect_stdout(f2):
        dev.print_device_info()
    assert f1.getvalue() == f2.getvalue(), "Mismatch in print_device_info"
    for pr in ["backend", "name", "driver_version"]:
        assert getattr(q, pr) == getattr(
            dev, pr
        ), "Mismatch found for property {}".format(pr)


def test_queue_submit_barrier(valid_filter):
    try:
        q = dpctl.SyclQueue(valid_filter)
    except dpctl.SyclQueueCreationError:
        pytest.skip("Failed to create device with supported filter")
    ev1 = q.submit_barrier()
    ev2 = q.submit_barrier()
    ev3 = q.submit_barrier([ev1, ev2])
    ev3.wait()
    ev1.wait()
    ev2.wait()
    with pytest.raises(TypeError):
        q.submit_barrier(range(3))


def test_queue__repr__():
    q1 = dpctl.SyclQueue()
    r1 = q1.__repr__()
    q2 = dpctl.SyclQueue(property="in_order")
    r2 = q2.__repr__()
    q3 = dpctl.SyclQueue(property="enable_profiling")
    r3 = q3.__repr__()
    q4 = dpctl.SyclQueue(property="default")
    r4 = q4.__repr__()
    q5 = dpctl.SyclQueue(property=["in_order", "enable_profiling"])
    r5 = q5.__repr__()
    assert type(r1) is str
    assert type(r2) is str
    assert type(r3) is str
    assert type(r4) is str
    assert type(r5) is str


def test_queue_invalid_property():
    with pytest.raises(ValueError):
        dpctl.SyclQueue(property=4.5)
    with pytest.raises(ValueError):
        dpctl.SyclQueue(property=["abc", tuple()])


def test_queue_capsule():
    q = dpctl.SyclQueue()
    cap = q._get_capsule()
    cap2 = q._get_capsule()
    q2 = dpctl.SyclQueue(cap)
    assert q == q2
    del cap2  # call deleter on non-renamed capsule


def test_cpython_api():
    import ctypes
    import sys

    q = dpctl.SyclQueue()
    mod = sys.modules[q.__class__.__module__]
    # get capsule storign get_context_ref function ptr
    q_ref_fn_cap = mod.__pyx_capi__["get_queue_ref"]
    # construct Python callable to invoke "get_queue_ref"
    cap_ptr_fn = ctypes.pythonapi.PyCapsule_GetPointer
    cap_ptr_fn.restype = ctypes.c_void_p
    cap_ptr_fn.argtypes = [ctypes.py_object, ctypes.c_char_p]
    q_ref_fn_ptr = cap_ptr_fn(
        q_ref_fn_cap, b"DPCTLSyclQueueRef (struct PySyclQueueObject *)"
    )
    callable_maker = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object)
    get_queue_ref_fn = callable_maker(q_ref_fn_ptr)

    r2 = q.addressof_ref()
    r1 = get_queue_ref_fn(q)
    assert r1 == r2


def test_constructor_many_arg():
    with pytest.raises(TypeError):
        dpctl.SyclQueue(None, None, None, None)
    with pytest.raises(TypeError):
        dpctl.SyclQueue(None, None)
    ctx = dpctl.SyclContext()
    with pytest.raises(TypeError):
        dpctl.SyclQueue(ctx, None)
    with pytest.raises(TypeError):
        dpctl.SyclQueue(ctx)


def test_queue_wait():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Failed to create device with supported filter")
    q.wait()


def test_queue_memops():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Failed to create device with supported filter")
    from dpctl.memory import MemoryUSMDevice

    m1 = MemoryUSMDevice(512, queue=q)
    m2 = MemoryUSMDevice(512, queue=q)
    q.memcpy(m1, m2, 512)
    q.prefetch(m1, 512)
    q.mem_advise(m1, 512, 0)
    with pytest.raises(TypeError):
        q.memcpy(m1, list(), 512)
    with pytest.raises(TypeError):
        q.memcpy(list(), m2, 512)
    with pytest.raises(TypeError):
        q.prefetch(list(), 512)
    with pytest.raises(TypeError):
        q.mem_advise(list(), 512, 0)
