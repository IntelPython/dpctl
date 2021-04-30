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
        ctx = dpctl.SyclContext(device)
        devs = ctx.get_devices()
        assert len(devs) == 1
        check(devs[0])
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
    ctx = q.get_sycl_context()
    devs = ctx.get_devices()
    # add check that device is among devs
    check(devs[0])


def test_valid_filter_selectors(valid_filter, check):
    """
    Tests if we can create a SyclDevice using a supported filter selector
    string.
    """
    device = None
    try:
        ctx = dpctl.SyclContext(valid_filter)
        device = ctx.get_devices()
    except ValueError:
        pytest.skip("Failed to create context with supported filter")
    check(device[0])


def test_invalid_filter_selectors(invalid_filter):
    """
    An invalid filter string should always be caught and a
    SyclQueueCreationError raised.
    """
    with pytest.raises(ValueError):
        dpctl.SyclContext(invalid_filter)


def test_context_not_equals():
    try:
        ctx_gpu = dpctl.SyclContext("gpu")
    except ValueError:
        pytest.skip()
    try:
        ctx_cpu = dpctl.SyclContext("cpu")
    except ValueError:
        pytest.skip()
    assert ctx_cpu != ctx_gpu


def test_context_equals():
    try:
        ctx1 = dpctl.SyclContext("gpu")
        ctx0 = dpctl.SyclContext("gpu")
    except ValueError:
        pytest.skip()
    assert ctx0 == ctx1


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
