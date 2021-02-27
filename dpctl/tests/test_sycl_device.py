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

""" Defines unit test cases for the SyclDevice class.
"""

import dpctl
import pytest

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
    max_compute_units = device.get_max_compute_units()
    assert max_compute_units > 0


def check_get_max_work_item_dims(device):
    max_work_item_dims = device.get_max_work_item_dims()
    assert max_work_item_dims > 0


def check_get_max_work_item_sizes(device):
    max_work_item_sizes = device.get_max_work_item_sizes()
    for size in max_work_item_sizes:
        assert size is not None


def check_get_max_work_group_size(device):
    max_work_group_size = device.get_max_work_group_size()
    # Special case for FPGA simulator
    if device.is_accelerator():
        assert max_work_group_size >= 0
    else:
        assert max_work_group_size > 0


def check_get_max_num_sub_groups(device):
    max_num_sub_groups = device.get_max_num_sub_groups()
    # Special case for FPGA simulator
    if device.is_accelerator() or device.is_host():
        assert max_num_sub_groups >= 0
    else:
        assert max_num_sub_groups > 0


def check_has_int64_base_atomics(device):
    try:
        device.has_int64_base_atomics()
    except Exception:
        pytest.fail("has_int64_base_atomics call failed")


def check_has_int64_extended_atomics(device):
    try:
        device.has_int64_extended_atomics()
    except Exception:
        pytest.fail("has_int64_extended_atomics call failed")


def check_is_accelerator(device):
    try:
        device.is_accelerator()
    except Exception:
        pytest.fail("is_accelerator call failed")


def check_is_cpu(device):
    try:
        device.is_cpu()
    except Exception:
        pytest.fail("is_cpu call failed")


def check_is_gpu(device):
    try:
        device.is_gpu()
    except Exception:
        pytest.fail("is_gpu call failed")


def check_is_host(device):
    try:
        device.is_host()
    except Exception:
        pytest.fail("is_hostcall failed")


list_of_checks = [
    check_get_max_compute_units,
    check_get_max_work_item_dims,
    check_get_max_work_item_sizes,
    check_get_max_work_group_size,
    check_get_max_num_sub_groups,
    check_has_int64_base_atomics,
    check_has_int64_extended_atomics,
    check_is_accelerator,
    check_is_cpu,
    check_is_gpu,
    check_is_host,
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
    """Tests if the standard SYCL device_selectors are able to select a
    device.
    """
    try:
        device = device_selector()
        check(device)
    except ValueError:
        pytest.skip()


def test_current_device(check):
    """Test is the device for the current queue is valid."""
    try:
        q = dpctl.get_current_queue()
    except Exception:
        pytest.fail("Encountered an exception inside get_current_queue().")
    device = q.get_sycl_device()
    check(device)


def test_valid_filter_selectors(valid_filter, check):
    """Tests if we can create a SyclDevice using a supported filter selector string."""
    device = None
    try:
        device = dpctl.SyclDevice(valid_filter)
    except ValueError:
        pytest.skip("Failed to create device with supported filter")
    check(device)


def test_invalid_filter_selectors(invalid_filter):
    """An invalid filter string should always be caught and a ValueError
    raised.
    """
    with pytest.raises(ValueError):
        device = dpctl.SyclDevice(invalid_filter)
