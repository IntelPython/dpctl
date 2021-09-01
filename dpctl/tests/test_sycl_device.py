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

import pytest

import dpctl
from dpctl._sycl_device import SubDeviceCreationError

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
    "cuda:cpu:0",
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


def check_get_max_read_image_args(device):
    try:
        device.max_read_image_args
    except Exception:
        pytest.fail("max_read_image_args call failed")


def check_get_max_write_image_args(device):
    try:
        device.max_write_image_args
    except Exception:
        pytest.fail("max_write_image_args call failed")


def check_get_image_2d_max_width(device):
    try:
        device.image_2d_max_width
    except Exception:
        pytest.fail("image_2d_max_width call failed")


def check_get_image_2d_max_height(device):
    try:
        device.image_2d_max_height
    except Exception:
        pytest.fail("image_2d_max_height call failed")


def check_get_image_3d_max_width(device):
    try:
        device.image_3d_max_width
    except Exception:
        pytest.fail("image_3d_max_width call failed")


def check_get_image_3d_max_height(device):
    try:
        device.image_3d_max_height
    except Exception:
        pytest.fail("image_3d_max_height call failed")


def check_get_image_3d_max_depth(device):
    try:
        device.image_3d_max_depth
    except Exception:
        pytest.fail("image_3d_max_depth call failed")


def check_get_sub_group_independent_forward_progress(device):
    try:
        device.sub_group_independent_forward_progress
    except Exception:
        pytest.fail("sub_group_independent_forward_progress call failed")


def check_get_preferred_vector_width_char(device):
    try:
        device.preferred_vector_width_char
    except Exception:
        pytest.fail("preferred_vector_width_char call failed")


def check_get_preferred_vector_width_short(device):
    try:
        device.preferred_vector_width_short
    except Exception:
        pytest.fail("preferred_vector_width_short call failed")


def check_get_preferred_vector_width_int(device):
    try:
        device.preferred_vector_width_int
    except Exception:
        pytest.fail("preferred_vector_width_int call failed")


def check_get_preferred_vector_width_long(device):
    try:
        device.preferred_vector_width_long
    except Exception:
        pytest.fail("preferred_vector_width_long call failed")


def check_get_preferred_vector_width_float(device):
    try:
        device.preferred_vector_width_float
    except Exception:
        pytest.fail("preferred_vector_width_float call failed")


def check_get_preferred_vector_width_double(device):
    try:
        device.preferred_vector_width_double
    except Exception:
        pytest.fail("preferred_vector_width_double call failed")


def check_get_preferred_vector_width_half(device):
    try:
        device.preferred_vector_width_half
    except Exception:
        pytest.fail("preferred_vector_width_half call failed")


def check_create_sub_devices_equally(device):
    try:
        n = int(device.max_compute_units / 2)
        device.create_sub_devices(partition=n)
    except SubDeviceCreationError:
        pytest.skip(
            "create_sub_devices can't create sub-devices on this device"
        )
    except Exception:
        pytest.fail("create_sub_devices failed")


def check_create_sub_devices_equally_zeros(device):
    try:
        device.create_sub_devices(partition=0)
    except TypeError:
        pass


def check_create_sub_devices_by_counts(device):
    try:
        n = device.max_compute_units / 2
        device.create_sub_devices(partition=(n, n))
    except SubDeviceCreationError:
        pytest.skip(
            "create_sub_devices can't create sub-devices on this device"
        )
    except Exception:
        pytest.fail("create_sub_devices failed")


def check_create_sub_devices_by_counts_zeros(device):
    try:
        device.create_sub_devices(partition=(0, 1))
    except TypeError:
        pass


def check_create_sub_devices_by_affinity_not_applicable(device):
    try:
        device.create_sub_devices(partition="not_applicable")
    except SubDeviceCreationError:
        pytest.skip(
            "create_sub_devices can't create sub-devices on this device"
        )
    except Exception:
        pytest.fail("create_sub_devices failed")


def check_create_sub_devices_by_affinity_numa(device):
    try:
        device.create_sub_devices(partition="numa")
    except SubDeviceCreationError:
        pytest.skip(
            "create_sub_devices can't create sub-devices on this device"
        )
    except Exception:
        pytest.fail("create_sub_devices failed")


def check_create_sub_devices_by_affinity_L4_cache(device):
    try:
        device.create_sub_devices(partition="L4_cache")
    except SubDeviceCreationError:
        pytest.skip(
            "create_sub_devices can't create sub-devices on this device"
        )
    except Exception:
        pytest.fail("create_sub_devices failed")


def check_create_sub_devices_by_affinity_L3_cache(device):
    try:
        device.create_sub_devices(partition="L3_cache")
    except SubDeviceCreationError:
        pytest.skip(
            "create_sub_devices can't create sub-devices on this device"
        )
    except Exception:
        pytest.fail("create_sub_devices failed")


def check_create_sub_devices_by_affinity_L2_cache(device):
    try:
        device.create_sub_devices(partition="L2_cache")
    except SubDeviceCreationError:
        pytest.skip(
            "create_sub_devices can't create sub-devices on this device"
        )
    except Exception:
        pytest.fail("create_sub_devices failed")


def check_create_sub_devices_by_affinity_L1_cache(device):
    try:
        device.create_sub_devices(partition="L1_cache")
    except SubDeviceCreationError:
        pytest.skip(
            "create_sub_devices can't create sub-devices on this device"
        )
    except Exception:
        pytest.fail("create_sub_devices failed")


def check_create_sub_devices_by_affinity_next_partitionable(device):
    try:
        device.create_sub_devices(partition="next_partitionable")
    except SubDeviceCreationError:
        pytest.skip(
            "create_sub_devices can't create sub-devices on this device"
        )
    except Exception:
        pytest.fail("create_sub_devices failed")


def check_print_device_info(device):
    try:
        device.print_device_info()
    except Exception:
        pytest.fail("Encountered an exception inside print_device_info().")


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
    check_get_sub_group_independent_forward_progress,
    check_get_preferred_vector_width_char,
    check_get_preferred_vector_width_short,
    check_get_preferred_vector_width_int,
    check_get_preferred_vector_width_long,
    check_get_preferred_vector_width_float,
    check_get_preferred_vector_width_double,
    check_get_preferred_vector_width_half,
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
    check_get_max_read_image_args,
    check_get_max_write_image_args,
    check_get_image_2d_max_width,
    check_get_image_2d_max_height,
    check_get_image_3d_max_width,
    check_get_image_3d_max_height,
    check_get_image_3d_max_depth,
    check_create_sub_devices_equally,
    check_create_sub_devices_by_counts,
    check_create_sub_devices_by_affinity_not_applicable,
    check_create_sub_devices_by_affinity_numa,
    check_create_sub_devices_by_affinity_L4_cache,
    check_create_sub_devices_by_affinity_L3_cache,
    check_create_sub_devices_by_affinity_L2_cache,
    check_create_sub_devices_by_affinity_L1_cache,
    check_create_sub_devices_by_affinity_next_partitionable,
    check_print_device_info,
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
    """
    Tests if we can create a SyclDevice using a supported filter selector
    string.
    """
    device = None
    try:
        device = dpctl.SyclDevice(valid_filter)
    except ValueError:
        pytest.skip("Failed to create device with supported filter")
    check(device)


def test_invalid_filter_selectors(invalid_filter):
    """
    An invalid filter string should always be caught and a ValueError raised.
    """
    with pytest.raises(ValueError):
        dpctl.SyclDevice(invalid_filter)


def test_filter_string(valid_filter):
    """
    Test that filter_string reconstructs the same device.
    """
    device = None
    try:
        device = dpctl.SyclDevice(valid_filter)
    except ValueError:
        pytest.skip("Failed to create device with supported filter")
    dev_id = device.filter_string
    assert (
        dpctl.SyclDevice(dev_id) == device
    ), "Reconstructed device is different, ({}, {})".format(
        valid_filter, dev_id
    )


def test_filter_string_property():
    """
    Test that filter_string reconstructs the same device.
    """
    devices = dpctl.get_devices()
    for d in devices:
        if d.default_selector_score >= 0:
            dev_id = d.filter_string
            d_r = dpctl.SyclDevice(dev_id)
            assert d == d_r
            assert hash(d) == hash(d_r)


def test_filter_string_method():
    """
    Test that filter_string reconstructs the same device.
    """
    devices = dpctl.get_devices()
    for d in devices:
        for be in [True, False]:
            for dt in [True, False]:
                if d.default_selector_score >= 0:
                    dev_id = d.get_filter_string(
                        include_backend=be, include_device_type=dt
                    )
                    d_r = dpctl.SyclDevice(dev_id)
                    assert d == d_r, "Failed "
                    assert hash(d) == hash(
                        d_r
                    ), "Hash equality is inconsistent with __eq__"


def test_hashing_of_device():
    """
    Test that a :class:`dpctl.SyclDevice` object can be used as
    a dictionary key.

    """
    device_dict = {dpctl.SyclDevice(): "default_device"}
    assert device_dict


list_of_supported_aspects = [
    "cpu",
    "gpu",
    "accelerator",
    "custom",
    "fp16",
    "fp64",
    "image",
    "online_compiler",
    "online_linker",
    "queue_profiling",
    "usm_device_allocations",
    "usm_host_allocations",
    "usm_shared_allocations",
    "usm_system_allocator",
]

# SYCL 2020 spec aspects not presently
# supported in DPC++, and dpctl
list_of_unsupported_aspects = [
    "emulated",
    "host_debuggable",
    "atomic64",
    "usm_atomic_host_allocations",
    "usm_atomic_shared_allocations",
]


@pytest.fixture(params=list_of_supported_aspects)
def supported_aspect(request):
    return request.param


@pytest.fixture(params=list_of_unsupported_aspects)
def unsupported_aspect(request):
    return request.param


def test_supported_aspect(supported_aspect):
    try:
        dpctl.select_device_with_aspects(supported_aspect)
    except ValueError:
        # ValueError may be raised if no device with
        # requested aspect charateristics is available
        pass


def test_unsupported_aspect(unsupported_aspect):
    try:
        dpctl.select_device_with_aspects(unsupported_aspect)
        raise AttributeError(
            f"The {unsupported_aspect} aspect is now supported in dpctl"
        )
    except AttributeError:
        pytest.skip(
            f"The {unsupported_aspect} aspect is not supported in dpctl"
        )


def test_handle_no_device():
    with pytest.raises(ValueError):
        dpctl.select_device_with_aspects(["gpu", "cpu"])
    with pytest.raises(ValueError):
        dpctl.select_device_with_aspects("cpu", excluded_aspects="cpu")
