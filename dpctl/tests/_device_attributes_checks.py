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

import pytest

import dpctl

list_of_standard_selectors = [
    dpctl.select_accelerator_device,
    dpctl.select_cpu_device,
    dpctl.select_default_device,
    dpctl.select_gpu_device,
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
    "0",
]

list_of_invalid_filter_selectors = [
    "-1",
    "opencl:gpu:-1",
    "cuda:cpu:0",
    "abc",
    1,
]


# Unit test cases that will be run for every device
def check_max_compute_units(device):
    max_compute_units = device.max_compute_units
    assert max_compute_units > 0


def check_global_mem_size(device):
    global_mem_size = device.global_mem_size
    assert global_mem_size > 0


def check_local_mem_size(device):
    local_mem_size = device.local_mem_size
    assert local_mem_size > 0


def check_max_work_item_dims(device):
    max_work_item_dims = device.max_work_item_dims
    assert max_work_item_dims > 0


def check_max_work_item_sizes1d(device):
    max_work_item_sizes = device.max_work_item_sizes1d
    for size in max_work_item_sizes:
        assert size is not None


def check_max_work_item_sizes2d(device):
    max_work_item_sizes = device.max_work_item_sizes2d
    for size in max_work_item_sizes:
        assert size is not None


def check_max_work_item_sizes3d(device):
    max_work_item_sizes = device.max_work_item_sizes3d
    for size in max_work_item_sizes:
        assert size is not None


def check_max_work_item_sizes(device):
    with pytest.warns(DeprecationWarning):
        max_work_item_sizes = device.max_work_item_sizes
    for size in max_work_item_sizes:
        assert size is not None


def check_max_work_group_size(device):
    max_work_group_size = device.max_work_group_size
    # Special case for FPGA simulator
    if device.is_accelerator:
        assert max_work_group_size >= 0
    else:
        assert max_work_group_size > 0


def check_max_num_sub_groups(device):
    max_num_sub_groups = device.max_num_sub_groups
    # Special case for FPGA simulator
    if device.is_accelerator:
        assert max_num_sub_groups >= 0
    else:
        assert max_num_sub_groups > 0


def check_sub_group_sizes(device):
    sg_sizes = device.sub_group_sizes
    assert all(el > 0 for el in sg_sizes)


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


def check_has_aspect_atomic64(device):
    try:
        device.has_aspect_atomic64
    except Exception:
        pytest.fail("has_aspect_atomic64 call failed")


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


def check_has_aspect_usm_system_allocations(device):
    try:
        device.has_aspect_usm_system_allocations
    except Exception:
        pytest.fail("has_aspect_usm_system_allocations call failed")


def check_has_aspect_usm_atomic_host_allocations(device):
    try:
        device.has_aspect_usm_atomic_host_allocations
    except Exception:
        pytest.fail("has_aspect_usm_atomic_host_allocations call failed")


def check_has_aspect_usm_atomic_shared_allocations(device):
    try:
        device.has_aspect_usm_atomic_shared_allocations
    except Exception:
        pytest.fail("has_aspect_usm_atomic_shared_allocations call failed")


def check_has_aspect_host_debuggable(device):
    try:
        device.has_aspect_host_debuggable
    except Exception:
        pytest.fail("has_aspect_host_debuggable call failed")


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


def check_max_read_image_args(device):
    try:
        device.max_read_image_args
    except Exception:
        pytest.fail("max_read_image_args call failed")


def check_max_write_image_args(device):
    try:
        device.max_write_image_args
    except Exception:
        pytest.fail("max_write_image_args call failed")


def check_image_2d_max_width(device):
    try:
        device.image_2d_max_width
    except Exception:
        pytest.fail("image_2d_max_width call failed")


def check_image_2d_max_height(device):
    try:
        device.image_2d_max_height
    except Exception:
        pytest.fail("image_2d_max_height call failed")


def check_image_3d_max_width(device):
    try:
        device.image_3d_max_width
    except Exception:
        pytest.fail("image_3d_max_width call failed")


def check_image_3d_max_height(device):
    try:
        device.image_3d_max_height
    except Exception:
        pytest.fail("image_3d_max_height call failed")


def check_image_3d_max_depth(device):
    try:
        device.image_3d_max_depth
    except Exception:
        pytest.fail("image_3d_max_depth call failed")


def check_sub_group_independent_forward_progress(device):
    try:
        device.sub_group_independent_forward_progress
    except Exception:
        pytest.fail("sub_group_independent_forward_progress call failed")


def check_preferred_vector_width_char(device):
    try:
        device.preferred_vector_width_char
    except Exception:
        pytest.fail("preferred_vector_width_char call failed")


def check_preferred_vector_width_short(device):
    try:
        device.preferred_vector_width_short
    except Exception:
        pytest.fail("preferred_vector_width_short call failed")


def check_preferred_vector_width_int(device):
    try:
        device.preferred_vector_width_int
    except Exception:
        pytest.fail("preferred_vector_width_int call failed")


def check_preferred_vector_width_long(device):
    try:
        device.preferred_vector_width_long
    except Exception:
        pytest.fail("preferred_vector_width_long call failed")


def check_preferred_vector_width_float(device):
    try:
        device.preferred_vector_width_float
    except Exception:
        pytest.fail("preferred_vector_width_float call failed")


def check_preferred_vector_width_double(device):
    try:
        device.preferred_vector_width_double
    except Exception:
        pytest.fail("preferred_vector_width_double call failed")


def check_preferred_vector_width_half(device):
    try:
        device.preferred_vector_width_half
    except Exception:
        pytest.fail("preferred_vector_width_half call failed")


def check_native_vector_width_char(device):
    try:
        device.native_vector_width_char
    except Exception:
        pytest.fail("native_vector_width_char call failed")


def check_native_vector_width_short(device):
    try:
        device.native_vector_width_short
    except Exception:
        pytest.fail("native_vector_width_short call failed")


def check_native_vector_width_int(device):
    try:
        device.native_vector_width_int
    except Exception:
        pytest.fail("native_vector_width_int call failed")


def check_native_vector_width_long(device):
    try:
        device.native_vector_width_long
    except Exception:
        pytest.fail("native_vector_width_long call failed")


def check_native_vector_width_float(device):
    try:
        device.native_vector_width_float
    except Exception:
        pytest.fail("native_vector_width_float call failed")


def check_native_vector_width_double(device):
    try:
        device.native_vector_width_double
    except Exception:
        pytest.fail("native_vector_width_double call failed")


def check_native_vector_width_half(device):
    try:
        device.native_vector_width_half
    except Exception:
        pytest.fail("native_vector_width_half call failed")


def check_create_sub_devices_equally(device):
    try:
        n = int(device.max_compute_units / 2)
        device.create_sub_devices(partition=n)
    except dpctl.SyclSubDeviceCreationError:
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
    except dpctl.SyclSubDeviceCreationError:
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
    except dpctl.SyclSubDeviceCreationError:
        pytest.skip(
            "create_sub_devices can't create sub-devices on this device"
        )
    except Exception:
        pytest.fail("create_sub_devices failed")


def check_create_sub_devices_by_affinity_numa(device):
    try:
        device.create_sub_devices(partition="numa")
    except dpctl.SyclSubDeviceCreationError:
        pytest.skip(
            "create_sub_devices can't create sub-devices on this device"
        )
    except Exception:
        pytest.fail("create_sub_devices failed")


def check_create_sub_devices_by_affinity_L4_cache(device):
    try:
        device.create_sub_devices(partition="L4_cache")
    except dpctl.SyclSubDeviceCreationError:
        pytest.skip(
            "create_sub_devices can't create sub-devices on this device"
        )
    except Exception:
        pytest.fail("create_sub_devices failed")


def check_create_sub_devices_by_affinity_L3_cache(device):
    try:
        device.create_sub_devices(partition="L3_cache")
    except dpctl.SyclSubDeviceCreationError:
        pytest.skip(
            "create_sub_devices can't create sub-devices on this device"
        )
    except Exception:
        pytest.fail("create_sub_devices failed")


def check_create_sub_devices_by_affinity_L2_cache(device):
    try:
        device.create_sub_devices(partition="L2_cache")
    except dpctl.SyclSubDeviceCreationError:
        pytest.skip(
            "create_sub_devices can't create sub-devices on this device"
        )
    except Exception:
        pytest.fail("create_sub_devices failed")


def check_create_sub_devices_by_affinity_L1_cache(device):
    try:
        device.create_sub_devices(partition="L1_cache")
    except dpctl.SyclSubDeviceCreationError:
        pytest.skip(
            "create_sub_devices can't create sub-devices on this device"
        )
    except Exception:
        pytest.fail("create_sub_devices failed")


def check_create_sub_devices_by_affinity_next_partitionable(device):
    try:
        device.create_sub_devices(partition="next_partitionable")
    except dpctl.SyclSubDeviceCreationError:
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


def check_repr(device):
    assert type(repr(device)) is str


def check_profiling_timer_resolution(device):
    try:
        resol = device.profiling_timer_resolution
    except Exception:
        pytest.fail(
            "Encountered an exception inside "
            "profiling_timer_resolution property."
        )
    assert isinstance(resol, int) and resol > 0


def check_platform(device):
    p = device.sycl_platform
    assert isinstance(p, dpctl.SyclPlatform)


def check_parent_device(device):
    pd = device.parent_device
    assert pd is None or isinstance(pd, dpctl.SyclDevice)


def check_partition_max_sub_devices(device):
    max_part = device.partition_max_sub_devices
    assert isinstance(max_part, int)
    assert max_part >= 0
    assert max_part <= device.max_compute_units


def check_filter_string(device):
    try:
        fs = device.filter_string
        assert type(fs) is str
        dd = dpctl.SyclDevice(fs)
        assert device == dd
    except TypeError:
        pass


def check_name(device):
    dn = device.name
    assert dn
    assert type(dn) is str


def check_driver_version(device):
    dv = device.driver_version
    assert dv
    assert type(dv) is str


def check_vendor(device):
    ve = device.vendor
    assert ve
    assert type(ve) is str


def check_default_selector_score(device):
    sc = device.default_selector_score
    assert type(sc) is int
    assert sc > 0 or not (
        device.has_aspect_cpu
        or device.has_aspect_gpu
        or device.has_aspect_accelerator
    )


def check_backend(device):
    be = device.backend
    assert type(be) is dpctl.backend_type


def check_device_type(device):
    dt = device.device_type
    assert type(dt) is dpctl.device_type


def check_max_clock_frequency(device):
    freq = device.max_clock_frequency
    assert isinstance(freq, int)
    # FIXME: Change to freq > 0 after transition to 2024.1
    assert freq >= 0


def check_max_mem_alloc_size(device):
    mmas = device.max_mem_alloc_size
    assert isinstance(mmas, int)
    assert mmas > 0


def check_global_mem_cache_type(device):
    gmc_ty = device.global_mem_cache_type
    assert type(gmc_ty) is dpctl.global_mem_cache_type


def check_global_mem_cache_size(device):
    gmc_sz = device.global_mem_cache_size
    assert type(gmc_sz) is int
    assert gmc_sz


def check_global_mem_cache_line_size(device):
    gmc_sz = device.global_mem_cache_line_size
    assert type(gmc_sz) is int
    assert gmc_sz


list_of_checks = [
    check_max_compute_units,
    check_max_work_item_dims,
    check_max_work_item_sizes1d,
    check_max_work_item_sizes2d,
    check_max_work_item_sizes3d,
    check_max_work_item_sizes,
    check_max_work_group_size,
    check_max_num_sub_groups,
    check_sub_group_sizes,
    check_is_accelerator,
    check_is_cpu,
    check_is_gpu,
    check_sub_group_independent_forward_progress,
    check_preferred_vector_width_char,
    check_preferred_vector_width_short,
    check_preferred_vector_width_int,
    check_preferred_vector_width_long,
    check_preferred_vector_width_float,
    check_preferred_vector_width_double,
    check_preferred_vector_width_half,
    check_native_vector_width_char,
    check_native_vector_width_short,
    check_native_vector_width_int,
    check_native_vector_width_long,
    check_native_vector_width_float,
    check_native_vector_width_double,
    check_native_vector_width_half,
    check_has_aspect_cpu,
    check_has_aspect_gpu,
    check_has_aspect_accelerator,
    check_has_aspect_custom,
    check_has_aspect_fp16,
    check_has_aspect_fp64,
    check_has_aspect_atomic64,
    check_has_aspect_image,
    check_has_aspect_online_compiler,
    check_has_aspect_online_linker,
    check_has_aspect_queue_profiling,
    check_has_aspect_usm_device_allocations,
    check_has_aspect_usm_host_allocations,
    check_has_aspect_usm_shared_allocations,
    check_has_aspect_usm_system_allocations,
    check_has_aspect_usm_atomic_host_allocations,
    check_has_aspect_usm_atomic_shared_allocations,
    check_has_aspect_host_debuggable,
    check_max_read_image_args,
    check_max_write_image_args,
    check_image_2d_max_width,
    check_image_2d_max_height,
    check_image_3d_max_width,
    check_image_3d_max_height,
    check_image_3d_max_depth,
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
    check_repr,
    check_global_mem_size,
    check_local_mem_size,
    check_profiling_timer_resolution,
    check_platform,
    check_parent_device,
    check_partition_max_sub_devices,
    check_filter_string,
    check_vendor,
    check_driver_version,
    check_name,
    check_default_selector_score,
    check_backend,
    check_device_type,
    check_global_mem_cache_type,
    check_global_mem_cache_size,
    check_global_mem_cache_line_size,
    check_max_clock_frequency,
    check_max_mem_alloc_size,
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
