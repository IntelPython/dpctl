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

""" Defines unit test cases for the SyclDevice class.
"""

import pytest

import dpctl
from dpctl import SyclDeviceCreationError

from .helper import get_queue_or_skip


def test_standard_selectors(device_selector, check):
    """Tests if the standard SYCL device_selectors are able to select a
    device.
    """
    try:
        device = device_selector()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("Could not create default-selected device")
    check(device)


def test_current_device(check):
    """Test is the device for the current queue is valid."""
    q = get_queue_or_skip()
    try:
        device = q.get_sycl_device()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("Could not create default-selected device")
    check(device)


def test_valid_filter_selectors(valid_filter, check):
    """
    Tests if we can create a SyclDevice using a supported filter selector
    string.
    """
    device = None
    try:
        device = dpctl.SyclDevice(valid_filter)
    except SyclDeviceCreationError:
        pytest.skip("Failed to create device with supported filter")
    check(device)


def test_invalid_filter_selectors(invalid_filter):
    """
    An invalid filter string should always be caught and a TypeError raised.
    """
    exc = (
        SyclDeviceCreationError
        if isinstance(invalid_filter, str)
        else TypeError
    )
    with pytest.raises(exc):
        dpctl.SyclDevice(invalid_filter)


def test_filter_string(valid_filter):
    """
    Test that filter_string reconstructs the same device.
    """
    device = None
    try:
        device = dpctl.SyclDevice(valid_filter)
    except SyclDeviceCreationError:
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
    try:
        device = dpctl.SyclDevice()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("Could not create default-constructed device")
    device_dict = {device: "default_device"}
    assert device_dict


def test_equal():
    try:
        d1 = dpctl.SyclDevice()
        d2 = dpctl.SyclDevice()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("Could not create default-selected device")
    assert d1 != Ellipsis
    assert d1 == d2


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
    "usm_system_allocations",
    "host_debuggable",
    "atomic64",
    "usm_atomic_host_allocations",
    "usm_atomic_shared_allocations",
    "emulated",
    "is_component",
    "is_composite",
]

# SYCL 2020 spec aspects not presently
# supported in DPC++, and dpctl
list_of_unsupported_aspects = []


@pytest.fixture(params=list_of_supported_aspects)
def supported_aspect(request):
    return request.param


@pytest.fixture(params=list_of_unsupported_aspects)
def unsupported_aspect(request):
    return request.param


def test_supported_aspect(supported_aspect):
    try:
        d = dpctl.SyclDevice()
        has_it = getattr(d, "has_aspect_" + supported_aspect)
    except dpctl.SyclDeviceCreationError:
        has_it = False
    try:
        d_wa = dpctl.select_device_with_aspects(supported_aspect)
        assert getattr(d_wa, "has_aspect_" + supported_aspect)
    except dpctl.SyclDeviceCreationError:
        # ValueError may be raised if no device with
        # requested aspect charateristics is available
        assert not has_it


def test_unsupported_aspect(unsupported_aspect):
    try:
        d = dpctl.SyclDevice()
        has_it = hasattr(d, "has_aspect_" + unsupported_aspect)
    except dpctl.SyclDeviceCreationError:
        has_it = False
    if has_it:
        raise AttributeError(
            f"The {unsupported_aspect} aspect is now supported in dpctl"
        )


def test_handle_no_device():
    with pytest.raises(dpctl.SyclDeviceCreationError):
        dpctl.select_device_with_aspects(["gpu", "cpu"])
    with pytest.raises(dpctl.SyclDeviceCreationError):
        dpctl.select_device_with_aspects("cpu", excluded_aspects="cpu")


def test_cpython_api_SyclDevice_GetDeviceRef():
    import ctypes
    import sys

    try:
        d = dpctl.SyclDevice()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("Could not create default-constructed device")
    mod = sys.modules[d.__class__.__module__]
    # get capsule storing SyclDevice_GetDeviceRef function ptr
    d_ref_fn_cap = mod.__pyx_capi__["SyclDevice_GetDeviceRef"]
    # construct Python callable to invoke "SyclDevice_GetDeviceRef"
    cap_ptr_fn = ctypes.pythonapi.PyCapsule_GetPointer
    cap_ptr_fn.restype = ctypes.c_void_p
    cap_ptr_fn.argtypes = [ctypes.py_object, ctypes.c_char_p]
    d_ref_fn_ptr = cap_ptr_fn(
        d_ref_fn_cap, b"DPCTLSyclDeviceRef (struct PySyclDeviceObject *)"
    )
    callable_maker = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object)
    get_device_ref_fn = callable_maker(d_ref_fn_ptr)

    r2 = d.addressof_ref()
    r1 = get_device_ref_fn(d)
    assert r1 == r2


def test_cpython_api_SyclDevice_Make():
    import ctypes
    import sys

    try:
        d = dpctl.SyclDevice()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("Could not create default-constructed device")
    mod = sys.modules[d.__class__.__module__]
    # get capsule storign SyclContext_Make function ptr
    make_d_fn_cap = mod.__pyx_capi__["SyclDevice_Make"]
    # construct Python callable to invoke "SyclDevice_Make"
    cap_ptr_fn = ctypes.pythonapi.PyCapsule_GetPointer
    cap_ptr_fn.restype = ctypes.c_void_p
    cap_ptr_fn.argtypes = [ctypes.py_object, ctypes.c_char_p]
    make_d_fn_ptr = cap_ptr_fn(
        make_d_fn_cap, b"struct PySyclDeviceObject *(DPCTLSyclDeviceRef)"
    )
    callable_maker = ctypes.PYFUNCTYPE(ctypes.py_object, ctypes.c_void_p)
    make_d_fn = callable_maker(make_d_fn_ptr)

    d2 = make_d_fn(d.addressof_ref())
    assert d == d2


def test_get_device_id_method():
    """
    Test that the get_device_id method reconstructs the same device.
    """
    devices = dpctl.get_devices()
    for d in devices:
        dev_id = d.get_device_id()
        d_r = dpctl.SyclDevice(str(dev_id))
        assert dev_id == devices.index(d)
        assert d == d_r
        assert hash(d) == hash(d_r)


def test_get_unpartitioned_parent_device_method():
    """
    Test that the get_unpartitioned_parent method returns self for root
    devices.
    """
    devices = dpctl.get_devices()
    for d in devices:
        assert d == d.get_unpartitioned_parent_device()


def test_get_unpartitioned_parent_device_from_sub_device():
    """
    Test that the get_unpartitioned_parent method returns the parent device
    from the sub-device.
    """
    try:
        dev = dpctl.SyclDevice()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No default device available")
    try:
        sdevs = dev.create_sub_devices(partition="next_partitionable")
    except dpctl.SyclSubDeviceCreationError:
        sdevs = None
    try:
        if sdevs is None:
            sdevs = dev.create_sub_devices(partition=[1, 1])
    except dpctl.SyclSubDeviceCreationError:
        pytest.skip("Default device can not be partitioned")
    assert isinstance(sdevs, list) and len(sdevs) > 0
    assert dev == sdevs[0].get_unpartitioned_parent_device()


def test_composite_device_method():
    """
    Test that the composite_device method returns a composite
    device found in ``dpctl.get_composite_devices()``
    """
    devices = dpctl.get_devices()
    composite_devices = dpctl.get_composite_devices()
    for d in devices:
        if d.has_aspect_is_component:
            Cd = d.composite_device
            assert Cd in composite_devices


def test_get_component_devices_from_composite():
    """
    Test that the component_devices method returns component
    root devices.
    """
    devices = dpctl.get_devices()
    composite_devices = dpctl.get_composite_devices()
    for Cd in composite_devices:
        assert Cd.has_aspect_is_composite
        component_devices = Cd.component_devices()
        for d in component_devices:
            assert d.has_aspect_is_component
            # component devices are root devices
            assert d in devices


@pytest.mark.parametrize("platform_name", ["level_zero", "cuda", "hip"])
def test_can_access_peer(platform_name):
    """
    Test checks for peer access.
    """
    try:
        platform = dpctl.SyclPlatform(platform_name)
    except ValueError as e:
        pytest.skip(f"{str(e)} {platform_name}")
    devices = platform.get_devices()
    if len(devices) < 2:
        pytest.skip(
            f"Platform {platform_name} does not have enough devices to "
            "test peer access"
        )
    dev0 = devices[0]
    dev1 = devices[1]
    assert isinstance(dev0.can_access_peer(dev1), bool)
    assert isinstance(
        dev0.can_access_peer(dev1, value="atomics_supported"), bool
    )


@pytest.mark.parametrize("platform_name", ["level_zero", "cuda", "hip"])
def test_enable_disable_peer_access(platform_name):
    """
    Test that peer access can be enabled and disabled.
    """
    try:
        platform = dpctl.SyclPlatform(platform_name)
    except ValueError as e:
        pytest.skip(f"{str(e)} {platform_name}")
    devices = platform.get_devices()
    if len(devices) < 2:
        pytest.skip(
            f"Platform {platform_name} does not have enough devices to "
            "test peer access"
        )
    dev0 = devices[0]
    dev1 = devices[1]
    if dev0.can_access_peer(dev1):
        dev0.enable_peer_access(dev1)
        dev0.disable_peer_access(dev1)
    else:
        pytest.skip(
            f"Provided {platform_name} devices do not support peer access"
        )


@pytest.mark.parametrize(
    "method",
    [
        "can_access_peer",
        "enable_peer_access",
        "disable_peer_access",
    ],
)
def test_peer_device_arg_validation(method):
    """
    Test for validation of arguments to peer access related methods.
    """
    try:
        dev = dpctl.SyclDevice()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No default device available")
    bad_dev = dict()
    callable = getattr(dev, method)
    with pytest.raises(TypeError):
        callable(bad_dev)


@pytest.mark.parametrize("platform_name", ["level_zero", "cuda", "hip"])
def test_peer_access_to_self(platform_name):
    """
    Validate behavior of a device attempting to enable peer access to itself.
    """
    try:
        platform = dpctl.SyclPlatform(platform_name)
    except ValueError as e:
        pytest.skip(f"{str(e)} {platform_name}")
    dev = platform.get_devices()[0]
    with pytest.raises(ValueError):
        dev.enable_peer_access(dev)
    with pytest.raises(ValueError):
        dev.disable_peer_access(dev)


def test_peer_access_value_keyword_validation():
    """
    Validate behavior of `can_access_peer` for invalid `value` keyword.
    """
    # we pick an arbitrary platform that supports peer access
    platforms = dpctl.get_platforms()
    peer_access_backends = [
        dpctl.backend_type.cuda,
        dpctl.backend_type.hip,
        dpctl.backend_type.hip,
    ]
    devs = None
    for p in platforms:
        if p.backend in peer_access_backends:
            p_devs = p.get_devices()
            if len(p_devs) >= 2:
                devs = p_devs
                break
    if devs is None:
        pytest.skip("No platform available with enough devices")
    dev0 = devs[0]
    dev1 = devs[1]
    bad_type = 2
    with pytest.raises(TypeError):
        dev0.can_access_peer(dev1, value=bad_type)
    bad_value = "wrong"
    with pytest.raises(ValueError):
        dev0.can_access_peer(dev1, value=bad_value)
