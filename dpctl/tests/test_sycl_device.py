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

""" Defines unit test cases for the SyclDevice class.
"""

import pytest
from helper import get_queue_or_skip

import dpctl
from dpctl import SyclDeviceCreationError


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
    An invalid filter string should always be caught and a ValueError raised.
    """
    exc = (
        SyclDeviceCreationError
        if isinstance(invalid_filter, str)
        else ValueError
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
]

# SYCL 2020 spec aspects not presently
# supported in DPC++, and dpctl
list_of_unsupported_aspects = [
    "emulated",
]


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
        dpctl.select_device_with_aspects(unsupported_aspect)
        raise AttributeError(
            f"The {unsupported_aspect} aspect is now supported in dpctl"
        )
    except AttributeError:
        pytest.skip(
            f"The {unsupported_aspect} aspect is not supported in dpctl"
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
