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

"""Defines unit test cases for the SyclPlatform class.
"""

import sys

import pytest
from helper import has_sycl_platforms

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

list_of_invalid_filter_selectors = [
    "-1",
    "opencl:gpu:-1",
    "cuda:cpu:0",
    "abc",
]


def check_name(platform):
    try:
        platform.name
    except Exception:
        pytest.fail("Encountered an exception inside platform.name.")


def check_vendor(platform):
    try:
        platform.vendor
    except Exception:
        pytest.fail("Encountered an exception inside platform.vendor.")


def check_version(platform):
    try:
        platform.version
    except Exception:
        pytest.fail("Encountered an exception inside platform.version.")


def check_backend(platform):
    try:
        platform.backend
    except Exception:
        pytest.fail("Encountered an exception inside platform.backend.")


def check_print_info(platform):
    try:
        platform.print_platform_info()
    except Exception:
        pytest.fail("Encountered an exception inside print_info().")


def check_repr(platform):
    r = repr(platform)
    assert type(r) is str
    assert r != ""


def check_default_context(platform):
    if "linux" not in sys.platform:
        return
    r = platform.default_context
    assert type(r) is dpctl.SyclContext


def check_equal_and_hash(platform):
    assert platform == platform
    if "linux" not in sys.platform:
        return
    default_ctx = platform.default_context
    for d in default_ctx.get_devices():
        assert platform == d.sycl_platform
        assert hash(platform) == hash(d.sycl_platform)


def check_hash_in_dict(platform):
    map = {platform: 0}
    assert map[platform] == 0


list_of_checks = [
    check_name,
    check_vendor,
    check_version,
    check_backend,
    check_print_info,
    check_repr,
    check_default_context,
    check_equal_and_hash,
    check_hash_in_dict,
]


@pytest.fixture(params=list_of_valid_filter_selectors)
def valid_filter(request):
    return request.param


@pytest.fixture(params=list_of_invalid_filter_selectors)
def invalid_filter(request):
    return request.param


@pytest.fixture(params=list_of_checks)
def check(request):
    return request.param


def test_platform_creation(valid_filter, check):
    """Tests if we can create a SyclPlatform using a supported filter selector
    string.
    """
    platform = None
    try:
        platform = dpctl.SyclPlatform(valid_filter)
    except ValueError:
        pytest.skip("Failed to create platform with supported filter")
    check(platform)


def test_default_platform_creation(check):
    platform = None
    try:
        platform = dpctl.SyclPlatform()
    except ValueError:
        pytest.skip("Failed to create default platform")
    check(platform)


def test_invalid_platform_creation(invalid_filter, check):
    """Tests if we can create a SyclPlatform using a supported filter selector
    string.
    """
    with pytest.raises(ValueError):
        dpctl.SyclPlatform(invalid_filter)


def test_lsplatform():
    try:
        dpctl.lsplatform()
    except Exception:
        pytest.fail("Encountered an exception inside lsplatform().")


def test_lsplatform0():
    try:
        dpctl.lsplatform(0)
    except Exception:
        pytest.fail("Encountered an exception inside lsplatform().")


def test_lsplatform1():
    try:
        dpctl.lsplatform(1)
    except Exception:
        pytest.fail("Encountered an exception inside lsplatform().")


def test_lsplatform2():
    try:
        dpctl.lsplatform(2)
    except Exception:
        pytest.fail("Encountered an exception inside lsplatform().")


def test_lsplatform3():
    try:
        with pytest.warns(UserWarning):
            dpctl.lsplatform(3)
    except Exception:
        pytest.fail("Encountered an exception inside lsplatform().")


def test_get_platforms():
    try:
        platforms = dpctl.get_platforms()
        if platforms:
            assert has_sycl_platforms()
    except Exception:
        pytest.fail("Encountered an exception inside get_platforms().")
