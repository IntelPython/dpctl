#                      Data Parallel Control (dpCtl)
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

"""Defines unit test cases for the SyclPlatform class.
"""

import dpctl
import pytest
from ._helper import has_sycl_platforms

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


list_of_checks = [
    check_name,
    check_vendor,
    check_version,
    check_backend,
    check_print_info,
]


@pytest.fixture(params=list_of_valid_filter_selectors)
def valid_filter(request):
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


def test_lsplatform():
    try:
        dpctl.lsplatform()
    except Exception:
        pytest.fail("Encountered an exception inside lsplatform().")


def test_get_platforms():
    try:
        platforms = dpctl.get_platforms()
        if platforms:
            assert has_sycl_platforms()
    except Exception:
        pytest.fail("Encountered an exception inside get_platforms().")
