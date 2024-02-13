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

""" Configures pytest to discover helper/ module
"""

import os
import sys

import pytest
from _device_attributes_checks import (
    check,
    device_selector,
    invalid_filter,
    valid_filter,
)
from _numpy_warnings import suppress_invalid_numpy_warnings

sys.path.append(os.path.join(os.path.dirname(__file__), "helper"))

# common fixtures
__all__ = [
    "check",
    "device_selector",
    "invalid_filter",
    "suppress_invalid_numpy_warnings",
    "valid_filter",
]


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "broken_complex: Specified again to remove warnings ",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--runcomplex",
        action="store_true",
        default=False,
        help="run broken complex tests",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runcomplex"):
        return
    skip_complex = pytest.mark.skip(
        reason="need --runcomplex option to run",
    )
    for item in items:
        if "broken_complex" in item.keywords:
            item.add_marker(skip_complex)
