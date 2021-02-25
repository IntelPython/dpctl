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

""" Defines unit test cases for the _sycl_device_factory module
"""

import dpctl
from dpctl import backend_type as bty, device_type as dty
import pytest

list_of_backends = [
    bty.host,
    bty.level_zero,
    bty.opencl,
]

list_of_device_types = [
    dty.cpu,
    dty.host_device,
    dty.gpu,
]

list_of_backend_strs = [
    "host",
    "level_zero",
    "opencl",
]

list_of_device_type_strs = [
    "cpu",
    "host_device",
    "gpu",
]


def string_to_device_type(dty_str):
    if dty_str == "accelerator":
        return dty.accelerator
    elif dty_str == "cpu":
        return dty.cpu
    elif dty_str == "host":
        return dty.host
    elif dty_str == "gpu":
        return dty.gpu


def string_to_backend_type(bty_str):
    if bty_str == "cuda":
        return bty.cuda
    elif bty_str == "host":
        return bty.host
    elif bty_str == "level_zero":
        return bty.level_zero
    elif bty_str == "opencl":
        return bty.opencl


@pytest.fixture(params=list_of_backends)
def backend(request):
    return request.param


@pytest.fixture(params=list_of_backend_strs)
def bty_str(request):
    return request.param


@pytest.fixture(params=list_of_device_types)
def device_type(request):
    return request.param


@pytest.fixture(params=list_of_device_type_strs)
def dty_str(request):
    return request.param


def check_if_backend_matches(devices, backend):
    for d in devices:
        assert d.get_backend() == backend


def check_if_device_type_matches(devices, device_type):
    for d in devices:
        assert d.get_device_type() == device_type


def test_get_devices_with_string_args(bty_str, dty_str):
    devices = dpctl.get_devices(backend=bty_str, device_type=dty_str)
    if len(devices):
        d = string_to_device_type(dty_str)
        b = string_to_backend_type(bty_str)
        check_if_backend_matches(devices, b)
        check_if_device_type_matches(devices, d)
    else:
        pytest.skip()


def test_get_devices_with_enum_args(backend, device_type):
    devices = dpctl.get_devices(backend=backend, device_type=device_type)
    if len(devices):
        check_if_backend_matches(devices, backend)
        check_if_device_type_matches(devices, device_type)
    else:
        pytest.skip()
