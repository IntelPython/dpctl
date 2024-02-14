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

""" Defines unit test cases for the _sycl_device_factory module
"""

import pytest

import dpctl
from dpctl import backend_type as bty
from dpctl import device_type as dty

argument_list_1 = [
    (bty.level_zero, dty.gpu),
    (bty.opencl, dty.gpu),
    (bty.opencl, dty.cpu),
]

argument_list_2 = [
    ("level_zero", "gpu"),
    ("opencl", "gpu"),
    ("opencl", "cpu"),
]

list_of_backend_str = [
    "level_zero",
    "opencl",
]

list_of_device_type_str = [
    "gpu",
    "cpu",
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


@pytest.fixture(params=argument_list_1)
def enum_args(request):
    return request.param


@pytest.fixture(params=argument_list_2)
def str_args(request):
    return request.param


@pytest.fixture(params=[item for item in bty])
def backend(request):
    return request.param


@pytest.fixture(params=list_of_backend_str)
def backend_str(request):
    return request.param


@pytest.fixture(params=[item for item in dty])
def device_type(request):
    return request.param


@pytest.fixture(params=list_of_device_type_str)
def device_type_str(request):
    return request.param


def check_if_device_type_is_valid(devices):
    for d in devices:
        assert d.device_type in set(item for item in dty)


def check_if_backend_is_valid(devices):
    for d in devices:
        assert d.backend in set(item for item in bty)


def check_if_backend_matches(devices, backend):
    for d in devices:
        assert d.backend == backend


def check_if_device_type_matches(devices, device_type):
    for d in devices:
        assert d.device_type == device_type


def test_get_devices_with_string_args(str_args):
    devices = dpctl.get_devices(backend=str_args[0], device_type=str_args[1])
    if len(devices):
        d = string_to_device_type(str_args[1])
        b = string_to_backend_type(str_args[0])
        check_if_backend_matches(devices, b)
        check_if_device_type_matches(devices, d)
    else:
        pytest.skip()


def test_get_devices_with_enum_args(enum_args):
    devices = dpctl.get_devices(backend=enum_args[0], device_type=enum_args[1])
    if len(devices):
        check_if_backend_matches(devices, enum_args[0])
        check_if_device_type_matches(devices, enum_args[1])
    else:
        pytest.skip()


def test_get_devices_with_backend_enum(backend):
    devices = dpctl.get_devices(backend=backend)
    if len(devices):
        check_if_device_type_is_valid(devices)
        check_if_backend_is_valid(devices)
        if backend != bty.all:
            check_if_backend_matches(devices, backend)

    else:
        pytest.skip()


def test_get_devices_with_backend_str(backend_str):
    print(backend_str)
    devices = dpctl.get_devices(backend=backend_str)
    if len(devices):
        b = string_to_backend_type(backend_str)
        check_if_backend_matches(devices, b)
        check_if_device_type_is_valid(devices)
    else:
        pytest.skip()


def test_get_devices_with_device_type_enum(device_type):
    devices = dpctl.get_devices(device_type=device_type)
    if len(devices):
        if device_type != dty.all:
            check_if_device_type_matches(devices, device_type)
        check_if_device_type_is_valid(devices)
        check_if_backend_is_valid(devices)
    else:
        pytest.skip()


def test_get_devices_with_device_type_str(device_type_str):
    num_devices = dpctl.get_num_devices(device_type=device_type_str)
    if num_devices > 0:
        devices = dpctl.get_devices(device_type=device_type_str)
        assert len(devices) == num_devices
        dty = string_to_device_type(device_type_str)
        check_if_device_type_matches(devices, dty)
        check_if_device_type_is_valid(devices)
        # check for consistency of ordering between filter selector
        # where backend is omitted, but device type and id is specified
        for i in range(num_devices):
            dev = dpctl.SyclDevice(":".join((device_type_str, str(i))))
            assert dev == devices[i]
    else:
        pytest.skip()
