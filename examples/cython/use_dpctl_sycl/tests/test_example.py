#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2022 Intel Corporation
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
import use_dpctl_sycl

import dpctl


def test_device_name():
    try:
        d = dpctl.SyclDevice()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("Could not create default device. Nothing to do")
    d_n = use_dpctl_sycl.device_name(d)
    assert d_n.decode("utf-8") == d.name


def test_device_driver_version():
    try:
        d = dpctl.SyclDevice()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("Could not create default device. Nothing to do")
    d_dv = use_dpctl_sycl.device_driver_version(d)
    assert d_dv.decode("utf-8") == d.driver_version


def test_device_copy():
    try:
        d = dpctl.SyclDevice()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("Could not create default device. Nothing to do")
    d_copy = use_dpctl_sycl.device_copy(d)
    assert d_copy == d
    assert d_copy.addressof_ref() != d.addressof_ref()
