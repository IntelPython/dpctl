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

import ctypes

import pytest

import dpctl
import dpctl.tensor as dpt

device_oneAPI = 14  # DLDeviceType.kDLOneAPI

_usm_types_list = ["shared", "device", "host"]


@pytest.fixture(params=_usm_types_list)
def usm_type(request):
    return request.param


_typestrs_list = [
    "b1",
    "u1",
    "i1",
    "u2",
    "i2",
    "u4",
    "i4",
    "u8",
    "i8",
    "f2",
    "f4",
    "f8",
    "c8",
    "c16",
]


@pytest.fixture(params=_typestrs_list)
def typestr(request):
    return request.param


def test_dlpack_device(usm_type):
    all_root_devices = dpctl.get_devices()
    for sycl_dev in all_root_devices:
        X = dpt.empty((64,), dtype="u1", usm_type=usm_type, device=sycl_dev)
        dev = X.__dlpack_device__()
        assert type(dev) is tuple
        assert len(dev) == 2
        assert dev[0] == device_oneAPI
        assert sycl_dev == all_root_devices[dev[1]]


def test_dlpack_exporter(typestr, usm_type):
    caps_fn = ctypes.pythonapi.PyCapsule_IsValid
    caps_fn.restype = bool
    caps_fn.argtypes = [ctypes.py_object, ctypes.c_char_p]
    all_root_devices = dpctl.get_devices()
    for sycl_dev in all_root_devices:
        X = dpt.empty((64,), dtype=typestr, usm_type=usm_type, device=sycl_dev)
        caps = X.__dlpack__()
        assert caps_fn(caps, b"dltensor")
        Y = X[::2]
        caps2 = Y.__dlpack__()
        assert caps_fn(caps2, b"dltensor")


@pytest.mark.parametrize("shape", [tuple(), (2,), (3, 0, 1), (2, 2, 2)])
def test_from_dlpack(shape, typestr, usm_type):
    all_root_devices = dpctl.get_devices()
    for sycl_dev in all_root_devices:
        X = dpt.empty(shape, dtype=typestr, usm_type=usm_type, device=sycl_dev)
        Y = dpt.from_dlpack(X)
        assert X.shape == Y.shape
        assert X.dtype == Y.dtype or (
            str(X.dtype) == "bool" and str(Y.dtype) == "uint8"
        )
        assert X.sycl_device == Y.sycl_device
        assert X.usm_type == Y.usm_type
        assert X._pointer == Y._pointer
        if Y.ndim:
            V = Y[::-1]
            W = dpt.from_dlpack(V)
            assert V.strides == W.strides


def test_from_dlpack_input_validation():
    vstr = dpt._dlpack.get_build_dlpack_version()
    assert type(vstr) is str
    with pytest.raises(TypeError):
        dpt.from_dlpack(None)

    class DummyWithProperty:
        @property
        def __dlpack__(self):
            return None

    with pytest.raises(TypeError):
        dpt.from_dlpack(DummyWithProperty())

    class DummyWithMethod:
        def __dlpack__(self):
            return None

    with pytest.raises(TypeError):
        dpt.from_dlpack(DummyWithMethod())
