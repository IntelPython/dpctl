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

import ctypes

import pytest
from helper import skip_if_dtype_not_supported

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
        skip_if_dtype_not_supported(typestr, sycl_dev)
        X = dpt.empty((64,), dtype=typestr, usm_type=usm_type, device=sycl_dev)
        caps = X.__dlpack__()
        assert caps_fn(caps, b"dltensor")
        Y = X[::2]
        caps2 = Y.__dlpack__()
        assert caps_fn(caps2, b"dltensor")


def test_dlpack_exporter_empty(typestr, usm_type):
    caps_fn = ctypes.pythonapi.PyCapsule_IsValid
    caps_fn.restype = bool
    caps_fn.argtypes = [ctypes.py_object, ctypes.c_char_p]
    try:
        sycl_dev = dpctl.select_default_device()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    skip_if_dtype_not_supported(typestr, sycl_dev)
    X = dpt.empty((0,), dtype=typestr, usm_type=usm_type, device=sycl_dev)
    caps = X.__dlpack__()
    assert caps_fn(caps, b"dltensor")
    Y = dpt.empty(
        (
            1,
            0,
        ),
        dtype=typestr,
        usm_type=usm_type,
        device=sycl_dev,
    )
    caps = Y.__dlpack__()
    assert caps_fn(caps, b"dltensor")


def test_dlpack_exporter_stream():
    try:
        q1 = dpctl.SyclQueue()
        q2 = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Could not create default queues")
    X = dpt.empty((64,), dtype="u1", sycl_queue=q1)
    cap1 = X.__dlpack__(stream=q1)
    cap2 = X.__dlpack__(stream=q2)
    assert type(cap1) is type(cap2)


@pytest.mark.parametrize("shape", [tuple(), (2,), (3, 0, 1), (2, 2, 2)])
def test_from_dlpack(shape, typestr, usm_type):
    all_root_devices = dpctl.get_devices()
    for sycl_dev in all_root_devices:
        skip_if_dtype_not_supported(typestr, sycl_dev)
        X = dpt.empty(shape, dtype=typestr, usm_type=usm_type, device=sycl_dev)
        Y = dpt.from_dlpack(X)
        assert X.shape == Y.shape
        assert X.dtype == Y.dtype
        assert X.usm_type == Y.usm_type
        assert X._pointer == Y._pointer
        # we can only expect device to round-trip for USM-device and
        # USM-shared allocations, which are made for specific device
        assert (Y.usm_type == "host") or (X.sycl_device == Y.sycl_device)
        if Y.ndim:
            V = Y[::-1]
            W = dpt.from_dlpack(V)
            assert V.strides == W.strides


@pytest.mark.parametrize("mod", [2, 5])
def test_from_dlpack_strides(mod, typestr, usm_type):
    all_root_devices = dpctl.get_devices()
    for sycl_dev in all_root_devices:
        skip_if_dtype_not_supported(typestr, sycl_dev)
        X0 = dpt.empty(
            3 * mod, dtype=typestr, usm_type=usm_type, device=sycl_dev
        )
        for start in range(mod):
            X = X0[slice(-start - 1, None, -mod)]
            Y = dpt.from_dlpack(X)
            assert X.shape == Y.shape
            assert X.dtype == Y.dtype
            assert X.usm_type == Y.usm_type
            assert X._pointer == Y._pointer
            # we can only expect device to round-trip for USM-device and
            # USM-shared allocations, which are made for specific device
            assert (Y.usm_type == "host") or (X.sycl_device == Y.sycl_device)
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


def test_from_dlpack_fortran_contig_array_roundtripping():
    """Based on examples from issue gh-1241"""
    n0, n1 = 3, 5
    try:
        ar1d = dpt.arange(n0 * n1, dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No default device available")
    ar2d_c = dpt.reshape(ar1d, (n0, n1), order="C")
    ar2d_f = dpt.asarray(ar2d_c, order="F")
    ar2d_r = dpt.from_dlpack(ar2d_f)

    assert dpt.all(dpt.equal(ar2d_f, ar2d_r))
    assert dpt.all(dpt.equal(ar2d_c, ar2d_r))
