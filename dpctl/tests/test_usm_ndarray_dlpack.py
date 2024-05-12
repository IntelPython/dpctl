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

import collections
import ctypes

import pytest
from helper import skip_if_dtype_not_supported

import dpctl
import dpctl.tensor as dpt
import dpctl.tensor._dlpack as _dlp

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


@pytest.fixture
def all_root_devices():
    """
    Caches root devices. For the sake of speed
    of test suite execution, keep at most two
    devices from each platform
    """
    devs = dpctl.get_devices()
    devs_per_platform = collections.defaultdict(list)
    for dev in devs:
        devs_per_platform[dev.sycl_platform].append(dev)

    pruned = map(lambda li: li[:2], devs_per_platform.values())
    return sum(pruned, start=[])


def test_dlpack_device(usm_type, all_root_devices):
    for sycl_dev in all_root_devices:
        X = dpt.empty((64,), dtype="u1", usm_type=usm_type, device=sycl_dev)
        dev = X.__dlpack_device__()
        assert type(dev) is tuple
        assert len(dev) == 2
        assert dev[0] == device_oneAPI
        assert sycl_dev == all_root_devices[dev[1]]


def test_dlpack_exporter(typestr, usm_type, all_root_devices):
    caps_fn = ctypes.pythonapi.PyCapsule_IsValid
    caps_fn.restype = bool
    caps_fn.argtypes = [ctypes.py_object, ctypes.c_char_p]
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
def test_from_dlpack(shape, typestr, usm_type, all_root_devices):
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
def test_from_dlpack_strides(mod, typestr, usm_type, all_root_devices):
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
    v = dpt._dlpack.get_build_dlpack_version()
    assert type(v) is tuple
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


def test_dlpack_from_subdevice():
    """
    This test checks that array allocated on a sub-device,
    with memory bound to platform-default SyclContext can be
    exported and imported via DLPack.
    """
    n = 64
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
    try:
        ctx = sdevs[0].sycl_platform.default_context
    except dpctl.SyclContextCreationError:
        pytest.skip("Platform's default_context is not available")
    try:
        q = dpctl.SyclQueue(ctx, sdevs[0])
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    ar = dpt.arange(n, dtype=dpt.int32, sycl_queue=q)
    ar2 = dpt.from_dlpack(ar)
    assert ar2.sycl_device == sdevs[0]


def test_legacy_dlpack_capsule():
    try:
        x = dpt.arange(100, dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No default device available")

    legacy_ver = (0, 8)

    cap = x.__dlpack__(max_version=legacy_ver)
    y = _dlp.from_dlpack_capsule(cap)
    del cap
    assert x._pointer == y._pointer

    x = dpt.arange(100, dtype="u4")
    x2 = dpt.reshape(x, (10, 10)).mT
    cap = x2.__dlpack__(max_version=legacy_ver)
    y = _dlp.from_dlpack_capsule(cap)
    del cap
    assert x2._pointer == y._pointer
    del x2

    x = dpt.arange(100, dtype="f4")
    x2 = dpt.asarray(dpt.reshape(x, (10, 10)), order="F")
    cap = x2.__dlpack__(max_version=legacy_ver)
    y = _dlp.from_dlpack_capsule(cap)
    del cap
    assert x2._pointer == y._pointer

    x = dpt.arange(100, dtype="c8")
    x3 = x[::-2]
    cap = x3.__dlpack__(max_version=legacy_ver)
    y = _dlp.from_dlpack_capsule(cap)
    assert x3._pointer == y._pointer
    del x3, y, x
    del cap

    x = dpt.ones(100, dtype="?")
    x4 = x[::-2]
    cap = x4.__dlpack__(max_version=legacy_ver)
    y = _dlp.from_dlpack_capsule(cap)
    assert x4._pointer == y._pointer
    del x4, y, x
    del cap


def test_versioned_dlpack_capsule():
    try:
        x = dpt.arange(100, dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No default device available")

    max_supported_ver = _dlp.get_build_dlpack_version()
    cap = x.__dlpack__(max_version=max_supported_ver)
    y = _dlp.from_dlpack_versioned_capsule(cap)
    del cap
    assert x._pointer == y._pointer

    x2 = dpt.asarray(dpt.reshape(x, (10, 10)), order="F")
    cap = x2.__dlpack__(max_version=max_supported_ver)
    y = _dlp.from_dlpack_versioned_capsule(cap)
    del cap
    assert x2._pointer == y._pointer
    del x2

    x3 = x[::-2]
    cap = x3.__dlpack__(max_version=max_supported_ver)
    y = _dlp.from_dlpack_versioned_capsule(cap)
    assert x3._pointer == y._pointer
    del x3, y, x
    del cap

    # read-only array
    x = dpt.arange(100, dtype="i4")
    x.flags["W"] = False
    cap = x.__dlpack__(max_version=max_supported_ver)
    y = _dlp.from_dlpack_versioned_capsule(cap)
    assert x._pointer == y._pointer
    assert not y.flags.writable

    # read-only array, and copy
    cap = x.__dlpack__(max_version=max_supported_ver, copy=True)
    y = _dlp.from_dlpack_versioned_capsule(cap)
    assert x._pointer != y._pointer
    assert not y.flags.writable


def test_from_dlpack_kwargs():
    try:
        x = dpt.arange(100, dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No default device available")

    y = dpt.from_dlpack(x, copy=True)
    assert x._pointer != y._pointer

    z = dpt.from_dlpack(x, device=x.sycl_device)
    assert z._pointer == x._pointer


def test_dlpack_deleters():
    try:
        x = dpt.arange(100, dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No default device available")

    legacy_ver = (0, 8)
    cap = x.__dlpack__(max_version=legacy_ver)
    del cap

    max_supported_ver = _dlp.get_build_dlpack_version()
    cap = x.__dlpack__(max_version=max_supported_ver)
    del cap


def test_from_dlpack_device():
    try:
        x = dpt.arange(100, dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No default device available")

    out = dpt.from_dlpack(x, device=x.__dlpack_device__())
    assert x.device == out.device
    assert x._pointer == out._pointer

    out = dpt.from_dlpack(x, device=x.device)
    assert x.device == out.device
    assert x._pointer == out._pointer

    out = dpt.from_dlpack(x, device=x.sycl_device)
    assert x.device == out.device
    assert x._pointer == out._pointer


def test_used_dlpack_capsule():
    try:
        x = dpt.arange(100, dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No default device available")

    legacy_ver = (0, 8)
    cap = x.__dlpack__(max_version=legacy_ver)
    _dlp.from_dlpack_capsule(cap)
    with pytest.raises(
        ValueError,
        match="A DLPack tensor object can not be consumed multiple times",
    ):
        _dlp.from_dlpack_capsule(cap)
    del cap

    max_supported_ver = _dlp.get_build_dlpack_version()
    cap = x.__dlpack__(max_version=max_supported_ver)
    _dlp.from_dlpack_versioned_capsule(cap)
    with pytest.raises(
        ValueError,
        match="A DLPack tensor object can not be consumed multiple times",
    ):
        _dlp.from_dlpack_versioned_capsule(cap)
    del cap


def test_dlpack_size_0():
    try:
        x = dpt.ones(0, dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No default device available")

    legacy_ver = (0, 8)
    cap = x.__dlpack__(max_version=legacy_ver)
    y = _dlp.from_dlpack_capsule(cap)
    assert y._pointer == x._pointer

    max_supported_ver = _dlp.get_build_dlpack_version()
    cap = x.__dlpack__(max_version=max_supported_ver)
    y = _dlp.from_dlpack_versioned_capsule(cap)
    assert y._pointer == x._pointer


def test_dlpack_max_version_validation():
    try:
        x = dpt.ones(100, dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No default device available")

    with pytest.raises(
        TypeError,
        match=r"`__dlpack__` expects `max_version` to be a "
        r"2-tuple of integers `\(major, minor\)`, instead "
        r"got .*",
    ):
        x.__dlpack__(max_version=1)


def test_dlpack_kwargs():
    try:
        q1 = dpctl.SyclQueue()
        q2 = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Could not create default queues")
    x = dpt.arange(100, dtype="i4", sycl_queue=q1)

    legacy_ver = (0, 8)
    cap = x.__dlpack__(stream=q2, max_version=legacy_ver, copy=True)
    # `copy` ignored for legacy path
    y = _dlp.from_dlpack_capsule(cap)
    assert y._pointer == x._pointer
    del x, y
    del cap

    x1 = dpt.arange(100, dtype="i4", sycl_queue=q1)
    max_supported_ver = _dlp.get_build_dlpack_version()
    cap = x1.__dlpack__(stream=q2, max_version=max_supported_ver, copy=False)
    y = _dlp.from_dlpack_versioned_capsule(cap)
    assert y._pointer == x1._pointer
    del x1, y
    del cap

    x2 = dpt.arange(100, dtype="i4", sycl_queue=q1)
    cap = x2.__dlpack__(stream=q2, max_version=max_supported_ver, copy=True)
    y = _dlp.from_dlpack_versioned_capsule(cap)
    assert y._pointer != x2._pointer
    del x2, y
    del cap
