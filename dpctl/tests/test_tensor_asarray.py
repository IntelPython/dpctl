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

import numpy as np
import pytest
from helper import get_queue_or_skip

import dpctl
import dpctl.tensor as dpt


@pytest.mark.parametrize(
    "src_usm_type, dst_usm_type",
    [
        ("device", "shared"),
        ("device", "host"),
        ("shared", "device"),
        ("shared", "host"),
        ("host", "device"),
        ("host", "shared"),
    ],
)
def test_asarray_change_usm_type(src_usm_type, dst_usm_type):
    try:
        d = dpctl.SyclDevice()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    X = dpt.empty(10, dtype="u1", usm_type=src_usm_type)
    Y = dpt.asarray(X, usm_type=dst_usm_type)
    assert X.shape == Y.shape
    assert X.usm_type == src_usm_type
    assert Y.usm_type == dst_usm_type

    with pytest.raises(ValueError):
        # zero copy is not possible
        dpt.asarray(X, usm_type=dst_usm_type, copy=False)

    Y = dpt.asarray(X, usm_type=dst_usm_type, sycl_queue=X.sycl_queue)
    assert X.shape == Y.shape
    assert Y.usm_type == dst_usm_type

    Y = dpt.asarray(
        X,
        usm_type=dst_usm_type,
        sycl_queue=X.sycl_queue,
        device=d.get_filter_string(),
    )
    assert X.shape == Y.shape
    assert Y.usm_type == dst_usm_type


def test_asarray_from_numpy():
    Xnp = np.arange(10)
    try:
        Y = dpt.asarray(Xnp, usm_type="device")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    assert type(Y) is dpt.usm_ndarray
    assert Y.shape == Xnp.shape
    assert Y.dtype == Xnp.dtype
    # Fortan contiguous case
    Xnp = np.array([[1, 2, 3], [4, 5, 6]], dtype="f4", order="F")
    Y = dpt.asarray(Xnp, usm_type="shared")
    assert type(Y) is dpt.usm_ndarray
    assert Y.shape == Xnp.shape
    assert Y.dtype == Xnp.dtype
    # general strided case
    Xnp = np.array([[1, 2, 3], [4, 5, 6]], dtype="i8")
    Y = dpt.asarray(Xnp[::-1, ::-1], usm_type="host")
    assert type(Y) is dpt.usm_ndarray
    assert Y.shape == Xnp.shape
    assert Y.dtype == Xnp.dtype


def test_asarray_from_sequence():
    X = [1, 2, 3]
    try:
        Y = dpt.asarray(X, usm_type="device")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    assert type(Y) is dpt.usm_ndarray

    X = [(1, 1), (2.0, 2.0 + 1.0j), range(4, 6), np.array([3, 4], dtype="c16")]
    Y = dpt.asarray(X, usm_type="device")
    assert type(Y) is dpt.usm_ndarray
    assert Y.ndim == 2
    assert Y.shape == (len(X), 2)

    X = []
    Y = dpt.asarray(X, usm_type="device")
    assert type(Y) is dpt.usm_ndarray
    assert Y.shape == (0,)

    X = [[], []]
    Y = dpt.asarray(X, usm_type="device")
    assert type(Y) is dpt.usm_ndarray
    assert Y.shape == (2, 0)

    X = [True, False]
    Y = dpt.asarray(X, usm_type="device")
    assert type(Y) is dpt.usm_ndarray
    assert Y.dtype.kind == "b"


def test_asarray_from_object_with_suai():
    """Test that asarray can deal with opaque objects implementing SUAI"""

    class Dummy:
        def __init__(self, obj, iface):
            self.obj = obj
            self.__sycl_usm_array_interface__ = iface

    try:
        X = dpt.empty((2, 3, 4), dtype="f4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    Y = dpt.asarray(Dummy(X, X.__sycl_usm_array_interface__))
    assert Y.shape == X.shape
    assert X.usm_type == Y.usm_type
    assert X.dtype == Y.dtype
    assert X.sycl_device == Y.sycl_device


def test_asarray_input_validation():
    with pytest.raises(TypeError):
        # copy keyword is not of right type
        dpt.asarray([1], copy="invalid")
    with pytest.raises(TypeError):
        # order keyword is not valid
        dpt.asarray([1], order=1)
    with pytest.raises(TypeError):
        # dtype is not valid
        dpt.asarray([1], dtype="invalid")
    with pytest.raises(ValueError):
        # unexpected value of order
        dpt.asarray([1], order="Z")
    with pytest.raises(TypeError):
        # usm_type is of wrong type
        dpt.asarray([1], usm_type=dict())
    with pytest.raises(ValueError):
        # usm_type has wrong value
        dpt.asarray([1], usm_type="mistake")
    try:
        wrong_queue_type = dpctl.SyclContext()
    except dpctl.SyclContextCreationError:
        # use any other type
        wrong_queue_type = Ellipsis
    with pytest.raises(TypeError):
        # sycl_queue type is not right
        dpt.asarray([1], sycl_queue=wrong_queue_type)
    with pytest.raises(ValueError):
        # sequence is not rectangular
        dpt.asarray([[1], 2])


def test_asarray_input_validation2():
    d = dpctl.get_devices()
    if len(d) < 2:
        pytest.skip("Not enough SYCL devices available")

    d0, d1 = d[:2]
    try:
        q0 = dpctl.SyclQueue(d0)
    except dpctl.SyclQueueCreationError:
        pytest.skip(f"SyclQueue could not be created for {d0}")
    try:
        q1 = dpctl.SyclQueue(d1)
    except dpctl.SyclQueueCreationError:
        pytest.skip(f"SyclQueue could not be created for {d1}")
    with pytest.raises(TypeError):
        dpt.asarray([1, 2], sycl_queue=q0, device=q1)


def test_asarray_scalars():
    import ctypes

    try:
        Y = dpt.asarray(5)
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    assert Y.dtype == dpt.dtype(int)
    Y = dpt.asarray(5.2)
    if Y.sycl_device.has_aspect_fp64:
        assert Y.dtype == dpt.dtype(float)
    else:
        assert Y.dtype == dpt.dtype(dpt.float32)
    Y = dpt.asarray(np.float32(2.3))
    assert Y.dtype == dpt.dtype(dpt.float32)
    Y = dpt.asarray(1.0j)
    if Y.sycl_device.has_aspect_fp64:
        assert Y.dtype == dpt.dtype(complex)
    else:
        assert Y.dtype == dpt.dtype(dpt.complex64)
    Y = dpt.asarray(ctypes.c_int(8))
    assert Y.dtype == dpt.dtype(ctypes.c_int)


def test_asarray_copy_false():
    q = get_queue_or_skip()
    rng = np.random.default_rng()
    Xnp = rng.integers(low=-255, high=255, size=(10, 4), dtype=np.int64)
    X = dpt.from_numpy(Xnp, usm_type="device", sycl_queue=q)
    Y1 = dpt.asarray(X, copy=False, order="K")
    assert Y1 is X
    Y1c = dpt.asarray(X, copy=True, order="K")
    assert not (Y1c is X)
    Y2 = dpt.asarray(X, copy=False, order="C")
    assert Y2 is X
    Y3 = dpt.asarray(X, copy=False, order="A")
    assert Y3 is X
    with pytest.raises(ValueError):
        Y1 = dpt.asarray(X, copy=False, order="F")
    Xf = dpt.empty(
        X.shape,
        dtype=X.dtype,
        usm_type="device",
        sycl_queue=X.sycl_queue,
        order="F",
    )
    Xf[:] = X
    Y4 = dpt.asarray(Xf, copy=False, order="K")
    assert Y4 is Xf
    Y5 = dpt.asarray(Xf, copy=False, order="F")
    assert Y5 is Xf
    Y6 = dpt.asarray(Xf, copy=False, order="A")
    assert Y6 is Xf
    with pytest.raises(ValueError):
        dpt.asarray(Xf, copy=False, order="C")


def test_asarray_invalid_dtype():
    q = get_queue_or_skip()
    Xnp = np.array([1, 2, 3], dtype=object)
    with pytest.raises(TypeError):
        dpt.asarray(Xnp, sycl_queue=q)


def test_asarray_cross_device():
    q = get_queue_or_skip()
    qprof = dpctl.SyclQueue(property="enable_profiling")
    x = dpt.empty(10, dtype="i8", sycl_queue=q)
    y = dpt.asarray(x, sycl_queue=qprof)
    assert y.sycl_queue == qprof


def test_asarray_seq_of_arrays_simple():
    get_queue_or_skip()
    r = dpt.arange(10)
    m = dpt.asarray(
        [
            r,
        ]
        * 4
    )
    assert m.shape == (4,) + r.shape
    assert m.dtype == r.dtype
    assert m.device == r.device


def test_asarray_seq_of_arrays():
    get_queue_or_skip()
    m = dpt.ones((2, 4), dtype="i4")
    w = dpt.zeros(4)
    v = dpt.full(4, -1)
    ar = dpt.asarray([m, [w, v]])
    assert ar.shape == (2, 2, 4)
    assert ar.device == m.device
    assert ar.device == w.device
    assert ar.device == v.device


def test_asarray_seq_of_array_different_queue():
    get_queue_or_skip()
    m = dpt.ones((2, 4), dtype="i4")
    w = dpt.zeros(4)
    v = dpt.full(4, -1)
    qprof = dpctl.SyclQueue(property="enable_profiling")
    ar = dpt.asarray([m, [w, v]], sycl_queue=qprof)
    assert ar.shape == (2, 2, 4)
    assert ar.sycl_queue == qprof


def test_asarray_seq_of_suai():
    get_queue_or_skip()

    class Dummy:
        def __init__(self, obj, iface):
            self.obj = obj
            self.__sycl_usm_array_interface__ = iface

    o = dpt.empty(0, usm_type="shared")
    d = Dummy(o, o.__sycl_usm_array_interface__)
    x = dpt.asarray(d)
    assert x.shape == (0,)
    assert x.usm_type == o.usm_type
    assert x._pointer == o._pointer
    assert x.sycl_queue == o.sycl_queue

    x = dpt.asarray([d, d])
    assert x.shape == (2, 0)
    assert x.usm_type == o.usm_type
    assert x.sycl_queue == o.sycl_queue


def test_asarray_seq_of_suai_different_queue():
    q = get_queue_or_skip()

    class Dummy:
        def __init__(self, obj, iface):
            self.obj = obj
            self.__sycl_usm_array_interface__ = iface

        @property
        def shape(self):
            return self.__sycl_usm_array_interface__["shape"]

    q2 = dpctl.SyclQueue()
    assert q != q2
    o = dpt.empty((2, 2), usm_type="shared", sycl_queue=q2)
    d = Dummy(o, o.__sycl_usm_array_interface__)

    x = dpt.asarray(d, sycl_queue=q)
    assert x.sycl_queue == q
    assert x.shape == d.shape
    x = dpt.asarray([d], sycl_queue=q)
    assert x.sycl_queue == q
    assert x.shape == (1,) + d.shape
    x = dpt.asarray([d, d], sycl_queue=q)
    assert x.sycl_queue == q
    assert x.shape == (2,) + d.shape


def test_asarray_seq_of_arrays_on_different_queues():
    q = get_queue_or_skip()

    m = dpt.empty((2, 4), dtype="i2", sycl_queue=q)
    q2 = dpctl.SyclQueue()
    w = dpt.empty(4, dtype="i1", sycl_queue=q2)
    q3 = dpctl.SyclQueue()
    py_seq = [
        0,
    ] * w.shape[0]
    res = dpt.asarray([m, [w, py_seq]], sycl_queue=q3)
    assert res.sycl_queue == q3
    assert dpt.isdtype(res.dtype, "integral")

    res = dpt.asarray([m, [w, range(w.shape[0])]], sycl_queue=q3)
    assert res.sycl_queue == q3
    assert dpt.isdtype(res.dtype, "integral")

    res = dpt.asarray([m, [w, w]], sycl_queue=q)
    assert res.sycl_queue == q
    assert dpt.isdtype(res.dtype, "integral")

    res = dpt.asarray([m, [w, dpt.asnumpy(w)]], sycl_queue=q2)
    assert res.sycl_queue == q2
    assert dpt.isdtype(res.dtype, "integral")

    res = dpt.asarray([w, dpt.asnumpy(w)])
    assert res.sycl_queue == w.sycl_queue
    assert dpt.isdtype(res.dtype, "integral")

    with pytest.raises(dpctl.utils.ExecutionPlacementError):
        dpt.asarray([m, [w, py_seq]])


def test_ulonglong_gh_1167():
    get_queue_or_skip()
    x = dpt.asarray(9223372036854775807, dtype="u8")
    assert x.dtype == dpt.uint64
    x = dpt.asarray(9223372036854775808, dtype="u8")
    assert x.dtype == dpt.uint64
