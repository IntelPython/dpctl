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
    d = dpctl.SyclDevice()
    if d.is_host:
        pytest.skip(
            "Skip test of host device, which only "
            "supports host USM allocations"
        )
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
    Y = dpt.asarray(Xnp, usm_type="device")
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
    Y = dpt.asarray(X, usm_type="device")
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

    X = dpt.empty((2, 3, 4), dtype="f4")
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
    with pytest.raises(TypeError):
        # sycl_queue type is not right
        dpt.asarray([1], sycl_queue=dpctl.SyclContext())
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

    Y = dpt.asarray(5)
    assert Y.dtype == np.dtype(int)
    Y = dpt.asarray(5.2)
    if Y.sycl_device.has_aspect_fp64:
        assert Y.dtype == np.dtype(float)
    else:
        assert Y.dtype == np.dtype(np.float32)
    Y = dpt.asarray(np.float32(2.3))
    assert Y.dtype == np.dtype(np.float32)
    Y = dpt.asarray(1.0j)
    if Y.sycl_device.has_aspect_fp64:
        assert Y.dtype == np.dtype(complex)
    else:
        assert Y.dtype == np.dtype(np.complex64)
    Y = dpt.asarray(ctypes.c_int(8))
    assert Y.dtype == np.dtype(ctypes.c_int)


def test_asarray_copy_false():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Could not create a queue")
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
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Could not create a queue")
    Xnp = np.array([1, 2, 3], dtype=object)
    with pytest.raises(TypeError):
        dpt.asarray(Xnp, sycl_queue=q)
