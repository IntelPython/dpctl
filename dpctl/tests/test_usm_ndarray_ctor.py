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

import numbers

import numpy as np
import numpy.lib.stride_tricks as np_st
import pytest

import dpctl
import dpctl.memory as dpm
import dpctl.tensor as dpt
from dpctl.tensor._usmarray import Device


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (4,),
        (0,),
        (0, 1),
        (0, 0),
        (4, 5),
        (2, 5, 2),
        (2, 2, 2, 2, 2, 2, 2, 2),
        5,
    ],
)
@pytest.mark.parametrize("usm_type", ["shared", "host", "device"])
def test_allocate_usm_ndarray(shape, usm_type):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclCreationError:
        pytest.skip("Default SYCL queue could not be created")
    X = dpt.usm_ndarray(
        shape, dtype="d", buffer=usm_type, buffer_ctor_kwargs={"queue": q}
    )
    Xnp = np.ndarray(shape, dtype="d")
    assert X.usm_type == usm_type
    assert X.sycl_context == q.sycl_context
    assert X.sycl_device == q.sycl_device
    assert X.size == Xnp.size
    assert X.shape == Xnp.shape
    assert X.shape == X.__sycl_usm_array_interface__["shape"]


@pytest.mark.parametrize(
    "dtype",
    [
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
        b"float32",
        np.dtype("d"),
        np.half,
    ],
)
def test_dtypes(dtype):
    Xusm = dpt.usm_ndarray((1,), dtype=dtype)
    assert Xusm.itemsize == np.dtype(dtype).itemsize
    expected_fmt = (np.dtype(dtype).str)[1:]
    actual_fmt = Xusm.__sycl_usm_array_interface__["typestr"][1:]
    assert expected_fmt == actual_fmt


@pytest.mark.parametrize("dtype", ["", ">f4", "invalid", 123])
def test_dtypes_invalid(dtype):
    with pytest.raises((TypeError, ValueError)):
        dpt.usm_ndarray((1,), dtype=dtype)


def test_properties():
    """
    Test that properties execute
    """
    X = dpt.usm_ndarray((3, 4, 5), dtype="c16")
    assert isinstance(X.sycl_queue, dpctl.SyclQueue)
    assert isinstance(X.sycl_device, dpctl.SyclDevice)
    assert isinstance(X.sycl_context, dpctl.SyclContext)
    assert isinstance(X.dtype, np.dtype)
    assert isinstance(X.__sycl_usm_array_interface__, dict)
    assert isinstance(X.T, dpt.usm_ndarray)
    assert isinstance(X.imag, dpt.usm_ndarray)
    assert isinstance(X.real, dpt.usm_ndarray)
    assert isinstance(X.shape, tuple)
    assert isinstance(X.strides, tuple)
    assert X.usm_type in ("shared", "device", "host")
    assert isinstance(X.size, numbers.Integral)
    assert isinstance(X.nbytes, numbers.Integral)
    assert isinstance(X.ndim, numbers.Integral)


@pytest.mark.parametrize(
    "ind",
    [
        tuple(),
        (None,),
        (
            None,
            Ellipsis,
            None,
        ),
        (2, 2, None, 3, 4),
        (Ellipsis,),
        (None, slice(0, None, 2), Ellipsis, slice(0, None, 3)),
        (None, slice(1, None, 2), Ellipsis, slice(1, None, 3)),
        (None, slice(None, -1, -2), Ellipsis, slice(2, None, 3)),
        (
            slice(None, None, -1),
            slice(None, None, -1),
            slice(0, None, 3),
            slice(1, None, 2),
        ),
    ],
)
def test_basic_slice(ind):
    X = dpt.usm_ndarray((2 * 3, 2 * 4, 3 * 5, 2 * 7), dtype="u1")
    Xnp = np.empty(X.shape, dtype=X.dtype)
    S = X[ind]
    Snp = Xnp[ind]
    assert S.shape == Snp.shape
    assert S.strides == Snp.strides
    assert S.dtype == X.dtype


def _from_numpy(np_ary, device=None, usm_type="shared"):
    if type(np_ary) is np.ndarray:
        if np_ary.flags["FORC"]:
            x = np_ary
        else:
            x = np.ascontiguous(np_ary)
        R = dpt.usm_ndarray(
            np_ary.shape,
            dtype=np_ary.dtype,
            buffer=usm_type,
            buffer_ctor_kwargs={
                "queue": Device.create_device(device).sycl_queue
            },
        )
        R.usm_data.copy_from_host(x.reshape((-1)).view("|u1"))
        return R
    else:
        raise ValueError("Expected numpy.ndarray, got {}".format(type(np_ary)))


def _to_numpy(usm_ary):
    if type(usm_ary) is dpt.usm_ndarray:
        usm_buf = usm_ary.usm_data
        s = usm_buf.nbytes
        host_buf = usm_buf.copy_to_host().view(usm_ary.dtype)
        usm_ary_itemsize = usm_ary.itemsize
        R_offset = (
            usm_ary.__sycl_usm_array_interface__["offset"] * usm_ary_itemsize
        )
        R = np.ndarray((s,), dtype="u1", buffer=host_buf)
        R = R[R_offset:].view(usm_ary.dtype)
        R_strides = (usm_ary_itemsize * si for si in usm_ary.strides)
        return np_st.as_strided(R, shape=usm_ary.shape, strides=R_strides)
    else:
        raise ValueError(
            "Expected dpctl.tensor.usm_ndarray, got {}".format(type(usm_ary))
        )


def test_slice_constructor_1d():
    Xh = np.arange(37, dtype="i4")
    default_device = dpctl.select_default_device()
    Xusm = _from_numpy(Xh, device=default_device, usm_type="device")
    for ind in [
        slice(1, None, 2),
        slice(0, None, 3),
        slice(1, None, 3),
        slice(2, None, 3),
        slice(None, None, -1),
        slice(-2, 2, -2),
        slice(-1, 1, -2),
        slice(None, None, -13),
    ]:
        assert np.array_equal(
            _to_numpy(Xusm[ind]), Xh[ind]
        ), "Failed for {}".format(ind)


def test_slice_constructor_3d():
    Xh = np.empty((37, 24, 35), dtype="i4")
    default_device = dpctl.select_default_device()
    Xusm = _from_numpy(Xh, device=default_device, usm_type="device")
    for ind in [
        slice(1, None, 2),
        slice(0, None, 3),
        slice(1, None, 3),
        slice(2, None, 3),
        slice(None, None, -1),
        slice(-2, 2, -2),
        slice(-1, 1, -2),
        slice(None, None, -13),
        (slice(None, None, -2), Ellipsis, None, 15),
    ]:
        assert np.array_equal(
            _to_numpy(Xusm[ind]), Xh[ind]
        ), "Failed for {}".format(ind)


@pytest.mark.parametrize("usm_type", ["device", "shared", "host"])
def test_slice_suai(usm_type):
    Xh = np.arange(0, 10, dtype="u1")
    default_device = dpctl.select_default_device()
    Xusm = _from_numpy(Xh, device=default_device, usm_type=usm_type)
    for ind in [slice(2, 3, None), slice(5, 7, None), slice(3, 9, None)]:
        assert np.array_equal(
            dpm.as_usm_memory(Xusm[ind]).copy_to_host(), Xh[ind]
        ), "Failed for {}".format(ind)
