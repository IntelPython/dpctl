#                       Data Parallel Control (dpctl)
#
#  Copyright 2020-2023 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import ctypes

import numpy as np
import pytest

import dpctl
import dpctl.tensor as dpt
from dpctl.tensor._type_utils import _can_cast
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _all_dtypes, _compare_dtypes, _usm_types


@pytest.mark.parametrize("op1_dtype", _all_dtypes[1:])
@pytest.mark.parametrize("op2_dtype", _all_dtypes[1:])
def test_subtract_dtype_matrix(op1_dtype, op2_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op1_dtype, q)
    skip_if_dtype_not_supported(op2_dtype, q)

    sz = 127
    ar1 = dpt.ones(sz, dtype=op1_dtype)
    ar2 = dpt.ones_like(ar1, dtype=op2_dtype)

    r = dpt.subtract(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected_dtype = np.subtract(
        np.zeros(1, dtype=op1_dtype), np.zeros(1, dtype=op2_dtype)
    ).dtype
    assert _compare_dtypes(r.dtype, expected_dtype, sycl_queue=q)
    assert r.shape == ar1.shape
    assert (dpt.asnumpy(r) == np.full(r.shape, 0, dtype=r.dtype)).all()
    assert r.sycl_queue == ar1.sycl_queue

    r2 = dpt.empty_like(ar1, dtype=r.dtype)
    dpt.subtract(ar1, ar2, out=r2)
    assert (dpt.asnumpy(r2) == np.full(r2.shape, 0, dtype=r2.dtype)).all()

    ar3 = dpt.ones(sz, dtype=op1_dtype)
    ar4 = dpt.ones(2 * sz, dtype=op2_dtype)

    r = dpt.subtract(ar3[::-1], ar4[::2])
    assert isinstance(r, dpt.usm_ndarray)
    expected_dtype = np.subtract(
        np.zeros(1, dtype=op1_dtype), np.zeros(1, dtype=op2_dtype)
    ).dtype
    assert _compare_dtypes(r.dtype, expected_dtype, sycl_queue=q)
    assert r.shape == ar3.shape
    assert (dpt.asnumpy(r) == np.full(r.shape, 0, dtype=r.dtype)).all()

    r2 = dpt.empty_like(ar1, dtype=r.dtype)
    dpt.subtract(ar3[::-1], ar4[::2], out=r2)
    assert (dpt.asnumpy(r2) == np.full(r2.shape, 0, dtype=r2.dtype)).all()


@pytest.mark.parametrize("op1_usm_type", _usm_types)
@pytest.mark.parametrize("op2_usm_type", _usm_types)
def test_subtract_usm_type_matrix(op1_usm_type, op2_usm_type):
    get_queue_or_skip()

    sz = 128
    ar1 = dpt.ones(sz, dtype="i4", usm_type=op1_usm_type)
    ar2 = dpt.ones_like(ar1, dtype="i4", usm_type=op2_usm_type)

    r = dpt.subtract(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected_usm_type = dpctl.utils.get_coerced_usm_type(
        (op1_usm_type, op2_usm_type)
    )
    assert r.usm_type == expected_usm_type


def test_subtract_order():
    get_queue_or_skip()

    test_shape = (
        20,
        20,
    )
    test_shape2 = tuple(2 * dim for dim in test_shape)
    n = test_shape[-1]

    for dt1, dt2 in zip(["i4", "i4", "f4"], ["i4", "f4", "i4"]):
        ar1 = dpt.ones(test_shape, dtype=dt1, order="C")
        ar2 = dpt.ones(test_shape, dtype=dt2, order="C")
        r1 = dpt.subtract(ar1, ar2, order="C")
        assert r1.flags.c_contiguous
        r2 = dpt.subtract(ar1, ar2, order="F")
        assert r2.flags.f_contiguous
        r3 = dpt.subtract(ar1, ar2, order="A")
        assert r3.flags.c_contiguous
        r4 = dpt.subtract(ar1, ar2, order="K")
        assert r4.flags.c_contiguous

        ar1 = dpt.ones(test_shape, dtype=dt1, order="F")
        ar2 = dpt.ones(test_shape, dtype=dt2, order="F")
        r1 = dpt.subtract(ar1, ar2, order="C")
        assert r1.flags.c_contiguous
        r2 = dpt.subtract(ar1, ar2, order="F")
        assert r2.flags.f_contiguous
        r3 = dpt.subtract(ar1, ar2, order="A")
        assert r3.flags.f_contiguous
        r4 = dpt.subtract(ar1, ar2, order="K")
        assert r4.flags.f_contiguous

        ar1 = dpt.ones(test_shape2, dtype=dt1, order="C")[:20, ::-2]
        ar2 = dpt.ones(test_shape2, dtype=dt2, order="C")[:20, ::-2]
        r4 = dpt.subtract(ar1, ar2, order="K")
        assert r4.strides == (n, -1)
        r5 = dpt.subtract(ar1, ar2, order="C")
        assert r5.strides == (n, 1)

        ar1 = dpt.ones(test_shape2, dtype=dt1, order="C")[:20, ::-2].mT
        ar2 = dpt.ones(test_shape2, dtype=dt2, order="C")[:20, ::-2].mT
        r4 = dpt.subtract(ar1, ar2, order="K")
        assert r4.strides == (-1, n)
        r5 = dpt.subtract(ar1, ar2, order="C")
        assert r5.strides == (n, 1)


def test_subtract_broadcasting():
    get_queue_or_skip()

    m = dpt.ones((100, 5), dtype="i4")
    v = dpt.arange(5, dtype="i4")

    r = dpt.subtract(m, v)
    assert (
        dpt.asnumpy(r) == np.arange(1, -4, step=-1, dtype="i4")[np.newaxis, :]
    ).all()

    r2 = dpt.subtract(v, m)
    assert (
        dpt.asnumpy(r2) == np.arange(-1, 4, dtype="i4")[np.newaxis, :]
    ).all()


@pytest.mark.parametrize("arr_dt", _all_dtypes)
def test_subtract_python_scalar(arr_dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arr_dt, q)

    X = dpt.zeros((10, 10), dtype=arr_dt, sycl_queue=q)
    py_zeros = (
        bool(0),
        int(0),
        float(0),
        complex(0),
        np.float32(0),
        ctypes.c_int(0),
    )
    for sc in py_zeros:
        R = dpt.subtract(X, sc)
        assert isinstance(R, dpt.usm_ndarray)
        R = dpt.subtract(sc, X)
        assert isinstance(R, dpt.usm_ndarray)


@pytest.mark.parametrize("dtype", _all_dtypes[1:])
def test_subtract_inplace_python_scalar(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)
    X = dpt.zeros((10, 10), dtype=dtype, sycl_queue=q)
    dt_kind = X.dtype.kind
    if dt_kind in "ui":
        X -= int(0)
    elif dt_kind == "f":
        X -= float(0)
    elif dt_kind == "c":
        X -= complex(0)


@pytest.mark.parametrize("op1_dtype", _all_dtypes[1:])
@pytest.mark.parametrize("op2_dtype", _all_dtypes[1:])
def test_subtract_inplace_dtype_matrix(op1_dtype, op2_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op1_dtype, q)
    skip_if_dtype_not_supported(op2_dtype, q)

    sz = 127
    ar1 = dpt.ones(sz, dtype=op1_dtype)
    ar2 = dpt.ones_like(ar1, dtype=op2_dtype)

    dev = q.sycl_device
    _fp16 = dev.has_aspect_fp16
    _fp64 = dev.has_aspect_fp64
    if _can_cast(ar2.dtype, ar1.dtype, _fp16, _fp64):
        ar1 -= ar2
        assert (dpt.asnumpy(ar1) == np.zeros(ar1.shape, dtype=ar1.dtype)).all()

        ar3 = dpt.ones(sz, dtype=op1_dtype)
        ar4 = dpt.ones(2 * sz, dtype=op2_dtype)

        ar3[::-1] -= ar4[::2]
        assert (dpt.asnumpy(ar3) == np.zeros(ar3.shape, dtype=ar3.dtype)).all()

    else:
        with pytest.raises(TypeError):
            ar1 -= ar2


def test_subtract_inplace_broadcasting():
    get_queue_or_skip()

    m = dpt.ones((100, 5), dtype="i4")
    v = dpt.arange(5, dtype="i4")

    m -= v
    assert (
        dpt.asnumpy(m) == np.arange(1, -4, step=-1, dtype="i4")[np.newaxis, :]
    ).all()
