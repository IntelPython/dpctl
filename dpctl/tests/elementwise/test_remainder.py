#                       Data Parallel Control (dpctl)
#
#  Copyright 2020-2024 Intel Corporation
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

from .utils import _compare_dtypes, _no_complex_dtypes, _usm_types


@pytest.mark.parametrize("op1_dtype", _no_complex_dtypes)
@pytest.mark.parametrize("op2_dtype", _no_complex_dtypes)
def test_remainder_dtype_matrix(op1_dtype, op2_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op1_dtype, q)
    skip_if_dtype_not_supported(op2_dtype, q)

    sz = 127
    ar1 = dpt.ones(sz, dtype=op1_dtype)
    ar2 = dpt.ones_like(ar1, dtype=op2_dtype)

    r = dpt.remainder(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.remainder(
        np.ones(1, dtype=op1_dtype), np.ones(1, dtype=op2_dtype)
    )
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar1.shape
    assert (dpt.asnumpy(r) == expected.astype(r.dtype)).all()
    assert r.sycl_queue == ar1.sycl_queue

    ar3 = dpt.ones(sz, dtype=op1_dtype)
    ar4 = dpt.ones(2 * sz, dtype=op2_dtype)

    r = dpt.remainder(ar3[::-1], ar4[::2])
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.remainder(
        np.ones(1, dtype=op1_dtype), np.ones(1, dtype=op2_dtype)
    )
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar3.shape
    assert (dpt.asnumpy(r) == expected.astype(r.dtype)).all()


@pytest.mark.parametrize("op1_usm_type", _usm_types)
@pytest.mark.parametrize("op2_usm_type", _usm_types)
def test_remainder_usm_type_matrix(op1_usm_type, op2_usm_type):
    get_queue_or_skip()

    sz = 128
    ar1 = dpt.ones(sz, dtype="i4", usm_type=op1_usm_type)
    ar2 = dpt.ones_like(ar1, dtype="i4", usm_type=op2_usm_type)

    r = dpt.remainder(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected_usm_type = dpctl.utils.get_coerced_usm_type(
        (op1_usm_type, op2_usm_type)
    )
    assert r.usm_type == expected_usm_type


def test_remainder_order():
    get_queue_or_skip()

    ar1 = dpt.ones((20, 20), dtype="i4", order="C")
    ar2 = dpt.ones((20, 20), dtype="i4", order="C")
    r1 = dpt.remainder(ar1, ar2, order="C")
    assert r1.flags.c_contiguous
    r2 = dpt.remainder(ar1, ar2, order="F")
    assert r2.flags.f_contiguous
    r3 = dpt.remainder(ar1, ar2, order="A")
    assert r3.flags.c_contiguous
    r4 = dpt.remainder(ar1, ar2, order="K")
    assert r4.flags.c_contiguous

    ar1 = dpt.ones((20, 20), dtype="i4", order="F")
    ar2 = dpt.ones((20, 20), dtype="i4", order="F")
    r1 = dpt.remainder(ar1, ar2, order="C")
    assert r1.flags.c_contiguous
    r2 = dpt.remainder(ar1, ar2, order="F")
    assert r2.flags.f_contiguous
    r3 = dpt.remainder(ar1, ar2, order="A")
    assert r3.flags.f_contiguous
    r4 = dpt.remainder(ar1, ar2, order="K")
    assert r4.flags.f_contiguous

    ar1 = dpt.ones((40, 40), dtype="i4", order="C")[:20, ::-2]
    ar2 = dpt.ones((40, 40), dtype="i4", order="C")[:20, ::-2]
    r4 = dpt.remainder(ar1, ar2, order="K")
    assert r4.strides == (20, -1)

    ar1 = dpt.ones((40, 40), dtype="i4", order="C")[:20, ::-2].mT
    ar2 = dpt.ones((40, 40), dtype="i4", order="C")[:20, ::-2].mT
    r4 = dpt.remainder(ar1, ar2, order="K")
    assert r4.strides == (-1, 20)


@pytest.mark.parametrize("dt", _no_complex_dtypes[1:8:2])
def test_remainder_negative_integers(dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x = dpt.arange(-5, -1, 1, dtype=dt, sycl_queue=q)
    x_np = np.arange(-5, -1, 1, dtype=dt)
    val = 3

    r1 = dpt.remainder(x, val)
    expected = np.remainder(x_np, val)
    assert (dpt.asnumpy(r1) == expected).all()

    r2 = dpt.remainder(val, x)
    expected = np.remainder(val, x_np)
    assert (dpt.asnumpy(r2) == expected).all()


def test_remainder_integer_zero():
    get_queue_or_skip()

    for dt in ["i4", "u4"]:
        x = dpt.ones(1, dtype=dt)
        y = dpt.zeros_like(x)

        assert (dpt.asnumpy(dpt.remainder(x, y)) == np.zeros(1, dtype=dt)).all()

        x = dpt.astype(x, dt)
        y = dpt.zeros_like(x)

        assert (dpt.asnumpy(dpt.remainder(x, y)) == np.zeros(1, dtype=dt)).all()


@pytest.mark.parametrize("dt", _no_complex_dtypes[9:])
def test_remainder_negative_floats(dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x = dpt.linspace(-5, 5, 20, dtype=dt, sycl_queue=q)
    x_np = np.linspace(-5, 5, 20, dtype=dt)
    val = 3

    tol = 8 * dpt.finfo(dt).resolution

    r1 = dpt.remainder(x, val)
    expected = np.remainder(x_np, val)
    with np.errstate(invalid="ignore"):
        np.allclose(
            dpt.asnumpy(r1), expected, rtol=tol, atol=tol, equal_nan=True
        )

    r2 = dpt.remainder(val, x)
    expected = np.remainder(val, x_np)
    with np.errstate(invalid="ignore"):
        np.allclose(
            dpt.asnumpy(r2), expected, rtol=tol, atol=tol, equal_nan=True
        )


def test_remainder_special_cases():
    get_queue_or_skip()

    lhs = [dpt.nan, dpt.inf, 0.0, -0.0, -0.0, 1.0, dpt.inf, -dpt.inf]
    rhs = [dpt.nan, dpt.inf, -0.0, 1.0, 1.0, 0.0, 1.0, -1.0]

    x, y = dpt.asarray(lhs, dtype="f4"), dpt.asarray(rhs, dtype="f4")

    x_np, y_np = np.asarray(lhs, dtype="f4"), np.asarray(rhs, dtype="f4")

    res = dpt.remainder(x, y)

    with np.errstate(invalid="ignore"):
        np.allclose(dpt.asnumpy(res), np.remainder(x_np, y_np))


@pytest.mark.parametrize("arr_dt", _no_complex_dtypes)
def test_remainder_python_scalar(arr_dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arr_dt, q)

    X = dpt.ones((10, 10), dtype=arr_dt, sycl_queue=q)
    py_ones = (
        bool(1),
        int(1),
        float(1),
        np.float32(1),
        ctypes.c_int(1),
    )
    for sc in py_ones:
        R = dpt.remainder(X, sc)
        assert isinstance(R, dpt.usm_ndarray)
        R = dpt.remainder(sc, X)
        assert isinstance(R, dpt.usm_ndarray)


@pytest.mark.parametrize("dtype", _no_complex_dtypes[1:])
def test_remainder_inplace_python_scalar(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)
    X = dpt.ones((10, 10), dtype=dtype, sycl_queue=q)
    dt_kind = X.dtype.kind
    if dt_kind in "ui":
        X %= int(1)
    elif dt_kind == "f":
        X %= float(1)


@pytest.mark.parametrize("op1_dtype", _no_complex_dtypes[1:])
@pytest.mark.parametrize("op2_dtype", _no_complex_dtypes[1:])
def test_remainder_inplace_dtype_matrix(op1_dtype, op2_dtype):
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
        ar1 %= ar2
        assert dpt.all(ar1 == dpt.zeros(ar1.shape, dtype=ar1.dtype))

        ar3 = dpt.ones(sz, dtype=op1_dtype)
        ar4 = dpt.ones(2 * sz, dtype=op2_dtype)

        ar3[::-1] %= ar4[::2]
        assert dpt.all(ar3 == dpt.zeros(ar3.shape, dtype=ar3.dtype))

    else:
        with pytest.raises(ValueError):
            ar1 %= ar2


def test_remainder_inplace_basic():
    get_queue_or_skip()

    x = dpt.arange(10, dtype="i4")
    expected = x & 1
    x %= 2

    assert dpt.all(x == expected)
