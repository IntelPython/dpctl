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

import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _compare_dtypes, _no_complex_dtypes, _real_fp_dtypes


@pytest.mark.parametrize("op1_dtype", _no_complex_dtypes)
@pytest.mark.parametrize("op2_dtype", _no_complex_dtypes)
def test_copysign_dtype_matrix(op1_dtype, op2_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op1_dtype, q)
    skip_if_dtype_not_supported(op2_dtype, q)

    sz = 127
    ar1 = dpt.ones(sz, dtype=op1_dtype)
    ar2 = dpt.ones_like(ar1, dtype=op2_dtype)

    r = dpt.copysign(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.copysign(
        np.ones(1, dtype=op1_dtype), np.ones(1, dtype=op2_dtype)
    )
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar1.shape
    assert (dpt.asnumpy(r) == expected.astype(r.dtype)).all()
    assert r.sycl_queue == ar1.sycl_queue

    ar3 = dpt.ones(sz, dtype=op1_dtype)
    ar4 = dpt.ones(2 * sz, dtype=op2_dtype)

    r = dpt.copysign(ar3[::-1], ar4[::2])
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.copysign(
        np.ones(1, dtype=op1_dtype), np.ones(1, dtype=op2_dtype)
    )
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar3.shape
    assert (dpt.asnumpy(r) == expected.astype(r.dtype)).all()


@pytest.mark.parametrize("arr_dt", _real_fp_dtypes)
def test_copysign_python_scalar(arr_dt):
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
        R = dpt.copysign(X, sc)
        assert isinstance(R, dpt.usm_ndarray)
        R = dpt.copysign(sc, X)
        assert isinstance(R, dpt.usm_ndarray)


@pytest.mark.parametrize("dt", _real_fp_dtypes)
def test_copysign(dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x = dpt.arange(100, dtype=dt, sycl_queue=q)
    x[1::2] *= -1
    y = dpt.ones(100, dtype=dt, sycl_queue=q)
    y[::2] *= -1
    res = dpt.copysign(x, y)
    expected = dpt.negative(x)
    tol = dpt.finfo(dt).resolution
    assert dpt.allclose(res, expected, atol=tol, rtol=tol)


def test_copysign_special_values():
    get_queue_or_skip()

    x1 = dpt.asarray([1.0, 0.0, dpt.nan, dpt.nan], dtype="f4")
    y1 = dpt.asarray([-1.0, -0.0, -dpt.nan, -1], dtype="f4")
    res = dpt.copysign(x1, y1)
    assert dpt.all(dpt.signbit(res))
    x2 = dpt.asarray([-1.0, -0.0, -dpt.nan, -dpt.nan], dtype="f4")
    res = dpt.copysign(x2, y1)
    assert dpt.all(dpt.signbit(res))
    y2 = dpt.asarray([0.0, 1.0, dpt.nan, 1.0], dtype="f4")
    res = dpt.copysign(x2, y2)
    assert not dpt.any(dpt.signbit(res))
    res = dpt.copysign(x1, y2)
    assert not dpt.any(dpt.signbit(res))
