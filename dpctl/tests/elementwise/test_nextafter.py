#                       Data Parallel Control (dpctl)
#
#  Copyright 2020-2025 Intel Corporation
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
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _compare_dtypes, _no_complex_dtypes


@pytest.mark.parametrize("op1_dtype", _no_complex_dtypes[1:])
@pytest.mark.parametrize("op2_dtype", _no_complex_dtypes[1:])
def test_nextafter_dtype_matrix(op1_dtype, op2_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op1_dtype, q)
    skip_if_dtype_not_supported(op2_dtype, q)

    sz = 127
    ar1 = dpt.ones(sz, dtype=op1_dtype, sycl_queue=q)
    ar2 = dpt.ones_like(ar1, dtype=op2_dtype, sycl_queue=q)

    r = dpt.nextafter(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.nextafter(
        np.ones(sz, dtype=op1_dtype), np.ones(sz, dtype=op2_dtype)
    )
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar1.shape
    assert (dpt.asnumpy(r) == expected.astype(r.dtype)).all()
    assert r.sycl_queue == ar1.sycl_queue

    ar3 = dpt.ones(sz, dtype=op1_dtype, sycl_queue=q)
    ar4 = dpt.ones(2 * sz, dtype=op2_dtype, sycl_queue=q)

    r = dpt.nextafter(ar3[::-1], ar4[::2])
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.nextafter(
        np.ones(sz, dtype=op1_dtype), np.ones(sz, dtype=op2_dtype)
    )
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar3.shape
    assert (dpt.asnumpy(r) == expected.astype(r.dtype)).all()


@pytest.mark.parametrize("arr_dt", _no_complex_dtypes[1:])
def test_nextafter_python_scalar(arr_dt):
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
        R = dpt.nextafter(X, sc)
        assert isinstance(R, dpt.usm_ndarray)
        R = dpt.nextafter(sc, X)
        assert isinstance(R, dpt.usm_ndarray)


@pytest.mark.parametrize("dt", ["f2", "f4", "f8"])
def test_nextafter_special_cases_nan(dt):
    """If either x1_i or x2_i is NaN, the result is NaN."""
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x1 = dpt.asarray([2.0, dpt.nan, dpt.nan], dtype=dt)
    x2 = dpt.asarray([dpt.nan, 2.0, dpt.nan], dtype=dt)

    y = dpt.nextafter(x1, x2)
    assert dpt.all(dpt.isnan(y))


@pytest.mark.parametrize("dt", ["f2", "f4", "f8"])
def test_nextafter_special_cases_zero(dt):
    """If x1_i is equal to x2_i, the result is x2_i."""
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x1 = dpt.asarray([-0.0, 0.0, -0.0, 0.0], dtype=dt)
    x2 = dpt.asarray([0.0, -0.0, -0.0, 0.0], dtype=dt)

    y = dpt.nextafter(x1, x2)
    assert dpt.all(y == 0)

    skip_checking_signs = (
        x1.dtype == dpt.float16
        and x1.sycl_device.backend == dpctl.backend_type.cuda
    )
    if skip_checking_signs:
        pytest.skip(
            "Skipped checking signs for nextafter due to "
            "known issue in DPC++ support for CUDA devices"
        )
    else:
        assert dpt.all(dpt.signbit(y) == dpt.signbit(x2))


@pytest.mark.parametrize("dt", ["f2", "f4", "f8"])
def test_nextafter_basic(dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    s = 10
    x1 = dpt.ones(s, dtype=dt, sycl_queue=q)
    x2 = dpt.full(s, 2, dtype=dt, sycl_queue=q)

    r = dpt.nextafter(x1, x2)
    expected_diff = dpt.asarray(dpt.finfo(dt).eps, dtype=dt, sycl_queue=q)

    assert dpt.all(r > 0)
    assert dpt.all(r - x1 == expected_diff)

    x3 = dpt.zeros(s, dtype=dt, sycl_queue=q)

    r = dpt.nextafter(x3, x1)
    assert dpt.all(r > 0)

    r = dpt.nextafter(x1, x3)
    assert dpt.all((r - x1) < 0)

    r = dpt.nextafter(x1, 0)
    assert dpt.all(x1 - r == (expected_diff) / 2)

    r = dpt.nextafter(x3, dpt.inf)
    assert dpt.all(r > 0)

    r = dpt.nextafter(x3, -dpt.inf)
    assert dpt.all(r < 0)
