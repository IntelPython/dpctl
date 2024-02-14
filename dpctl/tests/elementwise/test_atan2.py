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
from numpy.testing import assert_allclose

import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _compare_dtypes, _no_complex_dtypes


@pytest.mark.parametrize("op1_dtype", _no_complex_dtypes[1:])
@pytest.mark.parametrize("op2_dtype", _no_complex_dtypes[1:])
def test_atan2_dtype_matrix(op1_dtype, op2_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op1_dtype, q)
    skip_if_dtype_not_supported(op2_dtype, q)

    sz = 127
    ar1 = dpt.ones(sz, dtype=op1_dtype, sycl_queue=q)
    ar2 = dpt.ones_like(ar1, dtype=op2_dtype, sycl_queue=q)

    r = dpt.atan2(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.arctan2(
        np.ones(sz, dtype=op1_dtype), np.ones(sz, dtype=op2_dtype)
    )
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar1.shape

    tol = 8 * max(
        dpt.finfo(r.dtype).resolution, dpt.finfo(expected.dtype).resolution
    )
    assert_allclose(dpt.asnumpy(r), expected, atol=tol, rtol=tol)
    assert r.sycl_queue == ar1.sycl_queue

    ar3 = dpt.ones(sz, dtype=op1_dtype, sycl_queue=q)
    ar4 = dpt.ones(2 * sz, dtype=op2_dtype, sycl_queue=q)

    r = dpt.atan2(ar3[::-1], ar4[::2])
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.arctan2(
        np.ones(sz, dtype=op1_dtype), np.ones(sz, dtype=op2_dtype)
    )
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar3.shape

    tol = 8 * max(
        dpt.finfo(r.dtype).resolution, dpt.finfo(expected.dtype).resolution
    )
    assert_allclose(dpt.asnumpy(r), expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("arr_dt", _no_complex_dtypes[1:])
def test_atan2_python_scalar(arr_dt):
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
        R = dpt.atan2(X, sc)
        assert isinstance(R, dpt.usm_ndarray)
        R = dpt.atan2(sc, X)
        assert isinstance(R, dpt.usm_ndarray)


@pytest.mark.parametrize("dt", ["f2", "f4", "f8"])
def test_atan2_special_case_one_nan(dt):
    """If either x1_i or x2_i is NaN, the result is NaN."""
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x1 = dpt.asarray([dpt.nan, dpt.nan, 1], dtype=dt)
    x2 = dpt.asarray([dpt.nan, 1, dpt.nan], dtype=dt)

    y = dpt.atan2(x1, x2)
    assert dpt.all(dpt.isnan(y))


@pytest.mark.parametrize("dt", ["f2", "f4", "f8"])
def test_atan2_special_case_positive_and_pzero(dt):
    """If x1_i is greater than 0 and x2_i is +0, the result
    is an approximation to +pi/2.
    """
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x1 = dpt.asarray([0.5, 1, 2, dpt.inf], dtype=dt)
    x2 = dpt.asarray([+0.0], dtype=dt)

    actual = dpt.atan2(x1, x2)
    expected = dpt.asarray(dpt.pi / 2, dtype=dt)

    diff = dpt.abs(dpt.subtract(actual, expected))
    atol = 8 * dpt.finfo(diff.dtype).eps
    assert dpt.all(dpt.less_equal(diff, atol))


@pytest.mark.parametrize("dt", ["f2", "f4", "f8"])
def test_atan2_special_case_positive_and_nzero(dt):
    """If x1_i is greater than 0 and x2_i is -0, the result
    is an approximation to +pi/2.
    """
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x1 = dpt.asarray([0.5, 1, 2, dpt.inf], dtype=dt)
    x2 = dpt.asarray([-0.0], dtype=dt)

    actual = dpt.atan2(x1, x2)
    expected = dpt.asarray(dpt.pi / 2, dtype=dt)

    diff = dpt.abs(dpt.subtract(actual, expected))
    atol = 8 * dpt.finfo(diff.dtype).eps
    assert dpt.all(dpt.less_equal(diff, atol))


@pytest.mark.parametrize("dt", ["f2", "f4", "f8"])
def test_atan2_special_case_pzero_and_positive(dt):
    """If x1_i is +0 and x2_i is greater than 0,
    the result is +0.
    """
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x1 = dpt.asarray(+0.0, dtype=dt)
    x2 = dpt.asarray([0.5, 1, 2, dpt.inf], dtype=dt)

    actual = dpt.atan2(x1, x2)
    expected = dpt.asarray(+0.0, dtype=dt)

    assert dpt.all(dpt.equal(actual, expected))
    assert not dpt.any(dpt.signbit(actual))


@pytest.mark.parametrize("dt", ["f2", "f4", "f8"])
def test_atan2_special_case_pzero_and_pzero(dt):
    """If x1_i is +0 and x2_i is +0, the result is +0."""
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x1 = dpt.asarray(+0.0, dtype=dt)
    x2 = dpt.asarray([+0.0], dtype=dt)

    actual = dpt.atan2(x1, x2)
    expected = dpt.asarray(+0.0, dtype=dt)

    assert dpt.all(dpt.equal(actual, expected))
    assert not dpt.any(dpt.signbit(actual))


@pytest.mark.parametrize("dt", ["f2", "f4", "f8"])
def test_atan2_special_case_pzero_and_nzero(dt):
    """
    If x1_i is +0 and x2_i is -0, the result is an
    approximation to +pi.
    """
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x1 = dpt.asarray(+0.0, dtype=dt)
    x2 = dpt.asarray([-0.0], dtype=dt)

    actual = dpt.atan2(x1, x2)
    expected = dpt.asarray(dpt.pi, dtype=dt)

    diff = dpt.abs(dpt.subtract(actual, expected))
    atol = 8 * dpt.finfo(diff.dtype).eps
    assert dpt.all(dpt.less_equal(diff, atol))


@pytest.mark.parametrize("dt", ["f2", "f4", "f8"])
def test_atan2_special_case_pzero_and_negatvie(dt):
    """
    If x1_i is +0 and x2_i is less than 0, the result
    is an approximation to +pi.
    """
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x1 = dpt.asarray(+0.0, dtype=dt)
    x2 = dpt.asarray([-0.5, -1, -2, -dpt.inf], dtype=dt)

    actual = dpt.atan2(x1, x2)
    expected = dpt.asarray(dpt.pi, dtype=dt)

    diff = dpt.abs(dpt.subtract(actual, expected))
    atol = 8 * dpt.finfo(diff.dtype).eps
    assert dpt.all(dpt.less_equal(diff, atol))


@pytest.mark.parametrize("dt", ["f2", "f4", "f8"])
def test_atan2_special_case_nzero_and_positive(dt):
    """If x1_i is -0 and x2_i is greater than 0,
    the result is -0.
    """
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x1 = dpt.asarray(-0.0, dtype=dt)
    x2 = dpt.asarray([0.5, 1, 2, dpt.inf], dtype=dt)

    actual = dpt.atan2(x1, x2)
    expected = dpt.asarray(-0.0, dtype=dt)

    assert dpt.all(dpt.equal(actual, expected))
    assert dpt.all(dpt.signbit(actual))


@pytest.mark.parametrize("dt", ["f2", "f4", "f8"])
def test_atan2_special_case_nzero_and_pzero(dt):
    """If x1_i is -0 and x2_i is +0, the result is -0."""
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x1 = dpt.asarray(-0.0, dtype=dt)
    x2 = dpt.asarray([+0.0], dtype=dt)

    actual = dpt.atan2(x1, x2)
    expected = dpt.asarray(-0.0, dtype=dt)

    assert dpt.all(dpt.equal(actual, expected))
    assert dpt.all(dpt.signbit(actual))


@pytest.mark.parametrize("dt", ["f2", "f4", "f8"])
def test_atan2_special_case_nzero_and_nzero(dt):
    """If x1_i is -0 and x2_i is -0, the result is
    an approximation to -pi.
    """
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x1 = dpt.asarray([-0.0], dtype=dt)
    x2 = dpt.asarray([-0.0], dtype=dt)

    actual = dpt.atan2(x1, x2)
    expected = dpt.asarray(-dpt.pi, dtype=dt)

    diff = dpt.abs(dpt.subtract(actual, expected))
    atol = 8 * dpt.finfo(diff.dtype).eps
    assert dpt.all(dpt.less_equal(diff, atol))


@pytest.mark.parametrize("dt", ["f2", "f4", "f8"])
def test_atan2_special_case_nzero_and_negative(dt):
    """If x1_i is -0 and x2_i is less than 0, the result
    is an approximation to -pi.
    """
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x1 = dpt.asarray([-0.0], dtype=dt)
    x2 = dpt.asarray([-dpt.inf, -2, -1, -0.5], dtype=dt)

    actual = dpt.atan2(x1, x2)
    expected = dpt.asarray(-dpt.pi, dtype=dt)

    diff = dpt.abs(dpt.subtract(actual, expected))
    atol = 8 * dpt.finfo(diff.dtype).eps
    assert dpt.all(dpt.less_equal(diff, atol))


@pytest.mark.parametrize("dt", ["f2", "f4", "f8"])
def test_atan2_special_case_negative_and_pzero(dt):
    """If x1_i is less than 0 and x2_i is +0, the result
    is an approximation to -pi/2.
    """
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x1 = dpt.asarray([-dpt.inf, -2, -1, -0.5], dtype=dt)
    x2 = dpt.asarray(+0.0, dtype=dt)

    actual = dpt.atan2(x1, x2)
    expected = dpt.asarray(-dpt.pi / 2, dtype=dt)

    diff = dpt.abs(dpt.subtract(actual, expected))
    atol = 8 * dpt.finfo(diff.dtype).eps
    assert dpt.all(dpt.less_equal(diff, atol))


@pytest.mark.parametrize("dt", ["f2", "f4", "f8"])
def test_atan2_special_case_negative_and_nzero(dt):
    """If x1_i is less than 0 and x2_i is -0, the result
    is an approximation to -pi/2."""
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x1 = dpt.asarray([-dpt.inf, -2, -1, -0.5], dtype=dt)
    x2 = dpt.asarray(-0.0, dtype=dt)

    actual = dpt.atan2(x1, x2)
    expected = dpt.asarray(-dpt.pi / 2, dtype=dt)

    diff = dpt.abs(dpt.subtract(actual, expected))
    atol = 8 * dpt.finfo(diff.dtype).eps
    assert dpt.all(dpt.less_equal(diff, atol))


@pytest.mark.parametrize("dt", ["f2", "f4", "f8"])
def test_atan2_special_case_pfinite_and_pinf(dt):
    """If x1_i is greater than 0, x1_i is a finite number,
    and x2_i is +infinity, the result is +0."""
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x1 = dpt.asarray([0.5, 1, 2, 5], dtype=dt)
    x2 = dpt.asarray(dpt.inf, dtype=dt)

    actual = dpt.atan2(x1, x2)
    expected = dpt.asarray(+0.0, dtype=dt)
    assert dpt.all(dpt.equal(actual, expected))
    assert not dpt.any(dpt.signbit(actual))


@pytest.mark.parametrize("dt", ["f2", "f4", "f8"])
def test_atan2_special_case_pfinite_and_ninf(dt):
    """If x1_i is greater than 0, x1_i is a finite number,
    and x2_i is -infinity, the result is an approximation
    to +pi."""
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x1 = dpt.asarray([0.5, 1, 2, 5], dtype=dt)
    x2 = dpt.asarray(-dpt.inf, dtype=dt)

    actual = dpt.atan2(x1, x2)
    expected = dpt.asarray(dpt.pi, dtype=dt)

    diff = dpt.abs(dpt.subtract(actual, expected))
    atol = 8 * dpt.finfo(diff.dtype).eps
    assert dpt.all(dpt.less_equal(diff, atol))


@pytest.mark.parametrize("dt", ["f2", "f4", "f8"])
def test_atan2_special_case_nfinite_and_pinf(dt):
    """If x1_i is less than 0, x1_i is a finite number,
    and x2_i is +infinity, the result is -0."""
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x1 = dpt.asarray([-0.5, -1, -2, -5], dtype=dt)
    x2 = dpt.asarray(dpt.inf, dtype=dt)

    actual = dpt.atan2(x1, x2)
    expected = dpt.asarray(-0.0, dtype=dt)
    assert dpt.all(dpt.equal(actual, expected))
    assert dpt.all(dpt.signbit(actual))


@pytest.mark.parametrize("dt", ["f2", "f4", "f8"])
def test_atan2_special_case_nfinite_and_ninf(dt):
    """If x1_i is less than 0, x1_i is a finite number, and
    x2_i is -infinity, the result is an approximation
    to -pi."""
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x1 = dpt.asarray([-0.5, -1, -2, -5], dtype=dt)
    x2 = dpt.asarray(-dpt.inf, dtype=dt)

    actual = dpt.atan2(x1, x2)
    expected = dpt.asarray(-dpt.pi, dtype=dt)

    diff = dpt.abs(dpt.subtract(actual, expected))
    atol = 8 * dpt.finfo(diff.dtype).eps
    assert dpt.all(dpt.less_equal(diff, atol))


@pytest.mark.parametrize("dt", ["f2", "f4", "f8"])
def test_atan2_special_case_pinf_and_finite(dt):
    """If x1_i is +infinity and x2_i is a finite number,
    the result is an approximation to +pi/2.
    """
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x1 = dpt.asarray(dpt.inf, dtype=dt)
    x2 = dpt.asarray([-2, -0.0, 0.0, 2], dtype=dt)

    actual = dpt.atan2(x1, x2)
    expected = dpt.asarray(dpt.pi / 2, dtype=dt)

    diff = dpt.abs(dpt.subtract(actual, expected))
    atol = 8 * dpt.finfo(diff.dtype).eps
    assert dpt.all(dpt.less_equal(diff, atol))


@pytest.mark.parametrize("dt", ["f2", "f4", "f8"])
def test_atan2_special_case_ninf_and_finite(dt):
    """If x1_i is -infinity and x2_i is a finite number,
    the result is an approximation to -pi/2.
    """
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x1 = dpt.asarray(-dpt.inf, dtype=dt)
    x2 = dpt.asarray([-2, -0.0, 0.0, 2], dtype=dt)

    actual = dpt.atan2(x1, x2)
    expected = dpt.asarray(-dpt.pi / 2, dtype=dt)

    diff = dpt.abs(dpt.subtract(actual, expected))
    atol = 8 * dpt.finfo(diff.dtype).eps
    assert dpt.all(dpt.less_equal(diff, atol))


@pytest.mark.parametrize("dt", ["f2", "f4", "f8"])
def test_atan2_special_case_pinf_and_pinf(dt):
    """If x1_i is +infinity and x2_i is +infinity,
    the result is an approximation to +pi/4.
    """
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x1 = dpt.asarray(dpt.inf, dtype=dt)
    x2 = dpt.asarray([dpt.inf], dtype=dt)

    actual = dpt.atan2(x1, x2)
    expected = dpt.asarray(dpt.pi / 4, dtype=dt)

    diff = dpt.abs(dpt.subtract(actual, expected))
    atol = 8 * dpt.finfo(diff.dtype).eps
    assert dpt.all(dpt.less_equal(diff, atol))


@pytest.mark.parametrize("dt", ["f2", "f4", "f8"])
def test_atan2_special_case_pinf_and_ninf(dt):
    """If x1_i is +infinity and x2_i is -infinity,
    the result is an approximation to +3*pi/4.
    """
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x1 = dpt.asarray(dpt.inf, dtype=dt)
    x2 = dpt.asarray([-dpt.inf], dtype=dt)

    actual = dpt.atan2(x1, x2)
    expected = dpt.asarray(3 * dpt.pi / 4, dtype=dt)

    diff = dpt.abs(dpt.subtract(actual, expected))
    atol = 8 * dpt.finfo(diff.dtype).eps
    assert dpt.all(dpt.less_equal(diff, atol))


@pytest.mark.parametrize("dt", ["f2", "f4", "f8"])
def test_atan2_special_case_ninf_and_pinf(dt):
    """If x1_i is -infinity and x2_i is +infinity,
    the result is an approximation to -pi/4.
    """
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x1 = dpt.asarray(-dpt.inf, dtype=dt)
    x2 = dpt.asarray([dpt.inf], dtype=dt)

    actual = dpt.atan2(x1, x2)
    expected = dpt.asarray(-dpt.pi / 4, dtype=dt)

    diff = dpt.abs(dpt.subtract(actual, expected))
    atol = 8 * dpt.finfo(diff.dtype).eps
    assert dpt.all(dpt.less_equal(diff, atol))


@pytest.mark.parametrize("dt", ["f2", "f4", "f8"])
def test_atan2_special_case_ninf_and_ninf(dt):
    """If x1_i is -infinity and x2_i is -infinity,
    the result is an approximation to -3*pi/4.
    """
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x1 = dpt.asarray(-dpt.inf, dtype=dt)
    x2 = dpt.asarray([-dpt.inf], dtype=dt)

    actual = dpt.atan2(x1, x2)
    expected = dpt.asarray(-3 * dpt.pi / 4, dtype=dt)

    diff = dpt.abs(dpt.subtract(actual, expected))
    atol = 8 * dpt.finfo(diff.dtype).eps
    assert dpt.all(dpt.less_equal(diff, atol))
