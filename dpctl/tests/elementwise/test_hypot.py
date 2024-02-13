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
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _compare_dtypes, _no_complex_dtypes, _usm_types


@pytest.mark.parametrize("op1_dtype", _no_complex_dtypes[1:])
@pytest.mark.parametrize("op2_dtype", _no_complex_dtypes[1:])
def test_hypot_dtype_matrix(op1_dtype, op2_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op1_dtype, q)
    skip_if_dtype_not_supported(op2_dtype, q)

    sz = 127
    ar1 = dpt.zeros(sz, dtype=op1_dtype, sycl_queue=q)
    ar2 = dpt.zeros_like(ar1, dtype=op2_dtype, sycl_queue=q)

    r = dpt.hypot(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.hypot(
        np.zeros(1, dtype=op1_dtype), np.zeros(1, dtype=op2_dtype)
    )
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar1.shape
    assert (dpt.asnumpy(r) == expected.astype(r.dtype)).all()
    assert r.sycl_queue == ar1.sycl_queue

    ar3 = dpt.zeros(sz, dtype=op1_dtype, sycl_queue=q)
    ar4 = dpt.zeros(2 * sz, dtype=op2_dtype, sycl_queue=q)

    r = dpt.hypot(ar3[::-1], ar4[::2])
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.hypot(
        np.zeros(1, dtype=op1_dtype), np.zeros(1, dtype=op2_dtype)
    )
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar3.shape
    assert (dpt.asnumpy(r) == expected.astype(r.dtype)).all()


@pytest.mark.parametrize("op1_usm_type", _usm_types)
@pytest.mark.parametrize("op2_usm_type", _usm_types)
def test_hypot_usm_type_matrix(op1_usm_type, op2_usm_type):
    get_queue_or_skip()

    sz = 128
    ar1 = dpt.ones(sz, dtype="i4", usm_type=op1_usm_type)
    ar2 = dpt.ones_like(ar1, dtype="i4", usm_type=op2_usm_type)

    r = dpt.hypot(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected_usm_type = dpctl.utils.get_coerced_usm_type(
        (op1_usm_type, op2_usm_type)
    )
    assert r.usm_type == expected_usm_type


def test_hypot_order():
    get_queue_or_skip()

    ar1 = dpt.ones((20, 20), dtype="i4", order="C")
    ar2 = dpt.ones((20, 20), dtype="i4", order="C")
    r1 = dpt.hypot(ar1, ar2, order="C")
    assert r1.flags.c_contiguous
    r2 = dpt.hypot(ar1, ar2, order="F")
    assert r2.flags.f_contiguous
    r3 = dpt.hypot(ar1, ar2, order="A")
    assert r3.flags.c_contiguous
    r4 = dpt.hypot(ar1, ar2, order="K")
    assert r4.flags.c_contiguous

    ar1 = dpt.ones((20, 20), dtype="i4", order="F")
    ar2 = dpt.ones((20, 20), dtype="i4", order="F")
    r1 = dpt.hypot(ar1, ar2, order="C")
    assert r1.flags.c_contiguous
    r2 = dpt.hypot(ar1, ar2, order="F")
    assert r2.flags.f_contiguous
    r3 = dpt.hypot(ar1, ar2, order="A")
    assert r3.flags.f_contiguous
    r4 = dpt.hypot(ar1, ar2, order="K")
    assert r4.flags.f_contiguous

    ar1 = dpt.ones((40, 40), dtype="i4", order="C")[:20, ::-2]
    ar2 = dpt.ones((40, 40), dtype="i4", order="C")[:20, ::-2]
    r4 = dpt.hypot(ar1, ar2, order="K")
    assert r4.strides == (20, -1)

    ar1 = dpt.ones((40, 40), dtype="i4", order="C")[:20, ::-2].mT
    ar2 = dpt.ones((40, 40), dtype="i4", order="C")[:20, ::-2].mT
    r4 = dpt.hypot(ar1, ar2, order="K")
    assert r4.strides == (-1, 20)


def test_hypot_broadcasting():
    get_queue_or_skip()

    m = dpt.ones((100, 5), dtype="i4")
    v = dpt.arange(1, 6, dtype="i4")

    r = dpt.hypot(m, v)

    expected = np.hypot(
        np.ones((100, 5), dtype="i4"), np.arange(1, 6, dtype="i4")
    )
    tol = 8 * np.finfo(r.dtype).resolution
    assert np.allclose(
        dpt.asnumpy(r), expected.astype(r.dtype), atol=tol, rtol=tol
    )

    r2 = dpt.hypot(v, m)
    expected2 = np.hypot(
        np.arange(1, 6, dtype="i4"), np.ones((100, 5), dtype="i4")
    )
    assert np.allclose(
        dpt.asnumpy(r2), expected2.astype(r2.dtype), atol=tol, rtol=tol
    )


@pytest.mark.parametrize("arr_dt", _no_complex_dtypes[1:])
def test_hypot_python_scalar(arr_dt):
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
        R = dpt.hypot(X, sc)
        assert isinstance(R, dpt.usm_ndarray)
        R = dpt.hypot(sc, X)
        assert isinstance(R, dpt.usm_ndarray)


class MockArray:
    def __init__(self, arr):
        self.data_ = arr

    @property
    def __sycl_usm_array_interface__(self):
        return self.data_.__sycl_usm_array_interface__


def test_hypot_mock_array():
    get_queue_or_skip()
    a = dpt.arange(10)
    b = dpt.ones(10)
    c = MockArray(b)
    r = dpt.hypot(a, c)
    assert isinstance(r, dpt.usm_ndarray)


def test_hypot_canary_mock_array():
    get_queue_or_skip()
    a = dpt.arange(10)

    class Canary:
        def __init__(self):
            pass

        @property
        def __sycl_usm_array_interface__(self):
            return None

    c = Canary()
    with pytest.raises(ValueError):
        dpt.hypot(a, c)
