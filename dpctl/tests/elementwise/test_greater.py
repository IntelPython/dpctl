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

import numpy as np
import pytest

import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _all_dtypes, _compare_dtypes


@pytest.mark.parametrize("op1_dtype", _all_dtypes)
@pytest.mark.parametrize("op2_dtype", _all_dtypes)
def test_greater_dtype_matrix(op1_dtype, op2_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op1_dtype, q)
    skip_if_dtype_not_supported(op2_dtype, q)

    sz = 127
    ar1 = dpt.zeros(sz, dtype=op1_dtype)
    ar2 = dpt.ones_like(ar1, dtype=op2_dtype)

    r = dpt.greater(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.greater(
        np.zeros(1, dtype=op1_dtype), np.ones(1, dtype=op2_dtype)
    )
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar1.shape
    assert (dpt.asnumpy(r) == expected.astype(r.dtype)).all()
    assert r.sycl_queue == ar1.sycl_queue

    ar3 = dpt.zeros(sz, dtype=op1_dtype)
    ar4 = dpt.ones(2 * sz, dtype=op2_dtype)

    r = dpt.greater(ar3[::-1], ar4[::2])
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.greater(
        np.zeros(1, dtype=op1_dtype), np.ones(1, dtype=op2_dtype)
    )
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar3.shape
    assert (dpt.asnumpy(r) == expected.astype(r.dtype)).all()


@pytest.mark.parametrize("op_dtype", ["c8", "c16"])
def test_greater_complex_matrix(op_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op_dtype, q)

    sz = 127
    ar1_np_real = np.random.randint(0, 10, sz)
    ar1_np_imag = np.random.randint(0, 10, sz)
    ar1_np = ar1_np_real + 1j * ar1_np_imag
    ar1 = dpt.asarray(ar1_np, dtype=op_dtype)

    ar2_np_real = np.random.randint(0, 10, sz)
    ar2_np_imag = np.random.randint(0, 10, sz)
    ar2_np = ar2_np_real + 1j * ar2_np_imag
    ar2 = dpt.asarray(ar2_np, dtype=op_dtype)

    r = dpt.greater(ar1, ar2)
    expected = np.greater(ar1_np, ar2_np)
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == expected.shape
    assert (dpt.asnumpy(r) == expected).all()

    r1 = dpt.greater(ar1[::-2], ar2[::2])
    expected1 = np.greater(ar1_np[::-2], ar2_np[::2])
    assert _compare_dtypes(r.dtype, expected1.dtype, sycl_queue=q)
    assert r1.shape == expected1.shape
    assert (dpt.asnumpy(r1) == expected1).all()

    ar3 = dpt.asarray([1.0 + 9j, 2.0 + 0j, 2.0 + 1j, 2.0 + 2j], dtype=op_dtype)
    ar4 = dpt.asarray([2.0 + 0j, dpt.nan, dpt.inf, -dpt.inf], dtype=op_dtype)

    ar3_np = dpt.asnumpy(ar3)
    ar4_np = dpt.asnumpy(ar4)

    r2 = dpt.greater(ar3, ar4)
    with np.errstate(invalid="ignore"):
        expected2 = np.greater(ar3_np, ar4_np)
    assert (dpt.asnumpy(r2) == expected2).all()

    r3 = dpt.greater(ar4, ar4)
    with np.errstate(invalid="ignore"):
        expected3 = np.greater(ar4_np, ar4_np)
    assert (dpt.asnumpy(r3) == expected3).all()


def test_greater_complex_float():
    get_queue_or_skip()

    ar1 = dpt.asarray([1.0 + 9j, 2.0 + 0j, 2.0 + 1j, 2.0 + 2j], dtype="c8")
    ar2 = dpt.full((4,), 2, dtype="f4")

    ar1_np = dpt.asnumpy(ar1)
    ar2_np = dpt.asnumpy(ar2)

    r = dpt.greater(ar1, ar2)
    expected = np.greater(ar1_np, ar2_np)
    assert (dpt.asnumpy(r) == expected).all()

    r1 = dpt.greater(ar2, ar1)
    expected1 = np.greater(ar2_np, ar1_np)
    assert (dpt.asnumpy(r1) == expected1).all()
    with np.errstate(invalid="ignore"):
        for tp in [dpt.nan, dpt.inf, -dpt.inf]:

            ar3 = dpt.full((4,), tp)
            ar3_np = dpt.asnumpy(ar3)

            r2 = dpt.greater(ar1, ar3)
            expected2 = np.greater(ar1_np, ar3_np)
            assert (dpt.asnumpy(r2) == expected2).all()

            r3 = dpt.greater(ar3, ar1)
            expected3 = np.greater(ar3_np, ar1_np)
            assert (dpt.asnumpy(r3) == expected3).all()


def test_greater_mixed_integer_kinds():
    get_queue_or_skip()

    x1 = dpt.flip(dpt.arange(-9, 1, dtype="i8"))
    x2 = dpt.arange(10, dtype="u8")

    # u8 - i8
    res = dpt.greater(x2, x1)
    assert dpt.all(res[1:])
    assert not res[0]
    # i8 - u8
    assert not dpt.any(dpt.greater(x1, x2))

    # Python scalar
    assert dpt.all(dpt.greater(x2, -1))
    assert not dpt.any(dpt.greater(-1, x2))


def test_greater_very_large_py_int():
    get_queue_or_skip()

    py_int = dpt.iinfo(dpt.int64).max + 10

    x = dpt.asarray(3, dtype="u8")
    assert py_int > x
    assert not dpt.greater(x, py_int)

    x = dpt.asarray(py_int, dtype="u8")
    assert x > -1
    assert not dpt.greater(-1, x)
