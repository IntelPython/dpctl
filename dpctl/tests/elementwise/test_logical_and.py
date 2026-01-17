#                       Data Parallel Control (dpctl)
#
#  Copyright 2023-2025 Intel Corporation
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
def test_logical_and_dtype_matrix(op1_dtype, op2_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op1_dtype, q)
    skip_if_dtype_not_supported(op2_dtype, q)

    sz = 127
    ar1 = dpt.asarray(np.random.randint(0, 2, sz), dtype=op1_dtype)
    ar2 = dpt.asarray(np.random.randint(0, 2, sz), dtype=op2_dtype)

    r = dpt.logical_and(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)

    expected = np.logical_and(dpt.asnumpy(ar1), dpt.asnumpy(ar2))
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar1.shape
    assert (dpt.asnumpy(r) == expected).all()
    assert r.sycl_queue == ar1.sycl_queue

    r2 = dpt.empty_like(r, dtype=r.dtype)
    dpt.logical_and(ar1, ar2, out=r2)
    assert (dpt.asnumpy(r) == dpt.asnumpy(r2)).all()

    ar3 = dpt.zeros(sz, dtype=op1_dtype)
    ar4 = dpt.ones(2 * sz, dtype=op2_dtype)

    r = dpt.logical_and(ar3[::-1], ar4[::2])
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.logical_and(
        np.zeros(1, dtype=op1_dtype), np.ones(1, dtype=op2_dtype)
    )
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar3.shape
    assert (dpt.asnumpy(r) == expected).all()

    r2 = dpt.empty_like(r, dtype=r.dtype)
    dpt.logical_and(ar3[::-1], ar4[::2], out=r2)
    assert (dpt.asnumpy(r) == dpt.asnumpy(r2)).all()


@pytest.mark.parametrize("op_dtype", ["c8", "c16"])
def test_logical_and_complex_matrix(op_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op_dtype, q)

    sz = 127
    ar1_np_real = np.random.randint(0, 2, sz)
    ar1_np_imag = np.random.randint(0, 2, sz)
    ar1_np = ar1_np_real + 1j * ar1_np_imag
    ar1 = dpt.asarray(ar1_np, dtype=op_dtype)

    ar2_np_real = np.random.randint(0, 2, sz)
    ar2_np_imag = np.random.randint(0, 2, sz)
    ar2_np = ar2_np_real + 1j * ar2_np_imag
    ar2 = dpt.asarray(ar2_np, dtype=op_dtype)

    r = dpt.logical_and(ar1, ar2)
    expected = np.logical_and(ar1_np, ar2_np)
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == expected.shape
    assert (dpt.asnumpy(r) == expected).all()

    r1 = dpt.logical_and(ar1[::-2], ar2[::2])
    expected1 = np.logical_and(ar1_np[::-2], ar2_np[::2])
    assert _compare_dtypes(r.dtype, expected1.dtype, sycl_queue=q)
    assert r1.shape == expected1.shape
    assert (dpt.asnumpy(r1) == expected1).all()

    ar3 = dpt.asarray(
        [
            2.0 + 0j,
            dpt.nan,
            dpt.nan * 1j,
            dpt.inf,
            dpt.inf * 1j,
            -dpt.inf,
            -dpt.inf * 1j,
        ],
        dtype=op_dtype,
    )
    ar4 = dpt.full(ar3.shape, fill_value=1.0 + 2j, dtype=op_dtype)

    ar3_np = dpt.asnumpy(ar3)
    ar4_np = dpt.asnumpy(ar4)

    r2 = dpt.logical_and(ar3, ar4)
    with np.errstate(invalid="ignore"):
        expected2 = np.logical_and(ar3_np, ar4_np)
    assert (dpt.asnumpy(r2) == expected2).all()

    r3 = dpt.logical_and(ar4, ar4)
    with np.errstate(invalid="ignore"):
        expected3 = np.logical_and(ar4_np, ar4_np)
    assert (dpt.asnumpy(r3) == expected3).all()


def test_logical_and_complex_float():
    get_queue_or_skip()

    ar1 = dpt.asarray([1j, 1.0 + 9j, 2.0 + 0j, 2.0 + 1j], dtype="c8")
    ar2 = dpt.full(ar1.shape, 2, dtype="f4")

    ar1_np = dpt.asnumpy(ar1)
    ar2_np = dpt.asnumpy(ar2)

    r = dpt.logical_and(ar1, ar2)
    expected = np.logical_and(ar1_np, ar2_np)
    assert (dpt.asnumpy(r) == expected).all()

    r1 = dpt.logical_and(ar2, ar1)
    expected1 = np.logical_and(ar2_np, ar1_np)
    assert (dpt.asnumpy(r1) == expected1).all()
    with np.errstate(invalid="ignore"):
        for tp in [
            dpt.nan,
            dpt.nan * 1j,
            dpt.inf,
            dpt.inf * 1j,
            -dpt.inf,
            -dpt.inf * 1j,
        ]:
            ar3 = dpt.full(ar1.shape, tp)
            ar3_np = dpt.asnumpy(ar3)
            r2 = dpt.logical_and(ar1, ar3)
            expected2 = np.logical_and(ar1_np, ar3_np)
            assert (dpt.asnumpy(r2) == expected2).all()

            r3 = dpt.logical_and(ar3, ar1)
            expected3 = np.logical_and(ar3_np, ar1_np)
            assert (dpt.asnumpy(r3) == expected3).all()
