#                       Data Parallel Control (dpctl)
#
#  Copyright 2023 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless_equal required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
import pytest

import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _integral_dtypes


@pytest.mark.parametrize("op1_dtype", _integral_dtypes)
@pytest.mark.parametrize("op2_dtype", _integral_dtypes)
def test_bitwise_right_shift_dtype_matrix_contig(op1_dtype, op2_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op1_dtype, q)
    skip_if_dtype_not_supported(op2_dtype, q)

    if op1_dtype != op2_dtype and "u8" in [op1_dtype, op2_dtype]:
        return

    sz = 7
    n = 2 * sz
    dt1 = dpt.dtype(op1_dtype)
    dt2 = dpt.dtype(op2_dtype)

    x1_range_begin = -sz if dpt.iinfo(dt1).min < 0 else 0
    x1 = dpt.arange(x1_range_begin, x1_range_begin + n, dtype=dt1)
    x2 = dpt.arange(0, n, dtype=dt2)

    r = dpt.bitwise_right_shift(x1, x2)
    assert isinstance(r, dpt.usm_ndarray)
    assert r.sycl_queue == x1.sycl_queue
    assert r.sycl_queue == x2.sycl_queue

    x1_np = np.arange(x1_range_begin, x1_range_begin + n, dtype=op1_dtype)
    x2_np = np.arange(0, n, dtype=op2_dtype)
    r_np = np.right_shift(x1_np, x2_np)

    assert r.dtype == r_np.dtype
    assert (dpt.asnumpy(r) == r_np).all()


@pytest.mark.parametrize("op1_dtype", _integral_dtypes)
@pytest.mark.parametrize("op2_dtype", _integral_dtypes)
def test_bitwise_right_shift_dtype_matrix_strided(op1_dtype, op2_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op1_dtype, q)
    skip_if_dtype_not_supported(op2_dtype, q)

    if op1_dtype != op2_dtype and "u8" in [op1_dtype, op2_dtype]:
        return

    sz = 11
    n = 2 * sz
    dt1 = dpt.dtype(op1_dtype)
    dt2 = dpt.dtype(op2_dtype)

    x1_range_begin = -sz if dpt.iinfo(dt1).min < 0 else 0
    x1 = dpt.arange(x1_range_begin, x1_range_begin + n, dtype=dt1)[::-2]
    x2 = dpt.arange(0, n, dtype=dt2)[::2]

    r = dpt.bitwise_right_shift(x1, x2)
    assert isinstance(r, dpt.usm_ndarray)
    assert r.sycl_queue == x1.sycl_queue
    assert r.sycl_queue == x2.sycl_queue

    x1_np = np.arange(x1_range_begin, x1_range_begin + n, dtype=dt1)[::-2]
    x2_np = np.arange(0, n, dtype=dt2)[::2]
    r_np = np.right_shift(x1_np, x2_np)

    assert r.dtype == r_np.dtype
    assert (dpt.asnumpy(r) == r_np).all()


@pytest.mark.parametrize("op_dtype", _integral_dtypes)
def test_bitwise_right_shift_range(op_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op_dtype, q)

    x = dpt.ones(255, dtype=op_dtype)
    y = dpt.asarray(64, dtype=op_dtype)

    z = dpt.bitwise_right_shift(x, y)
    assert dpt.all(dpt.equal(z, 0))
