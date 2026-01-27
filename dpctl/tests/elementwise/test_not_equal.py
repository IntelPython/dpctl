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
def test_not_equal_dtype_matrix(op1_dtype, op2_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op1_dtype, q)
    skip_if_dtype_not_supported(op2_dtype, q)

    sz = 127
    ar1 = dpt.ones(sz, dtype=op1_dtype)
    ar2 = dpt.ones_like(ar1, dtype=op2_dtype)

    r = dpt.not_equal(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected_dtype = np.not_equal(
        np.zeros(1, dtype=op1_dtype), np.zeros(1, dtype=op2_dtype)
    ).dtype
    assert _compare_dtypes(r.dtype, expected_dtype, sycl_queue=q)
    assert r.shape == ar1.shape
    assert (dpt.asnumpy(r) == np.full(r.shape, False, dtype=r.dtype)).all()
    assert r.sycl_queue == ar1.sycl_queue

    ar3 = dpt.ones(sz, dtype=op1_dtype)
    ar4 = dpt.ones(2 * sz, dtype=op2_dtype)

    r = dpt.not_equal(ar3[::-1], ar4[::2])
    assert isinstance(r, dpt.usm_ndarray)
    expected_dtype = np.not_equal(
        np.ones(1, dtype=op1_dtype), np.ones(1, dtype=op2_dtype)
    ).dtype
    assert _compare_dtypes(r.dtype, expected_dtype, sycl_queue=q)
    assert r.shape == ar3.shape
    assert (dpt.asnumpy(r) == np.full(r.shape, False, dtype=r.dtype)).all()


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_not_equal_alignment(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n = 256
    s = dpt.concat((dpt.zeros(n, dtype=dtype), dpt.zeros(n, dtype=dtype)))

    mask = s[:-1] != s[1:]
    (pos,) = dpt.nonzero(mask)
    assert dpt.all(pos == n)

    out_arr = dpt.zeros(2 * n, dtype=mask.dtype)
    dpt.not_equal(s[:-1], s[1:], out=out_arr[1:])
    (pos,) = dpt.nonzero(mask)
    assert dpt.all(pos == (n + 1))
