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
#  Unless_equal required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
import pytest

import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _compare_dtypes, _integral_dtypes


@pytest.mark.parametrize(
    "op_dtype",
    [
        "b1",
    ]
    + _integral_dtypes,
)
def test_bitwise_invert_dtype_matrix(op_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op_dtype, q)

    sz = 7
    ar1 = dpt.asarray(np.random.randint(0, 2, sz), dtype=op_dtype)

    r = dpt.bitwise_invert(ar1)
    assert isinstance(r, dpt.usm_ndarray)
    assert r.dtype == ar1.dtype

    expected = np.bitwise_not(dpt.asnumpy(ar1))
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar1.shape
    assert (dpt.asnumpy(r) == expected).all()
    assert r.sycl_queue == ar1.sycl_queue

    r2 = dpt.empty_like(r, dtype=r.dtype)
    dpt.bitwise_invert(ar1, out=r2)
    assert dpt.all(dpt.equal(r, r2))

    ar2 = dpt.zeros(sz, dtype=op_dtype)
    r = dpt.bitwise_invert(ar2[::-1])
    assert isinstance(r, dpt.usm_ndarray)

    expected = np.bitwise_not(np.zeros(ar2.shape, dtype=op_dtype))
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar2.shape
    assert (dpt.asnumpy(r) == expected).all()

    ar3 = dpt.ones(sz, dtype=op_dtype)
    r2 = dpt.bitwise_invert(ar3[::2])
    assert isinstance(r, dpt.usm_ndarray)

    expected = np.bitwise_not(np.ones(ar3.shape, dtype=op_dtype)[::2])
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert (dpt.asnumpy(r2) == expected).all()

    r3 = dpt.empty_like(r, dtype=r.dtype)
    dpt.bitwise_invert(ar2[::-1], out=r3)
    assert dpt.all(dpt.equal(r, r3))


def test_bitwise_invert_large_boolean():
    get_queue_or_skip()

    x = dpt.tril(dpt.ones((32, 32), dtype="?"), k=-1)
    res = dpt.astype(dpt.bitwise_invert(x), "i4")

    assert dpt.all(res >= 0)
    assert dpt.all(res <= 1)
