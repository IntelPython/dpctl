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

import re

import numpy as np
import pytest
from numpy.testing import assert_allclose

import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _compare_dtypes, _no_complex_dtypes


@pytest.mark.parametrize("op1_dtype", _no_complex_dtypes)
@pytest.mark.parametrize("op2_dtype", _no_complex_dtypes)
def test_logaddexp_dtype_matrix(op1_dtype, op2_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op1_dtype, q)
    skip_if_dtype_not_supported(op2_dtype, q)

    sz = 127
    ar1 = dpt.ones(sz, dtype=op1_dtype)
    ar2 = dpt.ones_like(ar1, dtype=op2_dtype)

    r = dpt.logaddexp(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.logaddexp(dpt.asnumpy(ar1), dpt.asnumpy(ar2))
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar1.shape
    tol = 8 * max(
        np.finfo(r.dtype).resolution, np.finfo(expected.dtype).resolution
    )
    assert_allclose(
        dpt.asnumpy(r), expected.astype(r.dtype), atol=tol, rtol=tol
    )
    assert r.sycl_queue == ar1.sycl_queue

    ar3 = dpt.ones(sz, dtype=op1_dtype)
    ar4 = dpt.ones(2 * sz, dtype=op2_dtype)

    r = dpt.logaddexp(ar3[::-1], ar4[::2])
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.logaddexp(dpt.asnumpy(ar3)[::-1], dpt.asnumpy(ar4)[::2])
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar3.shape
    assert_allclose(
        dpt.asnumpy(r), expected.astype(r.dtype), atol=tol, rtol=tol
    )


def test_logaddexp_broadcasting_error():
    get_queue_or_skip()
    m = dpt.ones((10, 10), dtype="i4")
    v = dpt.ones((3,), dtype="i4")
    with pytest.raises(ValueError):
        dpt.logaddexp(m, v)


@pytest.mark.parametrize("dtype", _no_complex_dtypes)
def test_logaddexp_dtype_error(
    dtype,
):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    ar1 = dpt.ones(5, dtype=dtype)
    ar2 = dpt.ones_like(ar1, dtype="f4")

    y = dpt.zeros_like(ar1, dtype="int8")
    with pytest.raises(ValueError) as excinfo:
        dpt.logaddexp(ar1, ar2, out=y)
    assert re.match("Output array of type.*is needed", str(excinfo.value))
