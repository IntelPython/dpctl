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
from dpctl.tensor._type_utils import _can_cast
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _all_dtypes, _compare_dtypes


@pytest.mark.parametrize("op1_dtype", _all_dtypes[1:])
@pytest.mark.parametrize("op2_dtype", _all_dtypes[1:])
def test_power_dtype_matrix(op1_dtype, op2_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op1_dtype, q)
    skip_if_dtype_not_supported(op2_dtype, q)

    sz = 127
    ar1 = dpt.ones(sz, dtype=op1_dtype)
    ar2 = dpt.ones_like(ar1, dtype=op2_dtype)

    r = dpt.pow(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.power(
        np.ones(1, dtype=op1_dtype), np.ones(1, dtype=op2_dtype)
    )
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar1.shape
    assert (dpt.asnumpy(r) == expected.astype(r.dtype)).all()
    assert r.sycl_queue == ar1.sycl_queue

    ar3 = dpt.ones(sz, dtype=op1_dtype)
    ar4 = dpt.ones(2 * sz, dtype=op2_dtype)

    r = dpt.pow(ar3[::-1], ar4[::2])
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.power(
        np.ones(1, dtype=op1_dtype), np.ones(1, dtype=op2_dtype)
    )
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar3.shape
    assert (dpt.asnumpy(r) == expected.astype(r.dtype)).all()


@pytest.mark.parametrize("op1_dtype", _all_dtypes[1:])
@pytest.mark.parametrize("op2_dtype", _all_dtypes[1:])
def test_pow_inplace_dtype_matrix(op1_dtype, op2_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op1_dtype, q)
    skip_if_dtype_not_supported(op2_dtype, q)

    sz = 127
    ar1 = dpt.ones(sz, dtype=op1_dtype)
    ar2 = dpt.ones_like(ar1, dtype=op2_dtype)

    dev = q.sycl_device
    _fp16 = dev.has_aspect_fp16
    _fp64 = dev.has_aspect_fp64
    if _can_cast(ar2.dtype, ar1.dtype, _fp16, _fp64, casting="same_kind"):
        ar1 **= ar2
        assert (
            dpt.asnumpy(ar1) == np.full(ar1.shape, 1, dtype=ar1.dtype)
        ).all()

        ar3 = dpt.ones(sz, dtype=op1_dtype)
        ar4 = dpt.ones(2 * sz, dtype=op2_dtype)

        ar3[::-1] *= ar4[::2]
        assert (
            dpt.asnumpy(ar3) == np.full(ar3.shape, 1, dtype=ar3.dtype)
        ).all()

    else:
        with pytest.raises(ValueError):
            ar1 **= ar2


def test_pow_inplace_basic():
    get_queue_or_skip()

    x = dpt.arange(10, dtype="i4")
    expected = dpt.square(x)
    x **= 2

    assert dpt.all(x == expected)
