#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
from helper import get_queue_or_skip, skip_if_dtype_not_supported
from numpy.testing import assert_array_equal

import dpctl.tensor as dpt

_all_dtypes = [
    "u1",
    "i1",
    "u2",
    "i2",
    "u4",
    "i4",
    "u8",
    "i8",
    "e",
    "f",
    "d",
    "F",
    "D",
]


def test_where_basic():
    get_queue_or_skip()

    cond = dpt.asarray(
        [
            [True, False, False],
            [False, True, False],
            [False, False, True],
            [False, False, False],
            [True, True, True],
        ]
    )
    out = dpt.where(cond, dpt.asarray(1), dpt.asarray(0))
    out_expected = dpt.asarray(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [1, 1, 1]]
    )

    assert (dpt.asnumpy(out) == dpt.asnumpy(out_expected)).all()


def _dtype_all_close(x1, x2):
    if np.issubdtype(x2.dtype, np.floating) or np.issubdtype(
        x2.dtype, np.complexfloating
    ):
        x2_dtype = x2.dtype
        return np.allclose(
            x1, x2, atol=np.finfo(x2_dtype).eps, rtol=np.finfo(x2_dtype).eps
        )
    else:
        return np.allclose(x1, x2)


@pytest.mark.parametrize("dt1", _all_dtypes)
@pytest.mark.parametrize("dt2", _all_dtypes)
def test_where_all_dtypes(dt1, dt2):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt1, q)
    skip_if_dtype_not_supported(dt2, q)

    cond = dpt.asarray([False, False, False, True, True], sycl_queue=q)
    x1 = dpt.asarray(2, sycl_queue=q)
    x2 = dpt.asarray(3, sycl_queue=q)

    res = dpt.where(cond, x1, x2)
    res_check = np.asarray([3, 3, 3, 2, 2], dtype=res.dtype)

    dev = q.sycl_device

    if not dev.has_aspect_fp16 or not dev.has_aspect_fp64:
        assert res.dtype.kind == dpt.result_type(x1.dtype, x2.dtype).kind

    assert _dtype_all_close(dpt.asnumpy(res), res_check)


def test_where_empty():
    # check that numpy returns same results when
    # handling empty arrays
    get_queue_or_skip()

    empty = dpt.empty(0)
    m = dpt.asarray(True)
    x1 = dpt.asarray(1)
    x2 = dpt.asarray(2)
    res = dpt.where(empty, x1, x2)

    empty_np = np.empty(0)
    m_np = dpt.asnumpy(m)
    x1_np = dpt.asnumpy(x1)
    x2_np = dpt.asnumpy(x2)
    res_np = np.where(empty_np, x1_np, x2_np)

    assert_array_equal(dpt.asnumpy(res), res_np)

    res = dpt.where(m, empty, x2)
    res_np = np.where(m_np, empty_np, x2_np)

    assert_array_equal(dpt.asnumpy(res), res_np)


@pytest.mark.parametrize("dt", _all_dtypes)
@pytest.mark.parametrize("order", ["C", "F"])
def test_where_contiguous(dt, order):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    cond = dpt.asarray(
        [
            [[True, False, False], [False, True, True]],
            [[False, True, False], [True, False, True]],
            [[False, False, True], [False, False, True]],
            [[False, False, False], [True, False, True]],
            [[True, True, True], [True, False, True]],
        ],
        sycl_queue=q,
        order=order,
    )

    x1 = dpt.full(cond.shape, 2, dtype=dt, order=order, sycl_queue=q)
    x2 = dpt.full(cond.shape, 3, dtype=dt, order=order, sycl_queue=q)

    expected = np.where(dpt.asnumpy(cond), dpt.asnumpy(x1), dpt.asnumpy(x2))
    res = dpt.where(cond, x1, x2)

    assert _dtype_all_close(dpt.asnumpy(res), expected)
