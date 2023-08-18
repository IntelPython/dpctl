#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2023 Intel Corporation
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
from dpctl.tensor._search_functions import _where_result_type
from dpctl.tensor._type_utils import _all_data_types
from dpctl.utils import ExecutionPlacementError

_all_dtypes = [
    "?",
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


class mock_device:
    def __init__(self, fp16, fp64):
        self.has_aspect_fp16 = fp16
        self.has_aspect_fp64 = fp64


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

    out = dpt.where(cond, dpt.ones(cond.shape), dpt.zeros(cond.shape))
    assert (dpt.asnumpy(out) == dpt.asnumpy(out_expected)).all()

    out = dpt.where(
        cond,
        dpt.ones(cond.shape[0], dtype="i4")[:, dpt.newaxis],
        dpt.zeros(cond.shape[0], dtype="i4")[:, dpt.newaxis],
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
@pytest.mark.parametrize("fp16", [True, False])
@pytest.mark.parametrize("fp64", [True, False])
def test_where_result_types(dt1, dt2, fp16, fp64):
    dev = mock_device(fp16, fp64)

    dt1 = dpt.dtype(dt1)
    dt2 = dpt.dtype(dt2)
    res_t = _where_result_type(dt1, dt2, dev)

    if fp16 and fp64:
        assert res_t == dpt.result_type(dt1, dt2)
    else:
        if res_t:
            assert res_t.kind == dpt.result_type(dt1, dt2).kind
        else:
            # some illegal cases are covered above, but
            # this guarantees that _where_result_type
            # produces None only when one of the dtypes
            # is illegal given fp aspects of device
            all_dts = _all_data_types(fp16, fp64)
            assert dt1 not in all_dts or dt2 not in all_dts


@pytest.mark.parametrize("dt", _all_dtypes)
def test_where_mask_dtypes(dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    # mask dtype changes
    cond = dpt.asarray([0, 1, 3, 0, 10], dtype=dt, sycl_queue=q)
    x1 = dpt.asarray(0, dtype="f4", sycl_queue=q)
    x2 = dpt.asarray(1, dtype="f4", sycl_queue=q)
    res = dpt.where(cond, x1, x2)

    res_check = np.asarray([1, 0, 0, 1, 0], dtype=res.dtype)
    assert _dtype_all_close(dpt.asnumpy(res), res_check)

    # contiguous cases
    x1 = dpt.full(cond.shape, 0, dtype="f4", sycl_queue=q)
    x2 = dpt.full(cond.shape, 1, dtype="f4", sycl_queue=q)
    res = dpt.where(cond, x1, x2)
    assert _dtype_all_close(dpt.asnumpy(res), res_check)

    # input array dtype changes
    cond = dpt.asarray([False, True, True, False, True], sycl_queue=q)
    x1 = dpt.asarray(0, dtype=dt, sycl_queue=q)
    x2 = dpt.asarray(1, dtype=dt, sycl_queue=q)
    res = dpt.where(cond, x1, x2)

    res_check = np.asarray([1, 0, 0, 1, 0], dtype=res.dtype)
    assert _dtype_all_close(dpt.asnumpy(res), res_check)

    # contiguous cases
    x1 = dpt.full(cond.shape, 0, dtype=dt, sycl_queue=q)
    x2 = dpt.full(cond.shape, 1, dtype=dt, sycl_queue=q)
    res = dpt.where(cond, x1, x2)
    assert _dtype_all_close(dpt.asnumpy(res), res_check)


def test_where_asymmetric_dtypes():
    q = get_queue_or_skip()

    cond = dpt.asarray([0, 1, 3, 0, 10], dtype="?", sycl_queue=q)
    x1 = dpt.asarray(2, dtype="i4", sycl_queue=q)
    x2 = dpt.asarray(3, dtype="i8", sycl_queue=q)

    res = dpt.where(cond, x1, x2)
    res_check = np.asarray([3, 2, 2, 3, 2], dtype=res.dtype)
    assert _dtype_all_close(dpt.asnumpy(res), res_check)

    # flip order

    res = dpt.where(cond, x2, x1)
    res_check = np.asarray([2, 3, 3, 2, 3], dtype=res.dtype)
    assert _dtype_all_close(dpt.asnumpy(res), res_check)


def test_where_nan_inf():
    get_queue_or_skip()

    cond = dpt.asarray([True, False, True, False], dtype="?")
    x1 = dpt.asarray([np.nan, 2.0, np.inf, 3.0], dtype="f4")
    x2 = dpt.asarray([2.0, np.nan, 3.0, np.inf], dtype="f4")

    cond_np = dpt.asnumpy(cond)
    x1_np = dpt.asnumpy(x1)
    x2_np = dpt.asnumpy(x2)

    res = dpt.where(cond, x1, x2)
    res_np = np.where(cond_np, x1_np, x2_np)

    assert np.allclose(dpt.asnumpy(res), res_np, equal_nan=True)

    res = dpt.where(x1, cond, x2)
    res_np = np.where(x1_np, cond_np, x2_np)
    assert _dtype_all_close(dpt.asnumpy(res), res_np)


def test_where_empty():
    # check that numpy returns same results when
    # handling empty arrays
    get_queue_or_skip()

    empty = dpt.empty(0, dtype="i2")
    m = dpt.asarray(True)
    x1 = dpt.asarray(1, dtype="i2")
    x2 = dpt.asarray(2, dtype="i2")
    res = dpt.where(empty, x1, x2)

    empty_np = np.empty(0, dtype="i2")
    m_np = dpt.asnumpy(m)
    x1_np = dpt.asnumpy(x1)
    x2_np = dpt.asnumpy(x2)
    res_np = np.where(empty_np, x1_np, x2_np)

    assert_array_equal(dpt.asnumpy(res), res_np)

    res = dpt.where(m, empty, x2)
    res_np = np.where(m_np, empty_np, x2_np)

    assert_array_equal(dpt.asnumpy(res), res_np)

    # check that broadcasting is performed
    with pytest.raises(ValueError):
        dpt.where(empty, x1, dpt.empty((1, 2)))


@pytest.mark.parametrize("order", ["C", "F"])
def test_where_contiguous(order):
    get_queue_or_skip()

    cond = dpt.asarray(
        [
            [[True, False, False], [False, True, True]],
            [[False, True, False], [True, False, True]],
            [[False, False, True], [False, False, True]],
            [[False, False, False], [True, False, True]],
            [[True, True, True], [True, False, True]],
        ],
        order=order,
    )

    x1 = dpt.full(cond.shape, 2, dtype="i4", order=order)
    x2 = dpt.full(cond.shape, 3, dtype="i4", order=order)
    expected = np.where(dpt.asnumpy(cond), dpt.asnumpy(x1), dpt.asnumpy(x2))
    res = dpt.where(cond, x1, x2)

    assert _dtype_all_close(dpt.asnumpy(res), expected)


def test_where_contiguous1D():
    get_queue_or_skip()

    cond = dpt.asarray([True, False, True, False, False, True])

    x1 = dpt.full(cond.shape, 2, dtype="i4")
    x2 = dpt.full(cond.shape, 3, dtype="i4")
    expected = np.where(dpt.asnumpy(cond), dpt.asnumpy(x1), dpt.asnumpy(x2))
    res = dpt.where(cond, x1, x2)
    assert_array_equal(dpt.asnumpy(res), expected)

    # test with complex dtype (branch in kernel)
    x1 = dpt.astype(x1, dpt.complex64)
    x2 = dpt.astype(x2, dpt.complex64)
    expected = np.where(dpt.asnumpy(cond), dpt.asnumpy(x1), dpt.asnumpy(x2))
    res = dpt.where(cond, x1, x2)
    assert _dtype_all_close(dpt.asnumpy(res), expected)


def test_where_gh_1170():
    get_queue_or_skip()

    cond = dpt.asarray([False, True, True, False], dtype="?")
    x1 = dpt.ones((3, 4), dtype="i4")
    x2 = dpt.zeros((3, 4), dtype="i4")

    res = dpt.where(cond, x1, x2)
    expected = np.broadcast_to(dpt.asnumpy(cond).astype(x1.dtype), x1.shape)

    assert_array_equal(dpt.asnumpy(res), expected)


def test_where_strided():
    get_queue_or_skip()

    s0, s1 = 4, 9
    cond = dpt.reshape(
        dpt.asarray(
            [True, False, False, False, True, True, False, True, False] * s0
        ),
        (s0, s1),
    )[:, ::3]

    x1 = dpt.reshape(
        dpt.arange(cond.shape[0] * cond.shape[1] * 2, dtype="i4"),
        (cond.shape[0], cond.shape[1] * 2),
    )[:, ::2]
    x2 = dpt.reshape(
        dpt.arange(cond.shape[0] * cond.shape[1] * 3, dtype="i4"),
        (cond.shape[0], cond.shape[1] * 3),
    )[:, ::3]
    expected = np.where(dpt.asnumpy(cond), dpt.asnumpy(x1), dpt.asnumpy(x2))
    res = dpt.where(cond, x1, x2)

    assert_array_equal(dpt.asnumpy(res), expected)

    # negative strides
    res = dpt.where(cond, dpt.flip(x1), x2)
    expected = np.where(
        dpt.asnumpy(cond), np.flip(dpt.asnumpy(x1)), dpt.asnumpy(x2)
    )
    assert_array_equal(dpt.asnumpy(res), expected)

    res = dpt.where(dpt.flip(cond), x1, x2)
    expected = np.where(
        np.flip(dpt.asnumpy(cond)), dpt.asnumpy(x1), dpt.asnumpy(x2)
    )
    assert_array_equal(dpt.asnumpy(res), expected)


def test_where_invariants():
    get_queue_or_skip()

    test_sh = (
        6,
        8,
    )
    mask = dpt.asarray(np.random.choice([True, False], size=test_sh))
    p = dpt.ones(test_sh, dtype=dpt.int16)
    m = dpt.full(test_sh, -1, dtype=dpt.int16)
    inds_list = [
        (
            np.s_[:3],
            np.s_[::2],
        ),
        (
            np.s_[::2],
            np.s_[::2],
        ),
        (
            np.s_[::-1],
            np.s_[:],
        ),
    ]
    for ind in inds_list:
        r1 = dpt.where(mask, p, m)[ind]
        r2 = dpt.where(mask[ind], p[ind], m[ind])
        assert (dpt.asnumpy(r1) == dpt.asnumpy(r2)).all()


def test_where_arg_validation():
    get_queue_or_skip()

    check = dict()
    x1 = dpt.empty((1,), dtype="i4")
    x2 = dpt.empty((1,), dtype="i4")

    with pytest.raises(TypeError):
        dpt.where(check, x1, x2)
    with pytest.raises(TypeError):
        dpt.where(x1, check, x2)
    with pytest.raises(TypeError):
        dpt.where(x1, x2, check)


def test_where_compute_follows_data():
    q1 = get_queue_or_skip()
    q2 = get_queue_or_skip()
    q3 = get_queue_or_skip()

    x1 = dpt.empty((1,), dtype="i4", sycl_queue=q1)
    x2 = dpt.empty((1,), dtype="i4", sycl_queue=q2)

    with pytest.raises(ExecutionPlacementError):
        dpt.where(dpt.empty((1,), dtype="i4", sycl_queue=q1), x1, x2)
    with pytest.raises(ExecutionPlacementError):
        dpt.where(dpt.empty((1,), dtype="i4", sycl_queue=q3), x1, x2)
    with pytest.raises(ExecutionPlacementError):
        dpt.where(x1, x1, x2)


def test_where_order():
    get_queue_or_skip()

    test_sh = (
        20,
        20,
    )
    test_sh2 = tuple(2 * dim for dim in test_sh)
    n = test_sh[-1]

    for dt1, dt2 in zip(["i4", "i4", "f4"], ["i4", "f4", "i4"]):
        ar1 = dpt.zeros(test_sh, dtype=dt1, order="C")
        ar2 = dpt.ones(test_sh, dtype=dt2, order="C")
        condition = dpt.zeros(test_sh, dtype="?", order="C")
        res = dpt.where(condition, ar1, ar2)
        assert res.flags.c_contiguous

        ar1 = dpt.ones(test_sh, dtype=dt1, order="F")
        ar2 = dpt.ones(test_sh, dtype=dt2, order="F")
        condition = dpt.zeros(test_sh, dtype="?", order="F")
        res = dpt.where(condition, ar1, ar2)
        assert res.flags.f_contiguous

        ar1 = dpt.ones(test_sh2, dtype=dt1, order="C")[:20, ::-2]
        ar2 = dpt.ones(test_sh2, dtype=dt2, order="C")[:20, ::-2]
        condition = dpt.zeros(test_sh2, dtype="?", order="C")[:20, ::-2]
        res = dpt.where(condition, ar1, ar2)
        assert res.strides == (n, -1)

        ar1 = dpt.ones(test_sh2, dtype=dt1, order="C")[:20, ::-2].mT
        ar2 = dpt.ones(test_sh2, dtype=dt2, order="C")[:20, ::-2].mT
        condition = dpt.zeros(test_sh2, dtype="?", order="C")[:20, ::-2].mT
        res = dpt.where(condition, ar1, ar2)
        assert res.strides == (-1, n)

        ar1 = dpt.ones(n, dtype=dt1, order="C")
        ar2 = dpt.broadcast_to(dpt.ones(n, dtype=dt2, order="C"), test_sh)
        condition = dpt.zeros(n, dtype="?", order="C")
        res = dpt.where(condition, ar1, ar2)
        assert res.strides == (20, 1)
