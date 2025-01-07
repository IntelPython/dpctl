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

from math import prod

import pytest
from numpy.testing import assert_raises_regex

import dpctl.tensor as dpt
from dpctl.tensor._type_utils import _to_device_supported_dtype
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported
from dpctl.utils import ExecutionPlacementError

_all_dtypes = [
    "?",
    "i1",
    "u1",
    "i2",
    "u2",
    "i4",
    "u4",
    "i8",
    "u8",
    "f2",
    "f4",
    "f8",
    "c8",
    "c16",
]


@pytest.mark.parametrize("dt", _all_dtypes)
def test_diff_basic(dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x = dpt.asarray([9, 12, 7, 17, 10, 18, 15, 9, 8, 8], dtype=dt, sycl_queue=q)
    op = dpt.not_equal if x.dtype is dpt.bool else dpt.subtract

    # test both n=2 and n>2 branches
    for n in [1, 2, 5]:
        res = dpt.diff(x, n=n)
        expected_res = x
        for _ in range(n):
            expected_res = op(expected_res[1:], expected_res[:-1])
        if dpt.dtype(dt).kind in "fc":
            assert dpt.allclose(res, expected_res)
        else:
            assert dpt.all(res == expected_res)


def test_diff_axis():
    get_queue_or_skip()

    x = dpt.tile(
        dpt.asarray([9, 12, 7, 17, 10, 18, 15, 9, 8, 8], dtype="i4"), (3, 4, 1)
    )
    x[:, ::2, :] = 0

    for n in [1, 2, 3]:
        res = dpt.diff(x, n=n, axis=1)
        expected_res = x
        for _ in range(n):
            expected_res = dpt.subtract(
                expected_res[:, 1:, :], expected_res[:, :-1, :]
            )
        assert dpt.all(res == expected_res)


def test_diff_prepend_append_type_promotion():
    get_queue_or_skip()

    dts = [
        ("i1", "u1", "i8"),
        ("i1", "u8", "u1"),
        ("u4", "i4", "f4"),
        ("i8", "c8", "u8"),
    ]

    for dt0, dt1, dt2 in dts:
        x = dpt.ones(10, dtype=dt1)
        prepend = dpt.full(1, 2, dtype=dt0)
        append = dpt.full(1, 3, dtype=dt2)

        res = dpt.diff(x, prepend=prepend, append=append)
        assert res.dtype == _to_device_supported_dtype(
            dpt.result_type(prepend, x, append),
            x.sycl_queue.sycl_device,
        )

        res = dpt.diff(x, prepend=prepend)
        assert res.dtype == _to_device_supported_dtype(
            dpt.result_type(prepend, x),
            x.sycl_queue.sycl_device,
        )

        res = dpt.diff(x, append=append)
        assert res.dtype == _to_device_supported_dtype(
            dpt.result_type(x, append),
            x.sycl_queue.sycl_device,
        )


def test_diff_0d():
    get_queue_or_skip()

    x = dpt.ones(())
    with pytest.raises(ValueError):
        dpt.diff(x)


def test_diff_empty_array():
    get_queue_or_skip()

    x = dpt.ones((3, 0, 5))
    res = dpt.diff(x, axis=1)
    assert res.shape == x.shape

    res = dpt.diff(x, axis=0)
    assert res.shape == (2, 0, 5)

    append = dpt.ones((3, 2, 5))
    res = dpt.diff(x, axis=1, append=append)
    assert res.shape == (3, 1, 5)

    prepend = dpt.ones((3, 2, 5))
    res = dpt.diff(x, axis=1, prepend=prepend)
    assert res.shape == (3, 1, 5)


def test_diff_no_op():
    get_queue_or_skip()

    x = dpt.ones(10, dtype="i4")
    res = dpt.diff(x, n=0)
    assert dpt.all(x == res)

    x = dpt.reshape(x, (2, 5))
    res = dpt.diff(x, n=0, axis=0)
    assert dpt.all(x == res)


@pytest.mark.parametrize("sh,axis", [((1,), 0), ((3, 4, 5), 1)])
def test_diff_prepend_append_py_scalars(sh, axis):
    get_queue_or_skip()

    n = 1

    arr = dpt.ones(sh, dtype="i4")
    zero = 0

    # first and last elements along axis
    # will be checked for correctness
    sl1 = [slice(None)] * arr.ndim
    sl1[axis] = slice(1)
    sl1 = tuple(sl1)

    sl2 = [slice(None)] * arr.ndim
    sl2[axis] = slice(-1, None, None)
    sl2 = tuple(sl2)

    r = dpt.diff(arr, axis=axis, prepend=zero, append=zero)
    assert all(r.shape[i] == arr.shape[i] for i in range(arr.ndim) if i != axis)
    assert r.shape[axis] == arr.shape[axis] + 2 - n
    assert dpt.all(r[sl1] == 1)
    assert dpt.all(r[sl2] == -1)

    r = dpt.diff(arr, axis=axis, prepend=zero)
    assert all(r.shape[i] == arr.shape[i] for i in range(arr.ndim) if i != axis)
    assert r.shape[axis] == arr.shape[axis] + 1 - n
    assert dpt.all(r[sl1] == 1)

    r = dpt.diff(arr, axis=axis, append=zero)
    assert all(r.shape[i] == arr.shape[i] for i in range(arr.ndim) if i != axis)
    assert r.shape[axis] == arr.shape[axis] + 1 - n
    assert dpt.all(r[sl2] == -1)

    r = dpt.diff(arr, axis=axis, prepend=dpt.asarray(zero), append=zero)
    assert all(r.shape[i] == arr.shape[i] for i in range(arr.ndim) if i != axis)
    assert r.shape[axis] == arr.shape[axis] + 2 - n
    assert dpt.all(r[sl1] == 1)
    assert dpt.all(r[sl2] == -1)

    r = dpt.diff(arr, axis=axis, prepend=zero, append=dpt.asarray(zero))
    assert all(r.shape[i] == arr.shape[i] for i in range(arr.ndim) if i != axis)
    assert r.shape[axis] == arr.shape[axis] + 2 - n
    assert dpt.all(r[sl1] == 1)
    assert dpt.all(r[sl2] == -1)


def test_tensor_diff_append_prepend_arrays():
    get_queue_or_skip()

    n = 1
    axis = 0

    for sh in [(5,), (3, 4, 5)]:
        sz = prod(sh)
        arr = dpt.reshape(dpt.arange(sz, 2 * sz, dtype="i4"), sh)
        prepend = dpt.reshape(dpt.arange(sz, dtype="i4"), sh)
        append = dpt.reshape(dpt.arange(2 * sz, 3 * sz, dtype="i4"), sh)
        const_diff = sz / sh[axis]

        r = dpt.diff(arr, axis=axis, prepend=prepend, append=append)
        assert all(
            r.shape[i] == arr.shape[i] for i in range(arr.ndim) if i != axis
        )
        assert (
            r.shape[axis]
            == arr.shape[axis] + prepend.shape[axis] + append.shape[axis] - n
        )
        assert dpt.all(r == const_diff)

        r = dpt.diff(arr, axis=axis, prepend=prepend)
        assert all(
            r.shape[i] == arr.shape[i] for i in range(arr.ndim) if i != axis
        )
        assert r.shape[axis] == arr.shape[axis] + prepend.shape[axis] - n
        assert dpt.all(r == const_diff)

        r = dpt.diff(arr, axis=axis, append=append)
        assert all(
            r.shape[i] == arr.shape[i] for i in range(arr.ndim) if i != axis
        )
        assert r.shape[axis] == arr.shape[axis] + append.shape[axis] - n
        assert dpt.all(r == const_diff)


def test_diff_wrong_append_prepend_shape():
    get_queue_or_skip()

    arr = dpt.ones((3, 4, 5), dtype="i4")
    arr_bad_sh = dpt.ones(2, dtype="i4")

    assert_raises_regex(
        ValueError,
        ".*shape.*is invalid.*",
        dpt.diff,
        arr,
        prepend=arr_bad_sh,
        append=arr_bad_sh,
    )

    assert_raises_regex(
        ValueError,
        ".*shape.*is invalid.*",
        dpt.diff,
        arr,
        prepend=arr,
        append=arr_bad_sh,
    )

    assert_raises_regex(
        ValueError,
        ".*shape.*is invalid.*",
        dpt.diff,
        arr,
        prepend=arr_bad_sh,
    )

    assert_raises_regex(
        ValueError,
        ".*shape.*is invalid.*",
        dpt.diff,
        arr,
        append=arr_bad_sh,
    )


def test_diff_compute_follows_data():
    q1 = get_queue_or_skip()
    q2 = get_queue_or_skip()
    q3 = get_queue_or_skip()

    ar1 = dpt.ones(1, dtype="i4", sycl_queue=q1)
    ar2 = dpt.ones(1, dtype="i4", sycl_queue=q2)
    ar3 = dpt.ones(1, dtype="i4", sycl_queue=q3)

    with pytest.raises(ExecutionPlacementError):
        dpt.diff(ar1, prepend=ar2, append=ar3)

    with pytest.raises(ExecutionPlacementError):
        dpt.diff(ar1, prepend=ar2, append=0)

    with pytest.raises(ExecutionPlacementError):
        dpt.diff(ar1, prepend=0, append=ar2)

    with pytest.raises(ExecutionPlacementError):
        dpt.diff(ar1, prepend=ar2)

    with pytest.raises(ExecutionPlacementError):
        dpt.diff(ar1, append=ar2)


def test_diff_input_validation():
    bad_in = dict()
    assert_raises_regex(
        TypeError,
        "Expecting dpctl.tensor.usm_ndarray type, got.*",
        dpt.diff,
        bad_in,
    )


def test_diff_positive_order():
    get_queue_or_skip()

    x = dpt.ones(1, dtype="i4")
    n = -1
    assert_raises_regex(
        ValueError,
        ".*must be positive.*",
        dpt.diff,
        x,
        n=n,
    )
