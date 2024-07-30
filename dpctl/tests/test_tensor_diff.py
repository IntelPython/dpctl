#                       Data Parallel Control (dpctl)
#
#  Copyright 2020-2024 Intel Corporation
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

import pytest

import dpctl.tensor as dpt
from dpctl.tensor._type_utils import _to_device_supported_dtype
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

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
    res = dpt.diff(x)
    op = dpt.not_equal if x.dtype is dpt.bool else dpt.subtract
    expected_res = op(x[1:], x[:-1])
    if dpt.dtype(dt).kind in "fc":
        assert dpt.allclose(res, expected_res)
    else:
        assert dpt.all(res == expected_res)

    res = dpt.diff(x, n=5)
    expected_res = x
    for _ in range(5):
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
    res = dpt.diff(x, n=1, axis=1)
    expected_res = dpt.subtract(x[:, 1:, :], x[:, :-1, :])
    assert dpt.all(res == expected_res)

    res = dpt.diff(x, n=3, axis=1)
    expected_res = x
    for _ in range(3):
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

    for _dts in dts:
        x = dpt.ones(10, dtype=_dts[1])
        prepend = dpt.full(1, 2, dtype=_dts[0])
        append = dpt.full(1, 3, dtype=_dts[2])

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

    res = dpt.diff(dpt.reshape(x, (2, 5)), n=0, axis=0)
    assert dpt.all(x == res)


@pytest.mark.parametrize("sh,axis", [((1,), 0), ((3, 4, 5), 1)])
def test_diff_prepend_append_py_scalars(sh, axis):
    get_queue_or_skip()

    arrs = [
        dpt.ones(sh, dtype="?"),
        dpt.ones(sh, dtype="i4"),
        dpt.ones(sh, dtype="f4"),
        dpt.ones(sh, dtype="c8"),
    ]

    py_zeros = [
        False,
        0,
        0.0,
        complex(0, 0),
    ]

    py_ones = [
        True,
        1,
        1.0,
        complex(1, 0),
    ]

    for zero, one, arr in zip(py_zeros, py_ones, arrs):
        n = 1
        r = dpt.diff(arr, n=n, axis=axis, prepend=zero, append=one)
        assert isinstance(r, dpt.usm_ndarray)
        assert all(
            r.shape[i] == arr.shape[i] for i in range(arr.ndim) if i != axis
        )
        assert r.shape[axis] == arr.shape[axis] + 2 - n

        r = dpt.diff(arr, n=n, axis=axis, prepend=zero)
        assert isinstance(r, dpt.usm_ndarray)
        assert all(
            r.shape[i] == arr.shape[i] for i in range(arr.ndim) if i != axis
        )
        assert r.shape[axis] == arr.shape[axis] + 1 - n
