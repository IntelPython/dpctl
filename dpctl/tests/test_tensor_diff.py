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
