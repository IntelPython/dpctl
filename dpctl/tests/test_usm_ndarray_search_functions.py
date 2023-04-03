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
    get_queue_or_skip

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


@pytest.mark.parametrize("dt1", _all_dtypes)
@pytest.mark.parametrize("dt2", _all_dtypes)
def test_where_all_dtypes(dt1, dt2):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt1, q)
    skip_if_dtype_not_supported(dt2, q)

    cond_np = np.arange(5) > 2
    x1_np = np.asarray(2, dtype=dt1)
    x2_np = np.asarray(3, dtype=dt2)

    cond = dpt.asarray(cond_np, sycl_queue=q)
    x1 = dpt.asarray(x1_np, sycl_queue=q)
    x2 = dpt.asarray(x2_np, sycl_queue=q)

    res = dpt.where(cond, x1, x2)
    res_np = np.where(cond_np, x1_np, x2_np)

    if res.dtype != res_np.dtype:
        assert res.dtype.kind == res_np.dtype.kind
        assert_array_equal(dpt.asnumpy(res).astype(res_np.dtype), res_np)

    else:
        assert_array_equal(dpt.asnumpy(res), res_np)
