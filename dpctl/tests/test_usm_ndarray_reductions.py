#                       Data Parallel Control (dpctl)
#
#  Copyright 2020-2023 Intel Corporation
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


def test_max_min_axis():
    get_queue_or_skip()

    x = dpt.reshape(
        dpt.arange((3 * 4 * 5 * 6 * 7), dtype="i4"), (3, 4, 5, 6, 7)
    )

    m = dpt.max(x, axis=(1, 2, -1))
    assert m.shape == (3, 6)
    assert dpt.all(m == x[:, -1, -1, :, -1])

    m = dpt.min(x, axis=(1, 2, -1))
    assert m.shape == (3, 6)
    assert dpt.all(m == x[:, 0, 0, :, 0])


def test_reduction_keepdims():
    get_queue_or_skip()

    x = dpt.ones((3, 4, 5, 6, 7), dtype="i4")
    m = dpt.max(x, axis=(1, 2, -1), keepdims=True)

    assert m.shape == (3, 1, 1, 6, 1)
    assert dpt.all(m == dpt.reshape(x[:, 0, 0, :, 0], m.shape))


def test_max_scalar():
    get_queue_or_skip()

    x = dpt.ones(())
    m = dpt.max(x)

    assert m.shape == ()
    assert x == m


@pytest.mark.parametrize("arg_dtype", ["i4", "f4", "c8"])
def test_reduction_kernels(arg_dtype):
    # i4 - always uses atomics w/ sycl group reduction
    # f4 - always uses atomics w/ custom group reduction
    # c8 - always uses temps w/ custom group reduction
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arg_dtype, q)

    x = dpt.reshape(
        dpt.arange(24 * 1025, dtype=arg_dtype, sycl_queue=q), (24, 1025)
    )

    m = dpt.max(x)
    assert m == x[-1, -1]
    m = dpt.max(x, axis=0)
    assert dpt.all(m == x[-1, :])
    m = dpt.max(x, axis=1)
    assert dpt.all(m == x[:, -1])

    m = dpt.min(x)
    assert m == x[0, 0]
    m = dpt.min(x, axis=0)
    assert dpt.all(m == x[0, :])
    m = dpt.min(x, axis=1)
    assert dpt.all(m == x[:, 0])


def test_max_min_nan_propagation():
    get_queue_or_skip()

    # float, finites
    x = dpt.arange(4, dtype="f4")
    x[0] = dpt.nan
    assert dpt.isnan(dpt.max(x))
    assert dpt.isnan(dpt.min(x))

    # float, infinities
    x[1:] = dpt.inf
    assert dpt.isnan(dpt.max(x))
    x[1:] = -dpt.inf
    assert dpt.isnan(dpt.min(x))

    # complex
    x = dpt.arange(4, dtype="c8")
    x[0] = complex(dpt.nan, 0)
    assert dpt.isnan(dpt.max(x))
    assert dpt.isnan(dpt.min(x))

    x[0] = complex(0, dpt.nan)
    assert dpt.isnan(dpt.max(x))
    assert dpt.isnan(dpt.min(x))
