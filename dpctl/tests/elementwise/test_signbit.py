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

import numpy as np
import pytest

import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_signbit_out_type_contig(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    x = dpt.linspace(1, 10, num=256, dtype=arg_dt)
    sb = dpt.signbit(x)
    assert sb.dtype == dpt.bool

    assert not dpt.any(sb)

    x2 = dpt.linspace(-10, -1, num=256, dtype=arg_dt)
    sb2 = dpt.signbit(x2)
    assert dpt.all(sb2)


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_signbit_out_type_strided(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    x = dpt.linspace(1, 10, num=256, dtype=arg_dt)
    sb = dpt.signbit(x[::-3])
    assert sb.dtype == dpt.bool

    assert not dpt.any(sb)

    x2 = dpt.linspace(-10, -1, num=256, dtype=arg_dt)
    sb2 = dpt.signbit(x2[::-3])
    assert dpt.all(sb2)


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_signbit_special_cases_contig(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    n = 63
    x1 = dpt.full(n, -dpt.inf, dtype=arg_dt)
    x2 = dpt.full(n, -0.0, dtype=arg_dt)
    x3 = dpt.full(n, 0.0, dtype=arg_dt)
    x4 = dpt.full(n, dpt.inf, dtype=arg_dt)

    x = dpt.concat((x1, x2, x3, x4))
    actual = dpt.signbit(x)

    expected = dpt.concat(
        (
            dpt.full(x1.size, True),
            dpt.full(x2.size, True),
            dpt.full(x3.size, False),
            dpt.full(x4.size, False),
        )
    )

    assert dpt.all(dpt.equal(actual, expected))


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_signbit_special_cases_strided(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    x1 = dpt.full(63, -dpt.inf, dtype=arg_dt)
    x2 = dpt.full(63, -0.0, dtype=arg_dt)
    x3 = dpt.full(63, 0.0, dtype=arg_dt)
    x4 = dpt.full(63, dpt.inf, dtype=arg_dt)

    x = dpt.concat((x1, x2, x3, x4))
    actual = dpt.signbit(x[::-1])

    expected = dpt.concat(
        (
            dpt.full(x4.size, False),
            dpt.full(x3.size, False),
            dpt.full(x2.size, True),
            dpt.full(x1.size, True),
        )
    )

    assert dpt.all(dpt.equal(actual, expected))
