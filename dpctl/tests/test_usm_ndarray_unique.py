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

import pytest

import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported


@pytest.mark.parametrize(
    "dtype",
    [
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
    ],
)
def test_unique_values(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n, roll = 10000, 734
    inp = dpt.roll(
        dpt.concat((dpt.ones(n, dtype=dtype), dpt.zeros(n, dtype=dtype))),
        roll,
    )

    uv = dpt.unique_values(inp)
    assert dpt.all(uv == dpt.arange(2, dtype=dtype))


@pytest.mark.parametrize(
    "dtype",
    [
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
    ],
)
def test_unique_counts(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n, roll = 10000, 734
    inp = dpt.roll(
        dpt.concat((dpt.ones(n, dtype=dtype), dpt.zeros(n, dtype=dtype))),
        roll,
    )

    uv, uv_counts = dpt.unique_counts(inp)
    assert dpt.all(uv == dpt.arange(2, dtype=dtype))
    assert dpt.all(uv_counts == dpt.full(2, n, dtype=uv_counts.dtype))


@pytest.mark.parametrize(
    "dtype",
    [
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
    ],
)
def test_unique_inverse(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n, roll = 10000, 734
    inp = dpt.roll(
        dpt.concat((dpt.ones(n, dtype=dtype), dpt.zeros(n, dtype=dtype))),
        roll,
    )

    uv, inv = dpt.unique_inverse(inp)
    assert dpt.all(uv == dpt.arange(2, dtype=dtype))
    assert dpt.all(inp == uv[inv])


@pytest.mark.parametrize(
    "dtype",
    [
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
    ],
)
def test_unique_all(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n, roll = 10000, 734
    inp = dpt.roll(
        dpt.concat((dpt.ones(n, dtype=dtype), dpt.zeros(n, dtype=dtype))),
        roll,
    )

    uv, ind, inv, uv_counts = dpt.unique_all(inp)
    assert dpt.all(uv == dpt.arange(2, dtype=dtype))
    assert dpt.all(uv == inp[ind])
    assert dpt.all(inp == uv[inv])
    assert dpt.all(uv_counts == dpt.full(2, n, dtype=uv_counts.dtype))
