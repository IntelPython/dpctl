#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2024 Intel Corporation
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
@pytest.mark.parametrize("n", [33, 255, 511, 1021, 8193])
def test_topk_1d_largest(dtype, n):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    o = dpt.ones(n, dtype=dtype)
    z = dpt.zeros(n, dtype=dtype)
    zo = dpt.concat((o, z))
    inp = dpt.roll(zo, 734)
    k = 5

    s = dpt.top_k(inp, k, mode="largest")
    assert s.values.shape == (k,)
    assert s.values.dtype == inp.dtype
    assert s.indices.shape == (k,)
    assert dpt.all(s.values == dpt.ones(k, dtype=dtype))
    assert dpt.all(s.values == inp[s.indices])


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
@pytest.mark.parametrize("n", [33, 255, 257, 513, 1021, 8193])
def test_topk_1d_smallest(dtype, n):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    o = dpt.ones(n, dtype=dtype)
    z = dpt.zeros(n, dtype=dtype)
    zo = dpt.concat((o, z))
    inp = dpt.roll(zo, 734)
    k = 5

    s = dpt.top_k(inp, k, mode="smallest")
    assert s.values.shape == (k,)
    assert s.values.dtype == inp.dtype
    assert s.indices.shape == (k,)
    assert dpt.all(s.values == dpt.zeros(k, dtype=dtype))
    assert dpt.all(s.values == inp[s.indices])


# triage failing top k radix implementation on CPU
# replicates from Python behavior of radix sort topk implementation
@pytest.mark.parametrize("n", [33, 255, 511, 1021, 8193])
def test_topk_largest_1d_radix_i1_255(n):
    get_queue_or_skip()
    dt = "i1"

    o = dpt.ones(n, dtype=dt)
    z = dpt.zeros(n, dtype=dt)
    zo = dpt.concat((o, z))
    inp = dpt.roll(zo, 734)
    k = 5

    sorted = dpt.copy(dpt.sort(inp, descending=True, kind="radixsort")[:k])
    argsorted = dpt.copy(
        dpt.argsort(inp, descending=True, kind="radixsort")[:k]
    )
    assert dpt.all(sorted == dpt.ones(k, dtype=dt))
    assert dpt.all(sorted == inp[argsorted])
