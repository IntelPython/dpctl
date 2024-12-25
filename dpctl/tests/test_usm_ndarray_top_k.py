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


def _expected_largest_inds(inp, n, shift, k):
    "Computed expected top_k indices for mode='largest'"
    assert k < n
    ones_start_id = shift % (2 * n)

    alloc_dev = inp.device

    if ones_start_id < n:
        expected_inds = dpt.arange(
            ones_start_id, ones_start_id + k, dtype="i8", device=alloc_dev
        )
    else:
        # wrap-around
        ones_end_id = (ones_start_id + n) % (2 * n)
        if ones_end_id >= k:
            expected_inds = dpt.arange(k, dtype="i8", device=alloc_dev)
        else:
            expected_inds = dpt.concat(
                (
                    dpt.arange(ones_end_id, dtype="i8", device=alloc_dev),
                    dpt.arange(
                        ones_start_id,
                        ones_start_id + k - ones_end_id,
                        dtype="i8",
                        device=alloc_dev,
                    ),
                )
            )

    return expected_inds


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
@pytest.mark.parametrize("n", [33, 43, 255, 511, 1021, 8193])
def test_top_k_1d_largest(dtype, n):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    shift, k = 734, 5
    o = dpt.ones(n, dtype=dtype)
    z = dpt.zeros(n, dtype=dtype)
    oz = dpt.concat((o, z))
    inp = dpt.roll(oz, shift)

    expected_inds = _expected_largest_inds(oz, n, shift, k)

    s = dpt.top_k(inp, k, mode="largest")
    assert s.values.shape == (k,)
    assert s.values.dtype == inp.dtype
    assert s.indices.shape == (k,)
    assert dpt.all(s.indices == expected_inds)
    assert dpt.all(s.values == dpt.ones(k, dtype=dtype)), s.values
    assert dpt.all(s.values == inp[s.indices]), s.indices


def _expected_smallest_inds(inp, n, shift, k):
    "Computed expected top_k indices for mode='smallest'"
    assert k < n
    zeros_start_id = (n + shift) % (2 * n)
    zeros_end_id = (shift) % (2 * n)

    alloc_dev = inp.device

    if zeros_start_id < zeros_end_id:
        expected_inds = dpt.arange(
            zeros_start_id, zeros_start_id + k, dtype="i8", device=alloc_dev
        )
    else:
        if zeros_end_id >= k:
            expected_inds = dpt.arange(k, dtype="i8", device=alloc_dev)
        else:
            expected_inds = dpt.concat(
                (
                    dpt.arange(zeros_end_id, dtype="i8", device=alloc_dev),
                    dpt.arange(
                        zeros_start_id,
                        zeros_start_id + k - zeros_end_id,
                        dtype="i8",
                        device=alloc_dev,
                    ),
                )
            )

    return expected_inds


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
@pytest.mark.parametrize("n", [37, 39, 61, 255, 257, 513, 1021, 8193])
def test_top_k_1d_smallest(dtype, n):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    shift, k = 734, 5
    o = dpt.ones(n, dtype=dtype)
    z = dpt.zeros(n, dtype=dtype)
    oz = dpt.concat((o, z))
    inp = dpt.roll(oz, shift)

    expected_inds = _expected_smallest_inds(oz, n, shift, k)

    s = dpt.top_k(inp, k, mode="smallest")
    assert s.values.shape == (k,)
    assert s.values.dtype == inp.dtype
    assert s.indices.shape == (k,)
    assert dpt.all(s.indices == expected_inds)
    assert dpt.all(s.values == dpt.zeros(k, dtype=dtype)), s.values
    assert dpt.all(s.values == inp[s.indices]), s.indices


# triage failing top k radix implementation on CPU
# replicates from Python behavior of radix sort topk implementation
@pytest.mark.parametrize(
    "n",
    [
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        61,
        137,
        255,
        511,
        1021,
        8193,
    ],
)
def test_top_k_largest_1d_radix_i1(n):
    get_queue_or_skip()
    dt = "i1"

    shift, k = 734, 5
    o = dpt.ones(n, dtype=dt)
    z = dpt.zeros(n, dtype=dt)
    oz = dpt.concat((o, z))
    inp = dpt.roll(oz, shift)

    expected_inds = _expected_largest_inds(oz, n, shift, k)

    sorted_v = dpt.sort(inp, descending=True, kind="radixsort")
    argsorted = dpt.argsort(inp, descending=True, kind="radixsort")

    assert dpt.all(sorted_v == inp[argsorted])

    topk_vals = dpt.copy(sorted_v[:k])
    topk_inds = dpt.copy(argsorted[:k])

    assert dpt.all(topk_vals == dpt.ones(k, dtype=dt))
    assert dpt.all(topk_inds == expected_inds)

    assert dpt.all(topk_vals == inp[topk_inds]), topk_inds
