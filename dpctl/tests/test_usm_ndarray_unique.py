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

import dpctl
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
    assert inp.shape == inv.shape


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
    assert inp.shape == inv.shape
    assert dpt.all(uv_counts == dpt.full(2, n, dtype=uv_counts.dtype))


def test_set_functions_empty_input():
    get_queue_or_skip()
    x = dpt.ones((10, 0, 1), dtype="i4")

    res = dpt.unique_values(x)
    assert isinstance(res, dpctl.tensor.usm_ndarray)
    assert res.size == 0
    assert res.dtype == x.dtype

    res = dpt.unique_inverse(x)
    assert type(res).__name__ == "UniqueInverseResult"
    uv, inv = res
    assert isinstance(uv, dpctl.tensor.usm_ndarray)
    assert uv.size == 0
    assert isinstance(inv, dpctl.tensor.usm_ndarray)
    assert inv.size == 0

    res = dpt.unique_counts(x)
    assert type(res).__name__ == "UniqueCountsResult"
    uv, uv_counts = res
    assert isinstance(uv, dpctl.tensor.usm_ndarray)
    assert uv.size == 0
    assert isinstance(uv_counts, dpctl.tensor.usm_ndarray)
    assert uv_counts.size == 0

    res = dpt.unique_all(x)
    assert type(res).__name__ == "UniqueAllResult"
    uv, ind, inv, uv_counts = res
    assert isinstance(uv, dpctl.tensor.usm_ndarray)
    assert uv.size == 0
    assert isinstance(ind, dpctl.tensor.usm_ndarray)
    assert ind.size == 0
    assert isinstance(inv, dpctl.tensor.usm_ndarray)
    assert inv.size == 0
    assert isinstance(uv_counts, dpctl.tensor.usm_ndarray)
    assert uv_counts.size == 0


def test_set_function_outputs():
    get_queue_or_skip()
    # check standard and early exit paths
    x1 = dpt.arange(10, dtype="i4")
    x2 = dpt.ones((10, 10), dtype="i4")

    assert isinstance(dpt.unique_values(x1), dpctl.tensor.usm_ndarray)
    assert isinstance(dpt.unique_values(x2), dpctl.tensor.usm_ndarray)

    assert type(dpt.unique_inverse(x1)).__name__ == "UniqueInverseResult"
    assert type(dpt.unique_inverse(x2)).__name__ == "UniqueInverseResult"

    assert type(dpt.unique_counts(x1)).__name__ == "UniqueCountsResult"
    assert type(dpt.unique_counts(x2)).__name__ == "UniqueCountsResult"

    assert type(dpt.unique_all(x1)).__name__ == "UniqueAllResult"
    assert type(dpt.unique_all(x2)).__name__ == "UniqueAllResult"


def test_set_functions_compute_follows_data():
    # tests that all intermediate calls and allocations
    # are compatible with an input with an arbitrary queue
    q = dpctl.SyclQueue()
    x = dpt.arange(10, dtype="i4", sycl_queue=q)

    assert isinstance(dpt.unique_values(x), dpctl.tensor.usm_ndarray)
    assert dpt.unique_counts(x)
    assert dpt.unique_inverse(x)
    assert dpt.unique_all(x)
