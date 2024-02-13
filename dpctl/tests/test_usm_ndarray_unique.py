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


def test_unique_values_strided():
    get_queue_or_skip()

    n, m = 1000, 20
    inp = dpt.ones((n, m), dtype="i4", order="F")
    inp[:, ::2] = 0

    uv = dpt.unique_values(inp)
    assert dpt.all(uv == dpt.arange(2, dtype="i4"))

    inp = dpt.reshape(inp, -1)
    inp = dpt.flip(dpt.reshape(inp, -1))

    uv = dpt.unique_values(inp)
    assert dpt.all(uv == dpt.arange(2, dtype="i4"))


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


def test_unique_counts_strided():
    get_queue_or_skip()

    n, m = 1000, 20
    inp = dpt.ones((n, m), dtype="i4", order="F")
    inp[:, ::2] = 0

    uv, uv_counts = dpt.unique_counts(inp)
    assert dpt.all(uv == dpt.arange(2, dtype="i4"))
    assert dpt.all(uv_counts == dpt.full(2, n / 2 * m, dtype=uv_counts.dtype))

    inp = dpt.flip(dpt.reshape(inp, -1))

    uv, uv_counts = dpt.unique_counts(inp)
    assert dpt.all(uv == dpt.arange(2, dtype="i4"))
    assert dpt.all(uv_counts == dpt.full(2, n / 2 * m, dtype=uv_counts.dtype))


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


def test_unique_inverse_strided():
    get_queue_or_skip()

    n, m = 1000, 20
    inp = dpt.ones((n, m), dtype="i4", order="F")
    inp[:, ::2] = 0

    uv, inv = dpt.unique_inverse(inp)
    assert dpt.all(uv == dpt.arange(2, dtype="i4"))
    assert dpt.all(inp == uv[inv])
    assert inp.shape == inv.shape

    inp = dpt.flip(dpt.reshape(inp, -1))

    uv, inv = dpt.unique_inverse(inp)
    assert dpt.all(uv == dpt.arange(2, dtype="i4"))
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


def test_unique_all_strided():
    get_queue_or_skip()

    n, m = 1000, 20
    inp = dpt.ones((n, m), dtype="i4", order="F")
    inp[:, ::2] = 0

    uv, ind, inv, uv_counts = dpt.unique_all(inp)
    assert dpt.all(uv == dpt.arange(2, dtype="i4"))
    assert dpt.all(uv == dpt.reshape(inp, -1)[ind])
    assert dpt.all(inp == uv[inv])
    assert inp.shape == inv.shape
    assert dpt.all(uv_counts == dpt.full(2, n / 2 * m, dtype=uv_counts.dtype))

    inp = dpt.flip(dpt.reshape(inp, -1))

    uv, ind, inv, uv_counts = dpt.unique_all(inp)
    assert dpt.all(uv == dpt.arange(2, dtype="i4"))
    assert dpt.all(uv == inp[ind])
    assert dpt.all(inp == uv[inv])
    assert inp.shape == inv.shape
    assert dpt.all(uv_counts == dpt.full(2, n / 2 * m, dtype=uv_counts.dtype))


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
    get_queue_or_skip()
    q = dpctl.SyclQueue()
    x = dpt.arange(10, dtype="i4", sycl_queue=q)

    uv = dpt.unique_values(x)
    assert isinstance(uv, dpctl.tensor.usm_ndarray)
    assert uv.sycl_queue == q
    uv, uc = dpt.unique_counts(x)
    assert isinstance(uv, dpctl.tensor.usm_ndarray)
    assert isinstance(uc, dpctl.tensor.usm_ndarray)
    assert uv.sycl_queue == q
    assert uc.sycl_queue == q
    uv, inv_ind = dpt.unique_inverse(x)
    assert isinstance(uv, dpctl.tensor.usm_ndarray)
    assert isinstance(inv_ind, dpctl.tensor.usm_ndarray)
    assert uv.sycl_queue == q
    assert inv_ind.sycl_queue == q
    uv, ind, inv_ind, uc = dpt.unique_all(x)
    assert isinstance(uv, dpctl.tensor.usm_ndarray)
    assert isinstance(ind, dpctl.tensor.usm_ndarray)
    assert isinstance(inv_ind, dpctl.tensor.usm_ndarray)
    assert isinstance(uc, dpctl.tensor.usm_ndarray)
    assert uv.sycl_queue == q
    assert ind.sycl_queue == q
    assert inv_ind.sycl_queue == q
    assert uc.sycl_queue == q
