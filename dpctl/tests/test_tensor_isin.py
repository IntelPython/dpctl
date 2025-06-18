#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2025 Intel Corporation
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
from dpctl.utils import ExecutionPlacementError


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
def test_isin_basic(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n = 100
    x = dpt.arange(n, dtype=dtype, sycl_queue=q)
    test = dpt.arange(n - 1, dtype=dtype, sycl_queue=q)
    r1 = dpt.isin(x, test)
    assert dpt.all(r1[:-1])
    assert not r1[-1]
    assert r1.shape == x.shape

    # test with invert keyword
    r2 = dpt.isin(x, test, invert=True)
    assert not dpt.any(r2[:-1])
    assert r2[-1]
    assert r2.shape == x.shape


def test_isin_basic_bool():
    dt = dpt.bool
    n = 100
    x = dpt.zeros(n, dtype=dt)
    x[-1] = True
    test = dpt.zeros((), dtype=dt)
    r1 = dpt.isin(x, test)
    assert dpt.all(r1[:-1])
    assert not r1[-1]
    assert r1.shape == x.shape

    r2 = dpt.isin(x, test, invert=True)
    assert not dpt.any(r2[:-1])
    assert r2[-1]
    assert r2.shape == x.shape


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
def test_isin_strided(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n, m = 100, 20
    x = dpt.zeros((n, m), dtype=dtype, order="F", sycl_queue=q)
    x[:, ::2] = dpt.arange(1, (m / 2) + 1, dtype=dtype, sycl_queue=q)
    x_s = x[:, ::2]
    test = dpt.arange(1, (m / 2), dtype=dtype, sycl_queue=q)
    r1 = dpt.isin(x_s, test)
    assert dpt.all(r1[:, :-1])
    assert not dpt.any(r1[:, -1])
    assert not dpt.any(x[:, 1::2])
    assert r1.shape == x_s.shape

    # test with invert keyword
    r2 = dpt.isin(x_s, test, invert=True)
    assert not dpt.any(r2[:, :-1])
    assert dpt.all(r2[:, -1])
    assert not dpt.any(x[:, 1:2])
    assert r2.shape == x_s.shape


def test_isin_strided_bool():
    dt = dpt.bool

    n, m = 100, 20
    x = dpt.zeros((n, m), dtype=dt, order="F")
    x[:, :-2:2] = True
    x_s = x[:, ::2]
    test = dpt.ones((), dtype=dt)
    r1 = dpt.isin(x_s, test)
    assert dpt.all(r1[:, :-1])
    assert not dpt.any(r1[:, -1])
    assert not dpt.any(x[:, 1::2])
    assert r1.shape == x_s.shape

    # test with invert keyword
    r2 = dpt.isin(x_s, test, invert=True)
    assert not dpt.any(r2[:, :-1])
    assert dpt.all(r2[:, -1])
    assert not dpt.any(x[:, 1:2])
    assert r2.shape == x_s.shape


def test_isin_empty_inputs():
    get_queue_or_skip()

    x = dpt.ones((10, 0, 1), dtype="i4")
    test = dpt.ones((), dtype="i4")
    res1 = dpt.isin(x, test)
    assert isinstance(res1, dpt.usm_ndarray)
    assert res1.size == 0
    assert res1.shape == x.shape
    assert res1.dtype == dpt.bool

    res2 = dpt.isin(x, test, invert=True)
    assert isinstance(res2, dpt.usm_ndarray)
    assert res2.size == 0
    assert res2.shape == x.shape
    assert res2.dtype == dpt.bool

    x = dpt.ones((3, 3), dtype="i4")
    test = dpt.ones(0, dtype="i4")
    res3 = dpt.isin(x, test)
    assert isinstance(res3, dpt.usm_ndarray)
    assert res3.shape == x.shape
    assert res3.dtype == dpt.bool
    assert not dpt.all(res3)

    res4 = dpt.isin(x, test, invert=True)
    assert isinstance(res4, dpt.usm_ndarray)
    assert res4.shape == x.shape
    assert res4.dtype == dpt.bool
    assert dpt.all(res4)


def test_isin_validation():
    with pytest.raises(ExecutionPlacementError):
        dpt.isin(1, 1)
