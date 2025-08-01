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

import itertools

import numpy as np
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
def test_sort_1d(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    inp = dpt.roll(
        dpt.concat(
            (dpt.ones(10000, dtype=dtype), dpt.zeros(10000, dtype=dtype))
        ),
        734,
    )

    s = dpt.sort(inp, descending=False)
    assert dpt.all(s[:-1] <= s[1:])

    s1 = dpt.sort(inp, descending=True)
    assert dpt.all(s1[:-1] >= s1[1:])


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
def test_sort_2d(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    fl = dpt.roll(
        dpt.concat(
            (dpt.ones(10000, dtype=dtype), dpt.zeros(10000, dtype=dtype))
        ),
        734,
    )
    inp = dpt.reshape(fl, (20, -1))

    s = dpt.sort(inp, axis=1, descending=False)
    assert dpt.all(s[:, :-1] <= s[:, 1:])

    s1 = dpt.sort(inp, axis=1, descending=True)
    assert dpt.all(s1[:, :-1] >= s1[:, 1:])


def test_sort_strides():
    get_queue_or_skip()

    fl = dpt.roll(
        dpt.concat((dpt.ones(10000, dtype="i4"), dpt.zeros(10000, dtype="i4"))),
        734,
    )
    inp = dpt.reshape(fl, (-1, 20))

    s = dpt.sort(inp, axis=0, descending=False)
    assert dpt.all(s[:-1, :] <= s[1:, :])

    s1 = dpt.sort(inp, axis=0, descending=True)
    assert dpt.all(s1[:-1, :] >= s1[1:, :])


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
def test_argsort_1d(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    inp = dpt.roll(
        dpt.concat(
            (dpt.ones(10000, dtype=dtype), dpt.zeros(10000, dtype=dtype))
        ),
        734,
    )

    s_idx = dpt.argsort(inp, descending=False)
    assert dpt.all(inp[s_idx[:-1]] <= inp[s_idx[1:]])

    s1_idx = dpt.argsort(inp, descending=True)
    assert dpt.all(inp[s1_idx[:-1]] >= inp[s1_idx[1:]])


def test_sort_validation():
    with pytest.raises(TypeError):
        dpt.sort(dict())


def test_sort_validation_kind():
    get_queue_or_skip()

    x = dpt.ones(128, dtype="u1")

    with pytest.raises(ValueError):
        dpt.sort(x, kind=Ellipsis)

    with pytest.raises(ValueError):
        dpt.sort(x, kind="invalid")


def test_argsort_validation():
    with pytest.raises(TypeError):
        dpt.argsort(dict())


def test_argsort_validation_kind():
    get_queue_or_skip()

    x = dpt.arange(127, stop=0, step=-1, dtype="i1")

    with pytest.raises(ValueError):
        dpt.argsort(x, kind=Ellipsis)

    with pytest.raises(ValueError):
        dpt.argsort(x, kind="invalid")


_all_kinds = ["stable", "mergesort", "radixsort"]


@pytest.mark.parametrize("kind", _all_kinds)
def test_sort_axis0(kind):
    get_queue_or_skip()

    n, m = 200, 30
    xf = dpt.arange(n * m, 0, step=-1, dtype="i4")
    x = dpt.reshape(xf, (n, m))
    s = dpt.sort(x, axis=0, kind=kind)

    assert dpt.all(s[:-1, :] <= s[1:, :])


@pytest.mark.parametrize("kind", _all_kinds)
def test_argsort_axis0(kind):
    get_queue_or_skip()

    n, m = 200, 30
    xf = dpt.arange(n * m, 0, step=-1, dtype="i4")
    x = dpt.reshape(xf, (n, m))
    idx = dpt.argsort(x, axis=0, kind=kind)

    s = dpt.take_along_axis(x, idx, axis=0)

    assert dpt.all(s[:-1, :] <= s[1:, :])


@pytest.mark.parametrize("kind", _all_kinds)
def test_argsort_axis1(kind):
    get_queue_or_skip()

    n, m = 200, 30
    xf = dpt.arange(n * m, 0, step=-1, dtype="i4")
    x = dpt.reshape(xf, (n, m))
    idx = dpt.argsort(x, axis=1, kind=kind)

    s = dpt.take_along_axis(x, idx, axis=1)

    assert dpt.all(s[:, :-1] <= s[:, 1:])


@pytest.mark.parametrize("kind", _all_kinds)
def test_sort_strided(kind):
    get_queue_or_skip()

    x_orig = dpt.arange(100, dtype="i4")
    x_flipped = dpt.flip(x_orig, axis=0)
    s = dpt.sort(x_flipped, kind=kind)

    assert dpt.all(s == x_orig)


@pytest.mark.parametrize("kind", _all_kinds)
def test_argsort_strided(kind):
    get_queue_or_skip()

    x_orig = dpt.arange(100, dtype="i4")
    x_flipped = dpt.flip(x_orig, axis=0)
    idx = dpt.argsort(x_flipped, kind=kind)
    s = dpt.take_along_axis(x_flipped, idx, axis=0)

    assert dpt.all(s == x_orig)


@pytest.mark.parametrize("kind", _all_kinds)
def test_sort_0d_array(kind):
    get_queue_or_skip()

    x = dpt.asarray(1, dtype="i4")
    expected = dpt.asarray(1, dtype="i4")
    assert dpt.sort(x, kind=kind) == expected


@pytest.mark.parametrize("kind", _all_kinds)
def test_argsort_0d_array(kind):
    get_queue_or_skip()

    x = dpt.asarray(1, dtype="i4")
    expected = dpt.asarray(0, dtype="i4")
    assert dpt.argsort(x, kind=kind) == expected


@pytest.mark.parametrize(
    "dtype",
    [
        "f2",
        "f4",
        "f8",
    ],
)
@pytest.mark.parametrize("kind", _all_kinds)
def test_sort_real_fp_nan(dtype, kind):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = dpt.asarray(
        [-0.0, 0.1, dpt.nan, 0.0, -0.1, dpt.nan, 0.2, -0.3], dtype=dtype
    )
    s = dpt.sort(x, kind=kind)

    expected = dpt.asarray(
        [-0.3, -0.1, -0.0, 0.0, 0.1, 0.2, dpt.nan, dpt.nan], dtype=dtype
    )

    assert dpt.allclose(s, expected, equal_nan=True)

    s = dpt.sort(x, descending=True, kind=kind)

    expected = dpt.asarray(
        [dpt.nan, dpt.nan, 0.2, 0.1, -0.0, 0.0, -0.1, -0.3], dtype=dtype
    )

    assert dpt.allclose(s, expected, equal_nan=True)


@pytest.mark.parametrize(
    "dtype",
    [
        "c8",
        "c16",
    ],
)
def test_sort_complex_fp_nan(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    rvs = [-0.0, 0.1, 0.0, 0.2, -0.3, dpt.nan]
    ivs = [-0.0, 0.1, 0.0, 0.2, -0.3, dpt.nan]

    cv = []
    for rv in rvs:
        for iv in ivs:
            cv.append(complex(rv, iv))

    inp = dpt.asarray(cv, dtype=dtype)
    s = dpt.sort(inp)

    expected = np.sort(dpt.asnumpy(inp))

    assert np.allclose(dpt.asnumpy(s), expected, equal_nan=True)

    pairs = []
    for i, j in itertools.permutations(range(inp.shape[0]), 2):
        pairs.append([i, j])
    sub_arrs = inp[dpt.asarray(pairs)]
    m1 = dpt.asnumpy(dpt.sort(sub_arrs, axis=1))
    m2 = np.sort(dpt.asnumpy(sub_arrs), axis=1)
    for k in range(len(pairs)):
        i, j = pairs[k]
        r1 = m1[k]
        r2 = m2[k]
        assert np.array_equal(
            r1.view(np.int64), r2.view(np.int64)
        ), f"Failed for {i} and {j}"


def test_radix_sort_size_1_axis():
    get_queue_or_skip()

    x1 = dpt.ones((), dtype="i1")
    r1 = dpt.sort(x1, kind="radixsort")
    assert r1 == x1

    x2 = dpt.ones([1], dtype="i1")
    r2 = dpt.sort(x2, kind="radixsort")
    assert r2 == x2

    x3 = dpt.reshape(dpt.arange(10, dtype="i1"), (10, 1))
    r3 = dpt.sort(x3, kind="radixsort")
    assert dpt.all(r3 == x3)

    x4 = dpt.reshape(dpt.arange(10, dtype="i1"), (1, 10))
    r4 = dpt.sort(x4, axis=0, kind="radixsort")
    assert dpt.all(r4 == x4)


def test_radix_argsort_size_1_axis():
    get_queue_or_skip()

    x1 = dpt.ones((), dtype="i1")
    r1 = dpt.argsort(x1, kind="radixsort")
    assert r1 == 0

    x2 = dpt.ones([1], dtype="i1")
    r2 = dpt.argsort(x2, kind="radixsort")
    assert r2 == 0

    x3 = dpt.reshape(dpt.arange(10, dtype="i1"), (10, 1))
    r3 = dpt.argsort(x3, kind="radixsort")
    assert dpt.all(r3 == 0)

    x4 = dpt.reshape(dpt.arange(10, dtype="i1"), (1, 10))
    r4 = dpt.argsort(x4, axis=0, kind="radixsort")
    assert dpt.all(r4 == 0)
