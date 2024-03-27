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
from helper import get_queue_or_skip, skip_if_dtype_not_supported

import dpctl.tensor as dpt

sint_types = [
    dpt.int8,
    dpt.int16,
    dpt.int32,
    dpt.int64,
]
uint_types = [
    dpt.uint8,
    dpt.uint16,
    dpt.uint32,
    dpt.uint64,
]
rfp_types = [
    dpt.float16,
    dpt.float32,
    dpt.float64,
]
cfp_types = [
    dpt.complex64,
    dpt.complex128,
]

no_complex_types = [dpt.bool] + sint_types + uint_types + rfp_types

all_types = [dpt.bool] + sint_types + uint_types + rfp_types + cfp_types


@pytest.mark.parametrize("dt", sint_types)
def test_contig_cumsum_sint(dt):
    get_queue_or_skip()
    n = 10000
    x = dpt.repeat(dpt.asarray([1, -1], dtype=dt), n)

    res = dpt.cumulative_sum(x, dtype=dt)

    ar = dpt.arange(n, dtype=dt)
    expected = dpt.concat((1 + ar, dpt.flip(ar)))
    assert dpt.all(res == expected)


@pytest.mark.parametrize("dt", sint_types)
def test_strided_cumsum_sint(dt):
    get_queue_or_skip()
    n = 10000
    x = dpt.repeat(dpt.asarray([1, -1], dtype=dt), 2 * n)[1::2]

    res = dpt.cumulative_sum(x, dtype=dt)

    ar = dpt.arange(n, dtype=dt)
    expected = dpt.concat((1 + ar, dpt.flip(ar)))
    assert dpt.all(res == expected)

    x2 = dpt.repeat(dpt.asarray([-1, 1], dtype=dt), 2 * n)[-1::-2]

    res = dpt.cumulative_sum(x2, dtype=dt)

    ar = dpt.arange(n, dtype=dt)
    expected = dpt.concat((1 + ar, dpt.flip(ar)))
    assert dpt.all(res == expected)


@pytest.mark.parametrize("dt", sint_types)
def test_contig_cumsum_axis_sint(dt):
    get_queue_or_skip()
    n0, n1 = 1000, 173
    x = dpt.repeat(dpt.asarray([1, -1], dtype=dt), n0)
    m = dpt.tile(dpt.expand_dims(x, axis=1), (1, n1))

    res = dpt.cumulative_sum(m, dtype=dt, axis=0)

    ar = dpt.arange(n0, dtype=dt)
    expected = dpt.concat((1 + ar, dpt.flip(ar)))
    assert dpt.all(res == dpt.expand_dims(expected, axis=1))


@pytest.mark.parametrize("dt", sint_types)
def test_strided_cumsum_axis_sint(dt):
    get_queue_or_skip()
    n0, n1 = 1000, 173
    x = dpt.repeat(dpt.asarray([1, -1], dtype=dt), 2 * n0)
    m = dpt.tile(dpt.expand_dims(x, axis=1), (1, n1))[1::2, ::-1]

    res = dpt.cumulative_sum(m, dtype=dt, axis=0)

    ar = dpt.arange(n0, dtype=dt)
    expected = dpt.concat((1 + ar, dpt.flip(ar)))
    assert dpt.all(res == dpt.expand_dims(expected, axis=1))


def test_accumulate_scalar():
    get_queue_or_skip()

    s = dpt.asarray(1, dtype="i8")
    r = dpt.cumulative_sum(s)
    assert r == s
    assert r.ndim == s.ndim

    r = dpt.cumulative_sum(s, include_initial=True)
    r_expected = dpt.asarray([0, 1], dtype="i8")
    assert dpt.all(r == r_expected)


def test_cumulative_sum_include_initial():
    get_queue_or_skip()

    n0, n1 = 3, 5
    x = dpt.ones((n0, n1), dtype="i4")
    r = dpt.cumulative_sum(x, axis=0, include_initial=True)
    assert dpt.all(r[0, :] == 0)

    r = dpt.cumulative_sum(x, axis=1, include_initial=True)
    assert dpt.all(r[:, 0] == 0)

    x = dpt.ones(n1, dtype="i4")
    r = dpt.cumulative_sum(x, include_initial=True)
    assert r.shape == (n1 + 1,)
    assert r[0] == 0


def test_cumulative_prod_identity():
    get_queue_or_skip()

    x = dpt.zeros(5, dtype="i4")
    r = dpt.cumulative_prod(x, include_initial=True)
    assert r[0] == 1


def test_cumulative_logsumexp_identity():
    get_queue_or_skip()

    x = dpt.ones(5, dtype="f4")
    r = dpt.cumulative_logsumexp(x, include_initial=True)
    assert r[0] == -dpt.inf


def test_accumulate_empty_array():
    get_queue_or_skip()

    n0, n1, n2 = 3, 0, 5
    x = dpt.ones((n0, n1, n2), dtype="i8")
    r = dpt.cumulative_sum(x, axis=1)
    assert r.shape == x.shape
    assert r.size == 0

    r = dpt.cumulative_sum(x, axis=0)
    assert r.shape == x.shape
    assert r.size == 0

    r = dpt.cumulative_sum(x, axis=1, include_initial=True)
    assert r.shape == (n0, n1 + 1, n2)
    assert r.size == (n0 * n2)

    r = dpt.cumulative_sum(x, axis=0, include_initial=True)
    assert r.shape == (n0 + 1, n1, n2)
    assert r.size == 0


@pytest.mark.parametrize("arg_dtype", all_types)
def test_cumsum_arg_dtype_default_output_dtype_matrix(arg_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arg_dtype, q)

    n = 100
    x = dpt.ones(n, dtype=arg_dtype)
    r = dpt.cumulative_sum(x)

    assert isinstance(r, dpt.usm_ndarray)
    if x.dtype.kind == "i":
        assert r.dtype.kind == "i"
    elif x.dtype.kind == "u":
        assert r.dtype.kind == "u"
    elif x.dtype.kind == "fc":
        assert r.dtype == arg_dtype

    r_expected = dpt.arange(1, n + 1, dtype=r.dtype)

    assert dpt.all(r == r_expected)


@pytest.mark.parametrize("arg_dtype", all_types)
@pytest.mark.parametrize("out_dtype", all_types)
def test_cumsum_arg_out_dtype_matrix(arg_dtype, out_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arg_dtype, q)
    skip_if_dtype_not_supported(out_dtype, q)

    n = 100
    x = dpt.ones(n, dtype=arg_dtype)
    r = dpt.cumulative_sum(x, dtype=out_dtype)

    assert isinstance(r, dpt.usm_ndarray)
    assert r.dtype == dpt.dtype(out_dtype)
    if out_dtype == dpt.bool:
        assert dpt.all(r)
    else:
        r_expected = dpt.arange(1, n + 1, dtype=out_dtype)
        assert dpt.all(r == r_expected)
