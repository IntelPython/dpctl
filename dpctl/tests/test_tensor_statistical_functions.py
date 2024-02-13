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

import pytest

import dpctl.tensor as dpt
from dpctl.tensor._tensor_impl import default_device_fp_type
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

_no_complex_dtypes = [
    "?",
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
]


@pytest.mark.parametrize("dt", _no_complex_dtypes)
def test_mean_dtypes(dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x = dpt.ones(10, dtype=dt)
    res = dpt.mean(x)
    assert res == 1
    if x.dtype.kind in "biu":
        assert res.dtype == dpt.dtype(default_device_fp_type(q))
    else:
        assert res.dtype == x.dtype


@pytest.mark.parametrize("dt", _no_complex_dtypes)
@pytest.mark.parametrize("py_zero", [float(0), int(0)])
def test_std_var_dtypes(dt, py_zero):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x = dpt.ones(10, dtype=dt)
    res = dpt.std(x, correction=py_zero)
    assert res == 0
    if x.dtype.kind in "biu":
        assert res.dtype == dpt.dtype(default_device_fp_type(q))
    else:
        assert res.dtype == x.dtype

    res = dpt.var(x, correction=py_zero)
    assert res == 0
    if x.dtype.kind in "biu":
        assert res.dtype == dpt.dtype(default_device_fp_type(q))
    else:
        assert res.dtype == x.dtype


def test_stat_fns_axis():
    get_queue_or_skip()

    x = dpt.ones((3, 4, 5, 6, 7), dtype="f4")
    m = dpt.mean(x, axis=(1, 2, -1))

    assert isinstance(m, dpt.usm_ndarray)
    assert m.shape == (3, 6)
    assert dpt.allclose(m, dpt.asarray(1, dtype=m.dtype))

    s = dpt.var(x, axis=(1, 2, -1))
    assert isinstance(s, dpt.usm_ndarray)
    assert s.shape == (3, 6)
    assert dpt.allclose(s, dpt.asarray(0, dtype=s.dtype))


@pytest.mark.parametrize("fn", [dpt.mean, dpt.var])
def test_stat_fns_empty(fn):
    get_queue_or_skip()
    x = dpt.empty((0,), dtype="f4")
    r = fn(x)
    assert r.shape == tuple()
    assert dpt.isnan(r)

    x = dpt.empty((10, 0, 2), dtype="f4")
    r = fn(x, axis=1)
    assert r.shape == (10, 2)
    assert dpt.all(dpt.isnan(r))

    r = fn(x, axis=0)
    assert r.shape == (0, 2)
    assert r.size == 0


def test_stat_fns_keepdims():
    get_queue_or_skip()

    x = dpt.ones((3, 4, 5, 6, 7), dtype="f4")
    m = dpt.mean(x, axis=(1, 2, -1), keepdims=True)

    assert isinstance(m, dpt.usm_ndarray)
    assert m.shape == (3, 1, 1, 6, 1)
    assert dpt.allclose(m, dpt.asarray(1, dtype=m.dtype))

    s = dpt.var(x, axis=(1, 2, -1), keepdims=True)
    assert isinstance(s, dpt.usm_ndarray)
    assert s.shape == (3, 1, 1, 6, 1)
    assert dpt.allclose(s, dpt.asarray(0, dtype=s.dtype))


def test_stat_fns_empty_axis():
    get_queue_or_skip()

    x = dpt.reshape(dpt.arange(3 * 4 * 5, dtype="f4"), (3, 4, 5))
    m = dpt.mean(x, axis=())

    assert x.shape == m.shape
    assert dpt.all(x == m)

    s = dpt.var(x, axis=())
    assert x.shape == s.shape
    assert dpt.all(s == 0)

    d = dpt.std(x, axis=())
    assert x.shape == d.shape
    assert dpt.all(d == 0)


def test_mean():
    get_queue_or_skip()

    x = dpt.reshape(dpt.arange(9, dtype="f4"), (3, 3))
    m = dpt.mean(x)
    expected = dpt.asarray(4, dtype="f4")
    assert dpt.allclose(m, expected)

    m = dpt.mean(x, axis=0)
    expected = dpt.arange(3, 6, dtype="f4")
    assert dpt.allclose(m, expected)

    m = dpt.mean(x, axis=1)
    expected = dpt.asarray([1, 4, 7], dtype="f4")
    assert dpt.allclose(m, expected)


def test_var_std():
    get_queue_or_skip()

    x = dpt.reshape(dpt.arange(9, dtype="f4"), (3, 3))
    r = dpt.var(x)
    expected = dpt.asarray(6.666666507720947, dtype="f4")
    assert dpt.allclose(r, expected)

    r1 = dpt.var(x, correction=3)
    expected1 = dpt.asarray(10.0, dtype="f4")
    assert dpt.allclose(r1, expected1)

    r = dpt.std(x)
    expected = dpt.sqrt(expected)
    assert dpt.allclose(r, expected)

    r1 = dpt.std(x, correction=3)
    expected1 = dpt.sqrt(expected1)
    assert dpt.allclose(r1, expected1)

    r = dpt.var(x, axis=0)
    expected = dpt.full(x.shape[1], 6, dtype="f4")
    assert dpt.allclose(r, expected)

    r1 = dpt.var(x, axis=0, correction=1)
    expected1 = dpt.full(x.shape[1], 9, dtype="f4")
    assert dpt.allclose(r1, expected1)

    r = dpt.std(x, axis=0)
    expected = dpt.sqrt(expected)
    assert dpt.allclose(r, expected)

    r1 = dpt.std(x, axis=0, correction=1)
    expected1 = dpt.sqrt(expected1)
    assert dpt.allclose(r1, expected1)

    r = dpt.var(x, axis=1)
    expected = dpt.full(x.shape[0], 0.6666666865348816, dtype="f4")
    assert dpt.allclose(r, expected)

    r1 = dpt.var(x, axis=1, correction=1)
    expected1 = dpt.ones(x.shape[0], dtype="f4")
    assert dpt.allclose(r1, expected1)

    r = dpt.std(x, axis=1)
    expected = dpt.sqrt(expected)
    assert dpt.allclose(r, expected)

    r1 = dpt.std(x, axis=1, correction=1)
    expected1 = dpt.sqrt(expected1)
    assert dpt.allclose(r1, expected1)


def test_var_axis_length_correction():
    get_queue_or_skip()

    x = dpt.reshape(dpt.arange(9, dtype="f4"), (3, 3))

    r = dpt.var(x, correction=x.size)
    assert dpt.isnan(r)

    r = dpt.var(x, axis=0, correction=x.shape[0])
    assert dpt.all(dpt.isnan(r))

    r = dpt.var(x, axis=1, correction=x.shape[1])
    assert dpt.all(dpt.isnan(r))


def test_stat_function_errors():
    d = dict()
    with pytest.raises(TypeError):
        dpt.var(d)
    with pytest.raises(TypeError):
        dpt.std(d)
    with pytest.raises(TypeError):
        dpt.mean(d)

    get_queue_or_skip()
    x = dpt.empty(1, dtype="f4")
    with pytest.raises(TypeError):
        dpt.var(x, axis=d)
    with pytest.raises(TypeError):
        dpt.std(x, axis=d)
    with pytest.raises(TypeError):
        dpt.mean(x, axis=d)

    with pytest.raises(TypeError):
        dpt.var(x, correction=d)
    with pytest.raises(TypeError):
        dpt.std(x, correction=d)

    x = dpt.empty(1, dtype="c8")
    with pytest.raises(ValueError):
        dpt.var(x)
    with pytest.raises(ValueError):
        dpt.std(x)
