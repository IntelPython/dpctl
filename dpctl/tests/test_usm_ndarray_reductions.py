#                       Data Parallel Control (dpctl)
#
#  Copyright 2020-2023 Intel Corporation
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

from random import randrange

import numpy as np
import pytest
from numpy.testing import assert_allclose

import dpctl.tensor as dpt
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


_all_dtypes = _no_complex_dtypes + [
    "c8",
    "c16",
]


def test_max_min_axis():
    get_queue_or_skip()

    x = dpt.reshape(
        dpt.arange((3 * 4 * 5 * 6 * 7), dtype="i4"), (3, 4, 5, 6, 7)
    )

    m = dpt.max(x, axis=(1, 2, -1))
    assert m.shape == (3, 6)
    assert dpt.all(m == x[:, -1, -1, :, -1])

    m = dpt.min(x, axis=(1, 2, -1))
    assert m.shape == (3, 6)
    assert dpt.all(m == x[:, 0, 0, :, 0])


def test_reduction_keepdims():
    get_queue_or_skip()

    n0, n1 = 3, 6
    x = dpt.ones((n0, 4, 5, n1, 7), dtype="i4")
    m = dpt.max(x, axis=(1, 2, -1), keepdims=True)

    xx = dpt.reshape(dpt.permute_dims(x, (0, 3, 1, 2, -1)), (n0, n1, -1))
    p = dpt.argmax(xx, axis=-1, keepdims=True)

    assert m.shape == (n0, 1, 1, n1, 1)
    assert dpt.all(m == dpt.reshape(x[:, 0, 0, :, 0], m.shape))
    assert dpt.all(p == 0)


def test_max_scalar():
    get_queue_or_skip()

    x = dpt.ones(())
    m = dpt.max(x)

    assert m.shape == ()
    assert x == m


@pytest.mark.parametrize("arg_dtype", ["i4", "f4", "c8"])
def test_reduction_kernels(arg_dtype):
    # i4 - always uses atomics w/ sycl group reduction
    # f4 - always uses atomics w/ custom group reduction
    # c8 - always uses temps w/ custom group reduction
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arg_dtype, q)

    x = dpt.ones((24, 1025), dtype=arg_dtype, sycl_queue=q)
    x[x.shape[0] // 2, :] = 3
    x[:, x.shape[1] // 2] = 3

    m = dpt.max(x)
    assert m == 3
    m = dpt.max(x, axis=0)
    assert dpt.all(m == 3)
    m = dpt.max(x, axis=1)
    assert dpt.all(m == 3)

    x = dpt.ones((24, 1025), dtype=arg_dtype, sycl_queue=q)
    x[x.shape[0] // 2, :] = 0
    x[:, x.shape[1] // 2] = 0

    m = dpt.min(x)
    assert m == 0
    m = dpt.min(x, axis=0)
    assert dpt.all(m == 0)
    m = dpt.min(x, axis=1)
    assert dpt.all(m == 0)


def test_max_min_nan_propagation():
    get_queue_or_skip()

    # float, finites
    x = dpt.arange(4, dtype="f4")
    x[0] = dpt.nan
    assert dpt.isnan(dpt.max(x))
    assert dpt.isnan(dpt.min(x))

    # float, infinities
    x[1:] = dpt.inf
    assert dpt.isnan(dpt.max(x))
    x[1:] = -dpt.inf
    assert dpt.isnan(dpt.min(x))

    # complex
    x = dpt.arange(4, dtype="c8")
    x[0] = complex(dpt.nan, 0)
    assert dpt.isnan(dpt.max(x))
    assert dpt.isnan(dpt.min(x))

    x[0] = complex(0, dpt.nan)
    assert dpt.isnan(dpt.max(x))
    assert dpt.isnan(dpt.min(x))


def test_argmax_scalar():
    get_queue_or_skip()

    x = dpt.ones(())
    m = dpt.argmax(x)

    assert m.shape == ()
    assert m == 0


@pytest.mark.parametrize("arg_dtype", ["i4", "f4", "c8"])
def test_search_reduction_kernels(arg_dtype):
    # i4 - always uses atomics w/ sycl group reduction
    # f4 - always uses atomics w/ custom group reduction
    # c8 - always uses temps w/ custom group reduction
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arg_dtype, q)

    x = dpt.ones((24 * 1025), dtype=arg_dtype, sycl_queue=q)
    idx = randrange(x.size)
    idx_tup = np.unravel_index(idx, (24, 1025))
    x[idx] = 2

    m = dpt.argmax(x)
    assert m == idx

    x = dpt.reshape(x, (24, 1025))

    x[idx_tup[0], :] = 3
    m = dpt.argmax(x, axis=0)
    assert dpt.all(m == idx_tup[0])
    x[:, idx_tup[1]] = 4
    m = dpt.argmax(x, axis=1)
    assert dpt.all(m == idx_tup[1])

    x = x[:, ::-2]
    idx = randrange(x.shape[1])
    x[:, idx] = 5
    m = dpt.argmax(x, axis=1)
    assert dpt.all(m == idx)

    x = dpt.ones((24 * 1025), dtype=arg_dtype, sycl_queue=q)
    idx = randrange(x.size)
    idx_tup = np.unravel_index(idx, (24, 1025))
    x[idx] = 0

    m = dpt.argmin(x)
    assert m == idx

    x = dpt.reshape(x, (24, 1025))

    x[idx_tup[0], :] = -1
    m = dpt.argmin(x, axis=0)
    assert dpt.all(m == idx_tup[0])
    x[:, idx_tup[1]] = -2
    m = dpt.argmin(x, axis=1)
    assert dpt.all(m == idx_tup[1])

    x = x[:, ::-2]
    idx = randrange(x.shape[1])
    x[:, idx] = -3
    m = dpt.argmin(x, axis=1)
    assert dpt.all(m == idx)


def test_argmax_argmin_nan_propagation():
    get_queue_or_skip()

    sz = 4
    idx = randrange(sz)
    # floats
    x = dpt.arange(sz, dtype="f4")
    x[idx] = dpt.nan
    assert dpt.argmax(x) == idx
    assert dpt.argmin(x) == idx

    # complex
    x = dpt.arange(sz, dtype="c8")
    x[idx] = complex(dpt.nan, 0)
    assert dpt.argmax(x) == idx
    assert dpt.argmin(x) == idx

    x[idx] = complex(0, dpt.nan)
    assert dpt.argmax(x) == idx
    assert dpt.argmin(x) == idx


def test_argmax_argmin_identities():
    # make sure that identity arrays work as expected
    get_queue_or_skip()

    x = dpt.full(3, dpt.iinfo(dpt.int32).min, dtype="i4")
    assert dpt.argmax(x) == 0
    x = dpt.full(3, dpt.iinfo(dpt.int32).max, dtype="i4")
    assert dpt.argmin(x) == 0


def test_reduction_arg_validation():
    get_queue_or_skip()

    x = dict()
    with pytest.raises(TypeError):
        dpt.sum(x)
    with pytest.raises(TypeError):
        dpt.max(x)
    with pytest.raises(TypeError):
        dpt.argmax(x)

    x = dpt.zeros((0,), dtype="i4")
    with pytest.raises(ValueError):
        dpt.max(x)
    with pytest.raises(ValueError):
        dpt.argmax(x)


@pytest.mark.parametrize("arg_dtype", _no_complex_dtypes[1:])
def test_logsumexp_arg_dtype_default_output_dtype_matrix(arg_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arg_dtype, q)

    m = dpt.ones(100, dtype=arg_dtype)
    r = dpt.logsumexp(m)

    assert isinstance(r, dpt.usm_ndarray)
    assert r.dtype.kind == "f"
    tol = dpt.finfo(r.dtype).resolution
    assert_allclose(
        dpt.asnumpy(r),
        np.logaddexp.reduce(dpt.asnumpy(m), dtype=r.dtype),
        rtol=tol,
        atol=tol,
    )


def test_logsumexp_empty():
    get_queue_or_skip()
    x = dpt.empty((0,), dtype="f4")
    y = dpt.logsumexp(x)
    assert y.shape == tuple()
    assert y == -dpt.inf


def test_logsumexp_axis():
    get_queue_or_skip()

    m = dpt.ones((3, 4, 5, 6, 7), dtype="f4")
    s = dpt.logsumexp(m, axis=(1, 2, -1))

    assert isinstance(s, dpt.usm_ndarray)
    assert s.shape == (3, 6)
    tol = dpt.finfo(s.dtype).resolution
    assert_allclose(
        dpt.asnumpy(s),
        np.logaddexp.reduce(dpt.asnumpy(m), axis=(1, 2, -1), dtype=s.dtype),
        rtol=tol,
        atol=tol,
    )


@pytest.mark.parametrize("arg_dtype", _no_complex_dtypes[1:])
@pytest.mark.parametrize("out_dtype", _all_dtypes[1:])
def test_logsumexp_arg_out_dtype_matrix(arg_dtype, out_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arg_dtype, q)
    skip_if_dtype_not_supported(out_dtype, q)

    m = dpt.ones(100, dtype=arg_dtype)
    r = dpt.logsumexp(m, dtype=out_dtype)

    assert isinstance(r, dpt.usm_ndarray)
    assert r.dtype == dpt.dtype(out_dtype)


def test_logsumexp_keepdims():
    get_queue_or_skip()

    m = dpt.ones((3, 4, 5, 6, 7), dtype="i4")
    s = dpt.logsumexp(m, axis=(1, 2, -1), keepdims=True)

    assert isinstance(s, dpt.usm_ndarray)
    assert s.shape == (3, 1, 1, 6, 1)


def test_logsumexp_scalar():
    get_queue_or_skip()

    m = dpt.ones(())
    s = dpt.logsumexp(m)

    assert isinstance(s, dpt.usm_ndarray)
    assert m.sycl_queue == s.sycl_queue
    assert s.shape == ()


@pytest.mark.parametrize("arg_dtype", _no_complex_dtypes[1:])
def test_hypot_arg_dtype_default_output_dtype_matrix(arg_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arg_dtype, q)

    m = dpt.ones(100, dtype=arg_dtype)
    r = dpt.reduce_hypot(m)

    assert isinstance(r, dpt.usm_ndarray)
    assert r.dtype.kind == "f"
    tol = dpt.finfo(r.dtype).resolution
    assert_allclose(
        dpt.asnumpy(r),
        np.hypot.reduce(dpt.asnumpy(m), dtype=r.dtype),
        rtol=tol,
        atol=tol,
    )


def test_hypot_empty():
    get_queue_or_skip()
    x = dpt.empty((0,), dtype="f4")
    y = dpt.reduce_hypot(x)
    assert y.shape == tuple()
    assert y == 0


@pytest.mark.parametrize("arg_dtype", _no_complex_dtypes[1:])
@pytest.mark.parametrize("out_dtype", _all_dtypes[1:])
def test_hypot_arg_out_dtype_matrix(arg_dtype, out_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arg_dtype, q)
    skip_if_dtype_not_supported(out_dtype, q)

    m = dpt.ones(100, dtype=arg_dtype)
    r = dpt.reduce_hypot(m, dtype=out_dtype)

    assert isinstance(r, dpt.usm_ndarray)
    assert r.dtype == dpt.dtype(out_dtype)
