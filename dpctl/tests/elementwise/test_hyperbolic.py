#                       Data Parallel Control (dpctl)
#
#  Copyright 2020-2025 Intel Corporation
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

import numpy as np
import pytest
from numpy.testing import assert_allclose

import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _all_dtypes, _map_to_device_dtype

_hyper_funcs = [(np.sinh, dpt.sinh), (np.cosh, dpt.cosh), (np.tanh, dpt.tanh)]
_inv_hyper_funcs = [
    (np.arcsinh, dpt.asinh),
    (np.arccosh, dpt.acosh),
    (np.arctanh, dpt.atanh),
]
_all_funcs = _hyper_funcs + _inv_hyper_funcs


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", _all_dtypes)
def test_hyper_out_type(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    a = 1 if np_call == np.arccosh else 0

    x = dpt.asarray(a, dtype=dtype, sycl_queue=q)
    expected_dtype = np_call(np.array(a, dtype=dtype)).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    assert dpt_call(x).dtype == expected_dtype


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_hyper_real_contig(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 100
    n_rep = 137
    if np_call == np.arctanh:
        Xnp = np.linspace(-0.9, 0.9, num=n_seq, dtype=dtype)
    elif np_call == np.arccosh:
        Xnp = np.linspace(1.01, 10.0, num=n_seq, dtype=dtype)
    else:
        Xnp = np.linspace(-10.0, 10.0, num=n_seq, dtype=dtype)

    X = dpt.asarray(np.repeat(Xnp, n_rep), dtype=dtype, sycl_queue=q)
    Y = dpt_call(X)

    tol = 8 * dpt.finfo(Y.dtype).resolution
    assert_allclose(
        dpt.asnumpy(Y), np.repeat(np_call(Xnp), n_rep), atol=tol, rtol=tol
    )

    Z = dpt.empty_like(X, dtype=dtype)
    dpt_call(X, out=Z)

    assert_allclose(
        dpt.asnumpy(Z), np.repeat(np_call(Xnp), n_rep), atol=tol, rtol=tol
    )


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_hyper_complex_contig(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 100
    n_rep = 137
    low = -9.0
    high = 9.0
    x1 = np.random.uniform(low=low, high=high, size=n_seq)
    x2 = np.random.uniform(low=low, high=high, size=n_seq)
    Xnp = x1 + 1j * x2

    X = dpt.asarray(np.repeat(Xnp, n_rep), dtype=dtype, sycl_queue=q)
    Y = dpt_call(X)

    tol = 50 * dpt.finfo(dtype).resolution
    assert_allclose(
        dpt.asnumpy(Y), np.repeat(np_call(Xnp), n_rep), atol=tol, rtol=tol
    )

    Z = dpt.empty_like(X, dtype=dtype)
    dpt_call(X, out=Z)

    assert_allclose(
        dpt.asnumpy(Z), np.repeat(np_call(Xnp), n_rep), atol=tol, rtol=tol
    )


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_hyper_real_strided(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    np.random.seed(42)
    strides = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    sizes = [2, 4, 6, 8, 9, 24, 72]
    tol = 8 * dpt.finfo(dtype).resolution

    low = -10.0
    high = 10.0
    if np_call == np.arctanh:
        low = -0.9
        high = 0.9
    elif np_call == np.arccosh:
        low = 1.01
        high = 100.0

    for ii in sizes:
        Xnp = np.random.uniform(low=low, high=high, size=ii)
        Xnp.astype(dtype)
        X = dpt.asarray(Xnp)
        Ynp = np_call(Xnp)
        for jj in strides:
            assert_allclose(
                dpt.asnumpy(dpt_call(X[::jj])),
                Ynp[::jj],
                atol=tol,
                rtol=tol,
            )


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_hyper_complex_strided(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    np.random.seed(42)
    strides = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    sizes = [2, 4, 6, 8, 9, 24, 72]
    tol = 50 * dpt.finfo(dtype).resolution

    low = -8.0
    high = 8.0
    for ii in sizes:
        x1 = np.random.uniform(low=low, high=high, size=ii)
        x2 = np.random.uniform(low=low, high=high, size=ii)
        Xnp = np.array([complex(v1, v2) for v1, v2 in zip(x1, x2)], dtype=dtype)
        X = dpt.asarray(Xnp)
        Ynp = np_call(Xnp)
        for jj in strides:
            assert_allclose(
                dpt.asnumpy(dpt_call(X[::jj])),
                Ynp[::jj],
                atol=tol,
                rtol=tol,
            )


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_hyper_real_special_cases(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = [np.nan, np.inf, -np.inf, 2.0, -2.0, +0.0, -0.0, +1.0, -1.0]

    xf = np.array(x, dtype=dtype)
    yf = dpt.asarray(xf, dtype=dtype, sycl_queue=q)

    with np.errstate(all="ignore"):
        Y_np = np_call(xf)

    tol = 8 * dpt.finfo(dtype).resolution
    assert_allclose(dpt.asnumpy(dpt_call(yf)), Y_np, atol=tol, rtol=tol)
