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

_trig_funcs = [(np.sin, dpt.sin), (np.cos, dpt.cos), (np.tan, dpt.tan)]
_inv_trig_funcs = [
    (np.arcsin, dpt.asin),
    (np.arccos, dpt.acos),
    (np.arctan, dpt.atan),
]
_all_funcs = _trig_funcs + _inv_trig_funcs


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", _all_dtypes)
def test_trig_out_type(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = dpt.asarray(0, dtype=dtype, sycl_queue=q)
    expected_dtype = np_call(np.array(0, dtype=dtype)).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    assert dpt_call(x).dtype == expected_dtype


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_trig_real_basic(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 100
    n_rep = 137
    if np_call in _trig_funcs:
        Xnp = np.linspace(
            -np.pi / 2 * 0.99, np.pi / 2 * 0.99, num=n_seq, dtype=dtype
        )
    if np_call == np.arctan:
        Xnp = np.linspace(-100.0, 100.0, num=n_seq, dtype=dtype)
    else:
        Xnp = np.linspace(-1.0, 1.0, num=n_seq, dtype=dtype)

    X = dpt.asarray(np.repeat(Xnp, n_rep), dtype=dtype, sycl_queue=q)
    Y = dpt_call(X)

    tol = 8 * dpt.finfo(dtype).resolution
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
def test_trig_complex_basic(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 256
    n_rep = 137
    low = -9.0
    high = 9.0
    x1 = np.random.uniform(low=low, high=high, size=n_seq)
    x2 = np.random.uniform(low=low, high=high, size=n_seq)
    Xnp = x1 + 1j * x2

    # stay away from poles and branch lines
    modulus = np.abs(Xnp)
    sel = np.logical_or(
        modulus < 0.9,
        np.logical_and(
            modulus > 1.2, np.minimum(np.abs(x2), np.abs(x1)) > 0.05
        ),
    )
    Xnp = Xnp[sel]

    X = dpt.repeat(dpt.asarray(Xnp, dtype=dtype, sycl_queue=q), n_rep)
    Y = dpt_call(X)

    expected = np.repeat(np_call(Xnp), n_rep)

    tol = 50 * dpt.finfo(dtype).resolution
    assert_allclose(dpt.asnumpy(Y), expected, atol=tol, rtol=tol)

    Z = dpt.empty_like(X, dtype=dtype)
    dpt_call(X, out=Z)

    assert_allclose(dpt.asnumpy(Z), expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_trig_real_special_cases(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = [np.nan, np.inf, -np.inf, 2.0, -2.0, +0.0, -0.0, +1.0, -1.0]

    xf = np.array(x, dtype=dtype)
    yf = dpt.asarray(xf, dtype=dtype, sycl_queue=q)

    with np.errstate(all="ignore"):
        Y_np = np_call(xf)

    tol = 8 * dpt.finfo(dtype).resolution
    Y = dpt_call(yf)
    assert_allclose(dpt.asnumpy(Y), Y_np, atol=tol, rtol=tol)
