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

import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _all_dtypes, _map_to_device_dtype


@pytest.mark.parametrize("dtype", _all_dtypes[1:])
def test_round_out_type(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    X = dpt.asarray(0.1, dtype=dtype, sycl_queue=q)
    expected_dtype = np.round(np.array(0, dtype=dtype)).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    assert dpt.round(X).dtype == expected_dtype


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_round_real_basic(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 100
    n_rep = 137
    Xnp = np.linspace(0.01, 88.1, num=n_seq, dtype=dtype)
    X = dpt.asarray(np.repeat(Xnp, n_rep), dtype=dtype, sycl_queue=q)
    Y = dpt.round(X)
    Ynp = np.round(Xnp)

    tol = 8 * dpt.finfo(dtype).resolution
    assert_allclose(dpt.asnumpy(Y), np.repeat(Ynp, n_rep), atol=tol, rtol=tol)

    Z = dpt.empty_like(X, dtype=dtype)
    dpt.round(X, out=Z)

    assert_allclose(dpt.asnumpy(Z), np.repeat(Ynp, n_rep), atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_round_complex_basic(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 100
    n_rep = 137
    low = -88.0
    high = 88.0
    x1 = np.random.uniform(low=low, high=high, size=n_seq)
    x2 = np.random.uniform(low=low, high=high, size=n_seq)
    Xnp = np.array([complex(v1, v2) for v1, v2 in zip(x1, x2)], dtype=dtype)

    X = dpt.asarray(np.repeat(Xnp, n_rep), dtype=dtype, sycl_queue=q)
    Y = dpt.round(X)

    tol = 8 * dpt.finfo(dtype).resolution
    assert_allclose(
        dpt.asnumpy(Y), np.repeat(np.round(Xnp), n_rep), atol=tol, rtol=tol
    )

    Z = dpt.empty_like(X, dtype=dtype)
    dpt.round(X, out=Z)

    assert_allclose(
        dpt.asnumpy(Z), np.repeat(np.round(Xnp), n_rep), atol=tol, rtol=tol
    )


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_round_real_special_cases(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    tol = 8 * dpt.finfo(dtype).resolution
    x = [np.nan, np.inf, -np.inf, 1.5, 2.5, -1.5, -2.5, 0.0, -0.0]
    Xnp = np.array(x, dtype=dtype)
    X = dpt.asarray(x, dtype=dtype)

    Y = dpt.asnumpy(dpt.round(X))
    Ynp = np.round(Xnp)
    assert_allclose(Y, Ynp, atol=tol, rtol=tol)
    assert_array_equal(np.signbit(Y), np.signbit(Ynp))


@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_round_complex_special_cases(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = [np.nan, np.inf, -np.inf, 1.5, 2.5, -1.5, -2.5, 0.0, -0.0]
    xc = [complex(*val) for val in itertools.product(x, repeat=2)]

    Xc_np = np.array(xc, dtype=dtype)
    Xc = dpt.asarray(Xc_np, dtype=dtype, sycl_queue=q)

    Ynp = np.round(Xc_np)
    Y = dpt.round(Xc)

    tol = 8 * dpt.finfo(dtype).resolution
    assert_allclose(dpt.asnumpy(dpt.real(Y)), np.real(Ynp), atol=tol, rtol=tol)
    assert_allclose(dpt.asnumpy(dpt.imag(Y)), np.imag(Ynp), atol=tol, rtol=tol)
