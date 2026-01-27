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
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _all_dtypes, _map_to_device_dtype


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_complex_out_type(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    X = dpt.asarray(0, dtype=dtype, sycl_queue=q)
    expected_dtype = np.real(np.array(0, dtype=dtype)).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    assert dpt.real(X).dtype == expected_dtype

    expected_dtype = np.imag(np.array(0, dtype=dtype)).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    assert dpt.imag(X).dtype == expected_dtype

    expected_dtype = np.conj(np.array(0, dtype=dtype)).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    assert dpt.conj(X).dtype == expected_dtype


@pytest.mark.parametrize(
    "np_call, dpt_call",
    [(np.real, dpt.real), (np.imag, dpt.imag), (np.conj, dpt.conj)],
)
@pytest.mark.parametrize("dtype", _all_dtypes)
def test_complex_output(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 100

    x1 = np.linspace(0, 10, num=n_seq, dtype=dtype)
    x2 = np.linspace(0, 20, num=n_seq, dtype=dtype)
    Xnp = x1 + 1j * x2
    X = dpt.asarray(Xnp, sycl_queue=q)

    Y = dpt_call(X)
    tol = 8 * dpt.finfo(Y.dtype).resolution

    assert_allclose(dpt.asnumpy(Y), np_call(Xnp), atol=tol, rtol=tol)

    Z = dpt.empty_like(X, dtype=Y.dtype)
    dpt_call(X, out=Z)

    assert_allclose(dpt.asnumpy(Z), np_call(Xnp), atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_projection_complex(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    X = [
        complex(1, 2),
        complex(dpt.inf, -1),
        complex(0, -dpt.inf),
        complex(-dpt.inf, dpt.nan),
    ]
    Y = [
        complex(1, 2),
        complex(np.inf, -0.0),
        complex(np.inf, -0.0),
        complex(np.inf, 0.0),
    ]

    Xf = dpt.asarray(X, dtype=dtype, sycl_queue=q)
    Yf = np.array(Y, dtype=dtype)

    tol = 8 * dpt.finfo(Xf.dtype).resolution
    assert_allclose(dpt.asnumpy(dpt.proj(Xf)), Yf, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_projection(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    Xf = dpt.asarray(1, dtype=dtype, sycl_queue=q)
    out_dtype = dpt.proj(Xf).dtype
    Yf = np.array(complex(1, 0), dtype=out_dtype)

    tol = 8 * dpt.finfo(Yf.dtype).resolution
    assert_allclose(dpt.asnumpy(dpt.proj(Xf)), Yf, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_complex_special_cases(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = [np.nan, -np.nan, np.inf, -np.inf, +0.0, -0.0]
    xc = [complex(*val) for val in itertools.product(x, repeat=2)]

    Xc_np = np.array(xc, dtype=dtype)
    Xc = dpt.asarray(Xc_np, dtype=dtype, sycl_queue=q)

    tol = 8 * dpt.finfo(dtype).resolution

    actual = dpt.real(Xc)
    expected = np.real(Xc_np)
    assert_allclose(dpt.asnumpy(actual), expected, atol=tol, rtol=tol)

    actual = dpt.imag(Xc)
    expected = np.imag(Xc_np)
    assert_allclose(dpt.asnumpy(actual), expected, atol=tol, rtol=tol)

    actual = dpt.conj(Xc)
    expected = np.conj(Xc_np)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert_allclose(dpt.asnumpy(actual), expected, atol=tol, rtol=tol)
