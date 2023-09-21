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

import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _all_dtypes, _map_to_device_dtype, _usm_types


@pytest.mark.parametrize("dtype", _all_dtypes[1:])
def test_round_out_type(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    X = dpt.asarray(0.1, dtype=dtype, sycl_queue=q)
    expected_dtype = np.round(np.array(0, dtype=dtype)).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    assert dpt.round(X).dtype == expected_dtype


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_round_real_contig(dtype):
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
def test_round_complex_contig(dtype):
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


@pytest.mark.parametrize("usm_type", _usm_types)
def test_round_usm_type(usm_type):
    q = get_queue_or_skip()

    arg_dt = np.dtype("f4")
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, usm_type=usm_type, sycl_queue=q)
    X[..., 0::2] = 16.2
    X[..., 1::2] = 23.7

    Y = dpt.round(X)
    assert Y.usm_type == X.usm_type
    assert Y.sycl_queue == X.sycl_queue
    assert Y.flags.c_contiguous

    expected_Y = np.empty(input_shape, dtype=arg_dt)
    expected_Y[..., 0::2] = np.round(np.float32(16.2))
    expected_Y[..., 1::2] = np.round(np.float32(23.7))
    tol = 8 * dpt.finfo(Y.dtype).resolution

    assert_allclose(dpt.asnumpy(Y), expected_Y, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_round_order(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, sycl_queue=q)
    X[..., 0::2] = 8.8
    X[..., 1::2] = 11.3

    for perms in itertools.permutations(range(4)):
        U = dpt.permute_dims(X[:, ::-1, ::-1, :], perms)
        expected_Y = np.round(dpt.asnumpy(U))
        for ord in ["C", "F", "A", "K"]:
            Y = dpt.round(U, order=ord)
            assert_allclose(dpt.asnumpy(Y), expected_Y)


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


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_round_real_strided(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    np.random.seed(42)
    strides = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    sizes = [2, 4, 6, 8, 9, 24, 72]
    tol = 8 * dpt.finfo(dtype).resolution

    for ii in sizes:
        Xnp = np.random.uniform(low=0.01, high=88.1, size=ii)
        Xnp.astype(dtype)
        X = dpt.asarray(Xnp)
        Ynp = np.round(Xnp)
        for jj in strides:
            assert_allclose(
                dpt.asnumpy(dpt.round(X[::jj])),
                Ynp[::jj],
                atol=tol,
                rtol=tol,
            )


@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_round_complex_strided(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    np.random.seed(42)
    strides = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    sizes = [2, 4, 6, 8, 9, 24, 72]
    tol = 8 * dpt.finfo(dtype).resolution

    low = -88.0
    high = 88.0
    for ii in sizes:
        x1 = np.random.uniform(low=low, high=high, size=ii)
        x2 = np.random.uniform(low=low, high=high, size=ii)
        Xnp = np.array([complex(v1, v2) for v1, v2 in zip(x1, x2)], dtype=dtype)
        X = dpt.asarray(Xnp)
        Ynp = np.round(Xnp)
        for jj in strides:
            assert_allclose(
                dpt.asnumpy(dpt.round(X[::jj])),
                Ynp[::jj],
                atol=tol,
                rtol=tol,
            )


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
