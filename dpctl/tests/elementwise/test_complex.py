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
from numpy.testing import assert_allclose

import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _all_dtypes, _map_to_device_dtype, _usm_types


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
    X = dpt.asarray(Xnp, dtype=Xnp.dtype, sycl_queue=q)

    Y = dpt_call(X)
    tol = 8 * dpt.finfo(Y.dtype).resolution

    assert_allclose(dpt.asnumpy(Y), np_call(Xnp), atol=tol, rtol=tol)

    Z = dpt.empty_like(X, dtype=np_call(Xnp).dtype)
    dpt_call(X, out=Z)

    assert_allclose(dpt.asnumpy(Z), np_call(Xnp), atol=tol, rtol=tol)


@pytest.mark.parametrize(
    "np_call, dpt_call",
    [(np.real, dpt.real), (np.imag, dpt.imag), (np.conj, dpt.conj)],
)
@pytest.mark.parametrize("usm_type", _usm_types)
def test_complex_usm_type(np_call, dpt_call, usm_type):
    q = get_queue_or_skip()

    arg_dt = np.dtype("c8")
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, usm_type=usm_type, sycl_queue=q)
    X[..., 0::2] = np.pi / 6 + 1j * np.pi / 3
    X[..., 1::2] = np.pi / 3 + 1j * np.pi / 6

    Y = dpt_call(X)
    assert Y.usm_type == X.usm_type
    assert Y.sycl_queue == X.sycl_queue
    assert Y.flags.c_contiguous

    expected_Y = np.empty(input_shape, dtype=arg_dt)
    expected_Y[..., 0::2] = np_call(np.complex64(np.pi / 6 + 1j * np.pi / 3))
    expected_Y[..., 1::2] = np_call(np.complex64(np.pi / 3 + 1j * np.pi / 6))
    tol = 8 * dpt.finfo(Y.dtype).resolution

    assert_allclose(dpt.asnumpy(Y), expected_Y, atol=tol, rtol=tol)


@pytest.mark.parametrize(
    "np_call, dpt_call",
    [(np.real, dpt.real), (np.imag, dpt.imag), (np.conj, dpt.conj)],
)
@pytest.mark.parametrize("dtype", _all_dtypes)
def test_complex_order(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, sycl_queue=q)
    X[..., 0::2] = np.pi / 6 + 1j * np.pi / 3
    X[..., 1::2] = np.pi / 3 + 1j * np.pi / 6

    for ord in ["C", "F", "A", "K"]:
        for perms in itertools.permutations(range(4)):
            U = dpt.permute_dims(X[:, ::-1, ::-1, :], perms)
            Y = dpt_call(U, order=ord)
            expected_Y = np_call(dpt.asnumpy(U))
            assert np.allclose(dpt.asnumpy(Y), expected_Y)


@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_projection_complex(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    X = [complex(1, 2), complex(dpt.inf, -1), complex(0, -dpt.inf)]
    Y = [complex(1, 2), complex(dpt.inf, -0), complex(dpt.inf, -0)]

    Xf = dpt.asarray(X, dtype=dtype, sycl_queue=q)
    Yf = dpt.asarray(Y, dtype=dtype, sycl_queue=q)

    tol = 8 * dpt.finfo(Xf.dtype).resolution
    assert_allclose(
        dpt.asnumpy(dpt.proj(Xf)), dpt.asnumpy(Yf), atol=tol, rtol=tol
    )


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_projection(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    Xf = dpt.asarray(1, dtype=dtype, sycl_queue=q)
    out_dtype = dpt.proj(Xf).dtype
    Yf = dpt.asarray(complex(1, 0), dtype=out_dtype, sycl_queue=q)

    tol = 8 * dpt.finfo(Yf.dtype).resolution
    assert_allclose(
        dpt.asnumpy(dpt.proj(Xf)), dpt.asnumpy(Yf), atol=tol, rtol=tol
    )


@pytest.mark.parametrize(
    "np_call, dpt_call",
    [(np.real, dpt.real), (np.imag, dpt.imag), (np.conj, dpt.conj)],
)
@pytest.mark.parametrize("dtype", ["f", "d"])
@pytest.mark.parametrize("stride", [-1, 1, 2, 4, 5])
def test_complex_strided(np_call, dpt_call, dtype, stride):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    N = 100
    rng = np.random.default_rng(42)
    x1 = rng.standard_normal(N, dtype)
    x2 = 1j * rng.standard_normal(N, dtype)
    x = x1 + x2
    y = np_call(x[::stride])
    z = dpt_call(dpt.asarray(x[::stride]))

    tol = 8 * dpt.finfo(y.dtype).resolution
    assert_allclose(y, dpt.asnumpy(z), atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", ["e", "f", "d"])
def test_complex_special_cases(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = [np.nan, -np.nan, np.inf, -np.inf]
    with np.errstate(all="ignore"):
        Xnp = 1j * np.array(x, dtype=dtype)
    X = dpt.asarray(Xnp, dtype=Xnp.dtype)

    tol = 8 * dpt.finfo(dtype).resolution
    assert_allclose(dpt.asnumpy(dpt.real(X)), np.real(Xnp), atol=tol, rtol=tol)
    assert_allclose(dpt.asnumpy(dpt.imag(X)), np.imag(Xnp), atol=tol, rtol=tol)
    assert_allclose(dpt.asnumpy(dpt.conj(X)), np.conj(Xnp), atol=tol, rtol=tol)
