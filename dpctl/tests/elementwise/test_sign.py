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

import itertools

import numpy as np
import pytest

import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _all_dtypes, _no_complex_dtypes, _usm_types


@pytest.mark.parametrize("dtype", _all_dtypes[1:])
def test_sign_out_type(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    X = dpt.asarray(0, dtype=arg_dt, sycl_queue=q)
    assert dpt.sign(X).dtype == arg_dt

    r = dpt.empty_like(X, dtype=arg_dt)
    dpt.sign(X, out=r)
    assert np.allclose(dpt.asnumpy(r), dpt.asnumpy(dpt.sign(X)))


@pytest.mark.parametrize("usm_type", _usm_types)
def test_sign_usm_type(usm_type):
    q = get_queue_or_skip()

    arg_dt = np.dtype("i4")
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, usm_type=usm_type, sycl_queue=q)
    X[..., 0::2] = 1
    X[..., 1::2] = 0

    Y = dpt.sign(X)
    assert Y.usm_type == X.usm_type
    assert Y.sycl_queue == X.sycl_queue
    assert Y.flags.c_contiguous

    expected_Y = dpt.asnumpy(X)
    assert np.allclose(dpt.asnumpy(Y), expected_Y)


@pytest.mark.parametrize("dtype", _all_dtypes[1:])
def test_sign_order(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    expected_dt = np.sign(np.ones(tuple(), dtype=arg_dt)).dtype
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, sycl_queue=q)
    X[..., 0::2] = 1
    X[..., 1::2] = 0

    for perms in itertools.permutations(range(4)):
        U = dpt.permute_dims(X[:, ::-1, ::-1, :], perms)
        expected_Y = np.ones(U.shape, dtype=expected_dt)
        expected_Y[..., 1::2] = 0
        expected_Y = np.transpose(expected_Y, perms)
        for ord in ["C", "F", "A", "K"]:
            Y = dpt.sign(U, order=ord)
            assert np.allclose(dpt.asnumpy(Y), expected_Y)


@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_sign_complex(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, sycl_queue=q)
    Xnp = np.random.standard_normal(
        size=input_shape
    ) + 1j * np.random.standard_normal(size=input_shape)
    Xnp = Xnp.astype(arg_dt)
    X[...] = Xnp

    for ord in ["C", "F", "A", "K"]:
        for perms in itertools.permutations(range(4)):
            U = dpt.permute_dims(X[:, ::-1, ::-1, :], perms)
            Y = dpt.sign(U, order=ord)
            X_t = np.transpose(Xnp[:, ::-1, ::-1, :], perms)
            expected_Y = X_t / np.abs(X_t)
            tol = dpt.finfo(Y.dtype).resolution
            np.testing.assert_allclose(
                dpt.asnumpy(Y), expected_Y, atol=tol, rtol=tol
            )


# test for all signed real data types
@pytest.mark.parametrize(
    "dt", _no_complex_dtypes[1:8:2] + _no_complex_dtypes[9:]
)
def test_sign_negative(dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x = dpt.arange(-20, 20, 1, dtype=dt, sycl_queue=q)
    x_np = np.arange(-20, 20, 1, dtype=dt)
    res = dpt.sign(x)

    assert (dpt.asnumpy(res) == np.sign(x_np)).all()
