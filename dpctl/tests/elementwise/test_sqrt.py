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
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import (
    _all_dtypes,
    _complex_fp_dtypes,
    _map_to_device_dtype,
    _real_fp_dtypes,
    _usm_types,
)


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_sqrt_out_type(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    X = dpt.asarray(0, dtype=dtype, sycl_queue=q)
    expected_dtype = np.sqrt(np.array(0, dtype=dtype)).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    assert dpt.sqrt(X).dtype == expected_dtype


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8", "c8", "c16"])
def test_sqrt_output_contig(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 1027

    X = dpt.linspace(0, 13, num=n_seq, dtype=dtype, sycl_queue=q)
    Xnp = dpt.asnumpy(X)

    Y = dpt.sqrt(X)
    tol = 8 * dpt.finfo(Y.dtype).resolution

    assert_allclose(dpt.asnumpy(Y), np.sqrt(Xnp), atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8", "c8", "c16"])
def test_sqrt_output_strided(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 2054

    X = dpt.linspace(0, 13, num=n_seq, dtype=dtype, sycl_queue=q)[::-2]
    Xnp = dpt.asnumpy(X)

    Y = dpt.sqrt(X)
    tol = 8 * dpt.finfo(Y.dtype).resolution

    assert_allclose(dpt.asnumpy(Y), np.sqrt(Xnp), atol=tol, rtol=tol)


@pytest.mark.parametrize("usm_type", _usm_types)
def test_sqrt_usm_type(usm_type):
    q = get_queue_or_skip()

    arg_dt = np.dtype("f4")
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, usm_type=usm_type, sycl_queue=q)
    X[..., 0::2] = 16.0
    X[..., 1::2] = 23.0

    Y = dpt.sqrt(X)
    assert Y.usm_type == X.usm_type
    assert Y.sycl_queue == X.sycl_queue
    assert Y.flags.c_contiguous

    expected_Y = np.empty(input_shape, dtype=arg_dt)
    expected_Y[..., 0::2] = np.sqrt(np.float32(16.0))
    expected_Y[..., 1::2] = np.sqrt(np.float32(23.0))
    tol = 8 * dpt.finfo(Y.dtype).resolution

    assert_allclose(dpt.asnumpy(Y), expected_Y, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_sqrt_order(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, sycl_queue=q)
    X[..., 0::2] = 16.0
    X[..., 1::2] = 23.0

    for ord in ["C", "F", "A", "K"]:
        for perms in itertools.permutations(range(4)):
            U = dpt.permute_dims(X[:, ::-1, ::-1, :], perms)
            Y = dpt.sqrt(U, order=ord)
            expected_Y = np.sqrt(dpt.asnumpy(U))
            tol = 8 * max(
                dpt.finfo(Y.dtype).resolution,
                np.finfo(expected_Y.dtype).resolution,
            )
            assert_allclose(dpt.asnumpy(Y), expected_Y, atol=tol, rtol=tol)


@pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
def test_sqrt_special_cases():
    q = get_queue_or_skip()

    X = dpt.asarray(
        [dpt.nan, -1.0, 0.0, -0.0, dpt.inf, -dpt.inf], dtype="f4", sycl_queue=q
    )
    Xnp = dpt.asnumpy(X)

    assert_equal(dpt.asnumpy(dpt.sqrt(X)), np.sqrt(Xnp))


@pytest.mark.parametrize("dtype", _real_fp_dtypes)
def test_sqrt_real_fp_special_values(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    nans_ = [dpt.nan, -dpt.nan]
    infs_ = [dpt.inf, -dpt.inf]
    finites_ = [-1.0, -0.0, 0.0, 1.0]
    inps_ = nans_ + infs_ + finites_

    x = dpt.asarray(inps_, dtype=dtype)
    r = dpt.sqrt(x)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected_np = np.sqrt(np.asarray(inps_, dtype=dtype))

    expected = dpt.asarray(expected_np, dtype=dtype)
    tol = dpt.finfo(r.dtype).resolution

    assert dpt.allclose(r, expected, atol=tol, rtol=tol, equal_nan=True)


@pytest.mark.broken_complex
@pytest.mark.parametrize("dtype", _complex_fp_dtypes)
def test_sqrt_complex_fp_special_values(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    nans_ = [dpt.nan, -dpt.nan]
    infs_ = [dpt.inf, -dpt.inf]
    finites_ = [-1.0, -0.0, 0.0, 1.0]
    inps_ = nans_ + infs_ + finites_
    c_ = [complex(*v) for v in itertools.product(inps_, repeat=2)]

    z = dpt.asarray(c_, dtype=dtype)
    r = dpt.sqrt(z)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected_np = np.sqrt(np.asarray(c_, dtype=dtype))

    expected = dpt.asarray(expected_np, dtype=dtype)
    tol = dpt.finfo(r.dtype).resolution

    assert dpt.allclose(r, expected, atol=tol, rtol=tol, equal_nan=True)
