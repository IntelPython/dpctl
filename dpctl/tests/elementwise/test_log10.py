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
from numpy.testing import assert_equal

import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _all_dtypes, _map_to_device_dtype, _usm_types


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_log_out_type(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    X = dpt.asarray(1, dtype=dtype, sycl_queue=q)
    expected_dtype = np.log10(np.array(1, dtype=dtype)).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    assert dpt.log10(X).dtype == expected_dtype


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8", "c8", "c16"])
def test_log_output_contig(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 1027

    X = dpt.linspace(1, 13, num=n_seq, dtype=dtype, sycl_queue=q)
    Xnp = dpt.asnumpy(X)

    Y = dpt.log10(X)
    tol = 8 * dpt.finfo(Y.dtype).resolution

    np.testing.assert_allclose(
        dpt.asnumpy(Y), np.log10(Xnp), atol=tol, rtol=tol
    )


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8", "c8", "c16"])
def test_log_output_strided(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 2 * 1027

    X = dpt.linspace(1, 13, num=n_seq, dtype=dtype, sycl_queue=q)[::-2]
    Xnp = dpt.asnumpy(X)

    Y = dpt.log10(X)
    tol = 8 * dpt.finfo(Y.dtype).resolution

    np.testing.assert_allclose(
        dpt.asnumpy(Y), np.log10(Xnp), atol=tol, rtol=tol
    )


@pytest.mark.parametrize("usm_type", _usm_types)
def test_log_usm_type(usm_type):
    q = get_queue_or_skip()

    arg_dt = np.dtype("f4")
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, usm_type=usm_type, sycl_queue=q)
    X[..., 0::2] = 4 * dpt.e
    X[..., 1::2] = 10 * dpt.e

    Y = dpt.log10(X)
    assert Y.usm_type == X.usm_type
    assert Y.sycl_queue == X.sycl_queue
    assert Y.flags.c_contiguous

    expected_Y = np.empty(input_shape, dtype=arg_dt)
    expected_Y[..., 0::2] = np.log10(np.float32(4 * dpt.e))
    expected_Y[..., 1::2] = np.log10(np.float32(10 * dpt.e))
    tol = 8 * dpt.finfo(Y.dtype).resolution

    np.testing.assert_allclose(dpt.asnumpy(Y), expected_Y, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_log_order(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, sycl_queue=q)
    X[..., 0::2] = 4 * dpt.e
    X[..., 1::2] = 10 * dpt.e

    for perms in itertools.permutations(range(4)):
        U = dpt.permute_dims(X[:, ::-1, ::-1, :], perms)
        expected_Y = np.log10(dpt.asnumpy(U))
        for ord in ["C", "F", "A", "K"]:
            Y = dpt.log10(U, order=ord)
            tol = 8 * max(
                dpt.finfo(Y.dtype).resolution,
                np.finfo(expected_Y.dtype).resolution,
            )
            np.testing.assert_allclose(
                dpt.asnumpy(Y), expected_Y, atol=tol, rtol=tol
            )


def test_log_special_cases():
    q = get_queue_or_skip()

    X = dpt.asarray(
        [dpt.nan, -1.0, 0.0, -0.0, dpt.inf, -dpt.inf], dtype="f4", sycl_queue=q
    )
    Xnp = dpt.asnumpy(X)

    with np.errstate(invalid="ignore", divide="ignore"):
        assert_equal(dpt.asnumpy(dpt.log10(X)), np.log10(Xnp))
