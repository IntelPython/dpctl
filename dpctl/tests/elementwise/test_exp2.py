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
from numpy.testing import assert_allclose

import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _all_dtypes, _map_to_device_dtype, _usm_types


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_exp2_out_type(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    X = dpt.asarray(0, dtype=dtype, sycl_queue=q)
    expected_dtype = np.exp2(np.array(0, dtype=dtype)).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    assert dpt.exp2(X).dtype == expected_dtype


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8", "c8", "c16"])
def test_exp2_output_contig(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 1027

    X = dpt.linspace(1, 5, num=n_seq, dtype=dtype, sycl_queue=q)
    Xnp = dpt.asnumpy(X)

    Y = dpt.exp2(X)
    tol = 8 * dpt.finfo(Y.dtype).resolution

    assert_allclose(dpt.asnumpy(Y), np.exp2(Xnp), atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8", "c8", "c16"])
def test_exp2_output_strided(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 2 * 1027

    X = dpt.linspace(1, 5, num=n_seq, dtype=dtype, sycl_queue=q)[::-2]
    Xnp = dpt.asnumpy(X)

    Y = dpt.exp2(X)
    tol = 8 * dpt.finfo(Y.dtype).resolution

    assert_allclose(dpt.asnumpy(Y), np.exp2(Xnp), atol=tol, rtol=tol)


@pytest.mark.parametrize("usm_type", _usm_types)
def test_exp2_usm_type(usm_type):
    q = get_queue_or_skip()

    arg_dt = np.dtype("f4")
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, usm_type=usm_type, sycl_queue=q)
    X[..., 0::2] = 1 / 4
    X[..., 1::2] = 1 / 2

    Y = dpt.exp2(X)
    assert Y.usm_type == X.usm_type
    assert Y.sycl_queue == X.sycl_queue
    assert Y.flags.c_contiguous

    expected_Y = np.empty(input_shape, dtype=arg_dt)
    expected_Y[..., 0::2] = np.exp2(np.float32(1 / 4))
    expected_Y[..., 1::2] = np.exp2(np.float32(1 / 2))
    tol = 8 * dpt.finfo(Y.dtype).resolution

    assert_allclose(dpt.asnumpy(Y), expected_Y, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_exp2_order(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, sycl_queue=q)
    X[..., 0::2] = 1 / 4
    X[..., 1::2] = 1 / 2

    for perms in itertools.permutations(range(4)):
        U = dpt.permute_dims(X[:, ::-1, ::-1, :], perms)
        expected_Y = np.exp2(dpt.asnumpy(U))
        for ord in ["C", "F", "A", "K"]:
            Y = dpt.exp2(U, order=ord)
            tol = 8 * max(
                dpt.finfo(Y.dtype).resolution,
                np.finfo(expected_Y.dtype).resolution,
            )
            assert_allclose(dpt.asnumpy(Y), expected_Y, atol=tol, rtol=tol)


def test_exp2_special_cases():
    get_queue_or_skip()

    X = dpt.asarray([dpt.nan, 0.0, -0.0, dpt.inf, -dpt.inf], dtype="f4")
    res = np.asarray([np.nan, 1.0, 1.0, np.inf, 0.0], dtype="f4")

    tol = dpt.finfo(X.dtype).resolution
    assert_allclose(dpt.asnumpy(dpt.exp2(X)), res, atol=tol, rtol=tol)

    # special cases for complex variant
    num_finite = 1.0
    vals = [
        complex(0.0, 0.0),
        complex(num_finite, dpt.inf),
        complex(num_finite, dpt.nan),
        complex(dpt.inf, 0.0),
        complex(-dpt.inf, num_finite),
        complex(dpt.inf, num_finite),
        complex(-dpt.inf, dpt.inf),
        complex(dpt.inf, dpt.inf),
        complex(-dpt.inf, dpt.nan),
        complex(dpt.inf, dpt.nan),
        complex(dpt.nan, 0.0),
        complex(dpt.nan, num_finite),
        complex(dpt.nan, dpt.nan),
    ]
    X = dpt.asarray(vals, dtype=dpt.complex64)
    cis_1 = complex(np.cos(num_finite), np.sin(num_finite))
    c_nan = complex(np.nan, np.nan)
    res = np.asarray(
        [
            complex(1.0, 0.0),
            c_nan,
            c_nan,
            complex(np.inf, 0.0),
            0.0,
            np.inf * cis_1,
            complex(0.0, 0.0),
            complex(np.inf, np.nan),
            complex(0.0, 0.0),
            complex(np.inf, np.nan),
            complex(np.nan, 0.0),
            c_nan,
            c_nan,
        ],
        dtype=np.complex64,
    )

    tol = dpt.finfo(X.dtype).resolution
    with np.errstate(invalid="ignore"):
        assert_allclose(dpt.asnumpy(dpt.exp2(X)), res, atol=tol, rtol=tol)
