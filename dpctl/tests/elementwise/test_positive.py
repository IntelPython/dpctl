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

from .utils import _all_dtypes, _usm_types


@pytest.mark.parametrize("dtype", _all_dtypes[1:])
def test_positive_out_type(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    X = dpt.asarray(0, dtype=arg_dt, sycl_queue=q)
    assert dpt.positive(X).dtype == arg_dt

    r = dpt.empty_like(X, dtype=arg_dt)
    dpt.positive(X, out=r)
    assert np.allclose(dpt.asnumpy(r), dpt.asnumpy(dpt.positive(X)))


@pytest.mark.parametrize("usm_type", _usm_types)
def test_positive_usm_type(usm_type):
    q = get_queue_or_skip()

    arg_dt = np.dtype("i4")
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, usm_type=usm_type, sycl_queue=q)
    X[..., 0::2] = 1
    X[..., 1::2] = 0

    Y = dpt.positive(X)
    assert Y.usm_type == X.usm_type
    assert Y.sycl_queue == X.sycl_queue
    assert Y.flags.c_contiguous

    expected_Y = dpt.asnumpy(X)
    assert np.allclose(dpt.asnumpy(Y), expected_Y)


@pytest.mark.parametrize("dtype", _all_dtypes[1:])
def test_positive_order(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, sycl_queue=q)
    X[..., 0::2] = 1
    X[..., 1::2] = 0

    for perms in itertools.permutations(range(4)):
        U = dpt.permute_dims(X[:, ::-1, ::-1, :], perms)
        expected_Y = np.ones(U.shape, dtype=U.dtype)
        expected_Y[..., 1::2] = 0
        expected_Y = np.transpose(expected_Y, perms)
        for ord in ["C", "F", "A", "K"]:
            Y = dpt.positive(U, order=ord)
            assert np.allclose(dpt.asnumpy(Y), expected_Y)
