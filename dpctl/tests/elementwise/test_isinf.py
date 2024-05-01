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

from .utils import _all_dtypes


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_isinf_out_type(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    X = dpt.asarray(0, dtype=dtype, sycl_queue=q)
    assert dpt.isinf(X).dtype == dpt.bool


def test_isinf_output():
    q = get_queue_or_skip()

    Xnp = np.asarray(np.inf)
    X = dpt.asarray(np.inf, sycl_queue=q)
    assert dpt.asnumpy(dpt.isinf(X)) == np.isinf(Xnp)


@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_isinf_complex(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    y1 = complex(np.inf, np.inf)
    y2 = complex(1, np.inf)
    y3 = complex(np.inf, 1)
    y4 = complex(2, 1)
    y5 = complex(np.inf, 1)
    y6 = complex(np.inf, np.nan)

    Ynp = np.repeat(np.array([y1, y2, y3, y4, y5, y6], dtype=dtype), 123)
    Y = dpt.asarray(Ynp, sycl_queue=q)
    assert np.array_equal(dpt.asnumpy(dpt.isinf(Y)), np.isinf(Ynp))


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_isinf_floats(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    y1 = np.nan
    y2 = 1
    y3 = np.inf
    y4 = -np.inf

    for mult in [123, 137, 255, 271, 272]:
        Ynp = np.repeat(np.array([y1, y2, y3, y4], dtype=dtype), mult)
        Y = dpt.asarray(Ynp, sycl_queue=q)
        assert np.array_equal(dpt.asnumpy(dpt.isinf(Y)), np.isinf(Ynp))


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_isinf_order(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    input_shape = (10, 10, 10, 10)
    X = dpt.ones(input_shape, dtype=arg_dt, sycl_queue=q)

    for perms in itertools.permutations(range(4)):
        U = dpt.permute_dims(X[::2, ::-1, ::-1, ::5], perms)
        expected_Y = np.full(U.shape, fill_value=False, dtype=dpt.bool)
        for ord in ["C", "F", "A", "K"]:
            Y = dpt.isinf(U, order=ord)
            assert_allclose(dpt.asnumpy(Y), expected_Y)
