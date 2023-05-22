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

import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _all_dtypes


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_isnan_out_type(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    X = dpt.asarray(0, dtype=dtype, sycl_queue=q)
    assert dpt.isnan(X).dtype == dpt.bool


def test_isnan_output():
    q = get_queue_or_skip()

    Xnp = np.asarray(np.nan)
    X = dpt.asarray(np.nan, sycl_queue=q)
    assert dpt.asnumpy(dpt.isnan(X)) == np.isnan(Xnp)


@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_isnan_complex(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    y1 = complex(np.nan, np.nan)
    y2 = complex(1, np.nan)
    y3 = complex(np.nan, 1)
    y4 = complex(2, 1)
    y5 = complex(np.inf, 1)

    Ynp = np.repeat(np.array([y1, y2, y3, y4, y5], dtype=dtype), 123)
    Y = dpt.asarray(Ynp, sycl_queue=q)
    assert np.array_equal(dpt.asnumpy(dpt.isnan(Y)), np.isnan(Ynp))

    r = dpt.empty_like(Y, dtype="bool")
    dpt.isnan(Y, out=r)
    assert np.array_equal(dpt.asnumpy(r)[()], np.isnan(Ynp))


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_isnan_floats(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    y1 = np.nan
    y2 = 1
    y3 = np.inf

    for mult in [123, 137, 255, 271, 272]:
        Ynp = np.repeat(np.array([y1, y2, y3], dtype=dtype), mult)
        Y = dpt.asarray(Ynp, sycl_queue=q)
        assert np.array_equal(dpt.asnumpy(dpt.isnan(Y)), np.isnan(Ynp))

        r = dpt.empty_like(Y, dtype="bool")
        dpt.isnan(Y, out=r)
        assert np.array_equal(dpt.asnumpy(r)[()], np.isnan(Ynp))


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_isnan_order(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    input_shape = (10, 10, 10, 10)
    X = dpt.ones(input_shape, dtype=arg_dt, sycl_queue=q)

    for ord in ["C", "F", "A", "K"]:
        for perms in itertools.permutations(range(4)):
            U = dpt.permute_dims(X[::2, ::-1, ::-1, ::5], perms)
            Y = dpt.isnan(U, order=ord)
            expected_Y = np.full(Y.shape, False, dtype=Y.dtype)
            assert np.allclose(dpt.asnumpy(Y), expected_Y)
