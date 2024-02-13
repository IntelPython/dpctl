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

import pytest

import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _all_dtypes, _complex_fp_dtypes


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_reciprocal_out_type(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = dpt.asarray(1, dtype=dtype, sycl_queue=q)
    one = dpt.asarray(1, dtype=dtype, sycl_queue=q)
    expected_dtype = dpt.divide(one, x).dtype
    assert dpt.reciprocal(x).dtype == expected_dtype


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_reciprocal_output_contig(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 1027

    x = dpt.linspace(1, 13, num=n_seq, dtype=dtype, sycl_queue=q)
    res = dpt.reciprocal(x)
    expected = 1 / x
    tol = 8 * dpt.finfo(res.dtype).resolution
    assert dpt.allclose(res, expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_reciprocal_output_strided(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 2054

    x = dpt.linspace(1, 13, num=n_seq, dtype=dtype, sycl_queue=q)[::-2]
    res = dpt.reciprocal(x)
    expected = 1 / x
    tol = 8 * dpt.finfo(res.dtype).resolution
    assert dpt.allclose(res, expected, atol=tol, rtol=tol)


def test_reciprocal_special_cases():
    get_queue_or_skip()

    x = dpt.asarray([dpt.nan, 0.0, -0.0, dpt.inf, -dpt.inf], dtype="f4")
    res = dpt.reciprocal(x)
    expected = dpt.asarray([dpt.nan, dpt.inf, -dpt.inf, 0.0, -0.0], dtype="f4")
    assert dpt.allclose(res, expected, equal_nan=True)


@pytest.mark.parametrize("dtype", _complex_fp_dtypes)
def test_reciprocal_complex_special_cases(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    nans_ = [dpt.nan, -dpt.nan]
    infs_ = [dpt.inf, -dpt.inf]
    finites_ = [-1.0, -0.0, 0.0, 1.0]
    inps_ = nans_ + infs_ + finites_
    c_ = [complex(*v) for v in itertools.product(inps_, repeat=2)]

    z = dpt.asarray(c_, dtype=dtype)
    r = dpt.reciprocal(z)

    expected = 1 / z

    tol = dpt.finfo(r.dtype).resolution

    assert dpt.allclose(r, expected, atol=tol, rtol=tol, equal_nan=True)
