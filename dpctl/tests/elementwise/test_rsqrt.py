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

import numpy as np
import pytest
from numpy.testing import assert_allclose

import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _map_to_device_dtype, _no_complex_dtypes, _real_fp_dtypes


@pytest.mark.parametrize("dtype", _no_complex_dtypes)
def test_rsqrt_out_type(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = dpt.asarray(1, dtype=dtype, sycl_queue=q)
    expected_dtype = np.reciprocal(np.sqrt(np.array(1, dtype=dtype))).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    assert dpt.rsqrt(x).dtype == expected_dtype


@pytest.mark.parametrize("dtype", _real_fp_dtypes)
def test_rsqrt_output_contig(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 1027

    x = dpt.linspace(1, 13, num=n_seq, dtype=dtype, sycl_queue=q)
    res = dpt.rsqrt(x)
    expected = np.reciprocal(np.sqrt(dpt.asnumpy(x), dtype=dtype))
    tol = 8 * dpt.finfo(res.dtype).resolution
    assert_allclose(dpt.asnumpy(res), expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", _real_fp_dtypes)
def test_rsqrt_output_strided(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 2054

    x = dpt.linspace(1, 13, num=n_seq, dtype=dtype, sycl_queue=q)[::-2]
    res = dpt.rsqrt(x)
    expected = np.reciprocal(np.sqrt(dpt.asnumpy(x), dtype=dtype))
    tol = 8 * dpt.finfo(res.dtype).resolution
    assert_allclose(dpt.asnumpy(res), expected, atol=tol, rtol=tol)


def test_rsqrt_special_cases():
    get_queue_or_skip()

    x = dpt.asarray([dpt.nan, -1.0, 0.0, -0.0, dpt.inf, -dpt.inf], dtype="f4")
    res = dpt.rsqrt(x)
    expected = dpt.asarray(
        [dpt.nan, dpt.nan, dpt.inf, -dpt.inf, 0.0, dpt.nan], dtype="f4"
    )
    assert dpt.allclose(res, expected, equal_nan=True)
