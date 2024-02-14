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
def test_cbrt_out_type(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    X = dpt.asarray(0, dtype=dtype, sycl_queue=q)
    expected_dtype = np.cbrt(np.array(0, dtype=dtype)).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    assert dpt.cbrt(X).dtype == expected_dtype


@pytest.mark.parametrize("dtype", _real_fp_dtypes)
def test_cbrt_output_contig(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 1027

    X = dpt.linspace(0, 13, num=n_seq, dtype=dtype, sycl_queue=q)
    Xnp = dpt.asnumpy(X)

    Y = dpt.cbrt(X)
    tol = 8 * dpt.finfo(Y.dtype).resolution

    assert_allclose(dpt.asnumpy(Y), np.cbrt(Xnp), atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", _real_fp_dtypes)
def test_cbrt_output_strided(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 2054

    X = dpt.linspace(0, 13, num=n_seq, dtype=dtype, sycl_queue=q)[::-2]
    Xnp = dpt.asnumpy(X)

    Y = dpt.cbrt(X)
    tol = 8 * dpt.finfo(Y.dtype).resolution

    assert_allclose(dpt.asnumpy(Y), np.cbrt(Xnp), atol=tol, rtol=tol)


@pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
def test_cbrt_special_cases():
    get_queue_or_skip()

    X = dpt.asarray([dpt.nan, 0.0, -0.0, dpt.inf, -dpt.inf], dtype="f4")
    res = dpt.cbrt(X)
    expected = dpt.asarray([dpt.nan, 0.0, -0.0, dpt.inf, -dpt.inf], dtype="f4")
    tol = dpt.finfo(dpt.float32).resolution

    assert dpt.allclose(res, expected, atol=tol, rtol=tol, equal_nan=True)
