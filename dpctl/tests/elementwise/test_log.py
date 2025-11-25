#                       Data Parallel Control (dpctl)
#
#  Copyright 2020-2025 Intel Corporation
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
from numpy.testing import assert_allclose, assert_equal

import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _all_dtypes, _map_to_device_dtype


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_log_out_type(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    X = dpt.asarray(1, dtype=dtype, sycl_queue=q)
    expected_dtype = np.log(np.array(1, dtype=dtype)).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    assert dpt.log(X).dtype == expected_dtype


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8", "c8", "c16"])
def test_log_basic(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 1027

    X = dpt.linspace(1, 13, num=n_seq, dtype=dtype, sycl_queue=q)
    Xnp = dpt.asnumpy(X)

    Y = dpt.log(X)
    tol = 8 * dpt.finfo(Y.dtype).resolution

    assert_allclose(dpt.asnumpy(Y), np.log(Xnp), atol=tol, rtol=tol)


def test_log_special_cases():
    q = get_queue_or_skip()

    X = dpt.asarray(
        [dpt.nan, -dpt.inf, -1.0, -0.0, 0.0, dpt.inf], dtype="f4", sycl_queue=q
    )
    Y = dpt.log(X)

    expected = np.array(
        [np.nan, np.nan, np.nan, -np.inf, -np.inf, np.inf], dtype="f4"
    )

    assert_equal(dpt.asnumpy(Y), expected)
