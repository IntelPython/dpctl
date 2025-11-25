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

import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _all_dtypes


@pytest.mark.parametrize("dtype", _all_dtypes[1:])
def test_square_out_type(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    X = dpt.arange(5, dtype=arg_dt, sycl_queue=q)
    assert dpt.square(X).dtype == arg_dt

    r = dpt.empty_like(X, dtype=arg_dt)
    dpt.square(X, out=r)
    assert np.allclose(dpt.asnumpy(r), dpt.asnumpy(dpt.square(X)))


@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_square_special_cases(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    vals = [np.nan, np.inf, -np.inf, 0.0, -0.0]
    X = dpt.asarray(vals, dtype=dtype, sycl_queue=q)
    X_np = dpt.asnumpy(X)

    tol = 8 * dpt.finfo(dtype).resolution
    with np.errstate(all="ignore"):
        assert np.allclose(
            dpt.asnumpy(dpt.square(X)),
            np.square(X_np),
            atol=tol,
            rtol=tol,
            equal_nan=True,
        )
