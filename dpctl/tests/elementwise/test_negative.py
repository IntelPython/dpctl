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
def test_negative_out_type(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    X = dpt.asarray(0, dtype=arg_dt, sycl_queue=q)
    assert dpt.negative(X).dtype == arg_dt

    r = dpt.empty_like(X, dtype=arg_dt)
    dpt.negative(X, out=r)
    assert np.allclose(dpt.asnumpy(r), dpt.asnumpy(dpt.negative(X)))


def test_negative_bool():
    get_queue_or_skip()
    x = dpt.ones(64, dtype="?")
    with pytest.raises(ValueError):
        dpt.negative(x)
