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

import pytest

import dpctl.tensor as dpt
from dpctl.tensor._tensor_impl import default_device_fp_type
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

_no_complex_dtypes = [
    "?",
    "i1",
    "u1",
    "i2",
    "u2",
    "i4",
    "u4",
    "i8",
    "u8",
    "f2",
    "f4",
    "f8",
]


@pytest.mark.parametrize("dt", _no_complex_dtypes)
def test_mean_dtypes(dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x = dpt.ones(10, dtype=dt)
    res = dpt.mean(x)
    assert res == 1
    if x.dtype.kind in "biu":
        assert res.dtype == dpt.dtype(default_device_fp_type(q))
    else:
        assert res.dtype == x.dtype


@pytest.mark.parametrize("dt", _no_complex_dtypes)
@pytest.mark.parametrize("py_zero", [float(0), int(0)])
def test_std_var_dtypes(dt, py_zero):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x = dpt.ones(10, dtype=dt)
    res = dpt.std(x, correction=py_zero)
    assert res == 0
    if x.dtype.kind in "biu":
        assert res.dtype == dpt.dtype(default_device_fp_type(q))
    else:
        assert res.dtype == x.dtype

    res = dpt.var(x, correction=py_zero)
    assert res == 0
    if x.dtype.kind in "biu":
        assert res.dtype == dpt.dtype(default_device_fp_type(q))
    else:
        assert res.dtype == x.dtype
