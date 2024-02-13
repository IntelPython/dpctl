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
from dpctl.tensor._type_utils import _can_cast
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _all_dtypes, _complex_fp_dtypes, _no_complex_dtypes


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_angle_out_type(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = dpt.asarray(1, dtype=dtype, sycl_queue=q)
    dt = dpt.dtype(dtype)
    dev = q.sycl_device
    _fp16 = dev.has_aspect_fp16
    _fp64 = dev.has_aspect_fp64
    if _can_cast(dt, dpt.complex64, _fp16, _fp64):
        assert dpt.angle(x).dtype == dpt.float32
    else:
        assert dpt.angle(x).dtype == dpt.float64


@pytest.mark.parametrize("dtype", _no_complex_dtypes[1:])
def test_angle_real(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = dpt.arange(10, dtype=dtype, sycl_queue=q)
    r = dpt.angle(x)

    assert dpt.all(r == 0)


@pytest.mark.parametrize("dtype", _complex_fp_dtypes)
def test_angle_complex(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    tol = 8 * dpt.finfo(dtype).resolution
    vals = dpt.pi * dpt.arange(10, dtype=dpt.finfo(dtype).dtype, sycl_queue=q)

    x = dpt.zeros(10, dtype=dtype, sycl_queue=q)

    x.imag[...] = vals
    r = dpt.angle(x)
    expected = dpt.atan2(x.imag, x.real)
    assert dpt.allclose(r, expected, atol=tol, rtol=tol)

    x.real[...] += dpt.pi
    r = dpt.angle(x)
    expected = dpt.atan2(x.imag, x.real)
    assert dpt.allclose(r, expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_angle_special_cases(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    vals = [np.nan, -np.nan, np.inf, -np.inf, +0.0, -0.0]
    vals = [complex(*val) for val in itertools.product(vals, repeat=2)]

    x = dpt.asarray(vals, dtype=dtype, sycl_queue=q)

    r = dpt.angle(x)
    expected = dpt.atan2(x.imag, x.real)

    tol = 8 * dpt.finfo(dtype).resolution

    assert dpt.allclose(r, expected, atol=tol, rtol=tol, equal_nan=True)
