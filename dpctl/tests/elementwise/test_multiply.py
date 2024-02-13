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

import ctypes

import numpy as np
import pytest

import dpctl
import dpctl.tensor as dpt
from dpctl.tensor._type_utils import _can_cast
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _all_dtypes, _compare_dtypes, _usm_types


@pytest.mark.parametrize("op1_dtype", _all_dtypes)
@pytest.mark.parametrize("op2_dtype", _all_dtypes)
def test_multiply_dtype_matrix(op1_dtype, op2_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op1_dtype, q)
    skip_if_dtype_not_supported(op2_dtype, q)

    sz = 127
    ar1 = dpt.ones(sz, dtype=op1_dtype)
    ar2 = dpt.ones_like(ar1, dtype=op2_dtype)

    r = dpt.multiply(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.multiply(
        np.ones(1, dtype=op1_dtype), np.ones(1, dtype=op2_dtype)
    )
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar1.shape
    assert (dpt.asnumpy(r) == expected.astype(r.dtype)).all()
    assert r.sycl_queue == ar1.sycl_queue

    ar3 = dpt.ones(sz, dtype=op1_dtype)
    ar4 = dpt.ones(2 * sz, dtype=op2_dtype)

    r = dpt.multiply(ar3[::-1], ar4[::2])
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.multiply(
        np.ones(1, dtype=op1_dtype), np.ones(1, dtype=op2_dtype)
    )
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar3.shape
    assert (dpt.asnumpy(r) == expected.astype(r.dtype)).all()


@pytest.mark.parametrize("op1_usm_type", _usm_types)
@pytest.mark.parametrize("op2_usm_type", _usm_types)
def test_multiply_usm_type_matrix(op1_usm_type, op2_usm_type):
    get_queue_or_skip()

    sz = 128
    ar1 = dpt.ones(sz, dtype="i4", usm_type=op1_usm_type)
    ar2 = dpt.ones_like(ar1, dtype="i4", usm_type=op2_usm_type)

    r = dpt.multiply(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected_usm_type = dpctl.utils.get_coerced_usm_type(
        (op1_usm_type, op2_usm_type)
    )
    assert r.usm_type == expected_usm_type


def test_multiply_order():
    get_queue_or_skip()

    ar1 = dpt.ones((20, 20), dtype="i4", order="C")
    ar2 = dpt.ones((20, 20), dtype="i4", order="C")
    r1 = dpt.multiply(ar1, ar2, order="C")
    assert r1.flags.c_contiguous
    r2 = dpt.multiply(ar1, ar2, order="F")
    assert r2.flags.f_contiguous
    r3 = dpt.multiply(ar1, ar2, order="A")
    assert r3.flags.c_contiguous
    r4 = dpt.multiply(ar1, ar2, order="K")
    assert r4.flags.c_contiguous

    ar1 = dpt.ones((20, 20), dtype="i4", order="F")
    ar2 = dpt.ones((20, 20), dtype="i4", order="F")
    r1 = dpt.multiply(ar1, ar2, order="C")
    assert r1.flags.c_contiguous
    r2 = dpt.multiply(ar1, ar2, order="F")
    assert r2.flags.f_contiguous
    r3 = dpt.multiply(ar1, ar2, order="A")
    assert r3.flags.f_contiguous
    r4 = dpt.multiply(ar1, ar2, order="K")
    assert r4.flags.f_contiguous

    ar1 = dpt.ones((40, 40), dtype="i4", order="C")[:20, ::-2]
    ar2 = dpt.ones((40, 40), dtype="i4", order="C")[:20, ::-2]
    r4 = dpt.multiply(ar1, ar2, order="K")
    assert r4.strides == (20, -1)

    ar1 = dpt.ones((40, 40), dtype="i4", order="C")[:20, ::-2].mT
    ar2 = dpt.ones((40, 40), dtype="i4", order="C")[:20, ::-2].mT
    r4 = dpt.multiply(ar1, ar2, order="K")
    assert r4.strides == (-1, 20)


def test_multiply_broadcasting():
    get_queue_or_skip()

    m = dpt.ones((100, 5), dtype="i4")
    v = dpt.arange(1, 6, dtype="i4")

    r = dpt.multiply(m, v)

    expected = np.multiply(
        np.ones((100, 5), dtype="i4"), np.arange(1, 6, dtype="i4")
    )
    assert (dpt.asnumpy(r) == expected.astype(r.dtype)).all()

    r2 = dpt.multiply(v, m)
    expected2 = np.multiply(
        np.arange(1, 6, dtype="i4"), np.ones((100, 5), dtype="i4")
    )
    assert (dpt.asnumpy(r2) == expected2.astype(r2.dtype)).all()


@pytest.mark.parametrize("arr_dt", _all_dtypes)
def test_multiply_python_scalar(arr_dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arr_dt, q)

    X = dpt.ones((10, 10), dtype=arr_dt, sycl_queue=q)
    py_ones = (
        bool(1),
        int(1),
        float(1),
        complex(1),
        np.float32(1),
        ctypes.c_int(1),
    )
    for sc in py_ones:
        R = dpt.multiply(X, sc)
        assert isinstance(R, dpt.usm_ndarray)
        R = dpt.multiply(sc, X)
        assert isinstance(R, dpt.usm_ndarray)


@pytest.mark.parametrize("arr_dt", _all_dtypes)
@pytest.mark.parametrize("sc", [bool(1), int(1), float(1), complex(1)])
def test_multiply_python_scalar_gh1219(arr_dt, sc):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arr_dt, q)

    Xnp = np.ones(4, dtype=arr_dt)

    X = dpt.ones(4, dtype=arr_dt, sycl_queue=q)

    R = dpt.multiply(X, sc)
    Rnp = np.multiply(Xnp, sc)
    assert _compare_dtypes(R.dtype, Rnp.dtype, sycl_queue=q)

    # symmetric case
    R = dpt.multiply(sc, X)
    Rnp = np.multiply(sc, Xnp)
    assert _compare_dtypes(R.dtype, Rnp.dtype, sycl_queue=q)


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_multiply_inplace_python_scalar(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)
    X = dpt.ones((10, 10), dtype=dtype, sycl_queue=q)
    dt_kind = X.dtype.kind
    if dt_kind in "ui":
        X *= int(1)
    elif dt_kind == "f":
        X *= float(1)
    elif dt_kind == "c":
        X *= complex(1)
    elif dt_kind == "b":
        X *= bool(1)


@pytest.mark.parametrize("op1_dtype", _all_dtypes)
@pytest.mark.parametrize("op2_dtype", _all_dtypes)
def test_multiply_inplace_dtype_matrix(op1_dtype, op2_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op1_dtype, q)
    skip_if_dtype_not_supported(op2_dtype, q)

    sz = 127
    ar1 = dpt.ones(sz, dtype=op1_dtype)
    ar2 = dpt.ones_like(ar1, dtype=op2_dtype)

    dev = q.sycl_device
    _fp16 = dev.has_aspect_fp16
    _fp64 = dev.has_aspect_fp64
    if _can_cast(ar2.dtype, ar1.dtype, _fp16, _fp64):
        ar1 *= ar2
        assert (
            dpt.asnumpy(ar1) == np.full(ar1.shape, 1, dtype=ar1.dtype)
        ).all()

        ar3 = dpt.ones(sz, dtype=op1_dtype)
        ar4 = dpt.ones(2 * sz, dtype=op2_dtype)

        ar3[::-1] *= ar4[::2]
        assert (
            dpt.asnumpy(ar3) == np.full(ar3.shape, 1, dtype=ar3.dtype)
        ).all()

    else:
        with pytest.raises(ValueError):
            ar1 *= ar2


def test_multiply_inplace_broadcasting():
    get_queue_or_skip()

    m = dpt.ones((100, 5), dtype="i4")
    v = dpt.arange(5, dtype="i4")

    m *= v
    assert (dpt.asnumpy(m) == np.arange(0, 5, dtype="i4")[np.newaxis, :]).all()
