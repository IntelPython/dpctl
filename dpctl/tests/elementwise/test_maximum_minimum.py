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

import ctypes
import itertools

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import dpctl
import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _all_dtypes, _compare_dtypes, _usm_types


@pytest.mark.parametrize("op1_dtype", _all_dtypes)
@pytest.mark.parametrize("op2_dtype", _all_dtypes)
def test_maximum_minimum_dtype_matrix(op1_dtype, op2_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op1_dtype, q)
    skip_if_dtype_not_supported(op2_dtype, q)

    sz = 127
    ar1_np = np.arange(sz)
    np.random.shuffle(ar1_np)
    ar1 = dpt.asarray(ar1_np, dtype=op1_dtype)
    ar2_np = np.arange(sz)
    np.random.shuffle(ar2_np)
    ar2 = dpt.asarray(ar2_np, dtype=op2_dtype)

    r = dpt.maximum(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.maximum(ar1_np.astype(op1_dtype), ar2_np.astype(op2_dtype))

    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar1.shape
    assert (dpt.asnumpy(r) == expected).all()
    assert r.sycl_queue == ar1.sycl_queue

    r = dpt.minimum(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.minimum(ar1_np.astype(op1_dtype), ar2_np.astype(op2_dtype))

    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar1.shape
    assert (dpt.asnumpy(r) == expected).all()
    assert r.sycl_queue == ar1.sycl_queue

    ar3_np = np.arange(sz)
    np.random.shuffle(ar3_np)
    ar3 = dpt.asarray(ar3_np, dtype=op1_dtype)
    ar4_np = np.arange(2 * sz)
    np.random.shuffle(ar4_np)
    ar4 = dpt.asarray(ar4_np, dtype=op2_dtype)

    r = dpt.maximum(ar3[::-1], ar4[::2])
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.maximum(
        ar3_np[::-1].astype(op1_dtype), ar4_np[::2].astype(op2_dtype)
    )

    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar3.shape
    assert (dpt.asnumpy(r) == expected).all()

    r = dpt.minimum(ar3[::-1], ar4[::2])
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.minimum(
        ar3_np[::-1].astype(op1_dtype), ar4_np[::2].astype(op2_dtype)
    )

    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar3.shape
    assert (dpt.asnumpy(r) == expected).all()


@pytest.mark.parametrize("op_dtype", ["c8", "c16"])
def test_maximum_minimum_complex_matrix(op_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op_dtype, q)

    sz = 127
    ar1_np_real = np.random.randint(0, 10, sz)
    ar1_np_imag = np.random.randint(0, 10, sz)
    ar1 = dpt.asarray(ar1_np_real + 1j * ar1_np_imag, dtype=op_dtype)

    ar2_np_real = np.random.randint(0, 10, sz)
    ar2_np_imag = np.random.randint(0, 10, sz)
    ar2 = dpt.asarray(ar2_np_real + 1j * ar2_np_imag, dtype=op_dtype)

    r = dpt.maximum(ar1, ar2)
    expected = np.maximum(dpt.asnumpy(ar1), dpt.asnumpy(ar2))
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == expected.shape
    assert_array_equal(dpt.asnumpy(r), expected)

    r1 = dpt.maximum(ar1[::-2], ar2[::2])
    expected1 = np.maximum(dpt.asnumpy(ar1[::-2]), dpt.asnumpy(ar2[::2]))
    assert _compare_dtypes(r.dtype, expected1.dtype, sycl_queue=q)
    assert r1.shape == expected1.shape
    assert_array_equal(dpt.asnumpy(r1), expected1)

    r = dpt.minimum(ar1, ar2)
    expected = np.minimum(dpt.asnumpy(ar1), dpt.asnumpy(ar2))
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == expected.shape
    assert_array_equal(dpt.asnumpy(r), expected)

    r1 = dpt.minimum(ar1[::-2], ar2[::2])
    expected1 = np.minimum(dpt.asnumpy(ar1[::-2]), dpt.asnumpy(ar2[::2]))
    assert _compare_dtypes(r.dtype, expected1.dtype, sycl_queue=q)
    assert r1.shape == expected1.shape
    assert_array_equal(dpt.asnumpy(r1), expected1)


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_maximum_minimum_real_special_cases(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = [np.nan, np.inf, -np.inf, 5.0, -3.0]
    x = list(itertools.product(x, repeat=2))
    Xnp = np.array([tup[0] for tup in x], dtype=dtype)
    Ynp = np.array([tup[1] for tup in x], dtype=dtype)
    X = dpt.asarray(Xnp, dtype=dtype)
    Y = dpt.asarray(Ynp, dtype=dtype)

    R = dpt.maximum(X, Y)
    Rnp = np.maximum(Xnp, Ynp)
    assert_array_equal(dpt.asnumpy(R), Rnp)

    R = dpt.minimum(X, Y)
    Rnp = np.minimum(Xnp, Ynp)
    assert_array_equal(dpt.asnumpy(R), Rnp)


@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_maximum_minimum_complex_special_cases(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = [np.nan, -np.inf, -np.inf, +2.0, -1.0]
    x = [complex(*val) for val in itertools.product(x, repeat=2)]
    x = list(itertools.product(x, repeat=2))

    Xnp = np.array([tup[0] for tup in x], dtype=dtype)
    Ynp = np.array([tup[1] for tup in x], dtype=dtype)
    X = dpt.asarray(Xnp, dtype=dtype, sycl_queue=q)
    Y = dpt.asarray(Ynp, dtype=dtype, sycl_queue=q)

    R = dpt.maximum(X, Y)
    Rnp = np.maximum(Xnp, Ynp)
    assert_array_equal(dpt.asnumpy(dpt.real(R)), np.real(Rnp))
    assert_array_equal(dpt.asnumpy(dpt.imag(R)), np.imag(Rnp))

    R = dpt.minimum(X, Y)
    Rnp = np.minimum(Xnp, Ynp)
    assert_array_equal(dpt.asnumpy(dpt.real(R)), np.real(Rnp))
    assert_array_equal(dpt.asnumpy(dpt.imag(R)), np.imag(Rnp))


@pytest.mark.parametrize("op1_usm_type", _usm_types)
@pytest.mark.parametrize("op2_usm_type", _usm_types)
def test_maximum_minimum_usm_type_matrix(op1_usm_type, op2_usm_type):
    get_queue_or_skip()

    sz = 128
    ar1_np = np.arange(sz, dtype="i4")
    np.random.shuffle(ar1_np)
    ar1 = dpt.asarray(ar1_np, usm_type=op1_usm_type)
    ar2_np = np.arange(sz, dtype="i4")
    np.random.shuffle(ar2_np)
    ar2 = dpt.asarray(ar2_np, usm_type=op2_usm_type)

    r = dpt.maximum(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected_usm_type = dpctl.utils.get_coerced_usm_type(
        (op1_usm_type, op2_usm_type)
    )
    assert r.usm_type == expected_usm_type

    r = dpt.minimum(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected_usm_type = dpctl.utils.get_coerced_usm_type(
        (op1_usm_type, op2_usm_type)
    )
    assert r.usm_type == expected_usm_type


def test_maximum_minimum_order():
    get_queue_or_skip()

    ar1_np = np.arange(20 * 20, dtype="i4").reshape(20, 20)
    np.random.shuffle(ar1_np)
    ar1 = dpt.asarray(ar1_np, order="C")
    ar2_np = np.arange(20 * 20, dtype="i4").reshape(20, 20)
    np.random.shuffle(ar2_np)
    ar2 = dpt.asarray(ar2_np, order="C")

    r1 = dpt.maximum(ar1, ar2, order="C")
    assert r1.flags.c_contiguous
    r2 = dpt.maximum(ar1, ar2, order="F")
    assert r2.flags.f_contiguous
    r3 = dpt.maximum(ar1, ar2, order="A")
    assert r3.flags.c_contiguous
    r4 = dpt.maximum(ar1, ar2, order="K")
    assert r4.flags.c_contiguous

    ar1 = dpt.asarray(ar1_np, order="F")
    ar2 = dpt.asarray(ar2_np, order="F")
    r1 = dpt.maximum(ar1, ar2, order="C")
    assert r1.flags.c_contiguous
    r2 = dpt.maximum(ar1, ar2, order="F")
    assert r2.flags.f_contiguous
    r3 = dpt.maximum(ar1, ar2, order="A")
    assert r3.flags.f_contiguous
    r4 = dpt.maximum(ar1, ar2, order="K")
    assert r4.flags.f_contiguous

    ar1_np = np.arange(40 * 40, dtype="i4").reshape(40, 40)
    np.random.shuffle(ar1_np)
    ar1 = dpt.asarray(ar1_np, order="C")[:20, ::-2]
    ar2_np = np.arange(40 * 40, dtype="i4").reshape(40, 40)
    np.random.shuffle(ar2_np)
    ar2 = dpt.asarray(ar2_np, order="C")[:20, ::-2]
    r4 = dpt.maximum(ar1, ar2, order="K")
    assert r4.strides == (20, -1)

    ar1 = dpt.asarray(ar1_np, order="C")[:20, ::-2].mT
    ar2 = dpt.asarray(ar2_np, order="C")[:20, ::-2].mT
    r4 = dpt.maximum(ar1, ar2, order="K")
    assert r4.strides == (-1, 20)


@pytest.mark.parametrize("arr_dt", _all_dtypes)
def test_maximum_minimum_python_scalar(arr_dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arr_dt, q)

    X = dpt.zeros((10, 10), dtype=arr_dt, sycl_queue=q)
    py_ones = (
        bool(1),
        int(1),
        float(1),
        complex(1),
        np.float32(1),
        ctypes.c_int(1),
    )
    for sc in py_ones:
        R = dpt.maximum(X, sc)
        assert isinstance(R, dpt.usm_ndarray)
        R = dpt.maximum(sc, X)
        assert isinstance(R, dpt.usm_ndarray)

        R = dpt.minimum(X, sc)
        assert isinstance(R, dpt.usm_ndarray)
        R = dpt.minimum(sc, X)
        assert isinstance(R, dpt.usm_ndarray)


class MockArray:
    def __init__(self, arr):
        self.data_ = arr

    @property
    def __sycl_usm_array_interface__(self):
        return self.data_.__sycl_usm_array_interface__


def test_maximum_minimum_mock_array():
    get_queue_or_skip()
    a = dpt.arange(10)
    b = dpt.ones(10)
    c = MockArray(b)
    r = dpt.maximum(a, c)
    assert isinstance(r, dpt.usm_ndarray)

    r = dpt.minimum(a, c)
    assert isinstance(r, dpt.usm_ndarray)


def test_maximum_canary_mock_array():
    get_queue_or_skip()
    a = dpt.arange(10)

    class Canary:
        def __init__(self):
            pass

        @property
        def __sycl_usm_array_interface__(self):
            return None

    c = Canary()
    with pytest.raises(ValueError):
        dpt.maximum(a, c)

    with pytest.raises(ValueError):
        dpt.minimum(a, c)
