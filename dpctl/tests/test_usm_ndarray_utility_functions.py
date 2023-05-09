import numpy as np
import pytest
from numpy import AxisError
from numpy.testing import assert_array_equal, assert_equal

import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

_all_dtypes = [
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
    "c8",
    "c16",
]


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_all_dtypes_contig(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = dpt.ones(10, dtype=dtype, sycl_queue=q)
    res = dpt.all(x)

    assert_equal(dpt.asnumpy(res), True)

    x[x.size // 2] = 0
    res = dpt.all(x)
    assert_equal(dpt.asnumpy(res), False)


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_all_dtypes_strided(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = dpt.ones(20, dtype=dtype, sycl_queue=q)[::-2]
    res = dpt.all(x)
    assert_equal(dpt.asnumpy(res), True)

    x[x.size // 2] = 0
    res = dpt.all(x)
    assert_equal(dpt.asnumpy(res), False)


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_any_dtypes_contig(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = dpt.zeros(10, dtype=dtype, sycl_queue=q)
    res = dpt.any(x)

    assert_equal(dpt.asnumpy(res), False)

    x[x.size // 2] = 1
    res = dpt.any(x)
    assert_equal(dpt.asnumpy(res), True)


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_any_dtypes_strided(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = dpt.zeros(20, dtype=dtype, sycl_queue=q)[::-2]
    res = dpt.any(x)
    assert_equal(dpt.asnumpy(res), False)

    x[x.size // 2] = 1
    res = dpt.any(x)
    assert_equal(dpt.asnumpy(res), True)


def test_all_axis():
    get_queue_or_skip()

    x = dpt.ones((2, 3, 4, 5, 6), dtype="i4")
    res = dpt.all(x, axis=(1, 2, -1))

    assert res.shape == (2, 5)
    assert_array_equal(dpt.asnumpy(res), np.full(res.shape, True))

    # make first row of output false
    x[0, 0, 0, ...] = 0
    res = dpt.all(x, axis=(1, 2, -1))
    assert_array_equal(dpt.asnumpy(res[0]), np.full(res.shape[1], False))


def test_any_axis():
    get_queue_or_skip()

    x = dpt.zeros((2, 3, 4, 5, 6), dtype="i4")
    res = dpt.any(x, axis=(1, 2, -1))

    assert res.shape == (2, 5)
    assert_array_equal(dpt.asnumpy(res), np.full(res.shape, False))

    # make first row of output true
    x[0, 0, 0, ...] = 1
    res = dpt.any(x, axis=(1, 2, -1))
    assert_array_equal(dpt.asnumpy(res[0]), np.full(res.shape[1], True))


def test_all_any_keepdims():
    get_queue_or_skip()

    x = dpt.ones((2, 3, 4, 5, 6), dtype="i4")

    res = dpt.all(x, axis=(1, 2, -1), keepdims=True)
    assert res.shape == (2, 1, 1, 5, 1)
    assert_array_equal(dpt.asnumpy(res), np.full(res.shape, True))

    res = dpt.any(x, axis=(1, 2, -1), keepdims=True)
    assert res.shape == (2, 1, 1, 5, 1)
    assert_array_equal(dpt.asnumpy(res), np.full(res.shape, True))


# nan, inf, and -inf should evaluate to true
def test_all_any_nan_inf():
    q = get_queue_or_skip()

    x = dpt.asarray([dpt.nan, dpt.inf, -dpt.inf], dtype="f4", sycl_queue=q)
    res = dpt.all(x)
    assert_equal(dpt.asnumpy(res), True)

    x = x[:, dpt.newaxis]
    res = dpt.any(x, axis=1)
    assert_array_equal(dpt.asnumpy(res), np.full(res.shape, True))


def test_all_any_scalar():
    get_queue_or_skip()

    x = dpt.ones((), dtype="i4")
    dpt.all(x)
    dpt.any(x)


def test_arg_validation_all_any():
    get_queue_or_skip()

    x = dpt.ones((4, 5), dtype="i4")
    d = dict()

    with pytest.raises(TypeError):
        dpt.all(d)
    with pytest.raises(AxisError):
        dpt.all(x, axis=-3)

    with pytest.raises(TypeError):
        dpt.any(d)
    with pytest.raises(AxisError):
        dpt.any(x, axis=-3)
