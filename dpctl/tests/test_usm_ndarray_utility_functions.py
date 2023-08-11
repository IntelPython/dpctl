from random import randrange

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


@pytest.mark.parametrize("func,identity", [(dpt.all, True), (dpt.any, False)])
@pytest.mark.parametrize("dtype", _all_dtypes)
def test_boolean_reduction_dtypes_contig(func, identity, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = dpt.full(10, identity, dtype=dtype, sycl_queue=q)
    res = func(x)

    assert_equal(dpt.asnumpy(res), identity)

    x[randrange(x.size)] = not identity
    res = func(x)
    assert_equal(dpt.asnumpy(res), not identity)

    # test branch in kernel for large arrays
    wg_size = 4 * 32
    x = dpt.full((wg_size + 1), identity, dtype=dtype, sycl_queue=q)
    res = func(x)
    assert_equal(dpt.asnumpy(res), identity)

    x[randrange(x.size)] = not identity
    res = func(x)
    assert_equal(dpt.asnumpy(res), not identity)


@pytest.mark.parametrize("func,identity", [(dpt.all, True), (dpt.any, False)])
@pytest.mark.parametrize("dtype", _all_dtypes)
def test_boolean_reduction_dtypes_strided(func, identity, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = dpt.full(20, identity, dtype=dtype, sycl_queue=q)[::-2]
    res = func(x)
    assert_equal(dpt.asnumpy(res), identity)

    x[randrange(x.size)] = not identity
    res = func(x)
    assert_equal(dpt.asnumpy(res), not identity)


@pytest.mark.parametrize("func,identity", [(dpt.all, True), (dpt.any, False)])
def test_boolean_reduction_axis(func, identity):
    get_queue_or_skip()

    x = dpt.full((2, 3, 4, 5, 6), identity, dtype="i4")
    res = func(x, axis=(1, 2, -1))

    assert res.shape == (2, 5)
    assert_array_equal(dpt.asnumpy(res), np.full(res.shape, identity))

    # make first row of output negation of identity
    x[0, 0, 0, ...] = not identity
    res = func(x, axis=(1, 2, -1))
    assert_array_equal(dpt.asnumpy(res[0]), np.full(res.shape[1], not identity))


@pytest.mark.parametrize("func", [dpt.all, dpt.any])
def test_boolean_reduction_keepdims(func):
    get_queue_or_skip()

    x = dpt.ones((2, 3, 4, 5, 6), dtype="i4")
    res = func(x, axis=(1, 2, -1), keepdims=True)
    assert res.shape == (2, 1, 1, 5, 1)
    assert_array_equal(dpt.asnumpy(res), np.full(res.shape, True))

    res = func(x, axis=None, keepdims=True)
    assert res.shape == (1,) * x.ndim


@pytest.mark.parametrize("func,identity", [(dpt.all, True), (dpt.any, False)])
def test_boolean_reduction_empty(func, identity):
    get_queue_or_skip()

    x = dpt.empty((0,), dtype="i4")
    res = func(x)
    assert_equal(dpt.asnumpy(res), identity)


# nan, inf, and -inf should evaluate to true
@pytest.mark.parametrize("func", [dpt.all, dpt.any])
def test_boolean_reductions_nan_inf(func):
    q = get_queue_or_skip()

    x = dpt.asarray([dpt.nan, dpt.inf, -dpt.inf], dtype="f4", sycl_queue=q)[
        :, dpt.newaxis
    ]
    res = func(x, axis=1)
    assert_equal(dpt.asnumpy(res), True)


@pytest.mark.parametrize("func", [dpt.all, dpt.any])
def test_boolean_reduction_scalars(func):
    get_queue_or_skip()

    x = dpt.ones((), dtype="i4")
    assert_equal(dpt.asnumpy(func(x)), True)

    x = dpt.zeros((), dtype="i4")
    assert_equal(dpt.asnumpy(func(x)), False)


@pytest.mark.parametrize("func", [dpt.all, dpt.any])
def test_boolean_reduction_empty_axis(func):
    get_queue_or_skip()

    x = dpt.ones((5,), dtype="i4")
    res = func(x, axis=())
    assert_array_equal(dpt.asnumpy(res), dpt.asnumpy(x).astype(np.bool_))


@pytest.mark.parametrize("func", [dpt.all, dpt.any])
def test_arg_validation_boolean_reductions(func):
    get_queue_or_skip()

    x = dpt.ones((4, 5), dtype="i4")
    d = dict()

    with pytest.raises(TypeError):
        func(d)
    with pytest.raises(AxisError):
        func(x, axis=-3)


def test_boolean_reductions_3d_gh_1327():
    get_queue_or_skip()

    size = 24
    x = dpt.reshape(dpt.arange(-10, size - 10, 1, dtype="i4"), (2, 3, 4))
    res = dpt.all(x, axis=0)
    res_np = np.full(res.shape, True, dtype="?")
    res_np[2, 2] = False

    assert (dpt.asnumpy(res) == res_np).all()

    x = dpt.ones((2, 3, 4, 5), dtype="i4")
    res = dpt.any(x, axis=0)

    assert (dpt.asnumpy(res) == np.full(res.shape, True, dtype="?")).all()
