import ctypes

import numpy as np
import pytest
from numpy.testing import assert_raises_regex

import dpctl
import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported
from dpctl.utils import ExecutionPlacementError

from .utils import _all_dtypes, _compare_dtypes, _usm_types


@pytest.mark.parametrize("op1_dtype", _all_dtypes)
@pytest.mark.parametrize("op2_dtype", _all_dtypes)
def test_add_dtype_matrix(op1_dtype, op2_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op1_dtype, q)
    skip_if_dtype_not_supported(op2_dtype, q)

    sz = 127
    ar1 = dpt.ones(sz, dtype=op1_dtype)
    ar2 = dpt.ones_like(ar1, dtype=op2_dtype)

    r = dpt.add(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected_dtype = np.add(
        np.zeros(1, dtype=op1_dtype), np.zeros(1, dtype=op2_dtype)
    ).dtype
    assert _compare_dtypes(r.dtype, expected_dtype, sycl_queue=q)
    assert r.shape == ar1.shape
    assert (dpt.asnumpy(r) == np.full(r.shape, 2, dtype=r.dtype)).all()
    assert r.sycl_queue == ar1.sycl_queue

    out = dpt.empty_like(ar1, dtype=r.dtype)
    dpt.add(ar1, ar2, out)
    assert (dpt.asnumpy(out) == np.full(out.shape, 2, dtype=out.dtype)).all()

    ar3 = dpt.ones(sz, dtype=op1_dtype)
    ar4 = dpt.ones(2 * sz, dtype=op2_dtype)

    r = dpt.add(ar3[::-1], ar4[::2])
    assert isinstance(r, dpt.usm_ndarray)
    expected_dtype = np.add(
        np.zeros(1, dtype=op1_dtype), np.zeros(1, dtype=op2_dtype)
    ).dtype
    assert _compare_dtypes(r.dtype, expected_dtype, sycl_queue=q)
    assert r.shape == ar3.shape
    assert (dpt.asnumpy(r) == np.full(r.shape, 2, dtype=r.dtype)).all()

    out = dpt.empty_like(ar1, dtype=r.dtype)
    dpt.add(ar3[::-1], ar4[::2], out)
    assert (dpt.asnumpy(out) == np.full(out.shape, 2, dtype=out.dtype)).all()


@pytest.mark.parametrize("op1_usm_type", _usm_types)
@pytest.mark.parametrize("op2_usm_type", _usm_types)
def test_add_usm_type_matrix(op1_usm_type, op2_usm_type):
    get_queue_or_skip()

    sz = 128
    ar1 = dpt.ones(sz, dtype="i4", usm_type=op1_usm_type)
    ar2 = dpt.ones_like(ar1, dtype="i4", usm_type=op2_usm_type)

    r = dpt.add(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected_usm_type = dpctl.utils.get_coerced_usm_type(
        (op1_usm_type, op2_usm_type)
    )
    assert r.usm_type == expected_usm_type


def test_add_order():
    get_queue_or_skip()

    test_shape = (
        20,
        20,
    )
    test_shape2 = tuple(2 * dim for dim in test_shape)
    n = test_shape[-1]

    for dt1, dt2 in zip(["i4", "i4", "f4"], ["i4", "f4", "i4"]):
        ar1 = dpt.ones(test_shape, dtype=dt1, order="C")
        ar2 = dpt.ones(test_shape, dtype=dt2, order="C")
        r1 = dpt.add(ar1, ar2, order="C")
        assert r1.flags.c_contiguous
        r2 = dpt.add(ar1, ar2, order="F")
        assert r2.flags.f_contiguous
        r3 = dpt.add(ar1, ar2, order="A")
        assert r3.flags.c_contiguous
        r4 = dpt.add(ar1, ar2, order="K")
        assert r4.flags.c_contiguous

        ar1 = dpt.ones(test_shape, dtype=dt1, order="F")
        ar2 = dpt.ones(test_shape, dtype=dt2, order="F")
        r1 = dpt.add(ar1, ar2, order="C")
        assert r1.flags.c_contiguous
        r2 = dpt.add(ar1, ar2, order="F")
        assert r2.flags.f_contiguous
        r3 = dpt.add(ar1, ar2, order="A")
        assert r3.flags.f_contiguous
        r4 = dpt.add(ar1, ar2, order="K")
        assert r4.flags.f_contiguous

        ar1 = dpt.ones(test_shape2, dtype=dt1, order="C")[:20, ::-2]
        ar2 = dpt.ones(test_shape2, dtype=dt2, order="C")[:20, ::-2]
        r4 = dpt.add(ar1, ar2, order="K")
        assert r4.strides == (n, -1)
        r5 = dpt.add(ar1, ar2, order="C")
        assert r5.strides == (n, 1)

        ar1 = dpt.ones(test_shape2, dtype=dt1, order="C")[:20, ::-2].mT
        ar2 = dpt.ones(test_shape2, dtype=dt2, order="C")[:20, ::-2].mT
        r4 = dpt.add(ar1, ar2, order="K")
        assert r4.strides == (-1, n)
        r5 = dpt.add(ar1, ar2, order="C")
        assert r5.strides == (n, 1)


def test_add_broadcasting():
    get_queue_or_skip()

    m = dpt.ones((100, 5), dtype="i4")
    v = dpt.arange(5, dtype="i4")

    r = dpt.add(m, v)
    assert (dpt.asnumpy(r) == np.arange(1, 6, dtype="i4")[np.newaxis, :]).all()

    r2 = dpt.add(v, m)
    assert (dpt.asnumpy(r2) == np.arange(1, 6, dtype="i4")[np.newaxis, :]).all()

    out = dpt.empty_like(m)
    dpt.add(m, v, out)
    assert (
        dpt.asnumpy(out) == np.arange(1, 6, dtype="i4")[np.newaxis, :]
    ).all()

    out2 = dpt.empty_like(m)
    dpt.add(v, m, out2)
    assert (
        dpt.asnumpy(out2) == np.arange(1, 6, dtype="i4")[np.newaxis, :]
    ).all()


@pytest.mark.parametrize("arr_dt", _all_dtypes)
def test_add_python_scalar(arr_dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arr_dt, q)

    X = dpt.zeros((10, 10), dtype=arr_dt, sycl_queue=q)
    py_zeros = (
        bool(0),
        int(0),
        float(0),
        complex(0),
        np.float32(0),
        ctypes.c_int(0),
    )
    for sc in py_zeros:
        R = dpt.add(X, sc)
        assert isinstance(R, dpt.usm_ndarray)
        R = dpt.add(sc, X)
        assert isinstance(R, dpt.usm_ndarray)


class MockArray:
    def __init__(self, arr):
        self.data_ = arr

    @property
    def __sycl_usm_array_interface__(self):
        return self.data_.__sycl_usm_array_interface__


def test_add_mock_array():
    get_queue_or_skip()
    a = dpt.arange(10)
    b = dpt.ones(10)
    c = MockArray(b)
    r = dpt.add(a, c)
    assert isinstance(r, dpt.usm_ndarray)


def test_add_canary_mock_array():
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
        dpt.add(a, c)


def test_add_errors():
    get_queue_or_skip()
    try:
        gpu_queue = dpctl.SyclQueue("gpu")
    except dpctl.SyclQueueCreationError:
        pytest.skip("SyclQueue('gpu') failed, skipping")
    try:
        cpu_queue = dpctl.SyclQueue("cpu")
    except dpctl.SyclQueueCreationError:
        pytest.skip("SyclQueue('cpu') failed, skipping")

    ar1 = dpt.ones(2, dtype="float32", sycl_queue=gpu_queue)
    ar2 = dpt.ones_like(ar1, sycl_queue=gpu_queue)
    y = dpt.empty_like(ar1, sycl_queue=cpu_queue)
    assert_raises_regex(
        TypeError,
        "Input and output allocation queues are not compatible",
        dpt.add,
        ar1,
        ar2,
        y,
    )

    ar1 = dpt.ones(2, dtype="float32")
    ar2 = dpt.ones_like(ar1, dtype="int32")
    y = dpt.empty(3)
    assert_raises_regex(
        TypeError,
        "The shape of input and output arrays are inconsistent",
        dpt.add,
        ar1,
        ar2,
        y,
    )

    ar1 = dpt.ones(2, dtype="float32")
    ar2 = dpt.ones_like(ar1, dtype="int32")
    y = ar1
    assert_raises_regex(
        TypeError,
        "Input and output arrays have memory overlap",
        dpt.add,
        ar1,
        ar2,
        y,
    )

    ar1 = np.ones(2, dtype="float32")
    ar2 = np.ones_like(ar1, dtype="int32")
    assert_raises_regex(
        ExecutionPlacementError,
        "Execution placement can not be unambiguously inferred.*",
        dpt.add,
        ar1,
        ar2,
    )

    ar1 = dpt.ones(2, dtype="float32")
    ar2 = dpt.ones_like(ar1, dtype="int32")
    y = np.empty_like(ar1)
    assert_raises_regex(
        TypeError,
        "output array must be of usm_ndarray type",
        dpt.add,
        ar1,
        ar2,
        y,
    )


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_add_dtype_error(
    dtype,
):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    ar1 = dpt.ones(5, dtype=dtype)
    ar2 = dpt.ones_like(ar1, dtype="f4")

    y = dpt.zeros_like(ar1, dtype="int8")
    assert_raises_regex(
        TypeError, "Output array of type.*is needed", dpt.add, ar1, ar2, y
    )
