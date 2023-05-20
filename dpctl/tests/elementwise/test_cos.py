import itertools

import numpy as np
import pytest
from numpy.testing import assert_raises_regex

import dpctl
import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _all_dtypes, _map_to_device_dtype


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_cos_out_type(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    X = dpt.asarray(0, dtype=dtype, sycl_queue=q)
    expected_dtype = np.cos(np.array(0, dtype=dtype)).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    assert dpt.cos(X).dtype == expected_dtype

    X = dpt.asarray(0, dtype=dtype, sycl_queue=q)
    expected_dtype = np.cos(np.array(0, dtype=dtype)).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    Y = dpt.empty_like(X, dtype=expected_dtype)
    dpt.cos(X, Y)
    np.testing.assert_allclose(dpt.asnumpy(dpt.cos(X)), dpt.asnumpy(Y))


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8", "c8", "c16"])
def test_cos_output(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 100
    n_rep = 137

    Xnp = np.linspace(-np.pi / 4, np.pi / 4, num=n_seq, dtype=dtype)
    X = dpt.asarray(np.repeat(Xnp, n_rep), dtype=dtype, sycl_queue=q)

    Y = dpt.cos(X)
    tol = 8 * dpt.finfo(Y.dtype).resolution

    np.testing.assert_allclose(
        dpt.asnumpy(Y), np.repeat(np.cos(Xnp), n_rep), atol=tol, rtol=tol
    )

    Z = dpt.empty_like(X, dtype=dtype)
    dpt.cos(X, Z)

    np.testing.assert_allclose(
        dpt.asnumpy(Z), np.repeat(np.cos(Xnp), n_rep), atol=tol, rtol=tol
    )


@pytest.mark.parametrize("usm_type", ["device", "shared", "host"])
def test_cos_usm_type(usm_type):
    q = get_queue_or_skip()

    arg_dt = np.dtype("f4")
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, usm_type=usm_type, sycl_queue=q)
    X[..., 0::2] = np.pi / 6
    X[..., 1::2] = np.pi / 3

    Y = dpt.cos(X)
    assert Y.usm_type == X.usm_type
    assert Y.sycl_queue == X.sycl_queue
    assert Y.flags.c_contiguous

    expected_Y = np.empty(input_shape, dtype=arg_dt)
    expected_Y[..., 0::2] = np.cos(np.float32(np.pi / 6))
    expected_Y[..., 1::2] = np.cos(np.float32(np.pi / 3))
    tol = 8 * dpt.finfo(Y.dtype).resolution

    np.testing.assert_allclose(dpt.asnumpy(Y), expected_Y, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_cos_order(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, sycl_queue=q)
    X[..., 0::2] = np.pi / 6
    X[..., 1::2] = np.pi / 3

    for ord in ["C", "F", "A", "K"]:
        for perms in itertools.permutations(range(4)):
            U = dpt.permute_dims(X[:, ::-1, ::-1, :], perms)
            Y = dpt.cos(U, order=ord)
            expected_Y = np.cos(dpt.asnumpy(U))
            tol = 8 * max(
                dpt.finfo(Y.dtype).resolution,
                np.finfo(expected_Y.dtype).resolution,
            )
            np.testing.assert_allclose(
                dpt.asnumpy(Y), expected_Y, atol=tol, rtol=tol
            )


def test_cos_errors():
    get_queue_or_skip()
    try:
        gpu_queue = dpctl.SyclQueue("gpu")
    except dpctl.SyclQueueCreationError:
        pytest.skip("SyclQueue('gpu') failed, skipping")
    try:
        cpu_queue = dpctl.SyclQueue("cpu")
    except dpctl.SyclQueueCreationError:
        pytest.skip("SyclQueue('cpu') failed, skipping")

    x = dpt.zeros(2, sycl_queue=gpu_queue)
    y = dpt.empty_like(x, sycl_queue=cpu_queue)
    assert_raises_regex(
        TypeError,
        "Input and output allocation queues are not compatible",
        dpt.cos,
        x,
        y,
    )

    x = dpt.zeros(2)
    y = dpt.empty(3)
    assert_raises_regex(
        TypeError,
        "The shape of input and output arrays are inconsistent",
        dpt.cos,
        x,
        y,
    )

    x = dpt.zeros(2)
    y = x
    assert_raises_regex(
        TypeError, "Input and output arrays have memory overlap", dpt.cos, x, y
    )

    x = dpt.zeros(2, dtype="float32")
    y = np.empty_like(x)
    assert_raises_regex(
        TypeError, "output array must be of usm_ndarray type", dpt.cos, x, y
    )


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_cos_error_dtype(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = dpt.zeros(5, dtype=dtype)
    y = dpt.empty_like(x, dtype="int16")
    assert_raises_regex(
        TypeError, "Output array of type.*is needed", dpt.cos, x, y
    )
