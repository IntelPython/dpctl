import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose

import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _all_dtypes, _map_to_device_dtype, _usm_types


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_exp_out_type(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    X = dpt.asarray(0, dtype=dtype, sycl_queue=q)
    expected_dtype = np.exp(np.array(0, dtype=dtype)).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    assert dpt.exp(X).dtype == expected_dtype


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8", "c8", "c16"])
def test_exp_output_contig(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 1027

    X = dpt.linspace(0, 11, num=n_seq, dtype=dtype, sycl_queue=q)
    Xnp = dpt.asnumpy(X)

    Y = dpt.exp(X)
    tol = 8 * dpt.finfo(Y.dtype).resolution

    assert_allclose(dpt.asnumpy(Y), np.exp(Xnp), atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8", "c8", "c16"])
def test_exp_output_strided(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 2054

    X = dpt.linspace(0, 11, num=n_seq, dtype=dtype, sycl_queue=q)[::-2]
    Xnp = dpt.asnumpy(X)

    Y = dpt.exp(X)
    tol = 8 * dpt.finfo(Y.dtype).resolution

    assert_allclose(dpt.asnumpy(Y), np.exp(Xnp), atol=tol, rtol=tol)


@pytest.mark.parametrize("usm_type", _usm_types)
def test_exp_usm_type(usm_type):
    q = get_queue_or_skip()

    arg_dt = np.dtype("f4")
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, usm_type=usm_type, sycl_queue=q)
    X[..., 0::2] = 16.0
    X[..., 1::2] = 23.0

    Y = dpt.exp(X)
    assert Y.usm_type == X.usm_type
    assert Y.sycl_queue == X.sycl_queue
    assert Y.flags.c_contiguous

    expected_Y = np.empty(input_shape, dtype=arg_dt)
    expected_Y[..., 0::2] = np.exp(np.float32(16.0))
    expected_Y[..., 1::2] = np.exp(np.float32(23.0))
    tol = 8 * dpt.finfo(Y.dtype).resolution

    assert_allclose(dpt.asnumpy(Y), expected_Y, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_exp_order(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, sycl_queue=q)
    X[..., 0::2] = 8.0
    X[..., 1::2] = 11.0

    for ord in ["C", "F", "A", "K"]:
        for perms in itertools.permutations(range(4)):
            U = dpt.permute_dims(X[:, ::-1, ::-1, :], perms)
            Y = dpt.exp(U, order=ord)
            expected_Y = np.exp(dpt.asnumpy(U))
            tol = 8 * max(
                dpt.finfo(Y.dtype).resolution,
                np.finfo(expected_Y.dtype).resolution,
            )
            assert_allclose(dpt.asnumpy(Y), expected_Y, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", ["f", "d"])
def test_exp_values(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    tol = 8 * dpt.finfo(dtype).resolution
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    log2_ = 0.69314718055994530943
    Xnp = np.array(x, dtype=dtype) * log2_
    X = dpt.asarray(Xnp, dtype=dtype)
    assert_allclose(dpt.asnumpy(dpt.exp(X)), np.exp(Xnp), atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", ["e", "f", "d"])
def test_exp_special_cases(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    tol = 8 * dpt.finfo(dtype).resolution
    x = [np.nan, -np.nan, np.inf, -np.inf, -1.0, 1.0, 0.0, -0.0]
    Xnp = np.array(x, dtype=dtype)
    X = dpt.asarray(x, dtype=dtype)
    assert_allclose(dpt.asnumpy(dpt.exp(X)), np.exp(Xnp), atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_exp_strided(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    np.random.seed(42)
    strides = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    sizes = np.arange(2, 100)
    tol = 8 * dpt.finfo(dtype).resolution

    for ii in sizes:
        Xnp = dtype(np.random.uniform(low=0.01, high=88.1, size=ii))
        X = dpt.asarray(Xnp)
        Y_expected = np.exp(Xnp)
        for jj in strides:
            assert_allclose(
                dpt.asnumpy(dpt.exp(X[::jj])),
                Y_expected[::jj],
                atol=tol,
                rtol=tol,
            )


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8", "c8", "c16"])
def test_exp_out_overlap(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    X = dpt.linspace(0, 1, 15, dtype=dtype, sycl_queue=q)
    X = dpt.reshape(X, (3, 5))

    Xnp = dpt.asnumpy(X)
    Ynp = np.exp(Xnp, out=Xnp)

    Y = dpt.exp(X, out=X)
    tol = 8 * dpt.finfo(Y.dtype).resolution
    assert Y is X
    assert_allclose(dpt.asnumpy(X), Xnp, atol=tol, rtol=tol)

    Ynp = np.exp(Xnp, out=Xnp[::-1])
    Y = dpt.exp(X, out=X[::-1])
    assert Y is not X
    assert_allclose(dpt.asnumpy(X), Xnp, atol=tol, rtol=tol)
    assert_allclose(dpt.asnumpy(Y), Ynp, atol=tol, rtol=tol)
