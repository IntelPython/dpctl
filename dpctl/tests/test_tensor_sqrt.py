import itertools

import numpy as np
import pytest
from numpy.testing import assert_equal

import dpctl.tensor as dpt
import dpctl.tensor._type_utils as tu
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

_all_dtypes = [
    "b1",
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
_usm_types = ["device", "shared", "host"]


def _map_to_device_dtype(dt, dev):
    return tu._to_device_supported_dtype(dt, dev)


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_sqrt_out_type(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    X = dpt.asarray(0, dtype=dtype, sycl_queue=q)
    expected_dtype = np.sqrt(np.array(0, dtype=dtype)).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    assert dpt.sqrt(X).dtype == expected_dtype


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8", "c8", "c16"])
def test_sqrt_output_contig(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 1027

    X = dpt.linspace(0, 13, num=n_seq, dtype=dtype, sycl_queue=q)
    Xnp = dpt.asnumpy(X)

    Y = dpt.sqrt(X)
    tol = 8 * dpt.finfo(Y.dtype).resolution

    np.testing.assert_allclose(dpt.asnumpy(Y), np.sqrt(Xnp), atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8", "c8", "c16"])
def test_sqrt_output_strided(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 2054

    X = dpt.linspace(0, 13, num=n_seq, dtype=dtype, sycl_queue=q)[::-2]
    Xnp = dpt.asnumpy(X)

    Y = dpt.sqrt(X)
    tol = 8 * dpt.finfo(Y.dtype).resolution

    np.testing.assert_allclose(dpt.asnumpy(Y), np.sqrt(Xnp), atol=tol, rtol=tol)


@pytest.mark.parametrize("usm_type", _usm_types)
def test_sqrt_usm_type(usm_type):
    q = get_queue_or_skip()

    arg_dt = np.dtype("f4")
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, usm_type=usm_type, sycl_queue=q)
    X[..., 0::2] = 16.0
    X[..., 1::2] = 23.0

    Y = dpt.sqrt(X)
    assert Y.usm_type == X.usm_type
    assert Y.sycl_queue == X.sycl_queue
    assert Y.flags.c_contiguous

    expected_Y = np.empty(input_shape, dtype=arg_dt)
    expected_Y[..., 0::2] = np.sqrt(np.float32(16.0))
    expected_Y[..., 1::2] = np.sqrt(np.float32(23.0))
    tol = 8 * dpt.finfo(Y.dtype).resolution

    np.testing.assert_allclose(dpt.asnumpy(Y), expected_Y, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_sqrt_order(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, sycl_queue=q)
    X[..., 0::2] = 16.0
    X[..., 1::2] = 23.0

    for ord in ["C", "F", "A", "K"]:
        for perms in itertools.permutations(range(4)):
            U = dpt.permute_dims(X[:, ::-1, ::-1, :], perms)
            Y = dpt.sqrt(U, order=ord)
            expected_Y = np.sqrt(dpt.asnumpy(U))
            tol = 8 * max(
                dpt.finfo(Y.dtype).resolution,
                np.finfo(expected_Y.dtype).resolution,
            )
            np.testing.assert_allclose(
                dpt.asnumpy(Y), expected_Y, atol=tol, rtol=tol
            )


def test_sqrt_special_cases():
    q = get_queue_or_skip()

    X = dpt.asarray(
        [dpt.nan, -1.0, 0.0, -0.0, dpt.inf, -dpt.inf], dtype="f4", sycl_queue=q
    )
    Xnp = dpt.asnumpy(X)

    assert_equal(dpt.asnumpy(dpt.sqrt(X)), np.sqrt(Xnp))
