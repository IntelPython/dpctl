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

import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_raises_regex

import dpctl
import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _all_dtypes, _map_to_device_dtype


@pytest.mark.parametrize(
    "np_call, dpt_call", [(np.sin, dpt.sin), (np.cos, dpt.cos)]
)
@pytest.mark.parametrize("dtype", _all_dtypes)
def test_sincos_out_type(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    X = dpt.asarray(0, dtype=dtype, sycl_queue=q)
    expected_dtype = np_call(np.array(0, dtype=dtype)).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    assert dpt_call(X).dtype == expected_dtype

    X = dpt.asarray(0, dtype=dtype, sycl_queue=q)
    expected_dtype = np_call(np.array(0, dtype=dtype)).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    Y = dpt.empty_like(X, dtype=expected_dtype)
    dpt_call(X, out=Y)
    assert_allclose(dpt.asnumpy(dpt_call(X)), dpt.asnumpy(Y))


@pytest.mark.parametrize(
    "np_call, dpt_call", [(np.sin, dpt.sin), (np.cos, dpt.cos)]
)
@pytest.mark.parametrize("dtype", ["f2", "f4", "f8", "c8", "c16"])
def test_sincos_output(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 100
    n_rep = 137

    Xnp = np.linspace(-np.pi / 4, np.pi / 4, num=n_seq, dtype=dtype)
    X = dpt.asarray(np.repeat(Xnp, n_rep), dtype=dtype, sycl_queue=q)

    Y = dpt_call(X)
    tol = 8 * dpt.finfo(Y.dtype).resolution

    assert_allclose(
        dpt.asnumpy(Y), np.repeat(np_call(Xnp), n_rep), atol=tol, rtol=tol
    )

    Z = dpt.empty_like(X, dtype=dtype)
    dpt_call(X, out=Z)

    assert_allclose(
        dpt.asnumpy(Z), np.repeat(np_call(Xnp), n_rep), atol=tol, rtol=tol
    )


@pytest.mark.parametrize(
    "np_call, dpt_call", [(np.sin, dpt.sin), (np.cos, dpt.cos)]
)
@pytest.mark.parametrize("usm_type", ["device", "shared", "host"])
def test_sincos_usm_type(np_call, dpt_call, usm_type):
    q = get_queue_or_skip()

    arg_dt = np.dtype("f4")
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, usm_type=usm_type, sycl_queue=q)
    X[..., 0::2] = np.pi / 6
    X[..., 1::2] = np.pi / 3

    Y = dpt_call(X)
    assert Y.usm_type == X.usm_type
    assert Y.sycl_queue == X.sycl_queue
    assert Y.flags.c_contiguous

    expected_Y = np.empty(input_shape, dtype=arg_dt)
    expected_Y[..., 0::2] = np_call(np.float32(np.pi / 6))
    expected_Y[..., 1::2] = np_call(np.float32(np.pi / 3))
    tol = 8 * dpt.finfo(Y.dtype).resolution

    assert_allclose(dpt.asnumpy(Y), expected_Y, atol=tol, rtol=tol)


@pytest.mark.parametrize(
    "np_call, dpt_call", [(np.sin, dpt.sin), (np.cos, dpt.cos)]
)
@pytest.mark.parametrize("dtype", _all_dtypes)
def test_sincos_order(np_call, dpt_call, dtype):
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
            Y = dpt_call(U, order=ord)
            expected_Y = np_call(dpt.asnumpy(U))
            tol = 8 * max(
                dpt.finfo(Y.dtype).resolution,
                np.finfo(expected_Y.dtype).resolution,
            )
            assert_allclose(dpt.asnumpy(Y), expected_Y, atol=tol, rtol=tol)


@pytest.mark.parametrize("callable", [dpt.sin, dpt.cos])
def test_sincos_errors(callable):
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
        callable,
        x,
        y,
    )

    x = dpt.zeros(2)
    y = dpt.empty(3)
    assert_raises_regex(
        TypeError,
        "The shape of input and output arrays are inconsistent",
        callable,
        x,
        y,
    )

    x = dpt.zeros(2, dtype="float32")
    y = np.empty_like(x)
    assert_raises_regex(
        TypeError, "output array must be of usm_ndarray type", callable, x, y
    )


@pytest.mark.parametrize("callable", [dpt.sin, dpt.cos])
@pytest.mark.parametrize("dtype", _all_dtypes)
def test_sincos_error_dtype(callable, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = dpt.zeros(5, dtype=dtype)
    y = dpt.empty_like(x, dtype="int16")
    assert_raises_regex(
        TypeError, "Output array of type.*is needed", callable, x, y
    )


@pytest.mark.parametrize("dtype", ["e", "f", "d"])
def test_sincos_special_cases(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    tol = 8 * dpt.finfo(dtype).resolution
    x = [np.nan, np.nan, np.nan, np.nan]
    y = [np.nan, -np.nan, np.inf, -np.inf]
    xf = np.array(x, dtype=dtype)
    yf = dpt.asarray(y, dtype=dtype)
    assert_allclose(dpt.asnumpy(dpt.sin(yf)), xf, atol=tol, rtol=tol)
    assert_allclose(dpt.asnumpy(dpt.cos(yf)), xf, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sincos_strided(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    np.random.seed(42)
    strides = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    sizes = np.arange(2, 100)
    tol = 8 * dpt.finfo(dtype).resolution

    for ii in sizes:
        Xnp = dtype(np.random.uniform(low=0.01, high=88.1, size=ii))
        Xnp[3:-1:4] = 120000.0
        X = dpt.asarray(Xnp)
        sin_true = np.sin(Xnp)
        cos_true = np.cos(Xnp)
        for jj in strides:
            assert_allclose(
                dpt.asnumpy(dpt.sin(X[::jj])),
                sin_true[::jj],
                atol=tol,
                rtol=tol,
            )
            assert_allclose(
                dpt.asnumpy(dpt.cos(X[::jj])),
                cos_true[::jj],
                atol=tol,
                rtol=tol,
            )


@pytest.mark.parametrize(
    "np_call, dpt_call", [(np.sin, dpt.sin), (np.cos, dpt.cos)]
)
@pytest.mark.parametrize("dtype", ["f2", "f4", "f8", "c8", "c16"])
def test_sincos_out_overlap(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    X = dpt.linspace(-np.pi / 2, np.pi / 2, 60, dtype=dtype, sycl_queue=q)
    X = dpt.reshape(X, (3, 5, 4))

    Xnp = dpt.asnumpy(X)
    Ynp = np_call(Xnp, out=Xnp)

    Y = dpt_call(X, out=X)
    assert Y is X
    assert np.allclose(dpt.asnumpy(X), Xnp)

    Ynp = np_call(Xnp, out=Xnp[::-1])
    Y = dpt_call(X, out=X[::-1])
    assert Y is not X
    assert np.allclose(dpt.asnumpy(X), Xnp)
    assert np.allclose(dpt.asnumpy(Y), Ynp)
