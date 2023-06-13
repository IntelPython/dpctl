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

from .utils import _map_to_device_dtype, _no_complex_dtypes

_all_funcs = [(np.floor, dpt.floor), (np.ceil, dpt.ceil), (np.trunc, dpt.trunc)]


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", _no_complex_dtypes)
def test_floor_ceil_trunc_out_type(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)
    if dtype == "b1":
        skip_if_dtype_not_supported("f2", q)

    X = dpt.asarray(0.1, dtype=dtype, sycl_queue=q)
    expected_dtype = np_call(np.array(0.1, dtype=dtype)).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    assert dpt_call(X).dtype == expected_dtype

    X = dpt.asarray(0.1, dtype=dtype, sycl_queue=q)
    expected_dtype = np_call(np.array(0.1, dtype=dtype)).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    Y = dpt.empty_like(X, dtype=expected_dtype)
    dpt_call(X, out=Y)
    assert_allclose(dpt.asnumpy(dpt_call(X)), dpt.asnumpy(Y))


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("usm_type", ["device", "shared", "host"])
def test_floor_ceil_trunc_usm_type(np_call, dpt_call, usm_type):
    q = get_queue_or_skip()

    arg_dt = np.dtype("f4")
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, usm_type=usm_type, sycl_queue=q)
    X[..., 0::2] = -0.4
    X[..., 1::2] = 0.7

    Y = dpt_call(X)
    assert Y.usm_type == X.usm_type
    assert Y.sycl_queue == X.sycl_queue
    assert Y.flags.c_contiguous

    expected_Y = np.empty(input_shape, dtype=arg_dt)
    expected_Y = np_call(np.float32(X))
    tol = 8 * dpt.finfo(Y.dtype).resolution
    assert_allclose(dpt.asnumpy(Y), expected_Y, atol=tol, rtol=tol)


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", _no_complex_dtypes)
def test_floor_ceil_trunc_order(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)
    if dtype == "b1":
        skip_if_dtype_not_supported("f2", q)

    arg_dt = np.dtype(dtype)
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, sycl_queue=q)
    X[..., 0::2] = -0.4
    X[..., 1::2] = 0.7

    for ord in ["C", "F", "A", "K"]:
        for perms in itertools.permutations(range(4)):
            U = dpt.permute_dims(X[:, ::-1, ::-1, :], perms)
            Y = dpt_call(U, order=ord)
            with np.errstate(all="ignore"):
                expected_Y = np_call(dpt.asnumpy(U))
            tol = 8 * max(
                dpt.finfo(Y.dtype).resolution,
                np.finfo(expected_Y.dtype).resolution,
            )
            assert_allclose(dpt.asnumpy(Y), expected_Y, atol=tol, rtol=tol)


@pytest.mark.parametrize("callable", [dpt.floor, dpt.ceil, dpt.trunc])
def test_floor_ceil_trunc_errors(callable):
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

    x = dpt.zeros(2)
    y = x
    assert_raises_regex(
        TypeError, "Input and output arrays have memory overlap", callable, x, y
    )

    x = dpt.zeros(2, dtype="float32")
    y = np.empty_like(x)
    assert_raises_regex(
        TypeError, "output array must be of usm_ndarray type", callable, x, y
    )


@pytest.mark.parametrize("callable", [dpt.floor, dpt.ceil, dpt.trunc])
@pytest.mark.parametrize("dtype", _no_complex_dtypes)
def test_floor_ceil_trunc_error_dtype(callable, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)
    if dtype == "b1":
        skip_if_dtype_not_supported("f2", q)

    x = dpt.zeros(5, dtype=dtype)
    y = dpt.empty_like(x, dtype="int16")
    assert_raises_regex(
        TypeError, "Output array of type.*is needed", callable, x, y
    )


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_floor_ceil_trunc_contig(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 100
    n_rep = 137
    Xnp = np.linspace(-99.9, 99.9, num=n_seq, dtype=dtype)

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


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_floor_ceil_trunc_strided(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    np.random.seed(42)
    strides = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    sizes = np.arange(2, 100)
    tol = 8 * dpt.finfo(dtype).resolution

    for ii in sizes:
        Xnp = np.random.uniform(low=-99.9, high=99.9, size=ii)
        Xnp.astype(dtype)
        X = dpt.asarray(Xnp)
        Ynp = np_call(Xnp)
        for jj in strides:
            assert_allclose(
                dpt.asnumpy(dpt_call(X[::jj])),
                Ynp[::jj],
                atol=tol,
                rtol=tol,
            )


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_floor_ceil_trunc_special_cases(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = [np.nan, np.inf, -np.inf]

    xf = np.array(x, dtype=dtype)
    yf = dpt.asarray(xf, dtype=dtype, sycl_queue=q)

    with np.errstate(all="ignore"):
        Y_np = np_call(xf)

    tol = 8 * dpt.finfo(dtype).resolution
    assert_allclose(dpt.asnumpy(dpt_call(yf)), Y_np, atol=tol, rtol=tol)
