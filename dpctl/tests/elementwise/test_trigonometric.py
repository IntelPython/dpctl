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

import itertools
import os
import re

import numpy as np
import pytest
from numpy.testing import assert_allclose

import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _all_dtypes, _map_to_device_dtype

_trig_funcs = [(np.sin, dpt.sin), (np.cos, dpt.cos), (np.tan, dpt.tan)]
_inv_trig_funcs = [
    (np.arcsin, dpt.asin),
    (np.arccos, dpt.acos),
    (np.arctan, dpt.atan),
]
_all_funcs = _trig_funcs + _inv_trig_funcs
_dpt_funcs = [t[1] for t in _all_funcs]


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", _all_dtypes)
def test_trig_out_type(np_call, dpt_call, dtype):
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


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_trig_real_contig(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 100
    n_rep = 137
    if np_call in _trig_funcs:
        Xnp = np.linspace(
            -np.pi / 2 * 0.99, np.pi / 2 * 0.99, num=n_seq, dtype=dtype
        )
    if np_call == np.arctan:
        Xnp = np.linspace(-100.0, 100.0, num=n_seq, dtype=dtype)
    else:
        Xnp = np.linspace(-1.0, 1.0, num=n_seq, dtype=dtype)

    X = dpt.asarray(np.repeat(Xnp, n_rep), dtype=dtype, sycl_queue=q)
    Y = dpt_call(X)

    tol = 8 * dpt.finfo(dtype).resolution
    assert_allclose(
        dpt.asnumpy(Y), np.repeat(np_call(Xnp), n_rep), atol=tol, rtol=tol
    )

    Z = dpt.empty_like(X, dtype=dtype)
    dpt_call(X, out=Z)

    assert_allclose(
        dpt.asnumpy(Z), np.repeat(np_call(Xnp), n_rep), atol=tol, rtol=tol
    )


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_trig_complex_contig(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 256
    n_rep = 137
    low = -9.0
    high = 9.0
    x1 = np.random.uniform(low=low, high=high, size=n_seq)
    x2 = np.random.uniform(low=low, high=high, size=n_seq)
    Xnp = x1 + 1j * x2

    # stay away from poles and branch lines
    modulus = np.abs(Xnp)
    sel = np.logical_or(
        modulus < 0.9,
        np.logical_and(
            modulus > 1.2, np.minimum(np.abs(x2), np.abs(x1)) > 0.05
        ),
    )
    Xnp = Xnp[sel]

    X = dpt.repeat(dpt.asarray(Xnp, dtype=dtype, sycl_queue=q), n_rep)
    Y = dpt_call(X)

    expected = np.repeat(np_call(Xnp), n_rep)

    tol = 50 * dpt.finfo(dtype).resolution
    assert_allclose(dpt.asnumpy(Y), expected, atol=tol, rtol=tol)

    Z = dpt.empty_like(X, dtype=dtype)
    dpt_call(X, out=Z)

    assert_allclose(dpt.asnumpy(Z), expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("usm_type", ["device", "shared", "host"])
def test_trig_usm_type(np_call, dpt_call, usm_type):
    q = get_queue_or_skip()

    arg_dt = np.dtype("f4")
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, usm_type=usm_type, sycl_queue=q)
    if np_call in _trig_funcs:
        X[..., 0::2] = np.pi / 6
        X[..., 1::2] = np.pi / 3
    if np_call == np.arctan:
        X[..., 0::2] = -2.2
        X[..., 1::2] = 3.3
    else:
        X[..., 0::2] = -0.3
        X[..., 1::2] = 0.7

    Y = dpt_call(X)
    assert Y.usm_type == X.usm_type
    assert Y.sycl_queue == X.sycl_queue
    assert Y.flags.c_contiguous

    expected_Y = np_call(dpt.asnumpy(X))
    tol = 8 * dpt.finfo(Y.dtype).resolution
    assert_allclose(dpt.asnumpy(Y), expected_Y, atol=tol, rtol=tol)


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", _all_dtypes)
def test_trig_order(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    input_shape = (4, 4, 4, 4)
    X = dpt.empty(input_shape, dtype=arg_dt, sycl_queue=q)
    if np_call in _trig_funcs:
        X[..., 0::2] = np.pi / 6
        X[..., 1::2] = np.pi / 3
    if np_call == np.arctan:
        X[..., 0::2] = -2.2
        X[..., 1::2] = 3.3
    else:
        X[..., 0::2] = -0.3
        X[..., 1::2] = 0.7

    for perms in itertools.permutations(range(4)):
        U = dpt.permute_dims(X[:, ::-1, ::-1, :], perms)
        expected_Y = np_call(dpt.asnumpy(U))
        for ord in ["C", "F", "A", "K"]:
            Y = dpt_call(U, order=ord)
            tol = 8 * max(
                dpt.finfo(Y.dtype).resolution,
                np.finfo(expected_Y.dtype).resolution,
            )
            assert_allclose(dpt.asnumpy(Y), expected_Y, atol=tol, rtol=tol)


@pytest.mark.parametrize("callable", _dpt_funcs)
@pytest.mark.parametrize("dtype", _all_dtypes)
def test_trig_error_dtype(callable, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = dpt.zeros(5, dtype=dtype)
    y = dpt.empty_like(x, dtype="int16")
    with pytest.raises(ValueError) as excinfo:
        callable(x, out=y)
    assert re.match("Output array of type.*is needed", str(excinfo.value))


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_trig_real_strided(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    np.random.seed(42)
    strides = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    sizes = [2, 3, 4, 6, 8, 9, 24, 50, 72]
    tol = 8 * dpt.finfo(dtype).resolution

    low = -100.0
    high = 100.0
    if np_call in [np.arccos, np.arcsin]:
        low = -1.0
        high = 1.0
    elif np_call in [np.tan]:
        low = -np.pi / 2 * (0.99)
        high = np.pi / 2 * (0.99)

    for ii in sizes:
        Xnp = np.random.uniform(low=low, high=high, size=ii)
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
@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_trig_complex_strided(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    np.random.seed(42)
    strides = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    sizes = [2, 4, 6, 8, 9, 24, 72]
    tol = 50 * dpt.finfo(dtype).resolution

    low = -9.0
    high = 9.0
    while True:
        x1 = np.random.uniform(low=low, high=high, size=2 * sum(sizes))
        x2 = np.random.uniform(low=low, high=high, size=2 * sum(sizes))
        Xnp_all = np.array(
            [complex(v1, v2) for v1, v2 in zip(x1, x2)], dtype=dtype
        )

        # stay away from poles and branch lines
        modulus = np.abs(Xnp_all)
        sel = np.logical_or(
            modulus < 0.9,
            np.logical_and(
                modulus > 1.2, np.minimum(np.abs(x2), np.abs(x1)) > 0.05
            ),
        )
        Xnp_all = Xnp_all[sel]
        if Xnp_all.size > sum(sizes):
            break

    pos = 0
    for ii in sizes:
        pos = pos + ii
        Xnp = Xnp_all[:pos]
        Xnp = Xnp[-ii:]
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
def test_trig_real_special_cases(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = [np.nan, np.inf, -np.inf, 2.0, -2.0, +0.0, -0.0, +1.0, -1.0]

    xf = np.array(x, dtype=dtype)
    yf = dpt.asarray(xf, dtype=dtype, sycl_queue=q)

    with np.errstate(all="ignore"):
        Y_np = np_call(xf)

    tol = 8 * dpt.finfo(dtype).resolution
    Y = dpt_call(yf)
    assert_allclose(dpt.asnumpy(Y), Y_np, atol=tol, rtol=tol)


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_trig_complex_special_cases_conj_property(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = [np.nan, np.inf, -np.inf, +0.0, -0.0, +1.0, -1.0]
    xc = [complex(*val) for val in itertools.product(x, repeat=2)]

    Xc_np = np.array(xc, dtype=dtype)
    Xc = dpt.asarray(Xc_np, dtype=dtype, sycl_queue=q)

    tol = 50 * dpt.finfo(dtype).resolution
    Y = dpt_call(Xc)
    Yc = dpt_call(dpt.conj(Xc))

    dpt.allclose(Y, dpt.conj(Yc), atol=tol, rtol=tol)


@pytest.mark.skipif(
    os.name != "posix", reason="Known to fail on Windows due to bug in NumPy"
)
@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_trig_complex_special_cases(np_call, dpt_call, dtype):

    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = [np.nan, np.inf, -np.inf, +0.0, -0.0, +1.0, -1.0]
    xc = [complex(*val) for val in itertools.product(x, repeat=2)]

    Xc_np = np.array(xc, dtype=dtype)
    Xc = dpt.asarray(Xc_np, dtype=dtype, sycl_queue=q)

    with np.errstate(all="ignore"):
        Ynp = np_call(Xc_np)

    tol = 50 * dpt.finfo(dtype).resolution
    Y = dpt_call(Xc)
    assert_allclose(dpt.asnumpy(dpt.real(Y)), np.real(Ynp), atol=tol, rtol=tol)
    assert_allclose(dpt.asnumpy(dpt.imag(Y)), np.imag(Ynp), atol=tol, rtol=tol)
