#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
from sycl_gemm import (
    axpby_inplace,
    dot_blocking,
    gemv,
    norm_squared_blocking,
    sub,
)

import dpctl
import dpctl.memory as dpm


def _real_dtype_for_device(q: dpctl.SyclQueue) -> np.dtype:
    """
    If the device supports fp64, return np.float64, else np.float32.
    """
    _fp64 = q.sycl_device.has_aspect_fp64
    return np.float64 if _fp64 else np.float32


def test_gemv():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    dtype = _real_dtype_for_device(q)

    Mnp = np.random.randn(5, 3).astype(dtype, copy=False)
    vnp = np.random.randn(3).astype(dtype, copy=False)

    M = dpm.MemoryUSMDevice(Mnp.nbytes, queue=q)
    ev1 = q.memcpy_async(dest=M, src=Mnp, count=Mnp.nbytes)

    v = dpm.MemoryUSMDevice(vnp.nbytes, queue=q)
    ev2 = q.memcpy_async(dest=v, src=vnp, count=vnp.nbytes)

    rnp = np.empty((5,), dtype=dtype)
    r = dpm.MemoryUSMDevice(rnp.nbytes, queue=q)

    hev, ev3 = gemv(
        q,
        M,
        v,
        r,
        5,
        3,
        np.dtype(Mnp.dtype),
        3,
        [ev1, ev2],
    )

    ev4 = q.memcpy_async(dest=rnp, src=r, count=rnp.nbytes, dEvents=[ev3])
    ev4.wait()
    hev.wait()

    assert np.allclose(rnp, Mnp @ vnp)


def test_sub():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    dtype = _real_dtype_for_device(q)

    anp = np.random.randn(5).astype(dtype, copy=False)
    bnp = np.random.randn(5).astype(dtype, copy=False)

    a = dpm.MemoryUSMDevice(anp.nbytes, queue=q)
    ev1 = q.memcpy_async(dest=a, src=anp, count=anp.nbytes)

    b = dpm.MemoryUSMDevice(bnp.nbytes, queue=q)
    ev2 = q.memcpy_async(dest=b, src=bnp, count=bnp.nbytes)

    rnp = np.empty((5,), dtype=dtype)
    r = dpm.MemoryUSMDevice(rnp.nbytes, queue=q)

    hev, ev3 = sub(q, a, b, r, 5, np.dtype(anp.dtype), [ev1, ev2])

    ev4 = q.memcpy_async(dest=rnp, src=r, count=rnp.nbytes, dEvents=[ev3])
    ev4.wait()
    hev.wait()

    assert np.allclose(rnp + bnp, anp)


def test_axpby():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    dtype = _real_dtype_for_device(q)

    xnp = np.random.randn(5).astype(dtype, copy=False)
    pnp = np.random.randn(5).astype(dtype, copy=False)

    x = dpm.MemoryUSMDevice(xnp.nbytes, queue=q)
    ev1 = q.memcpy_async(dest=x, src=xnp, count=xnp.nbytes)

    p = dpm.MemoryUSMDevice(pnp.nbytes, queue=q)
    ev2 = q.memcpy_async(dest=p, src=pnp, count=pnp.nbytes)

    alpha = 0.5
    beta = -0.7

    hev, ev3 = axpby_inplace(
        q,
        alpha,
        x,
        beta,
        p,
        5,
        np.dtype(xnp.dtype),
        [ev1, ev2],
    )

    rnp = np.empty((5,), dtype=dtype)
    ev4 = q.memcpy_async(dest=rnp, src=p, count=rnp.nbytes, dEvents=[ev3])
    ev4.wait()
    hev.wait()

    assert np.allclose(rnp, alpha * xnp + beta * pnp)


def test_dot():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    dtype = _real_dtype_for_device(q)

    anp = np.random.randn(5).astype(dtype, copy=False)
    bnp = np.random.randn(5).astype(dtype, copy=False)

    a = dpm.MemoryUSMDevice(anp.nbytes, queue=q)
    ev1 = q.memcpy_async(dest=a, src=anp, count=anp.nbytes)

    b = dpm.MemoryUSMDevice(bnp.nbytes, queue=q)
    ev2 = q.memcpy_async(dest=b, src=bnp, count=bnp.nbytes)

    dot_res = dot_blocking(q, a, b, 5, np.dtype(anp.dtype), [ev1, ev2])

    assert np.allclose(dot_res, np.dot(anp, bnp))


def test_norm_squared():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    dtype = _real_dtype_for_device(q)

    anp = np.random.randn(5).astype(dtype, copy=False)
    a = dpm.MemoryUSMDevice(anp.nbytes, queue=q)
    ev1 = q.memcpy_async(dest=a, src=anp, count=anp.nbytes)

    dot_res = norm_squared_blocking(q, a, 5, np.dtype(anp.dtype), [ev1])

    assert np.allclose(dot_res, np.dot(anp, anp))
