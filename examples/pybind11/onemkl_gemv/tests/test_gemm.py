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
import dpctl.tensor as dpt


def test_gemv():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")
    Mnp, vnp = np.random.randn(5, 3), np.random.randn(3)
    r = dpt.empty((5,), dtype="d", sycl_queue=q)
    M = dpt.asarray(Mnp, sycl_queue=q)
    v = dpt.asarray(vnp, sycl_queue=q)
    hev, ev = gemv(q, M, v, r, [])
    hev.wait()
    rnp = dpt.asnumpy(r)
    assert np.allclose(rnp, Mnp @ vnp)


def test_sub():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")
    anp, bnp = np.random.randn(5), np.random.randn(5)
    r = dpt.empty((5,), dtype="d", sycl_queue=q)
    a = dpt.asarray(anp, sycl_queue=q)
    b = dpt.asarray(bnp, sycl_queue=q)
    hev, ev = sub(q, a, b, r, [])
    hev.wait()
    rnp = dpt.asnumpy(r)
    assert np.allclose(rnp + bnp, anp)


def test_axpby():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")
    xnp, pnp = np.random.randn(5), np.random.randn(5)
    x = dpt.asarray(xnp, sycl_queue=q)
    p = dpt.asarray(pnp, sycl_queue=q)
    hev, ev = axpby_inplace(q, 0.5, x, -0.7, p, [])
    hev.wait()
    rnp = dpt.asnumpy(p)
    assert np.allclose(rnp, 0.5 * xnp - 0.7 * pnp)


def test_dot():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")
    anp, bnp = np.random.randn(5), np.random.randn(5)
    a = dpt.asarray(anp, sycl_queue=q)
    b = dpt.asarray(bnp, sycl_queue=q)
    dot_res = dot_blocking(q, a, b)
    assert np.allclose(dot_res, np.dot(anp, bnp))


def test_norm_squared():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")
    anp = np.random.randn(5)
    a = dpt.asarray(anp, sycl_queue=q)
    dot_res = norm_squared_blocking(q, a)
    assert np.allclose(dot_res, np.dot(anp, anp))
