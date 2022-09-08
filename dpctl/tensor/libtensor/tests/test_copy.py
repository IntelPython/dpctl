#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2022 Intel Corporation
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

import dpctl
import dpctl.tensor as dpt
import dpctl.tensor._tensor_impl as ti

_usm_types_list = ["shared", "device", "host"]
_typestrs_list = [
    "b1",
    "u1",
    "i1",
    "u2",
    "i2",
    "u4",
    "i4",
    "u8",
    "i8",
    "f2",
    "f4",
    "f8",
    "c8",
    "c16",
]


@pytest.fixture(params=_usm_types_list)
def usm_type(request):
    return request.param


@pytest.fixture(params=_typestrs_list)
def src_typestr(request):
    return request.param


@pytest.fixture(params=_typestrs_list)
def dst_typestr(request):
    return request.param


def _random_vector(n, src_dt):
    src_dt = np.dtype(src_dt)
    if np.issubdtype(src_dt, np.integer):
        Xnp = np.random.randint(0, 2, size=n).astype(src_dt)
    elif np.issubdtype(src_dt, np.floating):
        Xnp = np.random.randn(n).astype(src_dt)
    elif np.issubdtype(src_dt, np.complexfloating):
        Xnp = np.random.randn(n) + 1j * np.random.randn(n)
        Xnp = Xnp.astype(src_dt)
    else:
        Xnp = np.random.randint(0, 2, size=n).astype(src_dt)
    return Xnp


def _force_cast(Xnp, dst_dt):
    if np.issubdtype(Xnp.dtype, np.complexfloating) and not np.issubdtype(
        dst_dt, np.complexfloating
    ):
        R = Xnp.real.astype(dst_dt, casting="unsafe", copy=True)
    else:
        R = Xnp.astype(dst_dt, casting="unsafe", copy=True)
    return R


def are_close(X1, X2):
    if np.issubdtype(X2.dtype, np.floating) or np.issubdtype(
        X2.dtype, np.complexfloating
    ):
        return np.allclose(
            X1, X2, atol=np.finfo(X2.dtype).eps, rtol=np.finfo(X2.dtype).eps
        )
    else:
        return np.allclose(X1, X2)


def test_copy1d_c_contig(src_typestr, dst_typestr):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")
    src_dt = np.dtype(src_typestr)
    dst_dt = np.dtype(dst_typestr)
    Xnp = _random_vector(4096, src_dt)

    X = dpt.asarray(Xnp, sycl_queue=q)
    Y = dpt.empty(Xnp.shape, dtype=dst_typestr, sycl_queue=q)
    hev, ev = ti._copy_usm_ndarray_into_usm_ndarray(src=X, dst=Y, sycl_queue=q)
    hev.wait()
    Ynp = _force_cast(Xnp, dst_dt)
    assert are_close(Ynp, dpt.asnumpy(Y))
    # q.wait()


def test_copy1d_strided(src_typestr, dst_typestr):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")
    src_dt = np.dtype(src_typestr)
    dst_dt = np.dtype(dst_typestr)
    Xnp = _random_vector(4096, src_dt)

    for s in (
        slice(None, None, 2),
        slice(None, None, -2),
    ):
        X = dpt.asarray(Xnp, sycl_queue=q)[s]
        Y = dpt.empty(X.shape, dtype=dst_typestr, sycl_queue=q)
        hev, ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=X, dst=Y, sycl_queue=q
        )
        hev.wait()
        Ynp = _force_cast(Xnp[s], dst_dt)
        assert are_close(Ynp, dpt.asnumpy(Y))

    # now 0-strided source
    X = dpt.usm_ndarray((4096,), dtype=src_typestr, strides=(0,))
    X[0] = Xnp[0]
    Y = dpt.empty(X.shape, dtype=dst_typestr, sycl_queue=q)
    hev, ev = ti._copy_usm_ndarray_into_usm_ndarray(src=X, dst=Y, sycl_queue=q)
    Ynp = _force_cast(np.broadcast_to(Xnp[0], X.shape), dst_dt)
    hev.wait()
    assert are_close(Ynp, dpt.asnumpy(Y))


def test_copy1d_strided2(src_typestr, dst_typestr):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")
    src_dt = np.dtype(src_typestr)
    dst_dt = np.dtype(dst_typestr)
    Xnp = _random_vector(4096, src_dt)

    for s in (
        slice(None, None, 2),
        slice(None, None, -2),
    ):
        X = dpt.asarray(Xnp, sycl_queue=q)[s]
        Y = dpt.empty(X.shape, dtype=dst_typestr, sycl_queue=q)[::-1]
        hev, ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=X, dst=Y, sycl_queue=q
        )
        Ynp = _force_cast(Xnp[s], dst_dt)
        hev.wait()
        assert are_close(Ynp, dpt.asnumpy(Y))


@pytest.mark.parametrize("sgn1", [-1, 1])
@pytest.mark.parametrize("sgn2", [-1, 1])
@pytest.mark.parametrize("st1", [5, 3, 1])
@pytest.mark.parametrize("st2", [1, 2])
def test_copy2d(src_typestr, dst_typestr, st1, sgn1, st2, sgn2):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    src_dt = np.dtype(src_typestr)
    dst_dt = np.dtype(dst_typestr)
    n1, n2 = 15, 12
    Snp = _random_vector(st1 * st2 * n1 * n2, src_dt).reshape(
        (st1 * n1, st2 * n2)
    )
    Xnp = Snp[
        slice(None, None, st1 * sgn1),
        slice(None, None, st2 * sgn2),
    ]
    S = dpt.asarray(Snp, sycl_queue=q)
    X = S[
        slice(None, None, st1 * sgn1),
        slice(None, None, st2 * sgn2),
    ]
    Y = dpt.empty((n1, n2), dtype=dst_dt)
    hev, ev = ti._copy_usm_ndarray_into_usm_ndarray(src=X, dst=Y, sycl_queue=q)
    Ynp = _force_cast(Xnp, dst_dt)
    hev.wait()
    assert are_close(Ynp, dpt.asnumpy(Y))
    Yst = dpt.empty((2 * n1, n2), dtype=dst_dt)[::2, ::-1]
    hev, ev = ti._copy_usm_ndarray_into_usm_ndarray(
        src=X, dst=Yst, sycl_queue=q
    )
    Y = dpt.empty((n1, n2), dtype=dst_dt)
    hev2, ev2 = ti._copy_usm_ndarray_into_usm_ndarray(
        src=Yst, dst=Y, sycl_queue=q, depends=[ev]
    )
    Ynp = _force_cast(Xnp, dst_dt)
    hev2.wait()
    hev.wait()
    assert are_close(Ynp, dpt.asnumpy(Y))
    # q.wait()


@pytest.mark.parametrize("sgn1", [-1, 1])
@pytest.mark.parametrize("sgn2", [-1, 1])
@pytest.mark.parametrize("sgn3", [-1, 1])
@pytest.mark.parametrize("st1", [3, 1])
@pytest.mark.parametrize("st2", [1, 2])
@pytest.mark.parametrize("st3", [3, 2])
def test_copy3d(src_typestr, dst_typestr, st1, sgn1, st2, sgn2, st3, sgn3):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    src_dt = np.dtype(src_typestr)
    dst_dt = np.dtype(dst_typestr)
    n1, n2, n3 = 5, 4, 6
    Snp = _random_vector(st1 * st2 * st3 * n1 * n2 * n3, src_dt).reshape(
        (st1 * n1, st2 * n2, st3 * n3)
    )
    Xnp = Snp[
        slice(None, None, st1 * sgn1),
        slice(None, None, st2 * sgn2),
        slice(None, None, st3 * sgn3),
    ]
    S = dpt.asarray(Snp, sycl_queue=q)
    X = S[
        slice(None, None, st1 * sgn1),
        slice(None, None, st2 * sgn2),
        slice(None, None, st3 * sgn3),
    ]
    Y = dpt.empty((n1, n2, n3), dtype=dst_dt)
    hev, ev = ti._copy_usm_ndarray_into_usm_ndarray(src=X, dst=Y, sycl_queue=q)
    Ynp = _force_cast(Xnp, dst_dt)
    hev.wait()
    assert are_close(Ynp, dpt.asnumpy(Y)), "1"
    Yst = dpt.empty((2 * n1, n2, n3), dtype=dst_dt)[::2, ::-1]
    hev2, ev2 = ti._copy_usm_ndarray_into_usm_ndarray(
        src=X, dst=Yst, sycl_queue=q
    )
    Y2 = dpt.empty((n1, n2, n3), dtype=dst_dt)
    hev3, ev3 = ti._copy_usm_ndarray_into_usm_ndarray(
        src=Yst, dst=Y2, sycl_queue=q, depends=[ev2]
    )
    hev3.wait()
    hev2.wait()
    assert are_close(Ynp, dpt.asnumpy(Y2)), "2"
    # q.wait()
