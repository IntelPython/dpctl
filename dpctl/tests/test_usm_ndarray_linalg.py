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

import dpctl
import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

_numeric_types = [
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


def test_matrix_transpose():
    get_queue_or_skip()

    X = dpt.reshape(dpt.arange(2 * 3, dtype="i4"), (2, 3))
    res = dpt.matrix_transpose(X)
    expected_res = X.mT

    assert expected_res.shape == res.shape
    assert expected_res.flags["C"] == res.flags["C"]
    assert expected_res.flags["F"] == res.flags["F"]
    assert dpt.all(X.mT == res)


def test_matrix_transpose_arg_validation():
    get_queue_or_skip()

    X = dpt.empty(5, dtype="i4")
    with pytest.raises(ValueError):
        dpt.matrix_transpose(X)

    X = dict()
    with pytest.raises(TypeError):
        dpt.matrix_transpose(X)

    X = dpt.empty((5, 5), dtype="i4")
    assert isinstance(dpt.matrix_transpose(X), dpt.usm_ndarray)


# @pytest.mark.parametrize("dtype", _numeric_types)
# def test_matmul_simple(dtype):
#     q = get_queue_or_skip()
#     skip_if_dtype_not_supported(dtype, q)

#     n, m = 235, 17
#     m1 = dpt.ones((m, n), dtype=dtype)
#     m2 = dpt.ones((n, m), dtype=dtype)

#     for k in [1, 2, 3, 4, 7, 8, 9, 15, 16, 17]:
#         r = dpt.matmul(m1[:k, :], m2[:, :k])
#         assert dpt.all(r == dpt.full((k, k), n, dtype=dtype))


@pytest.mark.parametrize("dtype", _numeric_types[::-1])
def test_matmul_simple2(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)
    dev = q.sycl_device
    if dev.is_cpu:
        cpu_count = dev.max_compute_units
        sub_devs = dev.create_sub_devices(partition=min(2, cpu_count // 2))
        ctx = dpctl.SyclContext(sub_devs[0])
        q = dpctl.SyclQueue(ctx, sub_devs[0])

    n, m = 235, 17
    m1 = dpt.ones((m, n), dtype=dtype, sycl_queue=q)
    m2 = dpt.ones((n, m), dtype=dtype, sycl_queue=q)

    for k in [1, 2, 3, 4, 7, 8, 9, 15, 16, 17]:
        r = dpt.matmul(m1[:k, :], m2[:, :k])
        assert dpt.all(r == dpt.full((k, k), n, dtype=dtype, sycl_queue=q))


@pytest.mark.parametrize("dtype", _numeric_types)
def test_matmul_nilpotent1(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n = 77
    N_mat = dpt.eye(n, k=1, dtype=dtype)
    I_mat = dpt.eye(n, dtype=dtype)
    R_mat = dpt.eye(n, dtype=dtype)
    for _ in range(n + 1):
        R_mat = I_mat + dpt.matmul(N_mat, R_mat)

    assert dpt.allclose(dpt.matmul(I_mat - N_mat, R_mat), I_mat)


@pytest.mark.parametrize("dtype", _numeric_types)
def test_matmul_nilpotent2(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n = 128
    u = dpt.ones((n, 1), dtype=dtype)
    v = dpt.ones((1, n), dtype=dtype)

    uv = dpt.matmul(u, v)
    uv_ref = u * v

    assert dpt.allclose(uv, uv_ref)


def test_matmul_null_axis():
    n = 3

    A_mat = dpt.ones((n, 0), dtype="f4")
    B_mat = dpt.ones((0, 1), dtype="f4")

    R_mat = dpt.matmul(A_mat, B_mat)
    assert R_mat.shape == (n, 1)

    R_mat = dpt.matmul(A_mat, B_mat[:, :0])
    assert R_mat.shape == (n, 0)


@pytest.mark.parametrize("dtype", ["i4", "f4"])
def test_matmul_dims(dtype):
    get_queue_or_skip()

    n, m, k, b = 4, 5, 7, 3
    v = dpt.ones(k, dtype=dtype)
    m1 = dpt.ones((n, k), dtype=dtype)
    m2 = dpt.ones((k, m), dtype=dtype)
    st1 = dpt.ones((b, n, k), dtype=dtype)
    st2 = dpt.ones((b, k, m), dtype=dtype)

    r = dpt.matmul(v, v)
    assert r.shape == tuple()
    assert dpt.round(r) == k

    r = dpt.matmul(m1, v)
    assert r.shape == (n,)
    assert dpt.all(dpt.round(r) == k)

    r = dpt.matmul(v, m2)
    assert r.shape == (m,)
    assert dpt.all(dpt.round(r) == k)

    r = dpt.matmul(m1, m2)
    assert r.shape == (
        n,
        m,
    )
    assert dpt.all(dpt.round(r) == k)

    r = dpt.matmul(v, st2)
    assert r.shape == (
        b,
        m,
    )
    assert dpt.all(dpt.round(r) == k)

    r = dpt.matmul(st1, v)
    assert r.shape == (
        b,
        n,
    )
    assert dpt.all(dpt.round(r) == k)

    r = dpt.matmul(st1, m2)
    assert r.shape == (
        b,
        n,
        m,
    )
    assert dpt.all(dpt.round(r) == k)

    r = dpt.matmul(m1, st2)
    assert r.shape == (
        b,
        n,
        m,
    )
    assert dpt.all(dpt.round(r) == k)

    r = dpt.matmul(st1, st2)
    assert r.shape == (
        b,
        n,
        m,
    )
    assert dpt.all(dpt.round(r) == k)


def test_matmul_arg_validation():
    get_queue_or_skip()

    s1, s2 = dpt.ones(tuple()), dpt.zeros(tuple())
    v1, v2 = dpt.ones(16), dpt.zeros(16)

    with pytest.raises(ValueError):
        dpt.matmul(s1, v2)

    with pytest.raises(ValueError):
        dpt.matmul(v1, s2)

    with pytest.raises(TypeError):
        dpt.matmul(dict(), v2)

    with pytest.raises(TypeError):
        dpt.matmul(v2, None)


def test_matmul_dims_validation():
    get_queue_or_skip()

    m1 = dpt.ones((16, 16))
    m2 = dpt.ones((16, 16))

    # contraction dimensions mismatch
    with pytest.raises(ValueError):
        dpt.matmul(m1[:, :7], m2[:3, :])

    m1 = dpt.ones((3, 4, 5))
    m2 = dpt.ones((2, 5, 3))
    # broadcasting dimensions mismatch
    with pytest.raises(ValueError):
        dpt.matmul(m1, m2)


def test_matmul_broadcasting():
    get_queue_or_skip()

    m1 = dpt.ones((7, 11, 16))
    m2 = dpt.ones((16, 13))

    r = dpt.matmul(m1, m2[dpt.newaxis, ...])

    assert r.shape == (7, 11, 13)


@pytest.mark.parametrize("dtype", ["i4", "i8", "f4", "c8"][::-1])
def test_matmul_strided(dtype):
    get_queue_or_skip()

    m1_shape = (14, 22, 32)
    m1_size = 1
    for el in m1_shape:
        m1_size = m1_size * el

    m1 = dpt.remainder(dpt.arange(1, m1_size + 1, dtype="i8"), 13)
    m1 = dpt.reshape(dpt.astype(m1, dtype), (14, 22, 32))[::2, ::-2, ::2]
    m2 = dpt.ones((14, 16, 13), dtype=dtype)[::2, :, :]

    r = dpt.matmul(m1, m2)

    assert r.shape == (7, 11, 13)
    ref = np.matmul(dpt.asnumpy(m1), dpt.asnumpy(m2))
    assert np.allclose(dpt.asnumpy(r), ref)


def test_matmul_out():
    get_queue_or_skip()

    m1 = (
        dpt.arange(14, dtype="f4")[:, dpt.newaxis, dpt.newaxis]
        + dpt.arange(17, dtype="f4")[dpt.newaxis, :, dpt.newaxis]
        + dpt.arange(128, dtype="f4")[dpt.newaxis, dpt.newaxis, :]
    )
    assert m1.shape == (14, 17, 128)
    m2 = dpt.tile(
        dpt.reshape(dpt.asarray([1, 2], dtype="f4"), (2, 1, 1)), (7, 128, 13)
    )
    assert m2.shape == (14, 128, 13)

    buf = dpt.zeros((2 * 14, 3 * 17, 13), dtype="f4")
    res = dpt.matmul(m1, m2, out=buf[::-2, 1::3, :])

    assert dpt.allclose(res, buf[::-2, 1::3, :])
    assert dpt.allclose(dpt.zeros_like(res), buf[::-2, 0::3, :])
    assert dpt.allclose(dpt.zeros_like(res), buf[::-2, 2::3, :])

    ref = np.matmul(dpt.asnumpy(m1), dpt.asnumpy(m2))
    assert np.allclose(ref, dpt.asnumpy(res))


@pytest.mark.parametrize("dtype", _numeric_types)
def test_tensordot_outer(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    t1 = dpt.ones((3, 8), dtype=dtype)
    t2 = dpt.ones((4, 12), dtype=dtype)

    r = dpt.tensordot(t1, t2, axes=0)
    assert r.shape == t1.shape + t2.shape
    assert dpt.allclose(r, dpt.ones_like(r))


@pytest.mark.parametrize("dtype", _numeric_types)
def test_tensordot_inner(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    t1 = dpt.ones((3, 8), dtype=dtype)
    t2 = dpt.ones((4, 8), dtype=dtype)

    r = dpt.tensordot(t1, t2.mT, axes=1)
    assert r.shape == t1.shape[:1] + t2.shape[:1]
    assert dpt.allclose(r, dpt.full_like(r, fill_value=t1.shape[1]))


@pytest.mark.parametrize("dtype", _numeric_types)
def test_tensordot_double(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    t1 = dpt.ones((2, 4, 8), dtype=dtype)
    t2 = dpt.ones((3, 4, 8), dtype=dtype)

    r = dpt.tensordot(t1, dpt.permute_dims(t2, (1, 2, 0)), axes=2)
    assert r.shape == t1.shape[:1] + t2.shape[:1]
    expected = dpt.prod(dpt.asarray(t1.shape[1:]))
    assert dpt.allclose(r, dpt.full_like(r, fill_value=expected))


@pytest.mark.parametrize("dtype", ["i4", "f4"])
def test_tensordot_axes_sequence(dtype):
    get_queue_or_skip()

    r = 4
    t1 = dpt.ones((2, 2, 4, 3), dtype=dtype)
    t2 = dpt.ones((3, 2, 4, 3), dtype=dtype)

    assert len(t1.shape) == r
    assert len(t2.shape) == r

    expected = dpt.prod(dpt.asarray(t1.shape[1:]))
    ps1 = itertools.permutations(range(r))
    ps2 = itertools.permutations(range(r))

    for p1 in ps1:
        assert len(p1) == r
        inv_p1 = sorted(range(r), key=p1.__getitem__)
        u1 = dpt.permute_dims(t1, p1)
        x1_axes = inv_p1[1:]
        for p2 in ps2:
            inv_p2 = sorted(range(r), key=p2.__getitem__)
            u2 = dpt.permute_dims(t2, p2)
            x2_axes = inv_p2[1:]

            tdr = dpt.tensordot(u1, u2, axes=(x1_axes, x2_axes))
            assert tdr.shape == t1.shape[:1] + t2.shape[:1]
            assert dpt.allclose(tdr, dpt.full_like(tdr, fill_value=expected))


@pytest.mark.parametrize("dtype", _numeric_types)
def test_vecdot_1d(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n = 511
    v1 = dpt.ones(n, dtype=dtype)

    v2 = dpt.ones(n, dtype=dtype)

    r = dpt.vecdot(v1, v2)

    assert r == n


@pytest.mark.parametrize("dtype", _numeric_types)
def test_vecdot_3d(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    m1, m2, n = 7, 3, 511
    v1 = dpt.ones((m1, m2, n), dtype=dtype)

    v2 = dpt.ones((m1, m2, n), dtype=dtype)

    r = dpt.vecdot(v1, v2)

    assert r.shape == (
        m1,
        m2,
    )
    assert dpt.all(r == n)


@pytest.mark.parametrize("dtype", _numeric_types)
def test_vecdot_axis(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    m1, m2, n = 7, 3, 511
    v1 = dpt.ones((m1, n, m2), dtype=dtype)

    v2 = dpt.ones((m1, n, m2), dtype=dtype)

    r = dpt.vecdot(v1, v2, axis=1)

    assert r.shape == (
        m1,
        m2,
    )
    assert dpt.all(r == n)


@pytest.mark.parametrize("dtype", _numeric_types)
def test_vecdot_strided(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    m1, m2, n = 7, 3, 511
    list1 = [1, 0, 2, 0]
    pattern1 = dpt.asarray(list1, dtype=dtype)
    n_padded1 = pattern1.size * (1 + ((n - 1) // pattern1.size))
    v1 = dpt.tile(dpt.reshape(pattern1, (1, -1, 1)), (m1, n_padded1, m2))[
        ::-1, :n, :
    ]

    list2 = [1, 2, 1, 2]
    pattern2 = dpt.asarray(list2, dtype=dtype)
    n_padded2 = pattern2.size * (1 + ((n - 1) // pattern2.size))
    v2 = dpt.tile(dpt.reshape(pattern2, (1, -1, 1)), (m1, n_padded2, m2))[
        :, :n, ::-1
    ]

    r = dpt.vecdot(v1, v2, axis=1)

    ref = sum(
        el1 * el2
        for el1, el2 in zip((list1 * n_padded1)[:n], (list2 * n_padded1)[:n])
    )

    assert r.shape == (
        m1,
        m2,
    )
    assert dpt.all(r == ref)


def test_vector_arg_validation():
    get_queue_or_skip()

    s1, s2 = dpt.ones(tuple()), dpt.zeros(tuple())
    v1, v2 = dpt.ones(16), dpt.zeros(16)

    with pytest.raises(ValueError):
        dpt.vecdot(s1, v2)

    with pytest.raises(ValueError):
        dpt.vecdot(v1, s2)

    with pytest.raises(TypeError):
        dpt.vecdot(dict(), v2)

    with pytest.raises(TypeError):
        dpt.vecdot(v2, None)

    with pytest.raises(ValueError):
        dpt.vecdot(v1[:5], v2[:4])

    with pytest.raises(ValueError):
        dpt.vecdot(v1, v2, axis=2)


@pytest.mark.parametrize("dt1", _numeric_types)
@pytest.mark.parametrize("dt2", _numeric_types)
def test_vecdot_type_promotion(dt1, dt2):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt1, q)
    skip_if_dtype_not_supported(dt2, q)

    v1 = dpt.ones(128, dtype=dt1)
    v2 = dpt.ones(128, dtype=dt2)

    r = dpt.vecdot(v1, v2)
    mul = v1 * v2
    assert r.shape == tuple()
    assert r.dtype == mul.dtype
    assert dpt.allclose(r, dpt.sum(mul, dtype=mul.dtype))
