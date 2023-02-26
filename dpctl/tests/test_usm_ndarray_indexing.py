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
from helper import get_queue_or_skip

# import dpctl
import dpctl.tensor as dpt

# from helper import skip_if_dtype_not_supported


def test_basic_slice1():
    q = get_queue_or_skip()
    x = dpt.empty(10, dtype="u2", sycl_queue=q)
    y = x[0]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.ndim == 0
    assert y.shape == tuple()
    assert y.strides == tuple()


def test_basic_slice2():
    q = get_queue_or_skip()
    x = dpt.empty(10, dtype="i2", sycl_queue=q)
    y = x[(0,)]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.ndim == 0
    assert y.shape == tuple()
    assert y.strides == tuple()


def test_basic_slice3():
    q = get_queue_or_skip()
    x = dpt.empty(10, dtype="i2", sycl_queue=q)
    y = x[:]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.ndim == x.ndim
    assert y.shape == x.shape
    assert y.strides == x.strides
    y = x[(slice(None, None, None),)]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.ndim == x.ndim
    assert y.shape == x.shape
    assert y.strides == x.strides


def test_basic_slice4():
    q = get_queue_or_skip()
    n0, n1 = 5, 3
    x = dpt.empty((n0, n1), dtype="f4", sycl_queue=q)
    y = x[::-1]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == x.shape
    assert y.strides == (-x.strides[0], x.strides[1])
    actual_offset = y.__sycl_usm_array_interface__["offset"]
    assert actual_offset == (n0 - 1) * n1


def test_basic_slice5():
    q = get_queue_or_skip()
    n0, n1 = 5, 3
    x = dpt.empty((n0, n1), dtype="c8", sycl_queue=q)
    y = x[:, ::-1]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == x.shape
    assert y.strides == (x.strides[0], -x.strides[1])
    actual_offset = y.__sycl_usm_array_interface__["offset"]
    assert actual_offset == (n1 - 1)


def test_basic_slice6():
    q = get_queue_or_skip()
    i0, n0, n1 = 2, 4, 3
    x = dpt.empty((n0, n1), dtype="c8", sycl_queue=q)
    y = x[i0, ::-1]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == (x.shape[1],)
    assert y.strides == (-x.strides[1],)
    actual_offset = y.__sycl_usm_array_interface__["offset"]
    expected_offset = i0 * x.strides[0] + (n1 - 1) * x.strides[1]
    assert actual_offset == expected_offset


def test_basic_slice7():
    q = get_queue_or_skip()
    n0, n1, n2 = 5, 3, 2
    x = dpt.empty((n0, n1, n2), dtype="?", sycl_queue=q)
    y = x[..., ::-1]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == x.shape
    assert y.strides == (
        x.strides[0],
        x.strides[1],
        -x.strides[2],
    )
    actual_offset = y.__sycl_usm_array_interface__["offset"]
    expected_offset = (n2 - 1) * x.strides[2]
    assert actual_offset == expected_offset


def test_basic_slice8():
    q = get_queue_or_skip()
    n0, n1 = 3, 7
    x = dpt.empty((n0, n1), dtype="u1", sycl_queue=q)
    y = x[..., dpt.newaxis]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == (n0, n1, 1)
    assert y.strides == (n1, 1, 0)


def test_basic_slice9():
    q = get_queue_or_skip()
    n0, n1 = 3, 7
    x = dpt.empty((n0, n1), dtype="u8", sycl_queue=q)
    y = x[dpt.newaxis, ...]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == (1, n0, n1)
    assert y.strides == (0, n1, 1)


def test_basic_slice10():
    q = get_queue_or_skip()
    n0, n1, n2 = 3, 7, 5
    x = dpt.empty((n0, n1, n2), dtype="u1", sycl_queue=q)
    y = x[dpt.newaxis, ..., :]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == (1, n0, n1, n2)
    assert y.strides == (0, n1 * n2, n2, 1)


def _all_equal(it1, it2):
    return all(dpt.asnumpy(x) == dpt.asnumpy(y) for x, y in zip(it1, it2))


def test_advanced_slice1():
    q = get_queue_or_skip()
    ii = dpt.asarray([1, 2], sycl_queue=q)
    x = dpt.arange(10, dtype="i4", sycl_queue=q)
    y = x[ii]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == ii.shape
    assert y.strides == (1,)
    # FIXME, once usm_ndarray.__equal__ is implemented,
    # use of asnumpy should be removed
    assert _all_equal(
        (x[ii[k]] for k in range(ii.shape[0])),
        (y[k] for k in range(ii.shape[0])),
    )
    y = x[(ii,)]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == ii.shape
    assert y.strides == (1,)
    # FIXME, once usm_ndarray.__equal__ is implemented,
    # use of asnumpy should be removed
    assert _all_equal(
        (x[ii[k]] for k in range(ii.shape[0])),
        (y[k] for k in range(ii.shape[0])),
    )


def test_advanced_slice1_negative_strides():
    q = get_queue_or_skip()
    ii = dpt.asarray([0, 1], sycl_queue=q)
    x = dpt.flip(dpt.arange(5, dtype="i4", sycl_queue=q))
    y = x[ii]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == ii.shape
    assert y.strides == (1,)
    # FIXME, once usm_ndarray.__equal__ is implemented,
    # use of asnumpy should be removed
    assert _all_equal(
        (x[ii[k]] for k in range(ii.shape[0])),
        (y[k] for k in range(ii.shape[0])),
    )


def test_advanced_slice2():
    q = get_queue_or_skip()
    ii = dpt.asarray([1, 2], sycl_queue=q)
    x = dpt.arange(10, dtype="i4", sycl_queue=q)
    y = x[ii, dpt.newaxis]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == ii.shape + (1,)
    assert y.flags["C"]


def test_advanced_slice3():
    q = get_queue_or_skip()
    ii = dpt.asarray([1, 2], sycl_queue=q)
    x = dpt.arange(10, dtype="i4", sycl_queue=q)
    y = x[dpt.newaxis, ii]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == (1,) + ii.shape
    assert y.flags["C"]


def _make_3d(dt, q):
    return dpt.reshape(
        dpt.arange(3 * 3 * 3, dtype=dt, sycl_queue=q),
        (
            3,
            3,
            3,
        ),
    )


def test_advanced_slice4():
    q = get_queue_or_skip()
    ii = dpt.asarray([1, 2], sycl_queue=q)
    x = _make_3d("i4", q)
    y = x[ii, ii, ii]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == ii.shape
    assert _all_equal(
        (x[ii[k], ii[k], ii[k]] for k in range(ii.shape[0])),
        (y[k] for k in range(ii.shape[0])),
    )


def test_advanced_slice5():
    q = get_queue_or_skip()
    ii = dpt.asarray([1, 2], sycl_queue=q)
    x = _make_3d("i4", q)
    with pytest.raises(IndexError):
        x[ii, 0, ii]


def test_advanced_slice6():
    q = get_queue_or_skip()
    ii = dpt.asarray([1, 2], sycl_queue=q)
    x = _make_3d("i4", q)
    y = x[:, ii, ii]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == (
        x.shape[0],
        ii.shape[0],
    )
    assert _all_equal(
        (
            x[i, ii[k], ii[k]]
            for i in range(x.shape[0])
            for k in range(ii.shape[0])
        ),
        (y[i, k] for i in range(x.shape[0]) for k in range(ii.shape[0])),
    )


def test_advanced_slice7():
    q = get_queue_or_skip()
    mask = dpt.asarray(
        [
            [[True, True, False], [False, True, True], [True, False, True]],
            [[True, False, False], [False, False, True], [False, True, False]],
            [[True, True, True], [False, False, False], [False, False, True]],
        ],
        sycl_queue=q,
    )
    x = _make_3d("i2", q)
    y = x[mask]
    expected = [0, 1, 4, 5, 6, 8, 9, 14, 16, 18, 19, 20, 26]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == (len(expected),)
    assert all(dpt.asnumpy(y[k]) == expected[k] for k in range(len(expected)))


def test_advanced_slice8():
    q = get_queue_or_skip()
    mask = dpt.asarray(
        [[True, False, False], [False, True, False], [False, True, False]],
        sycl_queue=q,
    )
    x = _make_3d("u2", q)
    y = x[mask]
    expected = dpt.asarray(
        [[0, 1, 2], [12, 13, 14], [21, 22, 23]], sycl_queue=q
    )
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == expected.shape
    assert (dpt.asnumpy(y) == dpt.asnumpy(expected)).all()


def test_advanced_slice9():
    q = get_queue_or_skip()
    mask = dpt.asarray(
        [[True, False, False], [False, True, False], [False, True, False]],
        sycl_queue=q,
    )
    x = _make_3d("u4", q)
    y = x[:, mask]
    expected = dpt.asarray([[0, 4, 7], [9, 13, 16], [18, 22, 25]], sycl_queue=q)
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == expected.shape
    assert (dpt.asnumpy(y) == dpt.asnumpy(expected)).all()


def lin_id(i, j, k):
    """global_linear_id for (3,3,3) range traversed in C-contiguous order"""
    return 9 * i + 3 * j + k


def test_advanced_slice10():
    q = get_queue_or_skip()
    x = _make_3d("u8", q)
    i0 = dpt.asarray([0, 1, 1], device=x.device)
    i1 = dpt.asarray([1, 1, 2], device=x.device)
    i2 = dpt.asarray([2, 0, 1], device=x.device)
    y = x[i0, i1, i2]
    res_expected = dpt.asarray(
        [
            lin_id(0, 1, 2),
            lin_id(1, 1, 0),
            lin_id(1, 2, 1),
        ],
        sycl_queue=q,
    )
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == res_expected.shape
    assert (dpt.asnumpy(y) == dpt.asnumpy(res_expected)).all()


def test_advanced_slice11():
    q = get_queue_or_skip()
    x = _make_3d("u8", q)
    i0 = dpt.asarray([0, 1, 1], device=x.device)
    i2 = dpt.asarray([2, 0, 1], device=x.device)
    with pytest.raises(IndexError):
        x[i0, :, i2]


def test_advanced_slice12():
    q = get_queue_or_skip()
    x = _make_3d("u8", q)
    i1 = dpt.asarray([1, 1, 2], device=x.device)
    i2 = dpt.asarray([2, 0, 1], device=x.device)
    y = x[:, dpt.newaxis, i1, i2, dpt.newaxis]
    res_expected = dpt.asarray(
        [
            [[[lin_id(0, 1, 2)], [lin_id(0, 1, 0)], [lin_id(0, 2, 1)]]],
            [[[lin_id(1, 1, 2)], [lin_id(1, 1, 0)], [lin_id(1, 2, 1)]]],
            [[[lin_id(2, 1, 2)], [lin_id(2, 1, 0)], [lin_id(2, 2, 1)]]],
        ],
        sycl_queue=q,
    )
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == res_expected.shape
    assert (dpt.asnumpy(y) == dpt.asnumpy(res_expected)).all()


def test_advanced_slice13():
    q = get_queue_or_skip()
    x = _make_3d("u8", q)
    i1 = dpt.asarray([[1], [2]], device=x.device)
    i2 = dpt.asarray([[0, 1]], device=x.device)
    y = x[i1, i2, 0]
    expected = dpt.asarray(
        [
            [lin_id(1, 0, 0), lin_id(1, 1, 0)],
            [lin_id(2, 0, 0), lin_id(2, 1, 0)],
        ],
        device=x.device,
    )
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == expected.shape
    assert (dpt.asnumpy(y) == dpt.asnumpy(expected)).all()


def test_integer_indexing_1d():
    get_queue_or_skip()
    x = dpt.arange(10, dtype="i4")
    ind_1d = dpt.asarray([7, 3, 1], dtype="u2")
    ind_2d = dpt.asarray([[2, 3, 4], [3, 4, 5], [5, 6, 7]], dtype="i4")

    y1 = x[ind_1d]
    assert y1.shape == ind_1d.shape
    y2 = x[ind_2d]
    assert y2.shape == ind_2d.shape
    assert (dpt.asnumpy(y1) == np.array([7, 3, 1], dtype="i4")).all()
    assert (
        dpt.asnumpy(y2)
        == np.array([[2, 3, 4], [3, 4, 5], [5, 6, 7]], dtype="i4")
    ).all()


def test_integer_indexing_2d():
    get_queue_or_skip()
    n0, n1 = 5, 7
    x = dpt.reshape(
        dpt.arange(n0 * n1, dtype="i4"),
        (
            n0,
            n1,
        ),
    )
    ind0 = dpt.arange(n0)
    ind1 = dpt.arange(n1)

    y = x[ind0[:2, dpt.newaxis], ind1[dpt.newaxis, -2:]]
    assert y.dtype == x.dtype
    assert (dpt.asnumpy(y) == np.array([[5, 6], [12, 13]])).all()


def test_integer_strided_indexing():
    get_queue_or_skip()
    n0, n1 = 5, 7
    x = dpt.reshape(
        dpt.arange(2 * n0 * n1, dtype="i4"),
        (
            2 * n0,
            n1,
        ),
    )
    ind0 = dpt.arange(n0)
    ind1 = dpt.arange(n1)

    z = x[::-2, :]
    y = z[ind0[:2, dpt.newaxis], ind1[dpt.newaxis, -2:]]
    assert y.dtype == x.dtype
    zc = dpt.copy(z, order="C")
    yc = zc[ind0[:2, dpt.newaxis], ind1[dpt.newaxis, -2:]]
    assert (dpt.asnumpy(y) == dpt.asnumpy(yc)).all()
