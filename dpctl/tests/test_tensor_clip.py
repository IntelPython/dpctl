#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2023 Intel Corporation
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
from helper import get_queue_or_skip, skip_if_dtype_not_supported
from numpy.testing import assert_raises_regex

import dpctl
import dpctl.tensor as dpt
from dpctl.tensor._type_utils import _can_cast
from dpctl.utils import ExecutionPlacementError

_all_dtypes = [
    "?",
    "u1",
    "i1",
    "u2",
    "i2",
    "u4",
    "i4",
    "u8",
    "i8",
    "e",
    "f",
    "d",
    "F",
    "D",
]

_usm_types = ["device", "shared", "host"]


@pytest.mark.parametrize("dt1", _all_dtypes)
@pytest.mark.parametrize("dt2", _all_dtypes)
def test_clip_dtypes(dt1, dt2):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt1, q)
    skip_if_dtype_not_supported(dt2, q)

    sz = 127
    ar1 = dpt.ones(sz, dtype=dt1, sycl_queue=q)
    ar2 = dpt.ones_like(ar1, dtype=dt1, sycl_queue=q)
    ar3 = dpt.ones_like(ar1, dtype=dt2, sycl_queue=q)

    dev = q.sycl_device
    _fp16 = dev.has_aspect_fp16
    _fp64 = dev.has_aspect_fp64
    # also covers cases where dt1 == dt2
    if _can_cast(ar3.dtype, ar1.dtype, _fp16, _fp64):
        r = dpt.clip(ar1, ar2, ar3)
        assert isinstance(r, dpt.usm_ndarray)
        assert r.dtype == ar1.dtype
        assert r.shape == ar1.shape
        assert dpt.all(r == ar1)
        assert r.sycl_queue == ar1.sycl_queue

        r = dpt.clip(ar1, min=ar3, max=None)
        assert isinstance(r, dpt.usm_ndarray)
        assert r.dtype == ar1.dtype
        assert r.shape == ar1.shape
        assert dpt.all(r == ar1)
        assert r.sycl_queue == ar1.sycl_queue

        r = dpt.clip(ar1, min=None, max=ar3)
        assert isinstance(r, dpt.usm_ndarray)
        assert r.dtype == ar1.dtype
        assert r.shape == ar1.shape
        assert dpt.all(r == ar1)
        assert r.sycl_queue == ar1.sycl_queue
    else:
        with pytest.raises(TypeError):
            dpt.clip(ar1, ar2, ar3)
        with pytest.raises(TypeError):
            dpt.clip(ar1, min=ar3, max=None)
        with pytest.raises(TypeError):
            dpt.clip(ar1, min=None, max=ar3)


def test_clip_empty():
    get_queue_or_skip()

    x = dpt.empty((2, 0, 3), dtype="i4")
    a_min = dpt.ones((2, 0, 3), dtype="i4")
    a_max = dpt.ones((2, 0, 3), dtype="i4")

    r = dpt.clip(x, a_min, a_max)
    assert r.size == 0
    assert r.shape == x.shape


def test_clip_python_scalars():
    get_queue_or_skip()

    arrs = [
        dpt.ones(1, dtype="?"),
        dpt.ones(1, dtype="i4"),
        dpt.ones(1, dtype="f4"),
        dpt.ones(1, dtype="c8"),
    ]

    py_zeros = [
        False,
        0,
        0.0,
        complex(0, 0),
    ]

    py_ones = [
        True,
        1,
        1.0,
        complex(1, 0),
    ]

    for zero, one, arr in zip(py_zeros, py_ones, arrs):
        r = dpt.clip(arr, zero, one)
        assert isinstance(r, dpt.usm_ndarray)


def test_clip_in_place():
    get_queue_or_skip()

    x = dpt.arange(10, dtype="i4")
    a_min = dpt.arange(1, 11, dtype="i4")
    a_max = dpt.arange(2, 12, dtype="i4")
    dpt.clip(x, a_min, a_max, out=x)
    assert dpt.all(x == a_min)

    x = dpt.arange(10, dtype="i4")
    dpt.clip(x, min=a_min, max=None, out=x)
    assert dpt.all(x == a_min)

    x = dpt.arange(10, dtype="i4")
    dpt.clip(x, a_min, a_max, out=a_max)
    assert dpt.all(a_max == a_min)

    a_min = dpt.arange(1, 11, dtype="i4")
    dpt.clip(x, min=a_min, max=None, out=a_min[::-1])
    assert dpt.all((x + 1)[::-1] == a_min)


def test_clip_special_cases():
    get_queue_or_skip()

    x = dpt.arange(10, dtype="f4")
    r = dpt.clip(x, -dpt.inf, dpt.inf)
    assert dpt.all(r == x)
    r = dpt.clip(x, dpt.nan, dpt.inf)
    assert dpt.all(dpt.isnan(r))
    r = dpt.clip(x, -dpt.inf, dpt.nan)
    assert dpt.all(dpt.isnan(r))


def test_clip_out_need_temporary():
    get_queue_or_skip()

    x = dpt.ones(10, dtype="i4")
    dpt.clip(x[:6], 2, 3, out=x[-6:])
    assert dpt.all(x[:-6] == 1) and dpt.all(x[-6:] == 2)

    x = dpt.full(6, 3, dtype="i4")
    a_min = dpt.full(10, 2, dtype="i4")
    dpt.clip(x, min=a_min[:6], max=4, out=a_min[-6:])
    assert dpt.all(a_min[:-6] == 2) and dpt.all(a_min[-6:] == 3)

    # with min/max == None
    a_min = dpt.full(10, 2, dtype="i4")
    dpt.clip(x, min=a_min[:6], max=None, out=a_min[-6:])
    assert dpt.all(a_min[:-6] == 2) and dpt.all(a_min[-6:] == 3)


def test_where_arg_validation():
    get_queue_or_skip()

    check = dict()
    x1 = dpt.empty((1,), dtype="i4")
    x2 = dpt.empty((1,), dtype="i4")

    with pytest.raises(TypeError):
        dpt.where(check, x1, x2)
    with pytest.raises(TypeError):
        dpt.where(x1, check, x2)
    with pytest.raises(TypeError):
        dpt.where(x1, x2, check)


def test_clip_order():
    get_queue_or_skip()

    test_shape = (
        20,
        20,
    )
    test_shape2 = tuple(2 * dim for dim in test_shape)
    n = test_shape[-1]

    ar1 = dpt.ones(test_shape, dtype="i4", order="C")
    ar2 = dpt.ones(test_shape, dtype="i4", order="C")
    ar3 = dpt.ones(test_shape, dtype="i4", order="C")
    r1 = dpt.clip(ar1, ar2, ar3, order="C")
    assert r1.flags.c_contiguous
    r2 = dpt.clip(ar1, ar2, ar3, order="F")
    assert r2.flags.f_contiguous
    r3 = dpt.clip(ar1, ar2, ar3, order="A")
    assert r3.flags.c_contiguous
    r4 = dpt.clip(ar1, ar2, ar3, order="K")
    assert r4.flags.c_contiguous

    r1 = dpt.clip(ar1, min=None, max=ar2, order="C")
    assert r1.flags.c_contiguous
    r2 = dpt.clip(ar1, min=None, max=ar2, order="F")
    assert r2.flags.f_contiguous
    r3 = dpt.clip(ar1, min=None, max=ar2, order="A")
    assert r3.flags.c_contiguous
    r4 = dpt.clip(ar1, min=None, max=ar2, order="K")
    assert r4.flags.c_contiguous

    ar1 = dpt.ones(test_shape, dtype="i4", order="F")
    ar2 = dpt.ones(test_shape, dtype="i4", order="F")
    ar3 = dpt.ones(test_shape, dtype="i4", order="F")
    r1 = dpt.clip(ar1, ar2, ar3, order="C")
    assert r1.flags.c_contiguous
    r2 = dpt.clip(ar1, ar2, ar3, order="F")
    assert r2.flags.f_contiguous
    r3 = dpt.clip(ar1, ar2, ar3, order="A")
    assert r3.flags.f_contiguous
    r4 = dpt.clip(ar1, ar2, ar3, order="K")
    assert r4.flags.f_contiguous

    r1 = dpt.clip(ar1, min=None, max=ar2, order="C")
    assert r1.flags.c_contiguous
    r2 = dpt.clip(ar1, min=None, max=ar2, order="F")
    assert r2.flags.f_contiguous
    r3 = dpt.clip(ar1, min=None, max=ar2, order="A")
    assert r3.flags.f_contiguous
    r4 = dpt.clip(ar1, min=None, max=ar2, order="K")
    assert r4.flags.f_contiguous

    ar1 = dpt.ones(test_shape2, dtype="i4", order="C")[:20, ::-2]
    ar2 = dpt.ones(test_shape2, dtype="i4", order="C")[:20, ::-2]
    ar3 = dpt.ones(test_shape2, dtype="i4", order="C")[:20, ::-2]
    r4 = dpt.clip(ar1, ar2, ar3, order="K")
    assert r4.strides == (n, -1)
    r5 = dpt.clip(ar1, ar2, ar3, order="C")
    assert r5.strides == (n, 1)

    r4 = dpt.clip(ar1, min=None, max=ar3, order="K")
    assert r4.strides == (n, -1)
    r5 = dpt.clip(ar1, min=None, max=ar3, order="C")
    assert r5.strides == (n, 1)

    ar1 = dpt.ones(test_shape2, dtype="i4", order="C")[:20, ::-2].mT
    ar2 = dpt.ones(test_shape2, dtype="i4", order="C")[:20, ::-2].mT
    ar3 = dpt.ones(test_shape2, dtype="i4", order="C")[:20, ::-2].mT
    r4 = dpt.clip(ar1, ar2, ar3, order="K")
    assert r4.strides == (-1, n)
    r5 = dpt.clip(ar1, ar2, ar3, order="C")
    assert r5.strides == (n, 1)

    r4 = dpt.clip(ar1, min=None, max=ar3, order="K")
    assert r4.strides == (-1, n)
    r5 = dpt.clip(ar1, min=None, max=ar3, order="C")
    assert r5.strides == (n, 1)


@pytest.mark.parametrize("usm_type1", _usm_types)
@pytest.mark.parametrize("usm_type2", _usm_types)
@pytest.mark.parametrize("usm_type3", _usm_types)
def test_clip_usm_type_matrix(usm_type1, usm_type2, usm_type3):
    get_queue_or_skip()

    sz = 128
    ar1 = dpt.ones(sz, dtype="i4", usm_type=usm_type1)
    ar2 = dpt.ones_like(ar1, dtype="i4", usm_type=usm_type2)
    ar3 = dpt.ones_like(ar1, dtype="i4", usm_type=usm_type3)

    r = dpt.clip(ar1, ar2, ar3)
    assert isinstance(r, dpt.usm_ndarray)
    expected_usm_type = dpctl.utils.get_coerced_usm_type(
        (usm_type1, usm_type2, usm_type3)
    )
    assert r.usm_type == expected_usm_type


@pytest.mark.parametrize("usm_type1", _usm_types)
@pytest.mark.parametrize("usm_type2", _usm_types)
def test_clip_usm_type_matrix_none_arg(usm_type1, usm_type2):
    get_queue_or_skip()

    sz = 128
    ar1 = dpt.ones(sz, dtype="i4", usm_type=usm_type1)
    ar2 = dpt.ones_like(ar1, dtype="i4", usm_type=usm_type2)

    r = dpt.clip(ar1, min=ar2, max=None)
    assert isinstance(r, dpt.usm_ndarray)
    expected_usm_type = dpctl.utils.get_coerced_usm_type((usm_type1, usm_type2))
    assert r.usm_type == expected_usm_type


def test_clip_dtype_error():
    get_queue_or_skip()

    ar1 = dpt.ones(1, dtype="i4")
    ar2 = dpt.ones(1, dtype="i4")
    ar3 = dpt.ones(1, dtype="i4")
    ar4 = dpt.empty_like(ar1, dtype="f4")

    assert_raises_regex(
        TypeError,
        "Output array of type.*is needed",
        dpt.clip,
        ar1,
        ar2,
        ar3,
        ar4,
    )
    assert_raises_regex(
        TypeError,
        "Output array of type.*is needed",
        dpt.clip,
        ar1,
        ar2,
        None,
        ar4,
    )


def test_clip_errors():
    get_queue_or_skip()
    try:
        gpu_queue = dpctl.SyclQueue("gpu")
    except dpctl.SyclQueueCreationError:
        pytest.skip("SyclQueue('gpu') failed, skipping")
    try:
        cpu_queue = dpctl.SyclQueue("cpu")
    except dpctl.SyclQueueCreationError:
        pytest.skip("SyclQueue('cpu') failed, skipping")

    ar1 = dpt.ones(2, dtype="float32", sycl_queue=gpu_queue)
    ar2 = dpt.ones_like(ar1, sycl_queue=gpu_queue)
    ar3 = dpt.ones_like(ar1, sycl_queue=gpu_queue)
    ar4 = dpt.empty_like(ar1, sycl_queue=cpu_queue)
    assert_raises_regex(
        ExecutionPlacementError,
        "Input and output allocation queues are not compatible",
        dpt.clip,
        ar1,
        ar2,
        ar3,
        ar4,
    )

    assert_raises_regex(
        ExecutionPlacementError,
        "Input and output allocation queues are not compatible",
        dpt.clip,
        ar1,
        None,
        ar3,
        ar4,
    )

    ar1 = dpt.ones(2, dtype="float32")
    ar2 = dpt.ones_like(ar1, dtype="float32")
    ar3 = dpt.ones_like(ar1, dtype="float32")
    ar4 = dpt.empty(3, dtype="float32")
    assert_raises_regex(
        ValueError,
        "The shape of input and output arrays are inconsistent",
        dpt.clip,
        ar1,
        ar2,
        ar3,
        ar4,
    )

    assert_raises_regex(
        ValueError,
        "The shape of input and output arrays are inconsistent",
        dpt.clip,
        ar1,
        ar2,
        None,
        ar4,
    )

    ar1 = np.ones(2, dtype="f4")
    ar2 = dpt.ones(2, dtype="f4")
    ar3 = dpt.ones(2, dtype="f4")
    assert_raises_regex(
        TypeError,
        "Expected `x` to be of dpctl.tensor.usm_ndarray type*",
        dpt.clip,
        ar1,
        ar2,
        ar3,
    )

    ar1 = dpt.ones(2, dtype="i4")
    ar2 = dpt.ones_like(ar1, dtype="i4")
    ar3 = dpt.ones_like(ar1, dtype="i4")
    ar4 = np.empty_like(ar1)
    assert_raises_regex(
        TypeError,
        "output array must be of usm_ndarray type",
        dpt.clip,
        ar1,
        ar2,
        ar3,
        ar4,
    )

    assert_raises_regex(
        TypeError,
        "output array must be of usm_ndarray type",
        dpt.clip,
        ar1,
        ar2,
        None,
        ar4,
    )


def test_clip_out_type_check():
    get_queue_or_skip()

    x1 = dpt.ones(10)
    x2 = dpt.ones(10)
    x3 = dpt.ones(10)

    out = range(10)

    with pytest.raises(TypeError):
        dpt.add(x1, x2, x3, out=out)


@pytest.mark.parametrize("dt", ["i4", "f4", "c8"])
def test_clip_basic(dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    sz = 1026
    x = dpt.arange(sz, dtype=dt, sycl_queue=q)
    r = dpt.clip(x, min=100, max=500)
    expected = dpt.arange(sz, dtype=dt, sycl_queue=q)
    expected[:100] = 100
    expected[500:] = 500
    assert dpt.all(expected == r)

    x = dpt.zeros(sz, dtype=dt, sycl_queue=q)
    a_max = dpt.full(sz, -1, dtype=dt, sycl_queue=q)
    a_max[::2] = -2
    r = dpt.clip(x, min=-3, max=a_max)
    assert dpt.all(a_max == r)


@pytest.mark.parametrize("dt", ["i4", "f4", "c8"])
def test_clip_strided(dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    sz = 2 * 1026
    x = dpt.arange(sz, dtype=dt, sycl_queue=q)[::-2]
    r = dpt.clip(x, min=100, max=500)
    expected = dpt.arange(sz, dtype=dt, sycl_queue=q)
    expected[:100] = 100
    expected[500:] = 500
    expected = expected[::-2]
    assert dpt.all(expected == r)

    x = dpt.zeros(sz, dtype=dt, sycl_queue=q)[::-2]
    a_max = dpt.full(sz, -1, dtype=dt, sycl_queue=q)
    a_max[::2] = -2
    a_max = a_max[::-2]
    r = dpt.clip(x, min=-3, max=a_max)
    assert dpt.all(a_max == r)
