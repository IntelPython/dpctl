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


# import numpy as np
# import pytest
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
    assert all(
        dpt.asnumpy(x[ii[k]]) == dpt.asnumpy(y[k]) for k in range(ii.shape[0])
    )
