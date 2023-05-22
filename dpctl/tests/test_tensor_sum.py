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

import numpy as np
import pytest

import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

_all_dtypes = [
    "?",
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
_usm_types = ["device", "shared", "host"]


@pytest.mark.parametrize("arg_dtype", _all_dtypes)
def test_sum_arg_dtype_default_output_dtype_matrix(arg_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arg_dtype, q)

    m = dpt.ones(100, dtype=arg_dtype)
    r = dpt.sum(m)

    assert isinstance(r, dpt.usm_ndarray)
    if m.dtype.kind == "i":
        assert r.dtype.kind == "i"
    elif m.dtype.kind == "u":
        assert r.dtype.kind == "u"
    elif m.dtype.kind == "f":
        assert r.dtype.kind == "f"
    elif m.dtype.kind == "c":
        assert r.dtype.kind == "c"
    assert (dpt.asnumpy(r) == 100).all()

    m = dpt.ones(200, dtype=arg_dtype)[:1:-2]
    r = dpt.sum(m)
    assert (dpt.asnumpy(r) == 99).all()


@pytest.mark.parametrize("arg_dtype", _all_dtypes)
@pytest.mark.parametrize("out_dtype", _all_dtypes[1:])
def test_sum_arg_out_dtype_matrix(arg_dtype, out_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arg_dtype, q)
    skip_if_dtype_not_supported(out_dtype, q)

    m = dpt.ones(100, dtype=arg_dtype)
    r = dpt.sum(m, dtype=out_dtype)

    assert isinstance(r, dpt.usm_ndarray)
    assert r.dtype == dpt.dtype(out_dtype)
    assert (dpt.asnumpy(r) == 100).all()


def test_sum_axis():
    get_queue_or_skip()

    m = dpt.ones((3, 4, 5, 6, 7), dtype="i4")
    s = dpt.sum(m, axis=(1, 2, -1))

    assert isinstance(s, dpt.usm_ndarray)
    assert s.shape == (3, 6)
    assert (dpt.asnumpy(s) == np.full(s.shape, 4 * 5 * 7)).all()


def test_sum_keepdims():
    get_queue_or_skip()

    m = dpt.ones((3, 4, 5, 6, 7), dtype="i4")
    s = dpt.sum(m, axis=(1, 2, -1), keepdims=True)

    assert isinstance(s, dpt.usm_ndarray)
    assert s.shape == (3, 1, 1, 6, 1)
    assert (dpt.asnumpy(s) == np.full(s.shape, 4 * 5 * 7)).all()
