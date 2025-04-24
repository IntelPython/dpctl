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

import dpctl.tensor as dpt
import dpctl.tensor._copy_utils as cu
from dpctl.tests.helper import get_queue_or_skip


def test_copy_utils_empty_like_orderK():
    get_queue_or_skip()
    a = dpt.empty((10, 10), dtype=dpt.int32, order="F")
    X = cu._empty_like_orderK(a, dpt.int32, a.usm_type, a.device)
    assert X.flags["F"]


def test_copy_utils_empty_like_orderK_invalid_args():
    get_queue_or_skip()
    with pytest.raises(TypeError):
        cu._empty_like_orderK([1, 2, 3], dpt.int32, "device", None)
    with pytest.raises(TypeError):
        cu._empty_like_pair_orderK(
            [1, 2, 3],
            (
                1,
                2,
                3,
            ),
            dpt.int32,
            (3,),
            "device",
            None,
        )

    a = dpt.empty(10, dtype=dpt.int32)
    with pytest.raises(TypeError):
        cu._empty_like_pair_orderK(
            a,
            (
                1,
                2,
                3,
            ),
            dpt.int32,
            (10,),
            "device",
            None,
        )


def test_copy_utils_from_numpy_empty_like_orderK():
    q = get_queue_or_skip()

    a = np.empty((10, 10), dtype=np.int32, order="C")
    r0 = cu._from_numpy_empty_like_orderK(a, dpt.int32, "device", q)
    assert r0.flags["C"]

    b = np.empty((10, 10), dtype=np.int32, order="F")
    r1 = cu._from_numpy_empty_like_orderK(b, dpt.int32, "device", q)
    assert r1.flags["F"]

    c = np.empty((2, 3, 4), dtype=np.int32, order="C")
    c = np.transpose(c, (1, 0, 2))
    r2 = cu._from_numpy_empty_like_orderK(c, dpt.int32, "device", q)
    assert not r2.flags["C"] and not r2.flags["F"]


def test_copy_utils_from_numpy_empty_like_orderK_invalid_args():
    with pytest.raises(TypeError):
        cu._from_numpy_empty_like_orderK([1, 2, 3], dpt.int32, "device", None)


def test_gh_2055():
    """
    Test that `dpt.asarray` works on contiguous NumPy arrays with `order="K"`
    when dimensions are permuted.

    See: https://github.com/IntelPython/dpctl/issues/2055
    """
    get_queue_or_skip()

    a = np.ones((2, 3, 4), dtype=dpt.int32)
    a_t = np.transpose(a, (2, 0, 1))
    r = dpt.asarray(a_t)
    assert not r.flags["C"] and not r.flags["F"]
