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

import pytest

import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip


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
