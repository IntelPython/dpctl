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
