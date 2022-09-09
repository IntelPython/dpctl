#                       Data Parallel Control (dpctl)
#
#  Copyright 2020-2022 Intel Corporation
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

"""Unit test cases for dpctl.tensor.numpy_usm_shared.
"""

import numpy

from dpctl.tensor import numpy_usm_shared as dparray


def get_arg():
    X = dparray.ndarray((256, 4), dtype="d")
    X.fill(1.0)
    return X


def test_dparray_type():
    X = get_arg()
    assert isinstance(X, dparray.ndarray)


def test_dparray_as_ndarray_self():
    X = get_arg()
    Y = X.as_ndarray()
    assert type(Y) == numpy.ndarray


def test_dparray_as_ndarray():
    X = get_arg()
    Y = dparray.as_ndarray(X)
    assert type(Y) == numpy.ndarray


def test_dparray_from_ndarray():
    X = get_arg()
    Y = dparray.as_ndarray(X)
    dp1 = dparray.from_ndarray(Y)
    assert isinstance(dp1, dparray.ndarray)


def test_multiplication_dparray():
    C = get_arg() * 5
    assert isinstance(C, dparray.ndarray)


def test_inplace_sub():
    X = get_arg()
    X -= 1


def test_dparray_through_python_func():
    def func_operation_with_const(dpctl_array):
        return dpctl_array * 2.0 + 13

    C = get_arg() * 5
    dp_func = func_operation_with_const(C)
    assert isinstance(dp_func, dparray.ndarray)


def test_dparray_mixing_dpctl_and_numpy():
    dp_numpy = numpy.ones((256, 4), dtype="d")
    X = get_arg()
    res = dp_numpy * X
    assert isinstance(X, dparray.ndarray)
    assert isinstance(res, dparray.ndarray)


def test_dparray_shape():
    X = get_arg()
    res = X.shape
    assert res == (256, 4)


def test_dparray_T():
    X = get_arg()
    res = X.T
    assert res.shape == (4, 256)


def test_numpy_ravel_with_dparray():
    X = get_arg()
    res = numpy.ravel(X)
    assert res.shape == (1024,)


def test_numpy_sum_with_dparray():
    X = get_arg()
    res = numpy.sum(X)
    assert res == 1024.0


def test_numpy_sum_with_dparray_out():
    X = get_arg()
    res = dparray.empty((X.shape[1],), dtype=X.dtype)
    res2 = numpy.sum(X, axis=0, out=res)
    assert res is res2
    assert isinstance(res2, dparray.ndarray)


def test_frexp_with_out():
    X = dparray.array([0.5, 4.7])
    mant = dparray.empty((2,), dtype="d")
    exp = dparray.empty((2,), dtype="i4")
    res = numpy.frexp(X, out=(mant, exp))
    assert res[0] is mant
    assert res[1] is exp
