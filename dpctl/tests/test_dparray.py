#                       Data Parallel Control (dpctl)
#
#  Copyright 2020-2021 Intel Corporation
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

import unittest

import numpy

from dpctl.tensor import numpy_usm_shared as dparray


class Test_dparray(unittest.TestCase):
    def setUp(self):
        self.X = dparray.ndarray((256, 4), dtype="d")
        self.X.fill(1.0)

    def test_dparray_type(self):
        self.assertIsInstance(self.X, dparray.ndarray)

    def test_dparray_as_ndarray_self(self):
        Y = self.X.as_ndarray()
        self.assertEqual(type(Y), numpy.ndarray)

    def test_dparray_as_ndarray(self):
        Y = dparray.as_ndarray(self.X)
        self.assertEqual(type(Y), numpy.ndarray)

    def test_dparray_from_ndarray(self):
        Y = dparray.as_ndarray(self.X)
        dp1 = dparray.from_ndarray(Y)
        self.assertIsInstance(dp1, dparray.ndarray)

    def test_multiplication_dparray(self):
        C = self.X * 5
        self.assertIsInstance(C, dparray.ndarray)

    def test_inplace_sub(self):
        self.X -= 1

    def test_dparray_through_python_func(self):
        def func_operation_with_const(dpctl_array):
            return dpctl_array * 2.0 + 13

        C = self.X * 5
        dp_func = func_operation_with_const(C)
        self.assertIsInstance(dp_func, dparray.ndarray)

    def test_dparray_mixing_dpctl_and_numpy(self):
        dp_numpy = numpy.ones((256, 4), dtype="d")
        res = dp_numpy * self.X
        self.assertIsInstance(self.X, dparray.ndarray)
        self.assertIsInstance(res, dparray.ndarray)

    def test_dparray_shape(self):
        res = self.X.shape
        self.assertEqual(res, (256, 4))

    def test_dparray_T(self):
        res = self.X.T
        self.assertEqual(res.shape, (4, 256))

    def test_numpy_ravel_with_dparray(self):
        res = numpy.ravel(self.X)
        self.assertEqual(res.shape, (1024,))

    def test_numpy_sum_with_dparray(self):
        res = numpy.sum(self.X)
        self.assertEqual(res, 1024.0)

    def test_numpy_sum_with_dparray_out(self):
        res = dparray.empty((self.X.shape[1],), dtype=self.X.dtype)
        res2 = numpy.sum(self.X, axis=0, out=res)
        self.assertTrue(res is res2)
        self.assertIsInstance(res2, dparray.ndarray)

    def test_frexp_with_out(self):
        X = dparray.array([0.5, 4.7])
        mant = dparray.empty((2,), dtype="d")
        exp = dparray.empty((2,), dtype="i4")
        res = numpy.frexp(X, out=(mant, exp))
        self.assertTrue(res[0] is mant)
        self.assertTrue(res[1] is exp)


if __name__ == "__main__":
    unittest.main()
