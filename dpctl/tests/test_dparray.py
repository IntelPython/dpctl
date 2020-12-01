import unittest
from dpctl import dparray
import numpy


def func_operation_with_const(dpctl_array):
    return dpctl_array * 2.0 + 13


def multiply_func(np_array, dpcrtl_array):
    return np_array * dpcrtl_array


class TestOverloadList(unittest.TestCase):
    maxDiff = None

    X = dparray.ndarray((256, 4), dtype='d')
    X.fill(1.0)

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

    def test_dparray_through_python_func(self):
        C = self.X * 5
        dp_func = func_operation_with_const(C)
        self.assertIsInstance(dp_func, dparray.ndarray)

    def test_dparray_mixing_dpctl_and_numpy(self):
        dp_numpy = numpy.ones((256, 4), dtype='d')
        res = multiply_func(dp_numpy, self.X)
        self.assertIsInstance(res, dparray.ndarray)

    def test_dparray_shape(self):
        res = self.X.shape
        self.assertEqual(res, (256, 4))

    def test_dparray_T(self):
        res = self.X.T
        self.assertEqual(res.shape, (4, 256))


if __name__ == '__main__':
    unittest.main()
