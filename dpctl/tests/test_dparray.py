import unittest
from dpctl.dptensor import dparray
import numpy


class TestOverloadList(unittest.TestCase):
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

    def test_dparray_mixing_dpctl_and_numpy(self):
        dp_numpy = numpy.ones((256, 4), dtype="d")
        res = dp_numpy * self.X
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

    @unittest.expectedFailure
    def test_numpy_sum_with_dparray(self):
        res = numpy.sum(self.X)
        self.assertEqual(res, 1024.0)


if __name__ == "__main__":
    unittest.main()
