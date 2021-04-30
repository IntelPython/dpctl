#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2021 Intel Corporation
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

"""Defines unit test cases for the Memory classes in _memory.pyx.
"""

import unittest

import numpy as np

import dpctl
from dpctl.memory import MemoryUSMDevice, MemoryUSMHost, MemoryUSMShared

from ._helper import has_cpu, has_gpu, has_sycl_platforms


class Dummy(MemoryUSMShared):
    """
    Class that exposes ``__sycl_usm_array_interface__`` with
    SYCL context for sycl object, instead of Sycl queue.
    """

    @property
    def __sycl_usm_array_interface(self):
        iface = super().__sycl_usm_array_interface__
        iface["syclob"] = iface["syclobj"].get_sycl_context()
        return iface


class TestMemory(unittest.TestCase):
    @unittest.skipUnless(
        has_sycl_platforms(), "No SYCL devices except the default host device."
    )
    def test_memory_create(self):
        nbytes = 1024
        queue = dpctl.get_current_queue()
        mobj = MemoryUSMShared(nbytes, alignment=64, queue=queue)
        self.assertEqual(mobj.nbytes, nbytes)
        self.assertTrue(hasattr(mobj, "__sycl_usm_array_interface__"))

    @unittest.skipUnless(
        has_sycl_platforms(), "No SYCL devices except the default host device."
    )
    def test_memory_create_with_np(self):
        nbytes = 16384
        mobj = dpctl.memory.MemoryUSMShared(np.int64(nbytes))
        self.assertEqual(mobj.nbytes, nbytes)
        self.assertTrue(hasattr(mobj, "__sycl_usm_array_interface__"))

    def _create_memory(self):
        nbytes = 1024
        queue = dpctl.get_current_queue()
        mobj = MemoryUSMShared(nbytes, alignment=64, queue=queue)
        return mobj

    def _create_host_buf(self, nbytes):
        ba = bytearray(nbytes)
        for i in range(nbytes):
            ba[i] = (i % 32) + ord("a")
        return ba

    @unittest.skipUnless(
        has_sycl_platforms(), "No SYCL devices except the default host device."
    )
    def test_memory_without_context(self):
        mobj = self._create_memory()

        # Without context
        self.assertEqual(mobj.get_usm_type(), "shared")

    @unittest.skipUnless(has_cpu(), "No OpenCL CPU queues available")
    def test_memory_cpu_context(self):
        mobj = self._create_memory()

        # CPU context
        with dpctl.device_context("opencl:cpu:0"):
            # type respective to the context in which
            # memory was created
            usm_type = mobj.get_usm_type()
            self.assertEqual(usm_type, "shared")

            current_queue = dpctl.get_current_queue()
            # type as view from current queue
            usm_type = mobj.get_usm_type(current_queue)
            # type can be unknown if current queue is
            # not in the same SYCL context
            self.assertTrue(usm_type in ["unknown", "shared"])

    @unittest.skipUnless(has_gpu(), "No OpenCL GPU queues available")
    def test_memory_gpu_context(self):
        mobj = self._create_memory()

        # GPU context
        with dpctl.device_context("opencl:gpu:0"):
            usm_type = mobj.get_usm_type()
            self.assertEqual(usm_type, "shared")
            current_queue = dpctl.get_current_queue()
            usm_type = mobj.get_usm_type(current_queue)
            self.assertTrue(usm_type in ["unknown", "shared"])

    @unittest.skipUnless(
        has_sycl_platforms(), "No SYCL devices except the default host device."
    )
    def test_buffer_protocol(self):
        mobj = self._create_memory()
        mv1 = memoryview(mobj)
        mv2 = memoryview(mobj)
        self.assertEqual(mv1, mv2)

    @unittest.skipUnless(
        has_sycl_platforms(), "No SYCL devices except the default host device."
    )
    def test_copy_host_roundtrip(self):
        mobj = self._create_memory()
        host_src_obj = self._create_host_buf(mobj.nbytes)
        mobj.copy_from_host(host_src_obj)
        host_dest_obj = mobj.copy_to_host()
        del mobj
        self.assertEqual(host_src_obj, host_dest_obj)

    @unittest.skipUnless(
        has_sycl_platforms(), "No SYCL devices except the default host device."
    )
    def test_zero_copy(self):
        mobj = self._create_memory()
        mobj2 = type(mobj)(mobj)

        self.assertTrue(mobj2.reference_obj is mobj)
        mobj_data = mobj.__sycl_usm_array_interface__["data"]
        mobj2_data = mobj2.__sycl_usm_array_interface__["data"]
        self.assertEqual(mobj_data, mobj2_data)

    @unittest.skipUnless(
        has_sycl_platforms(), "No SYCL devices except the default host device."
    )
    def test_pickling(self):
        import pickle

        mobj = self._create_memory()
        host_src_obj = self._create_host_buf(mobj.nbytes)
        mobj.copy_from_host(host_src_obj)

        mobj_reconstructed = pickle.loads(pickle.dumps(mobj))
        self.assertEqual(
            type(mobj),
            type(mobj_reconstructed),
            "Pickling should preserve type",
        )
        self.assertEqual(
            mobj.tobytes(),
            mobj_reconstructed.tobytes(),
            "Pickling should preserve buffer content",
        )
        self.assertNotEqual(
            mobj._pointer,
            mobj_reconstructed._pointer,
            "Pickling/unpickling changes pointer",
        )


class _TestMemoryUSMBase:
    """Base tests for MemoryUSM*"""

    def setUp(self):
        pass

    @unittest.skipUnless(
        has_sycl_platforms(), "No SYCL devices except the default host device."
    )
    def test_create_with_size_and_alignment_and_queue(self):
        q = dpctl.get_current_queue()
        m = self.MemoryUSMClass(1024, alignment=64, queue=q)
        self.assertEqual(m.nbytes, 1024)
        self.assertEqual(m.get_usm_type(), self.usm_type)

    @unittest.skipUnless(
        has_sycl_platforms(), "No SYCL devices except the default host device."
    )
    def test_create_with_size_and_queue(self):
        q = dpctl.get_current_queue()
        m = self.MemoryUSMClass(1024, queue=q)
        self.assertEqual(m.nbytes, 1024)
        self.assertEqual(m.get_usm_type(), self.usm_type)

    @unittest.skipUnless(
        has_sycl_platforms(), "No SYCL devices except the default host device."
    )
    def test_create_with_size_and_alignment(self):
        m = self.MemoryUSMClass(1024, alignment=64)
        self.assertEqual(m.nbytes, 1024)
        self.assertEqual(m.get_usm_type(), self.usm_type)

    @unittest.skipUnless(
        has_sycl_platforms(), "No SYCL devices except the default host device."
    )
    def test_create_with_only_size(self):
        m = self.MemoryUSMClass(1024)
        self.assertEqual(m.nbytes, 1024)
        self.assertEqual(m.get_usm_type(), self.usm_type)

    @unittest.skipUnless(
        has_sycl_platforms(), "No SYCL Devices except the default host device."
    )
    def test_sycl_usm_array_interface(self):
        m = self.MemoryUSMClass(256)
        m2 = Dummy(m.nbytes)
        hb = np.random.randint(0, 256, size=256, dtype="|u1")
        m2.copy_from_host(hb)
        # test that USM array interface works with SyclContext as 'syclobj'
        m.copy_from_device(m2)
        self.assertTrue(np.array_equal(m.copy_to_host(), hb))


class TestMemoryUSMShared(_TestMemoryUSMBase, unittest.TestCase):
    """Tests for MemoryUSMShared"""

    def setUp(self):
        self.MemoryUSMClass = MemoryUSMShared
        self.usm_type = "shared"


class TestMemoryUSMHost(_TestMemoryUSMBase, unittest.TestCase):
    """Tests for MemoryUSMHost"""

    def setUp(self):
        self.MemoryUSMClass = MemoryUSMHost
        self.usm_type = "host"


class TestMemoryUSMDevice(_TestMemoryUSMBase, unittest.TestCase):
    """Tests for MemoryUSMDevice"""

    def setUp(self):
        self.MemoryUSMClass = MemoryUSMDevice
        self.usm_type = "device"


class View:
    def __init__(self, buf, shape, strides, offset):
        self.buffer = buf
        self.shape = shape
        self.strides = strides
        self.offset = offset

    @property
    def __sycl_usm_array_interface__(self):
        sua_iface = self.buffer.__sycl_usm_array_interface__
        sua_iface["offset"] = self.offset
        sua_iface["shape"] = self.shape
        sua_iface["strides"] = self.strides
        return sua_iface


class TestMemoryWithView(unittest.TestCase):
    def test_suai_non_contig_1D(self):
        """
        Test of zero-copy using sycl_usm_array_interface with non-contiguous
        data.
        """

        MemoryUSMClass = MemoryUSMShared
        try:
            buf = MemoryUSMClass(32)
        except:
            self.skipTest("MemoryUSMShared could not be allocated")
        host_canary = np.full((buf.nbytes,), 77, dtype="|u1")
        buf.copy_from_host(host_canary)
        n1d = 10
        step_1d = 2
        offset = 8
        v = View(buf, shape=(n1d,), strides=(step_1d,), offset=offset)
        buf2 = MemoryUSMClass(v)
        expected_nbytes = (
            np.flip(
                host_canary[offset : offset + n1d * step_1d : step_1d]
            ).ctypes.data
            + 1
            - host_canary[offset:].ctypes.data
        )
        self.assertEqual(buf2.nbytes, expected_nbytes)
        inset_canary = np.arange(0, buf2.nbytes, dtype="|u1")
        buf2.copy_from_host(inset_canary)
        res = buf.copy_to_host()
        del buf
        del buf2
        expected_res = host_canary.copy()
        expected_res[offset : offset + (n1d - 1) * step_1d + 1] = inset_canary
        self.assertTrue(np.array_equal(res, expected_res))

    def test_suai_non_contig_2D(self):
        MemoryUSMClass = MemoryUSMDevice
        try:
            buf = MemoryUSMClass(20)
        except:
            self.skipTest("MemoryUSMShared could not be allocated")
        host_canary = np.arange(20, dtype="|u1")
        buf.copy_from_host(host_canary)
        shape_2d = (2, 2)
        strides_2d = (10, -2)
        offset = 9
        idx = []
        for i0 in range(shape_2d[0]):
            for i1 in range(shape_2d[1]):
                idx.append(offset + i0 * strides_2d[0] + i1 * strides_2d[1])
        idx.sort()
        v = View(buf, shape=shape_2d, strides=strides_2d, offset=offset)
        buf2 = MemoryUSMClass(v)
        expected_nbytes = idx[-1] - idx[0] + 1
        self.assertEqual(buf2.nbytes, expected_nbytes)
        inset_canary = np.full((buf2.nbytes), 255, dtype="|u1")
        buf2.copy_from_host(inset_canary)
        res = buf.copy_to_host()
        del buf
        del buf2
        expected_res = host_canary.copy()
        expected_res[idx[0] : idx[-1] + 1] = inset_canary
        self.assertTrue(np.array_equal(res, expected_res))


if __name__ == "__main__":
    unittest.main()
