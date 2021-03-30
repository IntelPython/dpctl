#                      Data Parallel Control (dpCtl)
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

"""Defines unit test cases for the SyclQueueManager class.
"""

import dpctl
import unittest
from ._helper import has_cpu, has_gpu, has_sycl_platforms


@unittest.skipIf(not has_sycl_platforms(), "No SYCL platforms available")
class TestIsInDeviceContext(unittest.TestCase):
    def test_is_in_device_context_outside_device_ctxt(self):
        self.assertFalse(dpctl.is_in_device_context())

    @unittest.skipUnless(has_gpu(), "No OpenCL GPU queues available")
    def test_is_in_device_context_inside_device_ctxt(self):
        with dpctl.device_context("opencl:gpu:0"):
            self.assertTrue(dpctl.is_in_device_context())

    @unittest.skipUnless(has_gpu(), "No OpenCL GPU queues available")
    @unittest.skipUnless(has_cpu(), "No OpenCL CPU queues available")
    def test_is_in_device_context_inside_nested_device_ctxt(self):
        with dpctl.device_context("opencl:cpu:0"):
            with dpctl.device_context("opencl:gpu:0"):
                self.assertTrue(dpctl.is_in_device_context())
            self.assertTrue(dpctl.is_in_device_context())
        self.assertFalse(dpctl.is_in_device_context())


@unittest.skipIf(not has_sycl_platforms(), "No SYCL platforms available")
class TestGetCurrentDevice(unittest.TestCase):
    def test_get_current_device_type_outside_device_ctxt(self):
        self.assertNotEqual(dpctl.get_current_device_type(), None)

    def test_get_current_device_type_inside_device_ctxt(self):
        self.assertNotEqual(dpctl.get_current_device_type(), None)

        with dpctl.device_context("opencl:gpu:0"):
            self.assertEqual(dpctl.get_current_device_type(), dpctl.device_type.gpu)

        self.assertNotEqual(dpctl.get_current_device_type(), None)

    @unittest.skipUnless(has_cpu(), "No OpenCL CPU queues available")
    def test_get_current_device_type_inside_nested_device_ctxt(self):
        self.assertNotEqual(dpctl.get_current_device_type(), None)

        with dpctl.device_context("opencl:cpu:0"):
            self.assertEqual(dpctl.get_current_device_type(), dpctl.device_type.cpu)

            with dpctl.device_context("opencl:gpu:0"):
                self.assertEqual(dpctl.get_current_device_type(), dpctl.device_type.gpu)
            self.assertEqual(dpctl.get_current_device_type(), dpctl.device_type.cpu)

        self.assertNotEqual(dpctl.get_current_device_type(), None)


@unittest.skipIf(not has_sycl_platforms(), "No SYCL platforms available")
class TestGetCurrentQueueInMultipleThreads(unittest.TestCase):
    def test_num_current_queues_outside_with_clause(self):
        self.assertEqual(dpctl.get_num_activated_queues(), 0)

    @unittest.skipUnless(has_gpu(), "No OpenCL GPU queues available")
    @unittest.skipUnless(has_cpu(), "No OpenCL CPU queues available")
    def test_num_current_queues_inside_with_clause(self):
        with dpctl.device_context("opencl:cpu:0"):
            self.assertEqual(dpctl.get_num_activated_queues(), 1)
            with dpctl.device_context("opencl:gpu:0"):
                self.assertEqual(dpctl.get_num_activated_queues(), 2)
        self.assertEqual(dpctl.get_num_activated_queues(), 0)

    @unittest.skipUnless(has_gpu(), "No OpenCL GPU queues available")
    @unittest.skipUnless(has_cpu(), "No OpenCL CPU queues available")
    def test_num_current_queues_inside_threads(self):
        from threading import Thread

        def SessionThread(self):
            self.assertEqual(dpctl.get_num_activated_queues(), 0)
            with dpctl.device_context("opencl:gpu:0"):
                self.assertEqual(dpctl.get_num_activated_queues(), 1)

        Session1 = Thread(target=SessionThread(self))
        Session2 = Thread(target=SessionThread(self))
        with dpctl.device_context("opencl:cpu:0"):
            self.assertEqual(dpctl.get_num_activated_queues(), 1)
            Session1.start()
            Session2.start()


if __name__ == "__main__":
    unittest.main()
