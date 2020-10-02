##===---------- test_sycl_queue_manager.py - dpctl  -------*- Python -*----===##
##
##                      Data Parallel Control (dpCtl)
##
## Copyright 2020 Intel Corporation
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##
##===----------------------------------------------------------------------===##
###
### \file
### Defines unit test cases for the SyclQueueManager class in sycl_core.pyx.
##===----------------------------------------------------------------------===##

import dpctl
import unittest

class TestGetNumPlatforms (unittest.TestCase):
    @unittest.skipIf(not dpctl.has_sycl_platforms(),
                     "No SYCL platforms available")
    def test_dpctl_get_num_platforms (self):
        if(dpctl.has_sycl_platforms):
            self.assertGreaterEqual(dpctl.get_num_platforms(), 1)

@unittest.skipIf(not dpctl.has_sycl_platforms(), "No SYCL platforms available")
class TestDumpMethods (unittest.TestCase):
    def test_dpctl_dump (self):
        try:
            dpctl.dump()
        except Exception:
            self.fail("Encountered an exception inside dump().")

    def test_dpctl_dump_device_info (self):
        q = dpctl.get_current_queue()
        try:
            q.get_sycl_device().dump_device_info()
        except Exception:
            self.fail("Encountered an exception inside dump_device_info().")

@unittest.skipIf(not dpctl.has_sycl_platforms(), "No SYCL platforms available")
class TestIsInDeviceContext (unittest.TestCase):

    def test_is_in_device_context_outside_device_ctxt (self):
        self.assertFalse(dpctl.is_in_device_context())

    @unittest.skipUnless(dpctl.get_num_queues(backend_ty="opencl",
                                              device_ty="gpu") > 0,
                         "No OpenCL GPU queues available")
    def test_is_in_device_context_inside_device_ctxt (self):
        with dpctl.device_context("opencl:gpu:0"):
            self.assertTrue(dpctl.is_in_device_context())

    @unittest.skipUnless(dpctl.get_num_queues(backend_ty="opencl",
                                              device_ty="gpu") > 0,
                         "No OpenCL GPU queues available")
    @unittest.skipUnless(dpctl.get_num_queues(backend_ty="opencl",
                                              device_ty="cpu") > 0,
                         "No OpenCL CPU queues available")
    def test_is_in_device_context_inside_nested_device_ctxt (self):
        with dpctl.device_context("opencl:cpu:0"):
            with dpctl.device_context("opencl:gpu:0"):
                self.assertTrue(dpctl.is_in_device_context())
            self.assertTrue(dpctl.is_in_device_context())
        self.assertFalse(dpctl.is_in_device_context())

@unittest.skipIf(not dpctl.has_sycl_platforms(), "No SYCL platforms available")
class TestGetCurrentQueueInMultipleThreads (unittest.TestCase):

    def test_num_current_queues_outside_with_clause (self):
        self.assertEqual(dpctl.get_num_activated_queues(), 0)

    @unittest.skipUnless(dpctl.get_num_queues(backend_ty="opencl",
                                              device_ty="gpu") > 0,
                         "No OpenCL GPU queues available")
    @unittest.skipUnless(dpctl.get_num_queues(backend_ty="opencl",
                                              device_ty="cpu") > 0,
                         "No OpenCL CPU queues available")
    def test_num_current_queues_inside_with_clause (self):
        with dpctl.device_context("opencl:cpu:0"):
            self.assertEqual(dpctl.get_num_activated_queues(), 1)
            with dpctl.device_context("opencl:gpu:0"):
                self.assertEqual(dpctl.get_num_activated_queues(), 2)
        self.assertEqual(dpctl.get_num_activated_queues(), 0)


    @unittest.skipUnless(dpctl.get_num_queues(backend_ty="opencl",
                                              device_ty="gpu") > 0,
                         "No OpenCL GPU queues available")
    @unittest.skipUnless(dpctl.get_num_queues(backend_ty="opencl",
                                              device_ty="cpu") > 0,
                         "No OpenCL CPU queues available")
    def test_num_current_queues_inside_threads (self):
        from threading import Thread, local
        def SessionThread (self):
            self.assertEqual(dpctl.get_num_activated_queues(), 0)
            with dpctl.device_context("opencl:gpu:0"):
                self.assertEqual(dpctl.get_num_activated_queues(), 1)

        Session1 = Thread(target=SessionThread(self))
        Session2 = Thread(target=SessionThread(self))
        with dpctl.device_context("opencl:cpu:0"):
            self.assertEqual(dpctl.get_num_activated_queues(), 1)
            Session1.start()
            Session2.start()

if __name__ == '__main__':
    unittest.main()
