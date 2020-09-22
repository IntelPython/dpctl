##===---------- test_sycl_queue_manager.py - dppl  -------*- Python -*-----===##
##
##               Python Data Parallel Processing Library (PyDPPL)
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
### This file has unit test cases to for the SyclQueueManager class
### in sycl_core.pyx.
##===----------------------------------------------------------------------===##

import dppl
import unittest

class TestGetNumPlatforms (unittest.TestCase):
    @unittest.skipIf(not dppl.has_sycl_platforms(), "No SYCL platforms available")
    def test_dppl_get_num_platforms (self):
        if(dppl.has_sycl_platforms):
            self.assertGreaterEqual(dppl.get_num_platforms(), 1)

@unittest.skipIf(not dppl.has_sycl_platforms(), "No SYCL platforms available")
class TestDumpMethods (unittest.TestCase):
    def test_dppl_dump (self):
        try:
            dppl.dump()
        except Exception:
            self.fail("Encountered an exception inside dump().")

    def test_dppl_dump_device_info (self):
        q = dppl.get_current_queue()
        try:
            q.get_sycl_device().dump_device_info()
        except Exception:
            self.fail("Encountered an exception inside dump_device_info().")

@unittest.skipIf(not dppl.has_sycl_platforms(), "No SYCL platforms available")
class TestDPPLIsInDPPLCtxt (unittest.TestCase):

    def test_is_in_dppl_ctxt_outside_device_ctxt (self):
        self.assertFalse(dppl.is_in_dppl_ctxt())

    def test_is_in_dppl_ctxt_inside_device_ctxt (self):
        with dppl.device_context(dppl.device_type.gpu):
            self.assertTrue(dppl.is_in_dppl_ctxt())

    def test_is_in_dppl_ctxt_inside_nested_device_ctxt (self):
        with dppl.device_context(dppl.device_type.cpu):
            with dppl.device_context(dppl.device_type.gpu):
                self.assertTrue(dppl.is_in_dppl_ctxt())
            self.assertTrue(dppl.is_in_dppl_ctxt())
        self.assertFalse(dppl.is_in_dppl_ctxt())

@unittest.skipIf(not dppl.has_sycl_platforms(), "No SYCL platforms available")
class TestGetCurrentQueueInMultipleThreads (unittest.TestCase):

    def test_num_current_queues_outside_with_clause (self):
        self.assertEqual(dppl.get_num_activated_queues(), 0)

    @unittest.skipIf(not dppl.has_gpu_queues(), "No GPU platforms available")
    def test_num_current_queues_inside_with_clause (self):
        with dppl.device_context(dppl.device_type.cpu):
            self.assertEqual(dppl.get_num_activated_queues(), 1)
            with dppl.device_context(dppl.device_type.gpu):
                self.assertEqual(dppl.get_num_activated_queues(), 2)
        self.assertEqual(dppl.get_num_activated_queues(), 0)

    @unittest.skipIf(not dppl.has_gpu_queues(), "No GPU platforms available")
    def test_num_current_queues_inside_threads (self):
        from threading import Thread, local
        def SessionThread (self):
            self.assertEqual(dppl.get_num_activated_queues(), 0)
            with dppl.device_context(dppl.device_type.gpu):
                self.assertEqual(dppl.get_num_activated_queues(), 1)

        Session1 = Thread(target=SessionThread(self))
        Session2 = Thread(target=SessionThread(self))
        with dppl.device_context(dppl.device_type.cpu):
            self.assertEqual(dppl.get_num_activated_queues(), 1)
            Session1.start()
            Session2.start()

if __name__ == '__main__':
    unittest.main()
