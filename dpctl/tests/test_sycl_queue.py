##===------------- test_sycl_queue.py - dpctl  -------*- Python -*---------===##
##
##                      Data Parallel Control (dpctl)
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
##
## \file
## Defines unit test cases for the SyclQueue classes defined in sycl_core.pyx.
##===----------------------------------------------------------------------===##

import dpctl
import unittest

class TestSyclQueue (unittest.TestCase):
    @unittest.skipUnless(
        dpctl.has_gpu_queues(), "No OpenCL GPU queues available"
    )
    @unittest.skipUnless(
        dpctl.has_cpu_queues(), "No OpenCL CPU queues available"
    )
    def test_queue_not_equals (self):
        with dpctl.device_context("opencl:gpu") as gpuQ0:
            with dpctl.device_context("opencl:cpu") as cpuQ:
                self.assertFalse(cpuQ.equals(gpuQ0))

    @unittest.skipUnless(
        dpctl.has_gpu_queues(), "No OpenCL GPU queues available"
    )
    def test_queue_equals (self):
        with dpctl.device_context("opencl:gpu") as gpuQ0:
            with dpctl.device_context("opencl:gpu") as gpuQ1:
                self.assertTrue(gpuQ0.equals(gpuQ1))

if __name__ == '__main__':
    unittest.main()
