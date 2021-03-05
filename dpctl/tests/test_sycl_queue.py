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

""" Defines unit test cases for the SyclQueue class.
"""

import dpctl
import unittest
from ._helper import (has_cpu, has_gpu)

class TestSyclQueue(unittest.TestCase):
    @unittest.skipUnless(has_gpu(), "No OpenCL GPU queues available")
    @unittest.skipUnless(has_cpu(), "No OpenCL CPU queues available")
    def test_queue_not_equals(self):
        with dpctl.device_context("opencl:gpu") as gpuQ:
            ctx_gpu = gpuQ.get_sycl_context()
            with dpctl.device_context("opencl:cpu") as cpuQ:
                ctx_cpu = cpuQ.get_sycl_context()
                self.assertFalse(ctx_cpu.equals(ctx_gpu))

    @unittest.skipUnless(has_gpu(), "No OpenCL GPU queues available")
    def test_queue_equals(self):
        with dpctl.device_context("opencl:gpu") as gpuQ0:
            ctx0 = gpuQ0.get_sycl_context()
            with dpctl.device_context("opencl:gpu") as gpuQ1:
                ctx1 = gpuQ1.get_sycl_context()
                self.assertTrue(ctx0.equals(ctx1))


if __name__ == "__main__":
    unittest.main()
