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

import unittest
import dppl
import dppl._memory as mem


class TestMemory (unittest.TestCase):
    # @unittest.skipIf(not dppl.has_sycl_platforms, "No SYCL platforms available")
    def test_memory_create (self):
        nbytes = 1024
        mobj = mem.Memory(nbytes)
        self.assertEqual(mobj.nbytes, nbytes)

        # Without context
        self.assertEqual(mem.SyclQueue().get_pointer_type(mobj.pointer), 'shared')
        self.assertEqual(mobj._usm_type(), 'shared')

        # CPU context
        with dppl.device_context(dppl.device_type.cpu):
            self.assertEqual(mem.SyclQueue().get_pointer_type(mobj.pointer), 'unknown')
            self.assertEqual(mobj._usm_type(), 'shared')

        # GPU context
        with dppl.device_context(dppl.device_type.gpu):
            self.assertEqual(mem.SyclQueue().get_pointer_type(mobj.pointer), 'unknown')
            self.assertEqual(mobj._usm_type(), 'shared')
