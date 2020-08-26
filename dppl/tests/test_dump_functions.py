#*******************************************************************************
# Copyright 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#******************************************************************************/

import unittest

import dppl
import dppl.ocldrv as drv


class TestDumpMethods(unittest.TestCase):

    def test_dppl_dump (self):
        try:
            dppl.dump()
        except Exception:
            self.fail("Encountered an exception inside dump().")

    def test_dppl_dump_device_info (self):
        q = dppl.get_current_queue()
        try:
            dppl.dump_device_info(q)
        except Exception:
            self.fail("Encountered an exception inside dump_device_info().")