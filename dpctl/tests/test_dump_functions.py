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

"""A basic canary unit test for the dpctl module.
"""


import unittest
import dpctl


class TestDumpMethods(unittest.TestCase):
    def test_dpctl_dump(self):
        try:
            dpctl.dump()
        except Exception:
            self.fail("Encountered an exception inside dump().")

    @unittest.skipUnless(
        dpctl.has_sycl_platforms(), "No SYCL devices except the default host device."
    )
    def test_dpctl_print_device_info(self):
        q = dpctl.get_current_queue()
        try:
            q.get_sycl_device().print_device_info()
        except Exception:
            self.fail("Encountered an exception inside print_device_info().")


if __name__ == "__main__":
    unittest.main()
