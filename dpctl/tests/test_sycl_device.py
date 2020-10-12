##===------------- test_sycl_device.py - dpctl  -------*- Python -*---------===##
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
## Defines unit test cases for the SyclDevice classes defined in sycl_core.pyx.
##===----------------------------------------------------------------------===##

import dpctl
import unittest

@unittest.skipIf(not dpctl.has_sycl_platforms(), "No SYCL platforms available")
class TestSyclDevice (unittest.TestCase):

    def test_get_max_compute_units (self):
        q = dpctl.get_current_queue()
        try:
            max_compute_units = q.get_sycl_device().get_max_compute_units()
        except Exception:
            self.fail("Encountered an exception inside get_max_compute_units().")
        self.assertNotEqual(max_compute_units, 0)

    def test_get_max_work_item_dims (self):
        q = dpctl.get_current_queue()
        try:
            max_work_item_dims = q.get_sycl_device().get_max_work_item_dims()
        except Exception:
            self.fail("Encountered an exception inside get_max_work_item_dims().")
        self.assertNotEqual(max_work_item_dims, 0)

    def test_get_max_work_item_sizes (self):
        q = dpctl.get_current_queue()
        try:
            max_work_item_sizes = q.get_sycl_device().get_max_work_item_sizes()
        except Exception:
            self.fail("Encountered an exception inside get_max_work_item_sizes().")
        self.assertNotEqual(max_work_item_sizes, (None, None, None))

    def test_get_max_work_group_size (self):
        q = dpctl.get_current_queue()
        try:
            max_work_group_size = q.get_sycl_device().get_max_work_group_size()
        except Exception:
            self.fail("Encountered an exception inside get_max_work_group_size().")
        self.assertNotEqual(max_work_group_size, 0)

    def test_get_max_num_sub_groups (self):
        q = dpctl.get_current_queue()
        try:
            max_num_sub_groups = q.get_sycl_device().get_max_num_sub_groups()
        except Exception:
            self.fail("Encountered an exception inside get_max_num_sub_groups().")
        self.assertNotEqual(max_num_sub_groups, 0)

if __name__ == '__main__':
    unittest.main()
