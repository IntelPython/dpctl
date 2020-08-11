##===---------- t_is_in_dppl_ctxt.py - dppl  -------------*- Python -*-----===##
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
### This file has unit test cases to check the value returned by the
### dppl.is_in_dppl_ctxt function.
##===----------------------------------------------------------------------===##

import dppl
import unittest

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

suite = unittest.TestLoader().loadTestsFromTestCase(TestDPPLIsInDPPLCtxt)
unittest.TextTestRunner(verbosity=2).run(suite)
