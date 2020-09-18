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
from dppl._memory import MemoryUSMShared, MemoryUSMHost, MemoryUSMDevice


class TestMemory (unittest.TestCase):

    def test_memory_create (self):
        nbytes = 1024
        mobj = MemoryUSMShared(nbytes)
        self.assertEqual(mobj.nbytes, nbytes)

    def _create_memory (self):
        nbytes = 1024
        mobj = MemoryUSMShared(nbytes)
        return mobj

    def test_memory_without_context (self):
        mobj = self._create_memory()

        # Without context
        self.assertEqual(mobj._usm_type(), 'shared')

    def test_memory_cpu_context (self):
        mobj = self._create_memory()

        # CPU context
        with dppl.device_context(dppl.device_type.cpu):
            self.assertEqual(mobj._usm_type(), 'shared')

    def test_memory_gpu_context (self):
        mobj = self._create_memory()

        # GPU context
        with dppl.device_context(dppl.device_type.gpu):
            self.assertEqual(mobj._usm_type(), 'shared')


class TestMemoryUSMShared(unittest.TestCase):
    """Tests for MemoryUSMShared
    """

    def test_create (self):
        m = MemoryUSMShared(1024)
        self.assertEqual(m._usm_type(), 'shared')


class TestMemoryUSMHost(unittest.TestCase):
    """Tests for MemoryUSMHost
    """

    def test_create (self):
        m = MemoryUSMHost(1024)
        self.assertEqual(m._usm_type(), 'host')


class TestMemoryUSMDevice(unittest.TestCase):
    """Tests for MemoryUSMDevice
    """

    def test_create (self):
        m = MemoryUSMDevice(1024)
        self.assertEqual(m._usm_type(), 'device')
