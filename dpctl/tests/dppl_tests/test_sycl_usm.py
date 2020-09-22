##===---------- test_sycl_queue_manager.py - dpctl  -------*- Python -*-----===##
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
import dpctl
from dpctl._memory import MemoryUSMShared, MemoryUSMHost, MemoryUSMDevice


class TestMemory (unittest.TestCase):

    def test_memory_create (self):
        nbytes = 1024
        queue = dpctl.get_current_queue()
        mobj = MemoryUSMShared(nbytes, queue)
        self.assertEqual(mobj.nbytes, nbytes)

    def _create_memory (self):
        nbytes = 1024
        queue = dpctl.get_current_queue()
        mobj = MemoryUSMShared(nbytes, queue)
        return mobj

    def test_memory_without_context (self):
        mobj = self._create_memory()

        # Without context
        self.assertEqual(mobj._usm_type(), 'shared')

    @unittest.skipIf(not dpctl.has_cpu_queues(), "No CPU platforms available")
    def test_memory_cpu_context (self):
        mobj = self._create_memory()

        # CPU context
        with dpctl.device_context(dpctl.device_type.cpu):
            # type respective to the context in which
            # memory was created
            usm_type = mobj._usm_type()
            self.assertEqual(usm_type, 'shared')

            current_queue = dpctl.get_current_queue()
            # type as view from current queue
            usm_type = mobj._usm_type(current_queue)
            # type can be unknown if current queue is
            # not in the same SYCL context
            self.assertTrue(usm_type in ['unknown', 'shared'])

    @unittest.skipIf(not dpctl.has_gpu_queues(), "No GPU platforms available")
    def test_memory_gpu_context (self):
        mobj = self._create_memory()

        # GPU context
        with dpctl.device_context(dpctl.device_type.gpu):
            usm_type = mobj._usm_type()
            self.assertEqual(usm_type, 'shared')
            current_queue = dpctl.get_current_queue()
            usm_type = mobj._usm_type(current_queue)
            self.assertTrue(usm_type in ['unknown', 'shared'])


class TestMemoryUSMBase:
    """ Base tests for MemoryUSM* """

    MemoryUSMClass = None
    usm_type = None

    def test_create_with_queue (self):
        q = dpctl.get_current_queue()
        m = self.MemoryUSMClass(1024, q)
        self.assertEqual(m.nbytes, 1024)
        self.assertEqual(m._usm_type(), self.usm_type)

    def test_create_without_queue (self):
        m = self.MemoryUSMClass(1024)
        self.assertEqual(m.nbytes, 1024)
        self.assertEqual(m._usm_type(), self.usm_type)


class TestMemoryUSMShared(TestMemoryUSMBase, unittest.TestCase):
    """ Tests for MemoryUSMShared """

    MemoryUSMClass = MemoryUSMShared
    usm_type = 'shared'


class TestMemoryUSMHost(TestMemoryUSMBase, unittest.TestCase):
    """ Tests for MemoryUSMHost """

    MemoryUSMClass = MemoryUSMHost
    usm_type = 'host'


class TestMemoryUSMDevice(TestMemoryUSMBase, unittest.TestCase):
    """ Tests for MemoryUSMDevice """

    MemoryUSMClass = MemoryUSMDevice
    usm_type = 'device'
