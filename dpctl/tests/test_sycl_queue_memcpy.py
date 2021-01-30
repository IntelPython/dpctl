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

"""Defines unit test cases for the SyclQueue.memcpy.
"""

import dpctl
import dpctl.memory
import unittest


class TestQueueMemcpy(unittest.TestCase):
    def _create_memory(self):
        nbytes = 1024
        mobj = dpctl.memory.MemoryUSMShared(nbytes)
        return mobj

    @unittest.skipUnless(
        dpctl.has_sycl_platforms(), "No SYCL devices except the default host device."
    )
    def test_memcpy_copy_usm_to_usm(self):
        mobj1 = self._create_memory()
        mobj2 = self._create_memory()
        q = dpctl.get_current_queue()

        mv1 = memoryview(mobj1)
        mv2 = memoryview(mobj2)

        mv1[:3] = b"123"

        q.memcpy(mobj2, mobj1, 3)

        self.assertEqual(mv2[:3], b"123")

    @unittest.skipUnless(
        dpctl.has_sycl_platforms(), "No SYCL devices except the default host device."
    )
    def test_memcpy_type_error(self):
        mobj = self._create_memory()
        q = dpctl.get_current_queue()

        with self.assertRaises(TypeError) as cm:
            q.memcpy(None, mobj, 3)

        self.assertEqual(type(cm.exception), TypeError)
        self.assertEqual(
            str(cm.exception), "Parameter `dest` should have type _Memory."
        )

        with self.assertRaises(TypeError) as cm:
            q.memcpy(mobj, None, 3)

        self.assertEqual(type(cm.exception), TypeError)
        self.assertEqual(str(cm.exception), "Parameter `src` should have type _Memory.")


if __name__ == "__main__":
    unittest.main()
