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

"""Defines unit test cases for kernel submission to a sycl::queue.
"""

import ctypes
import unittest

import numpy as np

import dpctl
import dpctl.memory as dpctl_mem
import dpctl.program as dpctl_prog

from ._helper import has_cpu, has_gpu


@unittest.skipUnless(has_gpu(), "No OpenCL GPU queues available")
class Test1DKernelSubmit(unittest.TestCase):
    def test_create_program_from_source(self):
        oclSrc = "                                                             \
        kernel void axpy(global int* a, global int* b, global int* c, int d) { \
            size_t index = get_global_id(0);                                   \
            c[index] = d*a[index] + b[index];                                  \
        }"
        q = dpctl.SyclQueue("opencl:gpu")
        prog = dpctl_prog.create_program_from_source(q, oclSrc)
        axpyKernel = prog.get_sycl_kernel("axpy")

        bufBytes = 1024 * np.dtype("i").itemsize
        abuf = dpctl_mem.MemoryUSMShared(bufBytes, queue=q)
        bbuf = dpctl_mem.MemoryUSMShared(bufBytes, queue=q)
        cbuf = dpctl_mem.MemoryUSMShared(bufBytes, queue=q)
        a = np.ndarray((1024), buffer=abuf, dtype="i")
        b = np.ndarray((1024), buffer=bbuf, dtype="i")
        c = np.ndarray((1024), buffer=cbuf, dtype="i")
        a[:] = np.arange(1024)
        b[:] = np.arange(1024, 0, -1)
        c[:] = 0
        d = 2
        args = []

        args.append(a.base)
        args.append(b.base)
        args.append(c.base)
        args.append(ctypes.c_int(d))

        r = [1024]

        q.submit(axpyKernel, args, r)
        self.assertTrue(np.allclose(c, a * d + b))


if __name__ == "__main__":
    unittest.main()
