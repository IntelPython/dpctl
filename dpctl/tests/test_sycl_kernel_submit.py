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

from ._helper import has_gpu


@unittest.skipUnless(has_gpu(), "No OpenCL GPU queues available")
class Test1DKernelSubmit(unittest.TestCase):
    def test_create_program_from_source(self):
        oclSrc = "                                                             \
        kernel void axpy(global int* a, global int* b, global int* c, int d) { \
            size_t index = get_global_id(0);                                   \
            c[index] = d*a[index] + b[index];                                  \
        }"
        q = dpctl.SyclQueue("opencl:gpu", property="enable_profiling")
        prog = dpctl_prog.create_program_from_source(q, oclSrc)
        axpyKernel = prog.get_sycl_kernel("axpy")

        n_elems = 1024 * 512
        bufBytes = n_elems * np.dtype("i").itemsize
        abuf = dpctl_mem.MemoryUSMShared(bufBytes, queue=q)
        bbuf = dpctl_mem.MemoryUSMShared(bufBytes, queue=q)
        cbuf = dpctl_mem.MemoryUSMShared(bufBytes, queue=q)
        a = np.ndarray((n_elems,), buffer=abuf, dtype="i")
        b = np.ndarray((n_elems,), buffer=bbuf, dtype="i")
        c = np.ndarray((n_elems,), buffer=cbuf, dtype="i")
        a[:] = np.arange(n_elems)
        b[:] = np.arange(n_elems, 0, -1)
        c[:] = 0
        d = 2
        args = []

        args.append(a.base)
        args.append(b.base)
        args.append(c.base)
        args.append(ctypes.c_int(d))

        r = [
            n_elems,
        ]

        timer = dpctl.SyclTimer()
        with timer(q):
            q.submit(axpyKernel, args, r)
            ref_c = a * d + b
        host_dt, device_dt = timer.dt
        self.assertTrue(host_dt > device_dt)
        self.assertTrue(np.allclose(c, ref_c))


if __name__ == "__main__":
    unittest.main()
