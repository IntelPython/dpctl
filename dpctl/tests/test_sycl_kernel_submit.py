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

import numpy as np
import pytest

import dpctl
import dpctl.memory as dpctl_mem
import dpctl.program as dpctl_prog


def test_create_program_from_source():
    try:
        q = dpctl.SyclQueue("opencl", property="enable_profiling")
    except dpctl.SyclQueueCreationError:
        pytest.skip("OpenCL queue could not be created")
    oclSrc = "                                                             \
    kernel void axpy(global int* a, global int* b, global int* c, int d) { \
        size_t index = get_global_id(0);                                   \
        c[index] = d*a[index] + b[index];                                  \
    }"
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
    assert host_dt > device_dt
    assert np.allclose(c, ref_c)
