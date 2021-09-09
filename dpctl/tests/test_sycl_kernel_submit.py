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


@pytest.mark.parametrize(
    "ctype_str,dtype,ctypes_ctor",
    [
        ("short", np.dtype("i2"), ctypes.c_short),
        ("int", np.dtype("i4"), ctypes.c_int),
        ("unsigned int", np.dtype("u4"), ctypes.c_uint),
        ("long", np.dtype(np.longlong), ctypes.c_longlong),
        ("unsigned long", np.dtype(np.ulonglong), ctypes.c_ulonglong),
        ("float", np.dtype("f4"), ctypes.c_float),
        ("double", np.dtype("f8"), ctypes.c_double),
    ],
)
def test_create_program_from_source(ctype_str, dtype, ctypes_ctor):
    try:
        q = dpctl.SyclQueue("opencl", property="enable_profiling")
    except dpctl.SyclQueueCreationError:
        pytest.skip("OpenCL queue could not be created")
    # OpenCL conventions for indexing global_id is opposite to
    # that of SYCL (and DPCTL)
    oclSrc = (
        "kernel void axpy("
        "   global " + ctype_str + " *a, global " + ctype_str + " *b,"
        "   global " + ctype_str + " *c, " + ctype_str + " d) {"
        "   size_t index = get_global_id(0);"
        "   c[index] = d * a[index] + b[index];"
        "}"
    )
    prog = dpctl_prog.create_program_from_source(q, oclSrc)
    axpyKernel = prog.get_sycl_kernel("axpy")

    n_elems = 1024 * 512
    lws = 128
    bufBytes = n_elems * dtype.itemsize
    abuf = dpctl_mem.MemoryUSMShared(bufBytes, queue=q)
    bbuf = dpctl_mem.MemoryUSMShared(bufBytes, queue=q)
    cbuf = dpctl_mem.MemoryUSMShared(bufBytes, queue=q)
    a = np.ndarray((n_elems,), buffer=abuf, dtype=dtype)
    b = np.ndarray((n_elems,), buffer=bbuf, dtype=dtype)
    c = np.ndarray((n_elems,), buffer=cbuf, dtype=dtype)
    a[:] = np.arange(n_elems)
    b[:] = np.arange(n_elems, 0, -1)
    c[:] = 0
    d = 2
    args = [a.base, b.base, c.base, ctypes_ctor(d)]

    assert n_elems % lws == 0

    for r in (
        [
            n_elems,
        ],
        [2, n_elems],
        [2, 2, n_elems],
    ):
        c[:] = 0
        timer = dpctl.SyclTimer()
        with timer(q):
            q.submit(axpyKernel, args, r).wait()
            ref_c = a * np.array(d, dtype=dtype) + b
        host_dt, device_dt = timer.dt
        assert host_dt > device_dt
        assert np.allclose(c, ref_c), "Failed for {}".format(r)

    for gr, lr in (
        (
            [
                n_elems,
            ],
            [lws],
        ),
        ([2, n_elems], [2, lws // 2]),
        ([2, 2, n_elems], [2, 2, lws // 4]),
    ):
        c[:] = 0
        timer = dpctl.SyclTimer()
        with timer(q):
            q.submit(axpyKernel, args, gr, lr, [dpctl.SyclEvent()]).wait()
            ref_c = a * np.array(d, dtype=dtype) + b
        host_dt, device_dt = timer.dt
        assert host_dt > device_dt
        assert np.allclose(c, ref_c), "Faled for {}, {}".formatg(r, lr)
