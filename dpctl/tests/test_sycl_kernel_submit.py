#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2022 Intel Corporation
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
import dpctl.tensor as dpt


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
    if dtype == np.dtype("f8") and q.sycl_device.has_aspect_fp64 is False:
        pytest.skip(
            "Device does not support double precision floating point type"
        )
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
    a = dpt.arange(n_elems, dtype=dtype, sycl_queue=q)
    b = dpt.arange(n_elems, stop=0, step=-1, dtype=dtype, sycl_queue=q)
    c = dpt.zeros(n_elems, dtype=dtype, sycl_queue=q)

    d = 2
    args = [a.usm_data, b.usm_data, c.usm_data, ctypes_ctor(d)]

    assert n_elems % lws == 0

    b_np = dpt.asnumpy(b)
    a_np = dpt.asnumpy(a)

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
            ref_c = a_np * np.array(d, dtype=dtype) + b_np
        host_dt, device_dt = timer.dt
        assert type(host_dt) is float and type(device_dt) is float
        assert np.allclose(dpt.asnumpy(c), ref_c), "Failed for {}".format(r)

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
            ref_c = a_np * np.array(d, dtype=dtype) + b_np
        host_dt, device_dt = timer.dt
        assert type(host_dt) is float and type(device_dt) is float
        assert np.allclose(dpt.asnumpy(c), ref_c), "Failed for {}, {}".formatg(
            r, lr
        )


def test_async_submit():
    try:
        q = dpctl.SyclQueue("opencl")
    except dpctl.SyclQueueCreationError:
        pytest.skip("OpenCL queue could not be created")
    oclSrc = (
        "kernel void kern1(global unsigned int *res, unsigned int mod) {"
        "   size_t index = get_global_id(0);"
        "   int ri = (index % mod);"
        "   res[index] = (ri * ri) % mod;"
        "}"
        " "
        "kernel void kern2(global unsigned int *res, unsigned int mod) {"
        "   size_t index = get_global_id(0);"
        "   int ri = (index % mod);"
        "   int ri2 = (ri * ri) % mod;"
        "   res[index] = (ri2 * ri) % mod;"
        "}"
        " "
        "kernel void kern3("
        "   global unsigned int *res, global unsigned int *arg1, "
        "   global unsigned int *arg2)"
        "{"
        "  size_t index = get_global_id(0);"
        "  size_t i = 0; "
        "  size_t unused_sum = 0;"
        "  for (i = 0; i < 4000; i++) { "
        "       unused_sum += i;"
        "  } "
        "  res[index] = "
        "      (arg1[index] < arg2[index]) ? arg1[index] : arg2[index];"
        "}"
    )
    prog = dpctl_prog.create_program_from_source(q, oclSrc)
    kern1Kernel = prog.get_sycl_kernel("kern1")
    kern2Kernel = prog.get_sycl_kernel("kern2")
    kern3Kernel = prog.get_sycl_kernel("kern3")

    assert isinstance(kern1Kernel, dpctl_prog.SyclKernel)
    assert isinstance(kern2Kernel, dpctl_prog.SyclKernel)
    assert isinstance(kern2Kernel, dpctl_prog.SyclKernel)

    n = 1024 * 1024
    X = dpt.empty((3, n), dtype="u4", usm_type="device", sycl_queue=q)
    first_row = dpctl_mem.as_usm_memory(X[0])
    second_row = dpctl_mem.as_usm_memory(X[1])
    third_row = dpctl_mem.as_usm_memory(X[2])

    e1 = q.submit(
        kern1Kernel,
        [
            first_row,
            ctypes.c_uint(17),
        ],
        [
            n,
        ],
    )
    e2 = q.submit(
        kern2Kernel,
        [
            second_row,
            ctypes.c_uint(27),
        ],
        [
            n,
        ],
    )
    e3 = q.submit(
        kern3Kernel,
        [third_row, first_row, second_row],
        [
            n,
        ],
        None,
        [e1, e2],
    )
    status_complete = dpctl.event_status_type.complete
    assert not all(
        [
            e == status_complete
            for e in (
                e1.execution_status,
                e2.execution_status,
                e3.execution_status,
            )
        ]
    )

    e3.wait()
    Xnp = dpt.asnumpy(X)
    Xref = np.empty((3, n), dtype="u4")
    for i in range(n):
        Xref[0, i] = (i * i) % 17
        Xref[1, i] = (i * i * i) % 27
        Xref[2, i] = min(Xref[0, i], Xref[1, i])

    assert np.array_equal(Xnp, Xref)
