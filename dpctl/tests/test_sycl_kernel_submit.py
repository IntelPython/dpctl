#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2024 Intel Corporation
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
from dpctl._sycl_queue import kernel_arg_type


@pytest.mark.parametrize(
    "ctype_str,dtype,ctypes_ctor",
    [
        ("short", dpt.dtype("i2"), ctypes.c_short),
        ("int", dpt.dtype("i4"), ctypes.c_int),
        ("unsigned int", dpt.dtype("u4"), ctypes.c_uint),
        ("long", dpt.dtype(np.longlong), ctypes.c_longlong),
        ("unsigned long", dpt.dtype(np.ulonglong), ctypes.c_ulonglong),
        ("float", dpt.dtype("f4"), ctypes.c_float),
        ("double", dpt.dtype("f8"), ctypes.c_double),
    ],
)
def test_create_program_from_source(ctype_str, dtype, ctypes_ctor):
    try:
        q = dpctl.SyclQueue("opencl", property="enable_profiling")
    except dpctl.SyclQueueCreationError:
        pytest.skip("OpenCL queue could not be created")
    if dtype == dpt.dtype("f8") and q.sycl_device.has_aspect_fp64 is False:
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


def test_submit_async():
    try:
        q = dpctl.SyclQueue("opencl")
    except dpctl.SyclQueueCreationError:
        pytest.skip("OpenCL queue could not be created")
    oclSrc = (
        "kernel void kern1(global unsigned int *res, unsigned int mod) {"
        "   size_t unused_sum = 0;"
        "   size_t i = 0; "
        "   for (i = 0; i < 4000; i++) { "
        "       unused_sum += i;"
        "   } "
        "   size_t index = get_global_id(0);"
        "   int ri = (index % mod);"
        "   res[index] = (ri * ri) % mod;"
        "}"
        " "
        "kernel void kern2(global unsigned int *res, unsigned int mod) {"
        "   size_t unused_sum = 0;"
        "   size_t i = 0; "
        "   for (i = 0; i < 4000; i++) { "
        "       unused_sum += i;"
        "   } "
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
        "   size_t index = get_global_id(0);"
        "   size_t i = 0; "
        "   size_t unused_sum = 0;"
        "   for (i = 0; i < 4000; i++) { "
        "       unused_sum += i;"
        "   } "
        "   res[index] = "
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

    status_complete = dpctl.event_status_type.complete

    # choose input size based on capability of the device
    f = q.sycl_device.max_work_group_size
    n = f * 1024
    n_alloc = 4 * n

    X = dpt.empty((3, n_alloc), dtype="u4", usm_type="device", sycl_queue=q)
    first_row = dpctl_mem.as_usm_memory(X[0])
    second_row = dpctl_mem.as_usm_memory(X[1])
    third_row = dpctl_mem.as_usm_memory(X[2])

    p1, p2 = 17, 27

    async_detected = False
    for attempt in range(5):
        e1 = q.submit_async(
            kern1Kernel,
            [
                first_row,
                ctypes.c_uint(p1),
            ],
            [
                n,
            ],
        )
        e2 = q.submit_async(
            kern2Kernel,
            [
                second_row,
                ctypes.c_uint(p2),
            ],
            [
                n,
            ],
        )
        e3 = q.submit_async(
            kern3Kernel,
            [third_row, first_row, second_row],
            [
                n,
            ],
            None,
            [e1, e2],
        )
        e3_st = e3.execution_status
        e2_st = e2.execution_status
        e1_st = e1.execution_status
        ht_e = q._submit_keep_args_alive(
            [first_row, second_row, third_row], [e1, e2, e3]
        )
        are_complete = [
            e == status_complete
            for e in (
                e1_st,
                e2_st,
                e3_st,
            )
        ]
        e3.wait()
        ht_e.wait()
        if not all(are_complete):
            async_detected = True
            break
        else:
            n = n * (1 if attempt % 2 == 0 else 2)
            if n > n_alloc:
                break

    assert async_detected, "No evidence of async submission detected, unlucky?"
    Xnp = dpt.asnumpy(X)
    Xref = np.empty((3, n), dtype="u4")
    for i in range(n):
        Xref[0, i] = (i * i) % p1
        Xref[1, i] = (i * i * i) % p2
        Xref[2, i] = min(Xref[0, i], Xref[1, i])

    assert np.array_equal(Xnp[:, :n], Xref[:, :n])


def _check_kernel_arg_type_instance(kati):
    assert isinstance(kati.name, str)
    assert isinstance(kati.value, int)
    assert isinstance(repr(kati), str)
    assert isinstance(str(kati), str)


def test_kernel_arg_type():
    """
    Check that enum values for kernel_arg_type start at 0,
    as numba_dpex expects. The next enumerated type must
    have next value.
    """
    assert isinstance(kernel_arg_type.__name__, str)
    assert isinstance(repr(kernel_arg_type), str)
    assert isinstance(str(kernel_arg_type), str)
    _check_kernel_arg_type_instance(kernel_arg_type.dpctl_int8)
    _check_kernel_arg_type_instance(kernel_arg_type.dpctl_uint8)
    _check_kernel_arg_type_instance(kernel_arg_type.dpctl_int16)
    _check_kernel_arg_type_instance(kernel_arg_type.dpctl_uint16)
    _check_kernel_arg_type_instance(kernel_arg_type.dpctl_int32)
    _check_kernel_arg_type_instance(kernel_arg_type.dpctl_uint32)
    _check_kernel_arg_type_instance(kernel_arg_type.dpctl_int64)
    _check_kernel_arg_type_instance(kernel_arg_type.dpctl_uint64)
    _check_kernel_arg_type_instance(kernel_arg_type.dpctl_float32)
    _check_kernel_arg_type_instance(kernel_arg_type.dpctl_float64)
    _check_kernel_arg_type_instance(kernel_arg_type.dpctl_void_ptr)
    _check_kernel_arg_type_instance(kernel_arg_type.dpctl_local_accessor)
