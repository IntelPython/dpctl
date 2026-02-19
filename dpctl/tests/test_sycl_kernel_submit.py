#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2025 Intel Corporation
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

"""Defines unit test cases for kernel submission to a sycl::queue."""

import ctypes
import os

import numpy as np
import pytest

import dpctl
import dpctl.memory as dpm
import dpctl.program as dpctl_prog
from dpctl._sycl_queue import kernel_arg_type


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
    if dtype.kind in "ui":
        n_elems = min(n_elems, np.iinfo(dtype).max)
        n_elems = (n_elems // lws) * lws
    a = np.arange(n_elems, dtype=dtype)
    b = np.arange(n_elems, stop=0, step=-1, dtype=dtype)
    c = np.zeros(n_elems, dtype=dtype)

    a_usm = dpm.MemoryUSMDevice(a.nbytes, queue=q)
    b_usm = dpm.MemoryUSMDevice(b.nbytes, queue=q)
    c_usm = dpm.MemoryUSMDevice(c.nbytes, queue=q)

    ev1 = q.memcpy_async(dest=a_usm, src=a, count=a.nbytes)
    ev2 = q.memcpy_async(dest=b_usm, src=b, count=b.nbytes)

    dpctl.SyclEvent.wait_for([ev1, ev2])

    d = 2
    args = [a_usm, b_usm, c_usm, ctypes_ctor(d)]

    assert n_elems % lws == 0

    for r in (
        [
            n_elems,
        ],
        [2, n_elems],
        [2, 2, n_elems],
    ):
        c_usm.memset()
        timer = dpctl.SyclTimer()
        with timer(q):
            q.submit(axpyKernel, args, r).wait()
            ref_c = a * np.array(d, dtype=dtype) + b
        host_dt, device_dt = timer.dt
        assert type(host_dt) is float and type(device_dt) is float
        q.memcpy(c, c_usm, c.nbytes)
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
        c_usm.memset()
        timer = dpctl.SyclTimer()
        with timer(q):
            q.submit(axpyKernel, args, gr, lr, [dpctl.SyclEvent()]).wait()
            ref_c = a * np.array(d, dtype=dtype) + b
        host_dt, device_dt = timer.dt
        assert type(host_dt) is float and type(device_dt) is float
        q.memcpy(c, c_usm, c.nbytes)
        assert np.allclose(c, ref_c), "Failed for {}, {}".format(gr, lr)


def test_submit_async():
    try:
        q = dpctl.SyclQueue("opencl")
    except dpctl.SyclQueueCreationError:
        pytest.skip("OpenCL queue could not be created")
    oclSrc = (
        "kernel void kern1("
        "   global unsigned int *res_base, ulong res_off, unsigned int mod) {"
        "   size_t unused_sum = 0;"
        "   size_t i = 0; "
        "   for (i = 0; i < 4000; i++) { "
        "       unused_sum += i;"
        "   } "
        "   global unsigned int *res = res_base + (size_t)res_off;"
        "   size_t index = get_global_id(0);"
        "   int ri = (index % mod);"
        "   res[index] = (ri * ri) % mod;"
        "}"
        " "
        "kernel void kern2("
        "   global unsigned int *res_base, ulong res_off, unsigned int mod) {"
        "   size_t unused_sum = 0;"
        "   size_t i = 0; "
        "   for (i = 0; i < 4000; i++) { "
        "       unused_sum += i;"
        "   } "
        "   global unsigned int *res = res_base + (size_t)res_off;"
        "   size_t index = get_global_id(0);"
        "   int ri = (index % mod);"
        "   int ri2 = (ri * ri) % mod;"
        "   res[index] = (ri2 * ri) % mod;"
        "}"
        " "
        "kernel void kern3("
        "   global unsigned int *res_base, ulong res_off,"
        "   global unsigned int *arg1_base, ulong arg1_off,"
        "   global unsigned int *arg2_base, ulong arg2_off)"
        "{"
        "   global unsigned int *res = res_base + (size_t)res_off;"
        "   global unsigned int *arg1 = arg1_base + (size_t)arg1_off;"
        "   global unsigned int *arg2 = arg2_base + (size_t)arg2_off;"
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

    x = np.empty((3, n_alloc), dtype="u4")
    x_usm = dpm.MemoryUSMDevice(x.nbytes, queue=q)

    e1 = q.memcpy_async(dest=x_usm, src=x, count=x.nbytes)

    p1, p2 = 17, 27

    async_detected = False
    for attempt in range(5):
        e1 = q.submit_async(
            kern1Kernel,
            [
                x_usm,
                ctypes.c_ulonglong(0),
                ctypes.c_uint(p1),
            ],
            [
                n,
            ],
            None,
            [e1],
        )
        e2 = q.submit_async(
            kern2Kernel,
            [
                x_usm,
                ctypes.c_ulonglong(n_alloc),
                ctypes.c_uint(p2),
            ],
            [
                n,
            ],
            None,
            [e1],
        )
        e3 = q.submit_async(
            kern3Kernel,
            [
                x_usm,
                ctypes.c_ulonglong(2 * n_alloc),
                x_usm,
                ctypes.c_ulonglong(0),
                x_usm,
                ctypes.c_ulonglong(n_alloc),
            ],
            [
                n,
            ],
            None,
            [e1, e2],
        )
        e3_st = e3.execution_status
        e2_st = e2.execution_status
        e1_st = e1.execution_status
        ht_e = q._submit_keep_args_alive([x_usm], [e1, e2, e3])
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
    q.memcpy(dest=x, src=x_usm, count=x.nbytes)
    x_ref = np.empty((3, n), dtype="u4")
    for i in range(n):
        x_ref[0, i] = (i * i) % p1
        x_ref[1, i] = (i * i * i) % p2
        x_ref[2, i] = min(x_ref[0, i], x_ref[1, i])
    assert np.array_equal(x[:, :n], x_ref[:, :n])


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
    _check_kernel_arg_type_instance(kernel_arg_type.dpctl_work_group_memory)
    _check_kernel_arg_type_instance(kernel_arg_type.dpctl_raw_kernel_arg)


def get_spirv_abspath(fn):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    spirv_file = os.path.join(curr_dir, "input_files", fn)
    return spirv_file


# the process for generating the .spv files in this test is documented in
# libsyclinterface/tests/test_sycl_queue_submit_local_accessor_arg.cpp
# in a comment starting on line 123
def test_submit_local_accessor_arg():
    try:
        q = dpctl.SyclQueue("level_zero")
    except dpctl.SyclQueueCreationError:
        pytest.skip("OpenCL queue could not be created")
    fn = get_spirv_abspath("local_accessor_kernel_inttys_fp32.spv")
    with open(fn, "br") as f:
        spirv_bytes = f.read()
    prog = dpctl_prog.create_program_from_spirv(q, spirv_bytes)
    krn = prog.get_sycl_kernel("_ZTS14SyclKernel_SLMIlE")
    lws = 32
    gws = lws * 10
    x = np.ones(gws, dtype="i8")
    res = np.empty_like(x)
    x_usm = dpm.MemoryUSMDevice(x.nbytes, queue=q)
    q.memcpy(dest=x_usm, src=x, count=x.nbytes)
    try:
        e = q.submit(
            krn,
            [x_usm, dpctl.LocalAccessor("i8", (lws,))],
            [gws],
            [lws],
        )
        e.wait()
    except dpctl._sycl_queue.SyclKernelSubmitError:
        pytest.skip(f"Kernel submission failed for device {q.sycl_device}")
    q.memcpy(dest=res, src=x_usm, count=x.nbytes)
    expected = np.arange(1, x.size + 1, dtype=x.dtype) * (2 * lws)
    assert np.all(res == expected)
