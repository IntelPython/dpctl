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

""" Defines unit test cases for the SyclEvent class.
"""

import numpy as np
import pytest

import dpctl
import dpctl.memory as dpctl_mem
import dpctl.program as dpctl_prog
from dpctl import event_status_type as esty

from ._helper import has_cpu


def produce_event(profiling=False):
    oclSrc = "                                                                 \
            kernel void add(global int* a) {                                   \
                size_t index = get_global_id(0);                               \
                a[index] = a[index] + 1;                                       \
            }"
    if profiling:
        q = dpctl.SyclQueue("opencl:cpu", property="enable_profiling")
    else:
        q = dpctl.SyclQueue("opencl:cpu")
    prog = dpctl_prog.create_program_from_source(q, oclSrc)
    addKernel = prog.get_sycl_kernel("add")

    bufBytes = 1024 * np.dtype("i").itemsize
    abuf = dpctl_mem.MemoryUSMShared(bufBytes, queue=q)
    a = np.ndarray((1024), buffer=abuf, dtype="i")
    a[:] = np.arange(1024)
    args = []

    args.append(a.base)
    r = [1024]
    ev = q.submit(addKernel, args, r)

    return ev


def test_create_default_event():
    try:
        dpctl.SyclEvent()
    except ValueError:
        pytest.fail("Failed to create a default event")


def test_create_event_from_capsule():
    try:
        event = dpctl.SyclEvent()
        event_capsule = event._get_capsule()
        dpctl.SyclEvent(event_capsule)
    except ValueError:
        pytest.fail("Failed to create an event from capsule")


def test_invalid_constructor_arg():
    with pytest.raises(TypeError):
        dpctl.SyclEvent(list())


def test_wait_with_event():
    event = dpctl.SyclEvent()
    try:
        dpctl.SyclEvent.wait_for(event)
    except ValueError:
        pytest.fail("Failed to wait_for(event)")
    event = dpctl.SyclEvent()
    try:
        event.wait()
    except ValueError:
        pytest.fail("Failed to wait for the event")


def test_wait_for_invalid():
    with pytest.raises(TypeError):
        dpctl.SyclEvent.wait_for(77)


def test_wait_with_list():
    event_1 = dpctl.SyclEvent()
    event_2 = dpctl.SyclEvent()
    try:
        dpctl.SyclEvent.wait_for([event_1, event_2])
    except ValueError:
        pytest.fail("Failed to wait for events from the list")


def test_execution_status():
    event = dpctl.SyclEvent()
    try:
        event_status = event.execution_status
    except ValueError:
        pytest.fail("Failed to get an event status")
    assert event_status == esty.complete


def test_execution_status_nondefault_event():
    try:
        event = produce_event()
    except dpctl.SyclQueueCreationError:
        pytest.skip("OpenCL CPU queue could not be created")
    try:
        event_status = event.execution_status
    except ValueError:
        pytest.fail("Failed to get an event status")
    assert type(event_status) is esty
    wl = event.get_wait_list()
    assert type(wl) is list


def test_backend():
    try:
        dpctl.SyclEvent().backend
    except ValueError:
        pytest.fail("Failed to get backend from event")
    try:
        event = produce_event()
    except dpctl.SyclQueueCreationError:
        pytest.skip("OpenCL CPU queue could not be created")
    try:
        event.backend
    except ValueError:
        pytest.fail("Failed to get backend from event")


@pytest.mark.skip(reason="event::get_wait_list() method returns wrong result")
def test_get_wait_list():
    if has_cpu():
        oclSrc = "                                                             \
            kernel void add_k(global float* a) {                               \
                size_t index = get_global_id(0);                               \
                a[index] = a[index] + 1;                                       \
            }                                                                  \
            kernel void sqrt_k(global float* a) {                              \
                size_t index = get_global_id(0);                               \
                a[index] = sqrt(a[index]);                                     \
            }                                                                  \
            kernel void sin_k(global float* a) {                               \
                size_t index = get_global_id(0);                               \
                a[index] = sin(a[index]);                                      \
            }"
        q = dpctl.SyclQueue("opencl:cpu")
        prog = dpctl_prog.create_program_from_source(q, oclSrc)
        addKernel = prog.get_sycl_kernel("add_k")
        sqrtKernel = prog.get_sycl_kernel("sqrt_k")
        sinKernel = prog.get_sycl_kernel("sin_k")

        bufBytes = 1024 * np.dtype("f").itemsize
        abuf = dpctl_mem.MemoryUSMShared(bufBytes, queue=q)
        a = np.ndarray((1024), buffer=abuf, dtype="f")
        a[:] = np.arange(1024)
        args = []

        args.append(a.base)
        r = [1024]
        ev_1 = q.submit(addKernel, args, r)
        ev_2 = q.submit(sqrtKernel, args, r, dEvents=[ev_1])
        ev_3 = q.submit(sinKernel, args, r, dEvents=[ev_2])

        try:
            wait_list = ev_3.get_wait_list()
        except ValueError:
            pytest.fail("Failed to get a list of waiting events from SyclEvent")
        assert len(wait_list)


def test_profiling_info():
    if has_cpu():
        event = produce_event(profiling=True)
        assert event.profiling_info_submit
        assert event.profiling_info_start
        assert event.profiling_info_end
    else:
        pytest.skip("No OpenCL CPU queues available")


def test_sycl_timer():
    try:
        q = dpctl.SyclQueue(property="enable_profiling")
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue creation of default device failed")
    timer = dpctl.SyclTimer()
    m1 = dpctl_mem.MemoryUSMDevice(256 * 1024, queue=q)
    m2 = dpctl_mem.MemoryUSMDevice(256 * 1024, queue=q)
    with timer(q):
        # device task
        m1.copy_from_device(m2)
        # host task
        [x ** 2 for x in range(1024)]
    host_dt, device_dt = timer.dt
    assert host_dt > device_dt
    q_no_profiling = dpctl.SyclQueue()
    assert q_no_profiling.has_enable_profiling is False
    with pytest.raises(ValueError):
        timer(queue=q_no_profiling)
    with pytest.raises(TypeError):
        timer(queue=None)


def test_event_capsule():
    ev = dpctl.SyclEvent()
    cap1 = ev._get_capsule()
    cap2 = ev._get_capsule()
    del ev
    del cap1  # test deleter
    del cap2


def test_addressof_ref():
    ev = dpctl.SyclEvent()
    ref = ev.addressof_ref()
    assert type(ref) is int


def test_cpython_api():
    import ctypes
    import sys

    ev = dpctl.SyclEvent()
    mod = sys.modules[ev.__class__.__module__]
    # get capsule storign get_event_ref function ptr
    ev_ref_fn_cap = mod.__pyx_capi__["get_event_ref"]
    # construct Python callable to invoke "get_event_ref"
    cap_ptr_fn = ctypes.pythonapi.PyCapsule_GetPointer
    cap_ptr_fn.restype = ctypes.c_void_p
    cap_ptr_fn.argtypes = [ctypes.py_object, ctypes.c_char_p]
    ev_ref_fn_ptr = cap_ptr_fn(
        ev_ref_fn_cap, b"DPCTLSyclEventRef (struct PySyclEventObject *)"
    )
    callable_maker = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object)
    get_event_ref_fn = callable_maker(ev_ref_fn_ptr)

    r2 = ev.addressof_ref()
    r1 = get_event_ref_fn(ev)
    assert r1 == r2
