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

""" Defines unit test cases for the SyclEvent class.
"""

import pytest
from helper import create_invalid_capsule

import dpctl
import dpctl.memory as dpctl_mem
import dpctl.program as dpctl_prog
import dpctl.tensor as dpt
from dpctl import event_status_type as esty


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

    n = 1024 * 1024
    a = dpt.arange(n, dtype="i", sycl_queue=q)
    args = [a.usm_data]

    r = [n]
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


def test_event_backend():
    if dpctl.get_num_devices() == 0:
        pytest.skip("No backends are available")
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


def test_get_wait_list():
    try:
        q = dpctl.SyclQueue("opencl:cpu")
    except dpctl.SyclQueueCreationError:
        pytest.skip("Sycl queue for OpenCL gpu device could not be created.")
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
    prog = dpctl_prog.create_program_from_source(q, oclSrc)
    addKernel = prog.get_sycl_kernel("add_k")
    sqrtKernel = prog.get_sycl_kernel("sqrt_k")
    sinKernel = prog.get_sycl_kernel("sin_k")

    n = 1024 * 1024
    a = dpt.arange(n, dtype="f", sycl_queue=q)
    args = [a.usm_data]

    r = [n]
    ev_1 = q.submit(addKernel, args, r)
    ev_2 = q.submit(sqrtKernel, args, r, dEvents=[ev_1])
    ev_3 = q.submit(sinKernel, args, r, dEvents=[ev_2])

    try:
        wait_list = ev_3.get_wait_list()
    except ValueError:
        pytest.fail("Failed to get a list of waiting events from SyclEvent")
    # FIXME: Due to an issue in underlying runtime the list returns is always
    #         empty. The proper expectation is `assert len(wait_list) > 0`
    assert len(wait_list) >= 0


def test_profiling_info():
    try:
        event = produce_event(profiling=True)
    except dpctl.SyclQueueCreationError:
        pytest.skip("No OpenCL CPU queues available")
    assert type(event.profiling_info_submit) is int
    assert type(event.profiling_info_start) is int
    assert type(event.profiling_info_end) is int


def test_sycl_timer():
    try:
        q = dpctl.SyclQueue(property="enable_profiling")
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue creation of default device failed")
    timer = dpctl.SyclTimer()
    m1 = dpctl_mem.MemoryUSMDevice(1024 * 1024, queue=q)
    m2 = dpctl_mem.MemoryUSMDevice(1024 * 1024, queue=q)
    with timer(q):
        # device task
        m1.copy_from_device(m2)
        # host operation
        [x**2 for x in range(128 * 1024)]
    elapsed = timer.dt
    host_dt, device_dt = elapsed
    assert isinstance(repr(elapsed), str)
    assert isinstance(str(elapsed), str)
    assert host_dt == elapsed.host_dt
    assert device_dt == elapsed.device_dt
    assert host_dt > device_dt or (host_dt > 0 and device_dt >= 0)
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


def test_event_invalid_capsule():
    cap = create_invalid_capsule()
    with pytest.raises(TypeError):
        dpctl.SyclEvent(cap)


def test_addressof_ref():
    ev = dpctl.SyclEvent()
    ref = ev.addressof_ref()
    assert type(ref) is int


def test_cpython_api_SyclEvent_GetEventRef():
    import ctypes
    import sys

    ev = dpctl.SyclEvent()
    mod = sys.modules[ev.__class__.__module__]
    # get capsule storign SyclEvent_GetEventRef function ptr
    ev_ref_fn_cap = mod.__pyx_capi__["SyclEvent_GetEventRef"]
    # construct Python callable to invoke "SyclEvent_GetEventRef"
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


def test_cpython_api_SyclEvent_Make():
    import ctypes
    import sys

    ev = dpctl.SyclEvent()
    mod = sys.modules[ev.__class__.__module__]
    # get capsule storing SyclEvent_Make function ptr
    make_e_fn_cap = mod.__pyx_capi__["SyclEvent_Make"]
    # construct Python callable to invoke "SyclDevice_Make"
    cap_ptr_fn = ctypes.pythonapi.PyCapsule_GetPointer
    cap_ptr_fn.restype = ctypes.c_void_p
    cap_ptr_fn.argtypes = [ctypes.py_object, ctypes.c_char_p]
    make_e_fn_ptr = cap_ptr_fn(
        make_e_fn_cap, b"struct PySyclEventObject *(DPCTLSyclEventRef)"
    )
    callable_maker = ctypes.PYFUNCTYPE(ctypes.py_object, ctypes.c_void_p)
    make_e_fn = callable_maker(make_e_fn_ptr)

    ev2 = make_e_fn(ev.addressof_ref())
    assert type(ev) is type(ev2)
