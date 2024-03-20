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

""" Defines unit test cases for the SyclQueue class.
"""

import ctypes
import sys

import pytest
from helper import create_invalid_capsule

import dpctl


def test_standard_selectors(device_selector, check):
    """
    Tests if the standard SYCL device_selectors are able to select a device.
    """
    try:
        device = device_selector()
        if device.default_selector_score < 0:
            pytest.skip()
        q = dpctl.SyclQueue(device)
        check(q.get_sycl_device())
    except dpctl.SyclDeviceCreationError:
        pytest.skip()


def test_valid_filter_selectors(valid_filter, check):
    """
    Tests if we can create a SyclDevice using a supported filter selector
    string.
    """
    device = None
    try:
        q = dpctl.SyclQueue(valid_filter)
        device = q.get_sycl_device()
        assert q.is_in_order is False
        q2 = dpctl.SyclQueue(valid_filter, property="in_order")
        # assert device == q2.get_sycl_device()
        assert q2.is_in_order is True
    except dpctl.SyclQueueCreationError:
        pytest.skip("Failed to create device with supported filter")
    check(device)


def test_invalid_filter_selectors(invalid_filter):
    """
    An invalid filter string should always be caught and a
    SyclQueueCreationError raised.
    """
    expected_exception = (
        dpctl.SyclQueueCreationError
        if isinstance(invalid_filter, str)
        else TypeError
    )
    with pytest.raises(expected_exception):
        dpctl.SyclQueue(invalid_filter)


def test_unexpected_keyword():
    """
    An unexpected keyword use raises TypeError.
    """
    with pytest.raises(TypeError):
        dpctl.SyclQueue(device="cpu")


def test_context_not_equals():
    try:
        gpuQ = dpctl.SyclQueue("gpu")
    except dpctl.SyclQueueCreationError:
        pytest.skip()
    ctx_gpu = gpuQ.get_sycl_context()
    try:
        cpuQ = dpctl.SyclQueue("cpu")
    except dpctl.SyclQueueCreationError:
        pytest.skip()
    ctx_cpu = cpuQ.get_sycl_context()
    assert ctx_cpu != ctx_gpu
    assert hash(ctx_cpu) != hash(ctx_gpu)
    assert gpuQ != cpuQ
    assert hash(cpuQ) != hash(gpuQ)


def test_context_equals():
    try:
        gpuQ1 = dpctl.SyclQueue("gpu")
        gpuQ0 = dpctl.SyclQueue("gpu")
    except dpctl.SyclQueueCreationError:
        pytest.skip()
    ctx0 = gpuQ0.get_sycl_context()
    ctx1 = gpuQ1.get_sycl_context()
    assert ctx0 == ctx1
    assert hash(ctx0) == hash(ctx1)


def test_has_enable_profiling():
    try:
        q = dpctl.SyclQueue(property="enable_profiling")
    except dpctl.SyclQueueCreationError:
        pytest.skip("Could not create queue with profiling property enabled")
    assert q.has_enable_profiling


def test_hashing_of_queue():
    """
    Test that a :class:`dpctl.SyclQueue` object can be used as
    a dictionary key.

    """
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default-constructed queue could not be created")
    queue_dict = {q: "default_queue"}
    assert queue_dict


def test_channeling_device_properties(capsys):
    try:
        q = dpctl.SyclQueue()
        dev = q.sycl_device
    except dpctl.SyclQueueCreationError:
        pytest.skip("Failed to create device from default selector")

    q.print_device_info()  # should execute without raising
    q_captured = capsys.readouterr()
    q_output = q_captured.out
    dev.print_device_info()
    d_captured = capsys.readouterr()
    d_output = d_captured.out
    assert q_output, "No output captured"
    assert q_output == d_output, "Mismatch in print_device_info"
    assert q_captured.err == "" and d_captured.err == ""
    for pr in ["backend", "name", "driver_version"]:
        assert getattr(q, pr) == getattr(
            dev, pr
        ), "Mismatch found for property {}".format(pr)


def test_queue_submit_barrier(valid_filter):
    try:
        q = dpctl.SyclQueue(valid_filter)
    except dpctl.SyclQueueCreationError:
        pytest.skip("Failed to create device with supported filter")
    ev1 = q.submit_barrier()
    ev2 = q.submit_barrier()
    ev3 = q.submit_barrier([ev1, ev2])
    ev3.wait()
    ev1.wait()
    ev2.wait()
    with pytest.raises(TypeError):
        q.submit_barrier(range(3))


def test_queue__repr__():
    try:
        q1 = dpctl.SyclQueue(property=0)
    except dpctl.SyclQueueCreationError:
        pytest.skip()
    r1 = q1.__repr__()
    assert type(r1) is str

    try:
        q2 = dpctl.SyclQueue(property="in_order")
    except dpctl.SyclQueueCreationError:
        pytest.skip()
    r2 = q2.__repr__()
    assert type(r2) is str

    try:
        q3 = dpctl.SyclQueue(property="enable_profiling")
    except dpctl.SyclQueueCreationError:
        pytest.skip()
    r3 = q3.__repr__()
    assert type(r3) is str

    try:
        q4 = dpctl.SyclQueue(property="default")
    except dpctl.SyclQueueCreationError:
        pytest.skip()
    r4 = q4.__repr__()
    assert type(r4) is str

    try:
        q5 = dpctl.SyclQueue(property=["in_order", "enable_profiling", 0])
    except dpctl.SyclQueueCreationError:
        pytest.skip()
    r5 = q5.__repr__()
    assert type(r5) is str


def test_queue_invalid_property():
    with pytest.raises(ValueError):
        dpctl.SyclQueue(property=4.5)
    with pytest.raises(ValueError):
        dpctl.SyclQueue(property=["abc", tuple()])


def test_queue_capsule():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Can not defaul-construct SyclQueue")
    cap = q._get_capsule()
    cap2 = q._get_capsule()
    q2 = dpctl.SyclQueue(cap)
    assert q == q2
    del cap2  # call deleter on non-renamed capsule
    assert q2 != []  # compare with other types


def test_queue_ctor():
    # construct from device
    try:
        d = dpctl.SyclDevice()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("Could not create default device")
    q = dpctl.SyclQueue(d)
    assert q.sycl_device == d

    ctx = dpctl.SyclContext(d)
    q = dpctl.SyclQueue(ctx, d)
    assert q.sycl_context == ctx
    assert q.sycl_device == d


def test_cpython_api_SyclQueue_GetQueueRef():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Can not defaul-construct SyclQueue")
    mod = sys.modules[q.__class__.__module__]
    # get capsule storign SyclQueue_GetQueueRef function ptr
    q_ref_fn_cap = mod.__pyx_capi__["SyclQueue_GetQueueRef"]
    # construct Python callable to invoke "SyclQueue_GetQueueRef"
    cap_ptr_fn = ctypes.pythonapi.PyCapsule_GetPointer
    cap_ptr_fn.restype = ctypes.c_void_p
    cap_ptr_fn.argtypes = [ctypes.py_object, ctypes.c_char_p]
    q_ref_fn_ptr = cap_ptr_fn(
        q_ref_fn_cap, b"DPCTLSyclQueueRef (struct PySyclQueueObject *)"
    )
    callable_maker = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object)
    get_queue_ref_fn = callable_maker(q_ref_fn_ptr)

    r2 = q.addressof_ref()
    r1 = get_queue_ref_fn(q)
    assert r1 == r2


def test_cpython_api_SyclQueue_Make():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Can not defaul-construct SyclQueue")
    mod = sys.modules[q.__class__.__module__]
    # get capsule storing SyclQueue_Make function ptr
    make_SyclQueue_fn_cap = mod.__pyx_capi__["SyclQueue_Make"]
    # construct Python callable to invoke "SyclQueue_Make"
    cap_ptr_fn = ctypes.pythonapi.PyCapsule_GetPointer
    cap_ptr_fn.restype = ctypes.c_void_p
    cap_ptr_fn.argtypes = [ctypes.py_object, ctypes.c_char_p]
    make_SyclQueue_fn_ptr = cap_ptr_fn(
        make_SyclQueue_fn_cap, b"struct PySyclQueueObject *(DPCTLSyclQueueRef)"
    )
    callable_maker = ctypes.PYFUNCTYPE(ctypes.py_object, ctypes.c_void_p)
    make_SyclQueue_fn = callable_maker(make_SyclQueue_fn_ptr)

    q2 = make_SyclQueue_fn(q.addressof_ref())
    assert q.sycl_device == q2.sycl_device
    assert q.sycl_context == q2.sycl_context


def test_constructor_many_arg():
    with pytest.raises(TypeError):
        dpctl.SyclQueue(None, None, None, None)
    with pytest.raises(TypeError):
        dpctl.SyclQueue(None, None)
    try:
        ctx = dpctl.SyclContext()
    except dpctl.SyclContextCreationError:
        pytest.skip()
    with pytest.raises(TypeError):
        dpctl.SyclQueue(ctx, None)
    with pytest.raises(TypeError):
        dpctl.SyclQueue(ctx)


def test_constructor_inconsistent_ctx_dev():
    try:
        q = dpctl.SyclQueue("cpu")
    except dpctl.SyclQueueCreationError:
        pytest.skip("Failed to create CPU queue")
    cpuD = q.sycl_device
    n_eu = cpuD.max_compute_units
    n_half = n_eu // 2
    try:
        d0, d1 = cpuD.create_sub_devices(partition=[n_half, n_eu - n_half])
    except Exception:
        pytest.skip("Could not create CPU sub-devices")
    ctx = dpctl.SyclContext(d0)
    with pytest.raises(dpctl.SyclQueueCreationError):
        dpctl.SyclQueue(ctx, d1)


def test_constructor_invalid_capsule():
    cap = create_invalid_capsule()
    with pytest.raises(TypeError):
        dpctl.SyclQueue(cap)


def test_queue_wait():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Failed to create default queue")
    q.wait()


def test_queue_memops():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Failed to create device with supported filter")
    from dpctl.memory import MemoryUSMDevice

    m1 = MemoryUSMDevice(512, queue=q)
    m2 = MemoryUSMDevice(512, queue=q)
    q.memcpy(m1, m2, 512)
    q.prefetch(m1, 512)
    q.mem_advise(m1, 512, 0)
    with pytest.raises(TypeError):
        q.memcpy(m1, list(), 512)
    with pytest.raises(TypeError):
        q.memcpy(list(), m2, 512)
    with pytest.raises(TypeError):
        q.prefetch(list(), 512)
    with pytest.raises(TypeError):
        q.mem_advise(list(), 512, 0)


@pytest.fixture(scope="session")
def dpctl_cython_extension(tmp_path_factory):
    import os
    import os.path
    import shutil
    import subprocess
    import sys
    import sysconfig

    curr_dir = os.path.dirname(__file__)
    dr = tmp_path_factory.mktemp("_cython_api")
    for fn in ["_cython_api.pyx", "setup_cython_api.py"]:
        shutil.copy(
            src=os.path.join(curr_dir, fn),
            dst=dr,
            follow_symlinks=False,
        )
    res = subprocess.run(
        [sys.executable, "setup_cython_api.py", "build_ext", "--inplace"],
        cwd=dr,
        env=os.environ,
    )
    if res.returncode == 0:
        import glob
        from importlib.util import module_from_spec, spec_from_file_location

        sfx = sysconfig.get_config_vars()["EXT_SUFFIX"]
        pth = glob.glob(os.path.join(dr, "_cython_api*" + sfx))
        if not pth:
            pytest.fail("Cython extension was not built")
        spec = spec_from_file_location("_cython_api", pth[0])
        builder_module = module_from_spec(spec)
        spec.loader.exec_module(builder_module)
        return builder_module
    else:
        pytest.fail("Cython extension could not be built")


def test_cython_api(dpctl_cython_extension):
    try:
        q = dpctl_cython_extension.call_create_from_context_and_devices()
    except (dpctl.SyclDeviceCreationError, dpctl.SyclQueueCreationError):
        pytest.skip()
    try:
        d = dpctl.SyclDevice()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("Default-construction of SyclDevice failed")
    assert q.sycl_device == d
