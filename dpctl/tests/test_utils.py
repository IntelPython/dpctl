#                      Data Parallel Control (dpctl)
#
# Copyright 2021 Intel Corporation
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

"""Defines unit test cases for utility functions."""

import pytest

import dpctl
import dpctl.memory
import dpctl.utils
from dpctl._sycl_queue import (
    _start_deferred_release_watcher,
    _stop_deferred_release_watcher,
)


@pytest.fixture
def stopped_release_watcher():
    """Leaves the deferred releases to the test, rather than to dpctl's thread.

    A test that expects an object to still be held, or to be released on the
    thread it runs on, has to be the only one running them.
    """
    _stop_deferred_release_watcher()
    yield
    _start_deferred_release_watcher()


@pytest.mark.filterwarnings("ignore:.*:RuntimeWarning")
def test_onetrace_enabled():
    import os

    v_name = "PTI_ENABLE_COLLECTION"
    v_v = os.getenv(v_name, None)
    with dpctl.utils.onetrace_enabled():
        assert os.getenv(v_name, None) == "1"
    assert os.getenv(v_name, None) == v_v


def test_intel_device_info():
    try:
        d = dpctl.select_default_device()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("Default device could not be created")
    descr = dpctl.utils.intel_device_info(d)
    assert isinstance(descr, dict)
    assert ("device_id" in descr) or not descr
    allowed_names = [
        "device_id",
        "gpu_slices",
        "gpu_eu_count",
        "gpu_eu_simd_width",
        "gpu_hw_threads_per_eu",
        "gpu_subslices_per_slice",
        "gpu_eu_count_per_subslice",
        "max_mem_bandwidth",
        "free_memory",
        "memory_clock_rate",
        "memory_bus_width",
    ]
    for descriptor_name in descr.keys():
        test = descriptor_name in allowed_names
        err_msg = f"Key '{descriptor_name}' is not recognized"
        assert test, err_msg


def test_intel_device_info_validation():
    invalid_device = {}
    with pytest.raises(TypeError):
        dpctl.utils.intel_device_info(invalid_device)


def test_order_manager():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created for default-selected device")
    _som = dpctl.utils.SequentialOrderManager
    _mngr = _som[q]
    assert isinstance(_mngr.num_submitted_events, int)
    assert isinstance(_mngr.submitted_events, list)
    assert isinstance(_mngr.num_cleanup_events, int)
    assert isinstance(_mngr.cleanup_events, list)
    _mngr.add_event(dpctl.SyclEvent())
    _mngr.add_event([dpctl.SyclEvent(), dpctl.SyclEvent()])
    _mngr.add_cleanup_event(dpctl.SyclEvent())
    _mngr.add_cleanup_event([dpctl.SyclEvent(), dpctl.SyclEvent()])
    _mngr.wait()
    cpy = _mngr.__copy__()
    _som.clear()
    del cpy

    try:
        _passed = False
        _som[None]
    except TypeError:
        _passed = True
    finally:
        assert _passed


def test_order_manager_waits_for_cleanup_events():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created for default-selected device")
    _som = dpctl.utils.SequentialOrderManager
    _mngr = _som[q]

    n_bytes = 4 * 1024 * 1024
    host_buf = bytearray(n_bytes)
    usm = dpctl.memory.MemoryUSMDevice(n_bytes, queue=q)

    copy_ev = q.copy_async(usm, host_buf, n_bytes)
    cleanup_ev = q.keep_args_alive((usm,), [copy_ev])
    # only the cleanup event is recorded, so waiting on the manager can only
    # wait for the copy through it
    _mngr.add_cleanup_event(cleanup_ev)
    _mngr.wait()
    assert copy_ev.execution_status == dpctl.event_status_type.complete
    assert _mngr.num_cleanup_events == 0
    _som.clear()


def test_order_manager_wait_drops_references(stopped_release_watcher):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created for default-selected device")
    _som = dpctl.utils.SequentialOrderManager
    _mngr = _som[q]

    released = []

    class Sentinel:
        def __del__(self):
            released.append(True)

    args = (Sentinel(),)
    ev = q.keep_args_alive(args)
    del args
    ev.wait()
    assert not released, "dpctl's thread is stopped, so nothing has run it"

    # waiting is a point where the references held for completed tasks can be
    # dropped, so it drops them
    _mngr.wait()
    assert released == [True]
    _som.clear()


def test_order_manager_deprecated_host_task_api():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created for default-selected device")
    _som = dpctl.utils.SequentialOrderManager
    _mngr = _som[q]

    with pytest.warns(DeprecationWarning):
        assert isinstance(_mngr.num_host_task_events, int)
    with pytest.warns(DeprecationWarning):
        assert isinstance(_mngr.host_task_events, list)
    with pytest.warns(DeprecationWarning):
        _mngr.add_event_pair(dpctl.SyclEvent(), dpctl.SyclEvent())
    with pytest.warns(DeprecationWarning):
        _mngr.add_event_pair([dpctl.SyclEvent()], dpctl.SyclEvent())
    with pytest.warns(DeprecationWarning):
        _mngr.add_event_pair(dpctl.SyclEvent(), [dpctl.SyclEvent()])
    _mngr.wait()
    _som.clear()
