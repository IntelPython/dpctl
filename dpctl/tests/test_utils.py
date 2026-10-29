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

import subprocess
import sys
import textwrap

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


def _run_python(script, timeout):
    """Run `script` in a fresh interpreter and return what it printed.

    Skips the test if the script prints "skip".
    """
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.strip()
    if out.startswith("skip"):
        pytest.skip(out)
    return out


# a release that blocks until `unblock` is set, if dpctl's thread runs it
_BLOCKING_RELEASE = """
import sysconfig
import threading
import time

if sysconfig.get_config_var("Py_GIL_DISABLED"):
    # the thread owning the object runs its __del__, not dpctl's thread
    print("skip: free-threaded build")
    raise SystemExit

import dpctl
from dpctl._sycl_queue import (
    _start_deferred_release_watcher,
    _stop_deferred_release_watcher,
)

entered = threading.Event()
unblock = threading.Event()
main_thread = threading.get_ident()


class Blocks:
    def __del__(self):
        # on a free-threaded build the owning thread runs this instead
        if threading.get_ident() == main_thread:
            return
        entered.set()
        unblock.wait(30)


try:
    q = dpctl.SyclQueue()
except dpctl.SyclQueueCreationError:
    print("skip: no default queue")
    raise SystemExit

args = (Blocks(),)
q.keep_args_alive(args).wait()
del args
if not entered.wait(5):
    print("skip: the release did not run on dpctl's thread")
    raise SystemExit
"""


def test_exit_does_not_wait_out_a_blocking_release():
    # the release blocks for 30 s unless the exit gives up on it
    _run_python(_BLOCKING_RELEASE, timeout=20)


def test_stop_does_not_wait_out_a_blocking_release():
    out = _run_python(
        _BLOCKING_RELEASE + """
started = time.monotonic()
_stop_deferred_release_watcher()
print(time.monotonic() - started)
unblock.set()
""",
        timeout=20,
    )
    assert float(out) < 5


def test_start_during_stop_does_not_deadlock():
    out = _run_python(
        _BLOCKING_RELEASE + """
stopper = threading.Thread(target=_stop_deferred_release_watcher)
stopper.start()
time.sleep(0.1)
starter = threading.Thread(target=_start_deferred_release_watcher)
starter.start()
time.sleep(0.1)
unblock.set()
stopper.join(10)
starter.join(10)
print("hung" if stopper.is_alive() or starter.is_alive() else "ok")
""",
        timeout=30,
    )
    assert out == "ok"


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="Requires fork"
)
def test_keep_args_alive_releases_in_forked_child():
    out = _run_python(
        """
import multiprocessing
import time

import dpctl
from dpctl._sycl_queue import _start_deferred_release_watcher


def work(_):
    released = []

    class Sentinel:
        def __del__(self):
            released.append(True)

    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        return None
    args = (Sentinel(),)
    q.keep_args_alive(args).wait()
    del args
    deadline = time.monotonic() + 5
    while not released and time.monotonic() < deadline:
        time.sleep(0.01)
    return bool(released)


if __name__ == "__main__":
    # the child inherits the state of a running thread, but not the thread
    _start_deferred_release_watcher()
    with multiprocessing.get_context("fork").Pool(2) as pool:
        results = pool.map(work, range(4))
    if None in results:
        print("skip: no default queue in the child")
    else:
        print(sum(results))
""",
        timeout=120,
    )
    assert out == "4"
