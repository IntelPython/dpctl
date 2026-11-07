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
import sysconfig
import textwrap
import threading
import time
import warnings

import numpy as np
import pytest

import dpctl
import dpctl.memory
import dpctl.utils
from dpctl._sycl_queue import (
    _drain_deferred_releases,
    _start_deferred_release_watcher,
    _stop_deferred_release_watcher,
)
from dpctl.utils import _order_manager

_free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


@pytest.fixture
def stopped_release_watcher():
    """Leaves the deferred releases to the test, rather than to dpctl's thread.

    A test that expects an object to still be held, or to be released on the
    thread it runs on, has to be the only one running them.
    """
    _stop_deferred_release_watcher()
    yield
    _start_deferred_release_watcher()


def _queue(**kwargs):
    try:
        return dpctl.SyclQueue(**kwargs)
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created for default-selected device")


def _submit_long_work(q, usm):
    """Copy into `usm` repeatedly, returning the event of the last copy."""
    # the first keep_args_alive on a device builds its kernel, which can take
    # longer than the copies
    q.keep_args_alive(()).wait()
    host_buf = bytearray(usm.nbytes)
    ev = None
    for _ in range(8):
        ev = q.copy_async(usm, host_buf, usm.nbytes, [ev] if ev else None)
    q.keep_args_alive((host_buf,), [ev])
    return ev


def _complete(ev):
    return ev.execution_status == dpctl.event_status_type.complete


class _Sentinel:
    """Appends to `released` when it is destroyed."""

    def __init__(self, released):
        self._released = released

    def __del__(self):
        self._released.append(True)


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
    q = _queue()
    _som = dpctl.utils.SequentialOrderManager
    _mngr = _som[q]

    usm = dpctl.memory.MemoryUSMDevice(32 * 1024 * 1024, queue=q)
    copy_ev = _submit_long_work(q, usm)
    cleanup_ev = q.keep_args_alive((usm,), [copy_ev])
    # only the cleanup event is recorded, so waiting on the manager can only
    # wait for the copies through it
    _mngr.add_cleanup_event(cleanup_ev)
    if _mngr.num_cleanup_events == 0:
        if _complete(cleanup_ev):
            pytest.skip("The copies completed before the event was recorded")
        pytest.fail("A pending cleanup event was not recorded")
    _mngr.wait()
    assert _complete(copy_ev)
    assert _mngr.num_cleanup_events == 0
    _som.clear()


def test_order_manager_wait_drops_references(stopped_release_watcher):
    q = _queue()
    _som = dpctl.utils.SequentialOrderManager
    _mngr = _som[q]

    released = []
    args = (_Sentinel(released),)
    ev = q.keep_args_alive(args)
    del args
    ev.wait()
    assert not released, "dpctl's thread is stopped, so nothing has run it"

    # waiting is a point where the references held for completed tasks can be
    # dropped, so it drops them
    _mngr.wait()
    assert released == [True]
    _som.clear()


def test_order_manager_wait_with_release_watcher_running():
    q = _queue()
    _som = dpctl.utils.SequentialOrderManager
    _mngr = _som[q]

    n = 200
    released = []
    for _ in range(n):
        args = (_Sentinel(released),)
        _mngr.add_cleanup_event(q.keep_args_alive(args))
        del args
        # races the thread for the same releases
        _mngr.wait()

    # what the thread took up is released by it rather than by `wait`
    deadline = time.monotonic() + 10
    while len(released) < n and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(released) == n
    _som.clear()


def test_keep_args_alive_on_in_order_queue_waits_for_earlier_work(
    stopped_release_watcher,
):
    q = _queue(property="in_order")

    usm = dpctl.memory.MemoryUSMDevice(32 * 1024 * 1024, queue=q)
    copy_ev = _submit_long_work(q, usm)

    released = []
    args = (_Sentinel(released),)
    # no events: the queue orders the release after the copies
    q.keep_args_alive(args)
    del args

    _drain_deferred_releases()
    if not _complete(copy_ev):
        assert not released
    q.wait()
    _drain_deferred_releases()
    assert released == [True]


def test_failed_allocation_drains_releases(stopped_release_watcher):
    q = _queue()

    released = []
    args = (_Sentinel(released),)
    q.keep_args_alive(args).wait()
    del args
    assert not released

    with pytest.raises(dpctl.memory.USMAllocationError):
        dpctl.memory.MemoryUSMDevice(1 << 62, queue=q)
    assert released == [True]


def test_order_manager_host_task_api():
    q = _queue()
    _som = dpctl.utils.SequentialOrderManager
    _mngr = _som[q]

    # deprecated in the documentation only, as dpnp still calls them
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert isinstance(_mngr.num_host_task_events, int)
        assert isinstance(_mngr.host_task_events, list)
        _mngr.add_event_pair(dpctl.SyclEvent(), dpctl.SyclEvent())
        _mngr.add_event_pair([dpctl.SyclEvent()], dpctl.SyclEvent())
        _mngr.add_event_pair(dpctl.SyclEvent(), [dpctl.SyclEvent()])
    _mngr.wait()
    _som.clear()


@pytest.mark.skipif(
    _free_threaded,
    reason="The thread owning the object runs its __del__",
)
def test_release_watcher_is_one_named_python_thread():
    q = _queue()

    ran_on = []

    class Records:
        def __del__(self):
            ran_on.append(threading.current_thread())

    def watcher_threads():
        main = threading.main_thread()
        return [t for t in ran_on if t is not main]

    # the caller drops the last reference instead if the release runs before
    # the call returns, so a release is repeated until two ran on the thread
    for _ in range(50):
        if len(watcher_threads()) >= 2:
            break
        n = len(ran_on)
        q.keep_args_alive((Records(),)).wait()
        deadline = time.monotonic() + 5
        while len(ran_on) == n and time.monotonic() < deadline:
            time.sleep(0.01)

    seen = watcher_threads()
    if len(seen) < 2:
        pytest.skip("The releases did not run on dpctl's thread")
    assert seen[0] is seen[1]
    assert seen[0].name == "dpctl-deferred-releases"


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


_RELEASED_AFTER_RESTART = """
released = []


class Sentinel:
    def __del__(self):
        released.append(True)


# deferred while the thread the stop let go of is still blocked
args = (Sentinel(),)
q.keep_args_alive(args).wait()
del args

deadline = time.monotonic() + 5
while not released and time.monotonic() < deadline:
    time.sleep(0.01)
print("released" if released else "held")
unblock.set()
"""


def test_start_after_timed_out_stop_runs_releases():
    # the stop gives up on the blocking release after a second
    out = _run_python(
        _BLOCKING_RELEASE + """
_stop_deferred_release_watcher()
_start_deferred_release_watcher()
""" + _RELEASED_AFTER_RESTART,
        timeout=30,
    )
    assert out == "released"


def test_start_during_timed_out_stop_runs_releases():
    out = _run_python(
        _BLOCKING_RELEASE + """
stopper = threading.Thread(target=_stop_deferred_release_watcher)
stopper.start()
time.sleep(0.1)
_start_deferred_release_watcher()
# the stop gives up on the blocking release after a second
stopper.join(10)
""" + _RELEASED_AFTER_RESTART,
        timeout=30,
    )
    assert out == "released"


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


def _in_order_queue(*args, cls=dpctl.SyclQueue):
    try:
        return cls(*args, property="in_order")
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created for default-selected device")


@pytest.fixture(params=[True, False], ids=["last_event", "added_events"])
def use_last_event(request, monkeypatch):
    if request.param and not dpctl.SyclQueue.is_get_last_event_available():
        pytest.skip("sycl_ext_oneapi_in_order_queue_events not available")
    monkeypatch.setattr(_order_manager, "_use_last_event", request.param)
    return request.param


def test_in_order_order_manager():
    q = _in_order_queue()
    _som = dpctl.utils.SequentialOrderManager
    _mngr = _som[q]
    assert _som[q] is _mngr
    assert isinstance(_mngr, _order_manager._InOrderManager)
    assert isinstance(_mngr.submitted_events, list)
    assert isinstance(_mngr.cleanup_events, list)
    _mngr.add_event(dpctl.SyclEvent())
    _mngr.add_event([dpctl.SyclEvent(), dpctl.SyclEvent()])
    _mngr.add_cleanup_event(dpctl.SyclEvent())
    _mngr.add_cleanup_event([dpctl.SyclEvent(), dpctl.SyclEvent()])
    # complete events are not kept
    assert _mngr.num_cleanup_events == 0
    _mngr.wait()
    cpy = _mngr.__copy__()
    assert isinstance(cpy, _order_manager._InOrderManager)
    assert cpy.num_cleanup_events == 0
    _som.clear()
    del cpy


@pytest.mark.parametrize("in_order", [True, False])
def test_order_manager_add_non_event(in_order):
    q = _in_order_queue() if in_order else dpctl.SyclQueue()
    _mngr = dpctl.utils.SequentialOrderManager[q]
    with pytest.raises(TypeError):
        _mngr.add_event(None)
    with pytest.raises(TypeError):
        _mngr.add_cleanup_event([None])
    dpctl.utils.SequentialOrderManager.clear()


def test_in_order_order_manager_host_task_api():
    q = _in_order_queue()
    _som = dpctl.utils.SequentialOrderManager
    _mngr = _som[q]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _mngr.add_event_pair(dpctl.SyclEvent(), dpctl.SyclEvent())
        _mngr.add_event_pair([dpctl.SyclEvent()], dpctl.SyclEvent())
        assert _mngr.host_task_events == []
        assert _mngr.num_host_task_events == 0
    _som.clear()


def test_in_order_order_manager_wait_drops_references(stopped_release_watcher):
    q = _in_order_queue()
    _som = dpctl.utils.SequentialOrderManager
    _mngr = _som[q]

    released = []
    args = (_Sentinel(released),)
    ev = q.keep_args_alive(args)
    del args
    ev.wait()
    assert not released, "dpctl's thread is stopped, so nothing has run it"

    _mngr.wait()
    assert released == [True]
    _som.clear()


def test_in_order_order_manager_submitted_events(use_last_event):
    q = _in_order_queue()
    _mngr = dpctl.utils.SequentialOrderManager[q]
    buf = dpctl.memory.MemoryUSMDevice(1024, queue=q)
    # work which is not added to the order manager
    q.memset_async(buf, 1, 1024)
    if use_last_event:
        assert _mngr.num_submitted_events == 1
    else:
        assert _mngr.num_submitted_events == 0
    dpctl.utils.SequentialOrderManager.clear()


def _fill_in_steps(q, buf, n_fills, mngr=None):
    nbytes = buf.nbytes
    ev = None
    for v in range(1, n_fills + 1):
        ev = q.memset_async(buf, v, nbytes, dEvents=[ev] if ev else None)
        if mngr is not None:
            mngr.add_event(ev)
            mngr.add_cleanup_event(q.keep_args_alive((buf,), [ev]))
    return ev


def test_in_order_order_manager_other_queue_depends():
    if not dpctl.SyclQueue.is_get_last_event_available():
        pytest.skip("sycl_ext_oneapi_in_order_queue_events not available")
    q_in = _in_order_queue()
    q_other = dpctl.SyclQueue(q_in.sycl_context, q_in.sycl_device)
    _mngr = dpctl.utils.SequentialOrderManager[q_in]
    n_fills = 8
    buf = dpctl.memory.MemoryUSMDevice(64 * 1024 * 1024, queue=q_in)
    host = np.empty(buf.nbytes, dtype=np.uint8)
    for _ in range(5):
        q_in.memset(buf, 0, buf.nbytes)
        _fill_in_steps(q_in, buf, n_fills - 1, _mngr)
        # the last fill is not added to the order manager
        q_in.memset_async(buf, n_fills, buf.nbytes)
        q_other.memcpy_async(
            host, buf, buf.nbytes, dEvents=_mngr.submitted_events
        ).wait()
        assert np.all(host == n_fills)
    dpctl.utils.SequentialOrderManager.clear()


def test_in_order_order_manager_other_queue_event(use_last_event):
    q_in = _in_order_queue()
    q_other = dpctl.SyclQueue(q_in.sycl_context, q_in.sycl_device)
    _mngr = dpctl.utils.SequentialOrderManager[q_in]
    n_fills = 8
    buf = dpctl.memory.MemoryUSMDevice(64 * 1024 * 1024, queue=q_in)
    host = np.empty(buf.nbytes, dtype=np.uint8)
    for _ in range(5):
        q_in.memset(buf, 0, buf.nbytes)
        _mngr.add_event(_fill_in_steps(q_other, buf, n_fills))
        q_in.memcpy_async(
            host, buf, buf.nbytes, dEvents=_mngr.submitted_events
        ).wait()
        assert np.all(host == n_fills)
        q_other.wait()
    dpctl.utils.SequentialOrderManager.clear()


class _WaitCountingQueue(dpctl.SyclQueue):
    def __init__(self, *args, **kwargs):
        self.num_waits = 0

    def wait(self):
        self.num_waits += 1
        super().wait()


def test_in_order_order_manager_waits_on_queue():
    q = _in_order_queue(cls=_WaitCountingQueue)
    _som = dpctl.utils.SequentialOrderManager
    _mngr = _som[q]
    _mngr.wait()
    assert q.num_waits == 1
    cpy = _mngr.__copy__()
    del cpy
    assert q.num_waits == 2
    del _mngr
    # clear() runs at exit, and must wait on in-order queues as well
    _som.clear()
    assert q.num_waits > 2
