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

"""
Demonstrates using the SequentialOrderManager with USM and the Python threading
module.
"""

import concurrent.futures

import numpy as np

import dpctl
import dpctl.memory as dpmem
from dpctl.utils import SequentialOrderManager


def _memset_async(q, usm_buf, fill_byte, om):
    """
    Fill ``usm_buf`` with ``fill_byte`` asynchronously and track in ``om``.

    ``_submit_keep_args_alive`` prevents the buffer and the target from being
    garbage-collected while the device is still reading/writing.
    """
    n = usm_buf.nbytes
    data = np.full(n, fill_byte, dtype=np.uint8)

    comp_ev = q.memcpy_async(usm_buf, data, n, dEvents=om.submitted_events)
    # keep Python objects alive until the copy finishes
    ht_ev = q._submit_keep_args_alive((usm_buf, data), [comp_ev])
    om.add_event_pair(ht_ev, comp_ev)
    return comp_ev


def independent_threads():
    """
    Each thread fills and reads back its own USM buffer independently, with
    each thread operating on separate memory, with separate order managers.
    """
    nbytes = 1024
    q = dpctl.SyclQueue()
    n_threads = 2

    def worker(thread_id):
        om = SequentialOrderManager[q]
        buf = dpmem.MemoryUSMShared(nbytes, queue=q)
        fill_value = thread_id + 1

        _memset_async(q, buf, fill_value, om)

        om.wait()

        arr = np.frombuffer(buf.copy_to_host(), dtype=np.uint8)
        return int(arr[0])

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=n_threads
    ) as executor:
        results = list(executor.map(worker, range(n_threads)))

    assert results == [1, 2], f"Unexpected results: {results}"
    print(f"independent_threads got what we expected: {results}")


def fork_join():
    """
    The main thread allocates, then each child thread creates a device buffer,
    then fills it via ``memcpy_async``, and waits.  After ``thread.join()``,
    the main thread copies each child's buffer into a single shared result
    buffer and verifies the contents.
    """
    nbytes = 1024
    q = dpctl.SyclQueue()
    n_threads = 2
    chunk = nbytes // n_threads

    def child_fill(thread_id):
        child_om = SequentialOrderManager[q]
        fill_val = (thread_id + 1) * 10

        host_data = np.full(chunk, fill_val, dtype=np.uint8)
        usm_data = dpmem.MemoryUSMShared(chunk, queue=q)
        usm_data.copy_from_host(host_data)

        usm_chunk = dpmem.MemoryUSMDevice(chunk, queue=q)

        comp_ev = q.memcpy_async(
            usm_chunk,
            usm_data,
            chunk,
            dEvents=child_om.submitted_events,
        )
        ht_ev = q._submit_keep_args_alive((usm_chunk, usm_data), [comp_ev])
        child_om.add_event_pair(ht_ev, comp_ev)

        child_om.wait()
        return usm_chunk

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=n_threads
    ) as executor:
        child_bufs = list(executor.map(child_fill, range(n_threads)))

    main_om = SequentialOrderManager[q]
    result_parts = []

    for child_buf in child_bufs:
        part = dpmem.MemoryUSMShared(chunk, queue=q)
        comp_ev = q.memcpy_async(
            part, child_buf, chunk, dEvents=main_om.submitted_events
        )
        ht_ev = q._submit_keep_args_alive((part, child_buf), [comp_ev])
        main_om.add_event_pair(ht_ev, comp_ev)
        result_parts.append(part)
    main_om.wait()

    arr = np.concatenate(
        [np.frombuffer(p.copy_to_host(), dtype=np.uint8) for p in result_parts]
    )
    assert np.all(
        arr[0:chunk] == 10
    ), f"Expected all values in first chunk to be 10, got {arr[0:chunk]}"
    assert np.all(
        arr[chunk:] == 20
    ), f"Expected all values in second chunk to be 20, got {arr[chunk:]}"
    print(f"fork-join got what we expected: [{arr[0]}, ..., {arr[chunk]}, ...]")


def explicit_event_passing():
    """
    Each child thread performs an async fill and gives its tracked events.
    The main thread adds those events to its own order manager so that a
    subsequent ``memcpy_async`` depends on the child work completing first.
    """
    nbytes = 1024
    q = dpctl.SyclQueue()
    n_threads = 2

    def child_prepare(thread_id):
        child_om = SequentialOrderManager[q]
        buf = dpmem.MemoryUSMShared(nbytes, queue=q)
        fill_val = (thread_id + 1) * 42  # 42 or 84

        _memset_async(q, buf, fill_val, child_om)

        return buf, child_om.host_task_events, child_om.submitted_events

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=n_threads
    ) as executor:
        futures_results = list(executor.map(child_prepare, range(n_threads)))

    child_buffers = []
    collected_ht_events = []
    collected_comp_events = []
    for buf, ht_events, comp_events in futures_results:
        child_buffers.append(buf)
        collected_ht_events.extend(ht_events)
        collected_comp_events.extend(comp_events)

    main_om = SequentialOrderManager[q]
    main_om.add_event_pair(collected_ht_events, collected_comp_events)

    results = []
    for buf in child_buffers:
        out = dpmem.MemoryUSMShared(nbytes, queue=q)
        comp_ev = q.memcpy_async(
            out, buf, nbytes, dEvents=main_om.submitted_events
        )
        ht_ev = q._submit_keep_args_alive((out, buf), [comp_ev])
        main_om.add_event_pair(ht_ev, comp_ev)
        results.append(out)

    main_om.wait()

    values = [
        int(np.frombuffer(r.copy_to_host(), dtype=np.uint8)[0]) for r in results
    ]
    assert values == [42, 84], f"Unexpected: {values}"
    print(f"explicit_event_passing got what we expected: {values}")


if __name__ == "__main__":
    import _runner as runner

    runner.run_examples(
        "Examples for working with SequentialOrderManager in dpctl.", globals()
    )
