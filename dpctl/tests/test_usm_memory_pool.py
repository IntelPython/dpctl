#                      Data Parallel Control (dpctl)
#
# Copyright 2026 Intel Corporation
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

"""Defines unit test cases for the USM memory pool."""

import os
import subprocess
import sys
import textwrap

import pytest

import dpctl
import dpctl.memory as dpm
from dpctl._sycl_queue import (
    _drain_deferred_releases,
    _start_deferred_release_watcher,
    _stop_deferred_release_watcher,
)

_MEMORY_CLASSES = [dpm.MemoryUSMDevice, dpm.MemoryUSMShared, dpm.MemoryUSMHost]

# sizes not used elsewhere, so that no other block of the same size is
# cached while a test runs
_SMALL = 3 * 1024 + 17
_LARGE = 3 * 1024 * 1024 + 17


@pytest.fixture
def queue():
    try:
        return dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("SyclQueue() failed")


@pytest.fixture
def caching():
    """Caches without a limit for the test, restoring the limit after."""
    limit = dpm.get_cache_limit()
    dpm.set_cache_limit(None)
    yield
    dpm.set_cache_limit(limit)


@pytest.mark.parametrize("cls", _MEMORY_CLASSES)
@pytest.mark.parametrize("nbytes", [_SMALL, _LARGE])
def test_freed_block_is_reused(cls, nbytes, queue, caching):
    m = cls(nbytes, queue=queue)
    ptr = m._pointer
    del m
    m = cls(nbytes, queue=queue)
    assert m._pointer == ptr


def test_block_of_nearby_size_is_reused(queue, caching):
    dpm.empty_cache()
    m = dpm.MemoryUSMDevice(_LARGE, queue=queue)
    ptr = m._pointer
    del m
    m = dpm.MemoryUSMDevice(_LARGE - 4096, queue=queue)
    assert m._pointer == ptr
    assert m.nbytes == _LARGE - 4096


def test_much_larger_block_is_not_reused(queue, caching):
    dpm.empty_cache()
    m = dpm.MemoryUSMDevice(4 * _LARGE, queue=queue)
    ptr = m._pointer
    del m
    m = dpm.MemoryUSMDevice(_LARGE, queue=queue)
    assert m._pointer != ptr


def test_block_is_not_shared_across_usm_kinds(queue, caching):
    m = dpm.MemoryUSMDevice(_SMALL, queue=queue)
    ptr = m._pointer
    del m
    m = dpm.MemoryUSMShared(_SMALL, queue=queue)
    assert m._pointer != ptr
    assert m.get_usm_type() == "shared"


def test_pool_stats(queue, caching):
    dpm.empty_cache()
    before = dpm.get_pool_stats()
    m = dpm.MemoryUSMDevice(_LARGE, queue=queue)
    during = dpm.get_pool_stats()
    assert during["used_bytes"] >= before["used_bytes"] + _LARGE
    del m
    after = dpm.get_pool_stats()
    assert after["cached_bytes"] >= _LARGE
    dpm.empty_cache()
    assert dpm.get_pool_stats()["cached_bytes"] == 0


def test_aligned_allocation_bypasses_pool(queue, caching):
    used = dpm.get_pool_stats()["used_bytes"]
    m = dpm.MemoryUSMDevice(_LARGE, alignment=4096, queue=queue)
    assert m._pointer % 4096 == 0
    assert dpm.get_pool_stats()["used_bytes"] == used


def test_cache_limit(queue, caching):
    assert dpm.get_cache_limit() is None
    m = dpm.MemoryUSMDevice(_LARGE, queue=queue)
    del m
    assert dpm.get_pool_stats()["cached_bytes"] > 0

    dpm.set_cache_limit(0)
    assert dpm.get_cache_limit() == 0
    assert dpm.get_pool_stats()["cached_bytes"] == 0
    m = dpm.MemoryUSMDevice(_LARGE, queue=queue)
    del m
    assert dpm.get_pool_stats()["cached_bytes"] == 0

    dpm.set_cache_limit(1024)
    assert dpm.get_cache_limit() == 1024
    m = dpm.MemoryUSMDevice(_LARGE, queue=queue)
    del m
    assert dpm.get_pool_stats()["cached_bytes"] <= 1024


def test_set_cache_limit_validation():
    with pytest.raises(ValueError):
        dpm.set_cache_limit(-1)
    with pytest.raises(TypeError):
        dpm.set_cache_limit("1")


def test_block_is_cached_only_once_released(queue, caching):
    _stop_deferred_release_watcher()
    try:
        m = dpm.MemoryUSMDevice(_LARGE, queue=queue)
        ptr = m._pointer
        ev = queue.memset_async(m, 0, _LARGE)
        gate = queue.keep_args_alive((m,), [ev])
        del m
        m2 = dpm.MemoryUSMDevice(_LARGE, queue=queue)
        assert m2._pointer != ptr
        del m2
        gate.wait()
        _drain_deferred_releases()
        m3 = dpm.MemoryUSMDevice(_LARGE, queue=queue)
        m4 = dpm.MemoryUSMDevice(_LARGE, queue=queue)
        assert ptr in (m3._pointer, m4._pointer)
    finally:
        _start_deferred_release_watcher()


def _run_python(script, env=None):
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.strip()
    if out.startswith("skip"):
        pytest.skip(out)
    return out


def test_env_var_disables_caching():
    env = dict(os.environ, DPCTL_USM_POOL="0")
    out = _run_python(
        """
        import dpctl.memory as dpm
        print(dpm.get_cache_limit())
        """,
        env=env,
    )
    assert out == "0"


def test_exit_with_cached_and_live_blocks():
    out = _run_python("""
        import dpctl
        import dpctl.memory as dpm
        try:
            q = dpctl.SyclQueue()
        except dpctl.SyclQueueCreationError:
            print("skip: no default queue")
            raise SystemExit(0)
        dpm.set_cache_limit(None)
        live = [dpm.MemoryUSMDevice(1024 * i + 1, queue=q) for i in range(8)]
        for i in range(8):
            dpm.MemoryUSMShared(4096 * i + 1, queue=q)
        print(dpm.get_pool_stats()["cached_bytes"] > 0)
        """)
    assert out == "True"
