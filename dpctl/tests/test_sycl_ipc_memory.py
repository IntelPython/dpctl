#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2026 Intel Corporation
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

"""Tests for IPC memory handle functionality."""

import gc

import pytest

import dpctl
from dpctl.memory import (
    MemoryIPCDevice,
    MemoryUSMDevice,
    MemoryUSMHost,
    MemoryUSMShared,
    SyclIPCCloseMemHandle,
    SyclIPCGetMemHandle,
    SyclIPCOpenMemHandle,
)


def _get_ipc_device():
    """Return a device with IPC memory support, or skip the test."""
    try:
        dev = dpctl.SyclDevice()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("Default device not available")
    if not dev.has_aspect_ext_oneapi_ipc_memory:
        pytest.skip("Device does not support IPC memory")
    return dev


def _get_ipc_queue():
    """Return a queue on a device with IPC memory support."""
    dev = _get_ipc_device()
    return dpctl.SyclQueue(dev)


# ─── Handle creation tests ─────────────────────────────────────────────


class TestIPCHandleCreation:
    def test_get_handle_from_device_memory(self):
        q = _get_ipc_queue()
        mem = MemoryUSMDevice(1024, queue=q)
        handle_bytes = SyclIPCGetMemHandle(mem)
        assert isinstance(handle_bytes, bytes)
        assert len(handle_bytes) > 0

    def test_get_handle_rejects_shared_memory(self):
        q = _get_ipc_queue()
        mem = MemoryUSMShared(1024, queue=q)
        with pytest.raises(TypeError):
            SyclIPCGetMemHandle(mem)

    def test_get_handle_rejects_host_memory(self):
        q = _get_ipc_queue()
        mem = MemoryUSMHost(1024, queue=q)
        with pytest.raises(TypeError):
            SyclIPCGetMemHandle(mem)


# ─── Open / close tests ───────────────────────────────────────────────


class TestIPCOpenClose:
    def test_open_returns_ipc_device_memory(self):
        q = _get_ipc_queue()
        dev = q.sycl_device
        mem = MemoryUSMDevice(4096, queue=q)
        handle_bytes = SyclIPCGetMemHandle(mem)

        mapped = SyclIPCOpenMemHandle(handle_bytes, dev, 4096)
        assert isinstance(mapped, MemoryIPCDevice)
        assert isinstance(mapped, MemoryUSMDevice)
        assert mapped.nbytes == 4096

    def test_close_mapping_nulls_pointer(self):
        q = _get_ipc_queue()
        dev = q.sycl_device
        mem = MemoryUSMDevice(4096, queue=q)
        handle_bytes = SyclIPCGetMemHandle(mem)

        mapped = SyclIPCOpenMemHandle(handle_bytes, dev, 4096)
        assert mapped._pointer != 0

        SyclIPCCloseMemHandle(mapped)
        assert mapped._pointer == 0
        assert mapped.nbytes == 0

    def test_close_mapping_idempotent(self):
        q = _get_ipc_queue()
        dev = q.sycl_device
        mem = MemoryUSMDevice(4096, queue=q)
        handle_bytes = SyclIPCGetMemHandle(mem)

        mapped = SyclIPCOpenMemHandle(handle_bytes, dev, 4096)
        SyclIPCCloseMemHandle(mapped)
        # Second close should be a no-op (ptr is NULL)
        SyclIPCCloseMemHandle(mapped)

    def test_close_mapping_rejects_non_ipc_memory(self):
        q = _get_ipc_queue()
        mem = MemoryUSMDevice(1024, queue=q)
        with pytest.raises(RuntimeError, match="non-IPC"):
            SyclIPCCloseMemHandle(mem)


# ─── Type identity tests ──────────────────────────────────────────────


class TestIPCTypeIdentity:
    def test_normal_memory_is_not_ipc(self):
        q = _get_ipc_queue()
        mem = MemoryUSMDevice(1024, queue=q)
        assert not isinstance(mem, MemoryIPCDevice)

    def test_shared_memory_is_not_ipc(self):
        q = _get_ipc_queue()
        mem = MemoryUSMShared(1024, queue=q)
        assert not isinstance(mem, MemoryIPCDevice)

    def test_host_memory_is_not_ipc(self):
        q = _get_ipc_queue()
        mem = MemoryUSMHost(1024, queue=q)
        assert not isinstance(mem, MemoryIPCDevice)

    def test_ipc_mapped_memory_is_ipc_type(self):
        q = _get_ipc_queue()
        dev = q.sycl_device
        mem = MemoryUSMDevice(4096, queue=q)
        handle_bytes = SyclIPCGetMemHandle(mem)

        mapped = SyclIPCOpenMemHandle(handle_bytes, dev, 4096)
        assert isinstance(mapped, MemoryIPCDevice)

    def test_type_persists_after_close_mapping(self):
        q = _get_ipc_queue()
        dev = q.sycl_device
        mem = MemoryUSMDevice(4096, queue=q)
        handle_bytes = SyclIPCGetMemHandle(mem)

        mapped = SyclIPCOpenMemHandle(handle_bytes, dev, 4096)
        SyclIPCCloseMemHandle(mapped)
        assert isinstance(mapped, MemoryIPCDevice)

    def test_is_closed_property(self):
        q = _get_ipc_queue()
        dev = q.sycl_device
        mem = MemoryUSMDevice(4096, queue=q)
        handle_bytes = SyclIPCGetMemHandle(mem)

        mapped = SyclIPCOpenMemHandle(handle_bytes, dev, 4096)
        assert mapped.is_closed is False

        SyclIPCCloseMemHandle(mapped)
        assert mapped.is_closed is True


# ─── Lifetime / deletion tests ────────────────────────────────────────


class TestIPCLifetime:
    def test_source_memory_outlives_mapped(self):
        """Source memory must stay alive while receiver uses the mapping."""
        q = _get_ipc_queue()
        dev = q.sycl_device
        mem = MemoryUSMDevice(4096, queue=q)
        handle_bytes = SyclIPCGetMemHandle(mem)

        mapped = SyclIPCOpenMemHandle(handle_bytes, dev, 4096)
        assert mapped.nbytes == 4096

        # Source is still alive — mapping is valid
        SyclIPCCloseMemHandle(mapped)
        del mem
        gc.collect()

    def test_del_mapped_memory_does_not_crash(self):
        """Deleting IPC-mapped memory should call CloseHandle, not
        sycl::free, and not crash."""
        q = _get_ipc_queue()
        dev = q.sycl_device
        mem = MemoryUSMDevice(4096, queue=q)
        handle_bytes = SyclIPCGetMemHandle(mem)

        mapped = SyclIPCOpenMemHandle(handle_bytes, dev, 4096)
        assert isinstance(mapped, MemoryIPCDevice)

        # del triggers __dealloc__ → CloseHandle path
        del mapped
        gc.collect()

    def test_del_mapped_memory_after_explicit_close(self):
        """If close_mapping was already called, __dealloc__ should
        be a no-op (ptr is NULL)."""
        q = _get_ipc_queue()
        dev = q.sycl_device
        mem = MemoryUSMDevice(4096, queue=q)
        handle_bytes = SyclIPCGetMemHandle(mem)

        mapped = SyclIPCOpenMemHandle(handle_bytes, dev, 4096)
        SyclIPCCloseMemHandle(mapped)

        # __dealloc__ sees NULL pointer, does nothing
        del mapped
        gc.collect()

    def test_multiple_opens_from_same_handle(self):
        """Each open() produces an independent mapping that must be
        closed independently."""
        q = _get_ipc_queue()
        dev = q.sycl_device
        mem = MemoryUSMDevice(4096, queue=q)
        handle_bytes = SyclIPCGetMemHandle(mem)

        mapped1 = SyclIPCOpenMemHandle(handle_bytes, dev, 4096)
        mapped2 = SyclIPCOpenMemHandle(handle_bytes, dev, 4096)

        assert isinstance(mapped1, MemoryIPCDevice)
        assert isinstance(mapped2, MemoryIPCDevice)

        SyclIPCCloseMemHandle(mapped1)
        # mapped2 is still valid
        assert mapped2._pointer != 0
        SyclIPCCloseMemHandle(mapped2)

    def test_del_mapped_via_gc(self):
        """Full lifecycle: get handle, open, use, let GC collect mapped."""
        q = _get_ipc_queue()
        dev = q.sycl_device
        mem = MemoryUSMDevice(2048, queue=q)
        handle_bytes = SyclIPCGetMemHandle(mem)

        mapped = SyclIPCOpenMemHandle(handle_bytes, dev, 2048)
        assert mapped.nbytes == 2048
        assert isinstance(mapped, MemoryIPCDevice)

        # GC collects mapped → MemoryIPCDevice.__dealloc__ → CloseHandle
        del mapped
        gc.collect()

        # Source is still valid after receiver is gone
        assert mem.nbytes == 2048
