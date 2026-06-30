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

# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True

"""Python bindings for SYCL IPC memory
(``sycl::ext::oneapi::experimental::ipc::memory``).

Typical usage for exporting a USM pointer::

    import dpctl
    from dpctl.memory import MemoryUSMDevice
    from dpctl.ipc import IPCMemoryHandle

    # Sender process
    mem = MemoryUSMDevice(1024)
    handle = IPCMemoryHandle(mem)
    raw = handle.to_bytes()   # send *raw* to another process

    # Receiver process (same system, different PID)
    remote_mem = IPCMemoryHandle.open(raw, device)
"""

from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_FromStringAndSize
from libc.stdlib cimport free as libc_free

from dpctl._backend cimport (
    DPCTLSyclContextRef,
    DPCTLSyclDeviceRef,
    DPCTLSyclUSMRef,
    DPCTLIPCMem_GetHandle,
    DPCTLIPCMem_OpenHandle,
    DPCTLIPCMem_CloseHandle,
    DPCTLIPCMem_FreeHandleData,
    DPCTLQueue_GetContext,
    DPCTLQueue_Delete,
    DPCTLContext_Delete,
    DPCTLDevice_Copy,
)

from .._sycl_context cimport SyclContext
from .._sycl_device cimport SyclDevice
from .._sycl_queue cimport SyclQueue
from ..memory._memory cimport _Memory, MemoryUSMDevice

import dpctl


__all__ = ["IPCMemoryHandle"]


cdef class IPCMemoryHandle:
    """Wrapper around a SYCL IPC memory handle.

    Instances are created by passing a :class:`dpctl.memory.MemoryUSMDevice`
    (or any ``_Memory`` subclass backed by a device USM pointer) to the
    constructor. The resulting object exposes :meth:`to_bytes` which
    returns an opaque ``bytes`` payload suitable for inter-process
    transport (e.g. via pickle, ZMQ, shared-memory).

    On the receiving side, call :meth:`IPCMemoryHandle.open` with the
    payload and a target device to obtain a
    :class:`dpctl.memory.MemoryUSMDevice` backed by the IPC-mapped
    memory.

    Parameters
    ----------
    usm_memory : dpctl.memory._Memory
        USM memory object whose device pointer to export.
    context : dpctl.SyclContext, optional
        SYCL context to use. Defaults to the context of *usm_memory*'s
        queue.

    Raises
    ------
    TypeError
        If *usm_memory* is not a ``_Memory`` instance.
    RuntimeError
        If the SYCL runtime fails to produce an IPC handle.
    """

    def __cinit__(self):
        self._handle_bytes = None
        self._ctx = None
        self._closed = False

    def __init__(self, _Memory usm_memory not None, SyclContext context=None):
        cdef DPCTLSyclUSMRef ptr = usm_memory.get_data_ptr()
        if ptr is NULL:
            raise ValueError("USM memory object has a null pointer")

        cdef SyclQueue q = usm_memory.queue
        if context is None:
            context = q.sycl_context

        cdef DPCTLSyclContextRef ctx_ref = context.get_context_ref()
        cdef char *data_out = NULL
        cdef size_t size_out = 0

        cdef int rc = DPCTLIPCMem_GetHandle(ptr, ctx_ref, &data_out, &size_out)
        if rc != 0:
            raise RuntimeError(
                "DPCTLIPCMem_GetHandle failed — device may not support "
                "aspect::ext_oneapi_ipc_memory"
            )

        try:
            self._handle_bytes = PyBytes_FromStringAndSize(data_out,
                                                          <Py_ssize_t>size_out)
        finally:
            DPCTLIPCMem_FreeHandleData(data_out)

        self._ctx = context
        self._closed = False

    def to_bytes(self):
        """Return the raw IPC handle data as ``bytes``.

        The returned object can be pickled, sent over a socket, or
        written to shared memory for another process to consume via
        :meth:`open`.
        """
        if self._closed:
            raise RuntimeError("IPC handle has already been closed")
        return self._handle_bytes

    @staticmethod
    def open(bytes handle_bytes not None,
             SyclDevice device not None,
             SyclContext context=None,
             Py_ssize_t nbytes=0):
        """Open an IPC handle in this process.

        Parameters
        ----------
        handle_bytes : bytes
            Opaque payload from :meth:`to_bytes` (possibly from another
            process).
        device : dpctl.SyclDevice
            Device to map the memory on.
        context : dpctl.SyclContext, optional
            SYCL context to use. Defaults to the default context for
            *device*'s platform.
        nbytes : int, optional
            Byte size of the original allocation. If 0, the size is
            determined by the driver (if supported).

        Returns
        -------
        dpctl.memory.MemoryUSMDevice
            A USM device memory object backed by the IPC-mapped pointer.
            The mapping is closed when the returned object is garbage
            collected.

        Raises
        ------
        RuntimeError
            If the handle cannot be opened.
        """
        cdef const char *raw = PyBytes_AS_STRING(handle_bytes)
        cdef size_t raw_size = <size_t>len(handle_bytes)

        if context is None:
            context = device.sycl_platform.default_context

        cdef DPCTLSyclContextRef ctx_ref = context.get_context_ref()
        cdef DPCTLSyclDeviceRef dev_ref = device.get_device_ref()

        cdef DPCTLSyclUSMRef mapped_ptr = DPCTLIPCMem_OpenHandle(
            raw, raw_size, ctx_ref, dev_ref)
        if mapped_ptr is NULL:
            raise RuntimeError("DPCTLIPCMem_OpenHandle failed")

        # Build a MemoryUSMDevice around the mapped pointer.
        # Use the device-cached queue so dpctl tracks the allocation.
        cdef SyclQueue q
        try:
            q = dpctl.SyclQueue(context, device)
        except Exception:
            DPCTLIPCMem_CloseHandle(mapped_ptr, ctx_ref)
            raise

        # Wrap as MemoryUSMDevice — nbytes must be known by the caller
        # or the driver. We require nbytes > 0 for safety.
        if nbytes <= 0:
            DPCTLIPCMem_CloseHandle(mapped_ptr, ctx_ref)
            raise ValueError("nbytes must be > 0 for IPC open")

        cdef object mem = MemoryUSMDevice.create_from_usm_pointer_size_qref(
            mapped_ptr, nbytes, q.get_queue_ref())
        return mem

    @staticmethod
    def close_mapping(_Memory usm_memory not None,
                      SyclContext context=None):
        """Explicitly close an IPC mapping.

        After calling this, *usm_memory* is invalidated and must not be
        used again. Its destructor will not attempt to free the pointer.

        Parameters
        ----------
        usm_memory : dpctl.memory._Memory
            The memory object returned by :meth:`open`.
        context : dpctl.SyclContext, optional
            Context used when opening. Defaults to the memory's queue
            context.
        """
        cdef DPCTLSyclUSMRef ptr = usm_memory.get_data_ptr()
        if ptr is NULL:
            return

        if context is None:
            context = usm_memory.queue.sycl_context

        cdef DPCTLSyclContextRef ctx_ref = context.get_context_ref()
        DPCTLIPCMem_CloseHandle(ptr, ctx_ref)

        # Prevent the _Memory destructor from calling sycl::free on the
        # now-unmapped pointer. Setting _opaque_ptr to NULL makes
        # __dealloc__ skip OpaqueSmartPtr_Delete.
        usm_memory._opaque_ptr = NULL
        usm_memory._memory_ptr = NULL
        usm_memory.nbytes = 0

    def close(self):
        """Mark this handle as closed (driver resources already released
        during construction via ``put``)."""
        self._closed = True

    def __dealloc__(self):
        self._closed = True

    def __repr__(self):
        return (
            f"IPCMemoryHandle(size={len(self._handle_bytes)}, "
            f"closed={self._closed})"
        )
