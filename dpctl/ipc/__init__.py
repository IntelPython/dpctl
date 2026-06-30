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

"""
**Data Parallel Control IPC** provides Python objects for inter-process
communication of SYCL USM memory.

- :class:`IPCMemoryHandle` wraps ``sycl::ext::oneapi::experimental::ipc::memory``
  to export/import USM device pointers across processes.

Requires oneAPI DPC++ compiler >= 2026.1 (with SYCL IPC memory runtime support).
"""

_IPC_MEMORY_AVAILABLE = False
_IPC_MEMORY_ERROR = ""

try:
    from ._ipc_memory import IPCMemoryHandle
    _IPC_MEMORY_AVAILABLE = True
except ImportError as _e:
    _IPC_MEMORY_ERROR = (
        "dpctl.ipc.IPCMemoryHandle is not available. "
        "SYCL IPC memory support was not detected at build time. "
        "This requires oneAPI DPC++ compiler >= 2026.0 with "
        "sycl::ext::oneapi::experimental::ipc::memory runtime support. "
        f"(Import error: {_e})"
    )

    class IPCMemoryHandle:
        """Placeholder for unavailable IPC memory support."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(_IPC_MEMORY_ERROR)

        @staticmethod
        def open(*args, **kwargs):
            raise RuntimeError(_IPC_MEMORY_ERROR)

        @staticmethod
        def close_mapping(*args, **kwargs):
            raise RuntimeError(_IPC_MEMORY_ERROR)


def is_ipc_memory_supported():
    """Return True if IPC memory is supported by the current SYCL runtime.

    Returns:
        bool: True if IPCMemoryHandle can be used, False otherwise.
    """
    return _IPC_MEMORY_AVAILABLE


def check_ipc_memory_support():
    """Raise RuntimeError if IPC memory is not supported.

    Use this for early failure at application startup.

    Raises:
        RuntimeError: if the SYCL IPC memory API is not available.
    """
    if not _IPC_MEMORY_AVAILABLE:
        raise RuntimeError(_IPC_MEMORY_ERROR)


__all__ = [
    "IPCMemoryHandle",
    "is_ipc_memory_supported",
    "check_ipc_memory_support",
]
