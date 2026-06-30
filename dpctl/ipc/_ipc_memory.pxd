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

"""Declarations for the IPCMemoryHandle Cython extension type."""

from .._backend cimport DPCTLSyclContextRef, DPCTLSyclDeviceRef, DPCTLSyclUSMRef
from .._sycl_context cimport SyclContext
from .._sycl_device cimport SyclDevice
from .._sycl_queue cimport SyclQueue
from ..memory._memory cimport _Memory, MemoryUSMDevice


cdef class IPCMemoryHandle:
    cdef bytes _handle_bytes
    cdef SyclContext _ctx
    cdef bint _closed
