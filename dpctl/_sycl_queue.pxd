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

# distutils: language = c++
# cython: language_level=3

""" This file declares the SyclQueue extension type.
"""

from libcpp cimport bool as cpp_bool

from ._backend cimport DPCTLSyclDeviceRef, DPCTLSyclQueueRef, _arg_data_type
from ._sycl_context cimport SyclContext
from ._sycl_device cimport SyclDevice
from ._sycl_event cimport SyclEvent
from .program._program cimport SyclKernel


cdef public api class _SyclQueue [
    object Py_SyclQueueObject, type Py_SyclQueueType
]:
    """ Python data owner class for a sycl::queue.
    """
    cdef DPCTLSyclQueueRef _queue_ref
    cdef SyclContext _context
    cdef SyclDevice _device


cdef public api class SyclQueue (_SyclQueue) [
    object PySyclQueueObject, type PySyclQueueType
]:
    """ Python wrapper class for a sycl::queue.
    """
    cdef int _init_queue_default(self, int)
    cdef int _init_queue_from__SyclQueue(self, _SyclQueue)
    cdef int _init_queue_from_DPCTLSyclDeviceRef(self, DPCTLSyclDeviceRef, int)
    cdef int _init_queue_from_device(self, SyclDevice, int)
    cdef int _init_queue_from_filter_string(self, const char *, int)
    cdef int _init_queue_from_context_and_device(
        self, SyclContext, SyclDevice, int
    )
    cdef int _init_queue_from_capsule(self, object)
    cdef int _populate_args(
        self,
        list args,
        void **kargs,
        _arg_data_type *kargty
    )
    cdef int _populate_range(self, size_t Range[3], list gS, size_t nGS)
    @staticmethod
    cdef  SyclQueue _create(DPCTLSyclQueueRef qref)
    @staticmethod
    cdef  SyclQueue _create_from_context_and_device(
        SyclContext ctx, SyclDevice dev, int props=*
    )
    cdef cpp_bool equals(self, SyclQueue q)
    cpdef SyclContext get_sycl_context(self)
    cpdef SyclDevice get_sycl_device(self)
    cdef  DPCTLSyclQueueRef get_queue_ref(self)
    cpdef SyclEvent _submit_keep_args_alive(
        self,
        object args,
        list dEvents
    )
    cpdef SyclEvent submit_async(
        self,
        SyclKernel kernel,
        list args,
        list gS,
        list lS=*,
        list dEvents=*
    )
    cpdef SyclEvent submit(
        self,
        SyclKernel kernel,
        list args,
        list gS,
        list lS=*,
        list dEvents=*
    )
    cpdef void wait(self)
    cdef DPCTLSyclQueueRef get_queue_ref(self)
    cpdef memcpy(self, dest, src, size_t count)
    cpdef SyclEvent memcpy_async(self, dest, src, size_t count, list dEvents=*)
    cpdef prefetch(self, ptr, size_t count=*)
    cpdef mem_advise(self, ptr, size_t count, int mem)
    cpdef SyclEvent submit_barrier(self, dependent_events=*)
