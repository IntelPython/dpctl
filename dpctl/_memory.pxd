##===--------------- _memory.pxd - dpctl module --------*- Cython -*-------===##
##
##                      Data Parallel Control (dpCtl)
##
## Copyright 2020 Intel Corporation
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##
##===----------------------------------------------------------------------===##

# distutils: language = c++
# cython: language_level=3

from ._backend cimport DPPLSyclUSMRef
from ._sycl_core cimport SyclQueue


cdef class Memory:
    cdef DPPLSyclUSMRef memory_ptr
    cdef Py_ssize_t nbytes
    cdef SyclQueue queue

    cdef _cinit(self, Py_ssize_t nbytes, ptr_type, SyclQueue queue)
    cdef _getbuffer(self, Py_buffer *buffer, int flags)


cdef class MemoryUSMShared(Memory):
    pass


cdef class MemoryUSMHost(Memory):
    pass


cdef class MemoryUSMDevice(Memory):
    pass
