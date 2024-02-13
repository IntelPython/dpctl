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

""" This file declares the SyclPlatform extension type and
    SYCL platform-related helper functions.
"""

from libcpp cimport bool

from ._backend cimport DPCTLSyclDeviceSelectorRef, DPCTLSyclPlatformRef


cdef class _SyclPlatform:
    ''' A helper metaclass to abstract a ``sycl::platform`` instance.
    '''
    cdef DPCTLSyclPlatformRef _platform_ref
    cdef const char *_vendor
    cdef const char *_name
    cdef const char *_version


cdef class SyclPlatform(_SyclPlatform):
    @staticmethod
    cdef SyclPlatform _create(DPCTLSyclPlatformRef dref)
    cdef int _init_from_cstring(self, const char *string)
    cdef int _init_from_selector(self, DPCTLSyclDeviceSelectorRef DSRef)
    cdef int _init_from__SyclPlatform(self, _SyclPlatform other)
    cdef DPCTLSyclPlatformRef get_platform_ref(self)
    cdef bool equals(self, SyclPlatform)


cpdef list get_platforms()
