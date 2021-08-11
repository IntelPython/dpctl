#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2021 Intel Corporation
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

""" This file declares the SyclEvent extension type.
"""

from ._backend cimport DPCTLSyclEventRef


cdef public api class SyclEvent [object PySyclEventObject, type PySyclEventType]:
    ''' Wrapper class for a Sycl Event
    '''
    cdef  DPCTLSyclEventRef _event_ref
    cdef list _args

    @staticmethod
    cdef  SyclEvent _create (DPCTLSyclEventRef e, list args)
    cdef  DPCTLSyclEventRef get_event_ref (self)
    cpdef void wait (self)


cdef class _SyclEventRaw:
    cdef DPCTLSyclEventRef _event_ref


cdef public class SyclEventRaw(_SyclEventRaw) [object PySyclEventRawObject, type PySyclEventRawType]:
    @staticmethod
    cdef SyclEventRaw _create (DPCTLSyclEventRef event)
    @staticmethod
    cdef void _init_helper(_SyclEventRaw event, DPCTLSyclEventRef ERef)
    cdef int _init_event_default(self)
    cdef int _init_event_from__SyclEventRaw(self, _SyclEventRaw other)
    cdef int _init_event_from_SyclEvent(self, SyclEvent event)
    cdef int _init_event_from_capsule(self, object caps)
    cdef  DPCTLSyclEventRef get_event_ref (self)
    cpdef void wait (self)
