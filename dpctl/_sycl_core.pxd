#                      Data Parallel Control (dpCtl)
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

""" This file declares the extension types and functions for the Cython API
    implemented in sycl_core.pyx.
"""

from ._backend cimport *


cdef class SyclContext:
    ''' Wrapper class for a Sycl Context
    '''
    cdef DPCTLSyclContextRef _ctxt_ref

    @staticmethod
    cdef SyclContext _create (DPCTLSyclContextRef ctxt)
    cpdef bool equals (self, SyclContext ctxt)
    cdef DPCTLSyclContextRef get_context_ref (self)


cdef class SyclEvent:
    ''' Wrapper class for a Sycl Event
    '''
    cdef  DPCTLSyclEventRef _event_ref
    cdef list _args

    @staticmethod
    cdef  SyclEvent _create (DPCTLSyclEventRef e, list args)
    cdef  DPCTLSyclEventRef get_event_ref (self)
    cpdef void wait (self)
