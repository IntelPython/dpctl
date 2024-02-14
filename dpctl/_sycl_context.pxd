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

""" This file declares the SyclContext extension type.
"""

from libcpp cimport bool

from ._backend cimport DPCTLSyclContextRef
from ._sycl_device cimport SyclDevice


cdef public api class _SyclContext [
    object Py_SyclContextObject,
    type Py_SyclContextType
]:
    """ Data owner for SyclContext
    """
    cdef DPCTLSyclContextRef _ctxt_ref


cdef public api class SyclContext(_SyclContext) [
    object PySyclContextObject,
    type PySyclContextType
]:
    ''' Wrapper class for a Sycl Context
    '''

    @staticmethod
    cdef SyclContext _create (DPCTLSyclContextRef CRef)
    cdef int _init_context_from__SyclContext(self, _SyclContext other)
    cdef int _init_context_from_one_device(self, SyclDevice device, int props)
    cdef int _init_context_from_devices(self, object devices, int props)
    cdef int _init_context_from_capsule(self, object caps)
    cdef bool equals (self, SyclContext ctxt)
    cdef DPCTLSyclContextRef get_context_ref (self)
