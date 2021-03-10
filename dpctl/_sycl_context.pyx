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

""" Implements SyclContext Cython extension type.
"""

from __future__ import print_function
import logging
from ._backend cimport (
    DPCTLSyclContextRef,
    DPCTLContext_Delete,
    DPCTLContext_AreEq,
)

__all__ = [
    "SyclContext",
]

_logger = logging.getLogger(__name__)


cdef class SyclContext:
    """ Python wrapper class for cl::sycl::context.
    """
    @staticmethod
    cdef SyclContext _create (DPCTLSyclContextRef ctxt):
        cdef SyclContext ret = SyclContext.__new__(SyclContext)
        ret._ctxt_ref = ctxt
        return ret

    def __dealloc__ (self):
        DPCTLContext_Delete(self._ctxt_ref)

    cpdef bool equals (self, SyclContext ctxt):
        """ Returns true if the SyclContext argument has the same _context_ref
            as this SyclContext.
        """
        return DPCTLContext_AreEq(self._ctxt_ref, ctxt.get_context_ref())

    cdef DPCTLSyclContextRef get_context_ref (self):
        return self._ctxt_ref

    def addressof_ref (self):
        """
        Returns the address of the DPCTLSyclContextRef pointer as a size_t.

        Returns:
            The address of the DPCTLSyclContextRef object used to create this
            SyclContext cast to a size_t.
        """
        return int(<size_t>self._ctx_ref)
