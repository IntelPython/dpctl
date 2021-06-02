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
# cython: linetrace=True

""" Implements SyclEvent Cython extension type.
"""

import logging

from ._backend cimport DPCTLEvent_Delete, DPCTLEvent_Wait, DPCTLSyclEventRef

__all__ = [
    "SyclEvent",
]

_logger = logging.getLogger(__name__)


cdef class SyclEvent:
    """ Python wrapper class for cl::sycl::event.
    """

    @staticmethod
    cdef SyclEvent _create(DPCTLSyclEventRef eref, list args):
        cdef SyclEvent ret = SyclEvent.__new__(SyclEvent)
        ret._event_ref = eref
        ret._args = args
        return ret

    def __dealloc__(self):
        self.wait()
        DPCTLEvent_Delete(self._event_ref)

    cdef DPCTLSyclEventRef get_event_ref(self):
        """ Returns the DPCTLSyclEventRef pointer for this class.
        """
        return self._event_ref

    cpdef void wait(self):
        DPCTLEvent_Wait(self._event_ref)

    def addressof_ref(self):
        """ Returns the address of the C API DPCTLSyclEventRef pointer as
        a size_t.

        Returns:
            The address of the DPCTLSyclEventRef object used to create this
            SyclEvent cast to a size_t.
        """
        return int(<size_t>self._event_ref)
