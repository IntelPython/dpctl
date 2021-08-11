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

from cpython cimport pycapsule

from ._backend cimport (  # noqa: E211
    DPCTLEvent_Copy,
    DPCTLEvent_Create,
    DPCTLEvent_Delete,
    DPCTLEvent_Wait,
    DPCTLSyclEventRef,
)

__all__ = [
    "SyclEvent",
    "SyclEventRaw",
]

_logger = logging.getLogger(__name__)


cdef api DPCTLSyclEventRef get_event_ref(SyclEvent ev):
    """ C-API function to access opaque event reference from
    Python object of type :class:`dpctl.SyclEvent`.
    """
    return ev.get_event_ref()


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

cdef void _event_capsule_deleter(object o):
    cdef DPCTLSyclEventRef ERef = NULL
    if pycapsule.PyCapsule_IsValid(o, "SyclEventRef"):
        ERef = <DPCTLSyclEventRef> pycapsule.PyCapsule_GetPointer(
            o, "SyclEventRef"
        )
        DPCTLEvent_Delete(ERef)


cdef class _SyclEventRaw:
    """ Python wrapper class for a ``cl::sycl::event``.
    """

    def __dealloc__(self):
        DPCTLEvent_Delete(self._event_ref)


cdef class SyclEventRaw(_SyclEventRaw):
    """ Python wrapper class for a ``cl::sycl::event``.
    """

    @staticmethod
    cdef void _init_helper(_SyclEventRaw event, DPCTLSyclEventRef ERef):
        event._event_ref = ERef

    @staticmethod
    cdef SyclEventRaw _create(DPCTLSyclEventRef eref):
        cdef _SyclEventRaw ret = _SyclEventRaw.__new__(_SyclEventRaw)
        SyclEventRaw._init_helper(ret, eref)
        return SyclEventRaw(ret)

    cdef int _init_event_default(self):
        self._event_ref = DPCTLEvent_Create()
        if (self._event_ref is NULL):
            return -1
        return 0

    cdef int _init_event_from__SyclEventRaw(self, _SyclEventRaw other):
        self._event_ref = DPCTLEvent_Copy(other._event_ref)
        if (self._event_ref is NULL):
            return -1
        return 0

    cdef int _init_event_from_SyclEvent(self, SyclEvent event):
        self._event_ref = DPCTLEvent_Copy(event._event_ref)
        if (self._event_ref is NULL):
            return -1
        return 0

    cdef int _init_event_from_capsule(self, object cap):
        cdef DPCTLSyclEventRef ERef = NULL
        cdef DPCTLSyclEventRef ERef_copy = NULL
        cdef int ret = 0
        if pycapsule.PyCapsule_IsValid(cap, "SyclEventRef"):
            ERef = <DPCTLSyclEventRef> pycapsule.PyCapsule_GetPointer(
                cap, "SyclEventRef"
            )
            if (ERef is NULL):
                return -2
            ret = pycapsule.PyCapsule_SetName(cap, "used_SyclEventRef")
            if (ret):
                return -2
            ERef_copy = DPCTLEvent_Copy(ERef)
            if (ERef_copy is NULL):
                return -3
            self._event_ref = ERef_copy
            return 0
        else:
            return -128

    def __cinit__(self, arg=None):
        cdef int ret = 0
        if arg is None:
            ret = self._init_event_default()
        elif type(arg) is _SyclEventRaw:
            ret = self._init_event_from__SyclEventRaw(<_SyclEventRaw> arg)
        elif isinstance(arg, SyclEvent):
            ret = self._init_event_from_SyclEvent(<SyclEvent> arg)
        elif pycapsule.PyCapsule_IsValid(arg, "SyclEventRef"):
            ret = self._init_event_from_capsule(arg)
        else:
            raise TypeError(
                    "Invalid argument."
                )
        if (ret < 0):
            if (ret == -1):
                raise ValueError("Event failed to be created.")
            elif (ret == -2):
                raise TypeError(
                    "Input capsule {} contains a null pointer or could not be"
                    " renamed".format(arg)
                )
            elif (ret == -3):
                raise ValueError(
                    "Internal Error: Could not create a copy of a sycl event."
                )
            raise ValueError(
                "Unrecognized error code ({}) encountered.".format(ret)
            )

    cdef DPCTLSyclEventRef get_event_ref(self):
        """ Returns the `DPCTLSyclEventRef` pointer for this class.
        """
        return self._event_ref

    cpdef void wait(self):
        DPCTLEvent_Wait(self._event_ref)

    def addressof_ref(self):
        """ Returns the address of the C API `DPCTLSyclEventRef` pointer as
        a size_t.

        Returns:
            The address of the `DPCTLSyclEventRef` object used to create this
            `SyclEvent` cast to a size_t.
        """
        return <size_t>self._event_ref

    def _get_capsule(self):
        cdef DPCTLSyclEventRef ERef = NULL
        ERef = DPCTLEvent_Copy(self._event_ref)
        if (ERef is NULL):
            raise ValueError("SyclEvent copy failed.")
        return pycapsule.PyCapsule_New(
            <void *>ERef,
            "SyclEventRef",
            &_event_capsule_deleter
        )
