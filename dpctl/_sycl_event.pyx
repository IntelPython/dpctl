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
from libc.stdint cimport uint64_t
import collections.abc

from ._backend cimport (  # noqa: E211
    DPCTLEvent_Copy,
    DPCTLEvent_Create,
    DPCTLEvent_Delete,
    DPCTLEvent_GetBackend,
    DPCTLEvent_GetCommandExecutionStatus,
    DPCTLEvent_GetProfilingInfoEnd,
    DPCTLEvent_GetProfilingInfoStart,
    DPCTLEvent_GetProfilingInfoSubmit,
    DPCTLEvent_GetWaitList,
    DPCTLEvent_Wait,
    DPCTLEvent_WaitAndThrow,
    DPCTLEventVector_Delete,
    DPCTLEventVector_GetAt,
    DPCTLEventVector_Size,
    DPCTLEventVectorRef,
    DPCTLSyclEventRef,
    _backend_type,
    _event_status_type,
)

from .enum_types import backend_type, event_status_type

__all__ = [
    "SyclEvent",
    "SyclEventRaw",
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

cdef void _event_capsule_deleter(object o):
    cdef DPCTLSyclEventRef ERef = NULL
    if pycapsule.PyCapsule_IsValid(o, "SyclEventRef"):
        ERef = <DPCTLSyclEventRef> pycapsule.PyCapsule_GetPointer(
            o, "SyclEventRef"
        )
        DPCTLEvent_Delete(ERef)

cdef void _init_helper(_SyclEventRaw event, DPCTLSyclEventRef ERef):
    "Populate attributes of class from opaque reference ERef"
    event._event_ref = ERef

cdef class _SyclEventRaw:
    """ Data owner for SyclEvent
    """

    def __dealloc__(self):
        DPCTLEvent_Delete(self._event_ref)


cdef class SyclEventRaw(_SyclEventRaw):
    """
    SyclEvent(arg=None)
    Python class representing ``cl::sycl::event``. There are multiple
    ways to create a :class:`dpctl.SyclEventRaw` object:

        - Invoking the constructor with no arguments creates a ready event
          using the default constructor of the ``cl::sycl::event``.

        :Example:
            .. code-block:: python

                import dpctl

                # Create a default SyclEventRaw
                e = dpctl.SyclEventRaw()

        - Invoking the constuctor with a :class:`dpctl.SyclEvent` object
          creates an event by copying the passed object.

        :Example:
            .. code-block:: python

                import dpctl

                # Create a SyclEventRaw by passing SyclEvent
                q = dpctl.SyclQueue()
                e = q.submit_barrier()
                e_r = dpctl.SyclEventRaw(e)

        - Invoking the constuctor with a :class:`dpctl.SyclEventRaw` object
          creates an event by copying the passed object.

        :Example:
            .. code-block:: python

                import dpctl

                # Create a SyclEventRaw by passing SyclEventRaw
                e = dpctl.SyclEventRaw()
                e_r = dpctl.SyclEventRaw(e)

        - Invoking the constuctor with a named ``PyCapsule`` with name
          **"SyclEventRef"** that carries a pointer to a ``sycl::event``
          object. The capsule will be renamed upon successful consumption
          to ensure one-time use. A new named capsule can be constructed by
          using :func:`dpctl.SyclEventRaw._get_capsule` method.

    Args:
        arg (optional): Defaults to ``None``.
            The argument can be a :class:`dpctl.SyclEvent`
            instance, a :class:`dpctl.SyclEventRaw` instance, or a
            named ``PyCapsule`` called **"SyclEventRef"**.

    Raises:
        ValueError: If the :class:`dpctl.SyclEventRaw` object creation failed.
        TypeError: In case of incorrect arguments given to constructors,
                   unexpected types of input arguments, or in the case the input
                   capsule contained a null pointer or could not be renamed.
    """

    @staticmethod
    cdef SyclEventRaw _create(DPCTLSyclEventRef eref):
        """"
        This function calls DPCTLEvent_Delete(eref).

        The user of this function must pass a copy to keep the
        eref argument alive.
        """
        cdef _SyclEventRaw ret = _SyclEventRaw.__new__(_SyclEventRaw)
        _init_helper(ret, eref)
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

    @staticmethod
    cdef void _wait(SyclEventRaw event):
        DPCTLEvent_WaitAndThrow(event._event_ref)

    @staticmethod
    def wait(event):
        """ Waits for a given event or a sequence of events.
        """
        if (isinstance(event, collections.abc.Sequence) and
           all((isinstance(el, SyclEventRaw) for el in event))):
            for e in event:
                SyclEventRaw._wait(e)
        elif isinstance(event, SyclEventRaw):
            SyclEventRaw._wait(event)
        else:
            raise TypeError(
                "The passed argument is not a SyclEventRaw type or "
                "a sequence of such objects"
            )

    def addressof_ref(self):
        """ Returns the address of the C API `DPCTLSyclEventRef` pointer as
        a size_t.

        Returns:
            The address of the `DPCTLSyclEventRef` object used to create this
            `SyclEvent` cast to a size_t.
        """
        return <size_t>self._event_ref

    def _get_capsule(self):
        """
        Returns a copy of the underlying ``cl::sycl::event`` pointer as a void
        pointer inside a named ``PyCapsule`` that has the name
        **SyclEventRef**. The ownership of the pointer inside the capsule is
        passed to the caller, and pointer is deleted when the capsule goes out
        of scope.
        Returns:
            :class:`pycapsule`: A capsule object storing a copy of the
            ``cl::sycl::event`` pointer belonging to thus
            :class:`dpctl.SyclEventRaw` instance.
        Raises:
            ValueError: If the ``DPCTLEvent_Copy`` fails to copy the
                        ``cl::sycl::event`` pointer.
        """
        cdef DPCTLSyclEventRef ERef = NULL
        ERef = DPCTLEvent_Copy(self._event_ref)
        if (ERef is NULL):
            raise ValueError("SyclEvent copy failed.")
        return pycapsule.PyCapsule_New(
            <void *>ERef,
            "SyclEventRef",
            &_event_capsule_deleter
        )

    @property
    def execution_status(self):
        """ Returns the event_status_type enum value for this event.
        """
        cdef _event_status_type ESTy = DPCTLEvent_GetCommandExecutionStatus(
                                        self._event_ref
        )
        if ESTy == _event_status_type._SUBMITTED:
            return event_status_type.submitted
        elif ESTy == _event_status_type._RUNNING:
            return event_status_type.running
        elif ESTy == _event_status_type._COMPLETE:
            return event_status_type.complete
        else:
            raise ValueError("Unknown event status.")

    @property
    def backend(self):
        """Returns the backend_type enum value for the device
        associated with this event.

        Returns:
            backend_type: The backend for the device.
        """
        cdef _backend_type BE = DPCTLEvent_GetBackend(self._event_ref)
        if BE == _backend_type._OPENCL:
            return backend_type.opencl
        elif BE == _backend_type._LEVEL_ZERO:
            return backend_type.level_zero
        elif BE == _backend_type._HOST:
            return backend_type.host
        elif BE == _backend_type._CUDA:
            return backend_type.cuda
        else:
            raise ValueError("Unknown backend type.")

    def get_wait_list(self):
        """
        Returns the list of :class:`dpctl.SyclEventRaw` objects that depend
        on this event.
        """
        cdef DPCTLEventVectorRef EVRef = DPCTLEvent_GetWaitList(
            self.get_event_ref()
        )
        cdef size_t num_events
        cdef size_t i
        cdef DPCTLSyclEventRef ERef
        if (EVRef is NULL):
            raise ValueError("Internal error: NULL event vector encountered")
        num_events = DPCTLEventVector_Size(EVRef)
        events = []
        for i in range(num_events):
            ERef = DPCTLEventVector_GetAt(EVRef, i)
            events.append(SyclEventRaw._create(ERef))
        DPCTLEventVector_Delete(EVRef)
        return events

    def profiling_info_submit(self):
        """
        Returns the 64-bit time value in nanoseconds
        when ``cl::sycl::command_group`` was submitted to the queue.
        """
        cdef uint64_t profiling_info_submit = 0
        profiling_info_submit = DPCTLEvent_GetProfilingInfoSubmit(
                                self._event_ref
        )
        return profiling_info_submit

    @property
    def profiling_info_start(self):
        """
        Returns the 64-bit time value in nanoseconds
        when ``cl::sycl::command_group`` started execution on the device.
        """
        cdef uint64_t profiling_info_start = 0
        profiling_info_start = DPCTLEvent_GetProfilingInfoStart(self._event_ref)
        return profiling_info_start

    @property
    def profiling_info_end(self):
        """
        Returns the 64-bit time value in nanoseconds
        when ``cl::sycl::command_group`` finished execution on the device.
        """
        cdef uint64_t profiling_info_end = 0
        profiling_info_end = DPCTLEvent_GetProfilingInfoEnd(self._event_ref)
        return profiling_info_end
