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

""" Implements SyclContext Cython extension type.
"""

import logging

from cpython cimport pycapsule
from cpython.mem cimport PyMem_Free, PyMem_Malloc

from ._backend cimport (  # noqa: E211
    DPCTLContext_AreEq,
    DPCTLContext_Copy,
    DPCTLContext_Create,
    DPCTLContext_CreateFromDevices,
    DPCTLContext_Delete,
    DPCTLContext_DeviceCount,
    DPCTLContext_GetDevices,
    DPCTLDevice_Copy,
    DPCTLDevice_Delete,
    DPCTLDeviceMgr_GetCachedContext,
    DPCTLDeviceVector_CreateFromArray,
    DPCTLDeviceVector_Delete,
    DPCTLDeviceVector_GetAt,
    DPCTLDeviceVector_Size,
    DPCTLDeviceVectorRef,
    DPCTLSyclContextRef,
    DPCTLSyclDeviceRef,
    error_handler_callback,
)
from ._sycl_device cimport SyclDevice
from ._sycl_queue cimport default_async_error_handler

__all__ = [
    "SyclContext",
]

_logger = logging.getLogger(__name__)

cdef void _context_capsule_deleter(object o):
    cdef DPCTLSyclContextRef CRef = NULL
    if pycapsule.PyCapsule_IsValid(o, "SyclContextRef"):
        CRef = <DPCTLSyclContextRef> pycapsule.PyCapsule_GetPointer(
            o, "SyclContextRef"
        )
        DPCTLContext_Delete(CRef)


cdef class _SyclContext:
    """ Data owner for SyclContext
    """

    def __dealloc__(self):
        DPCTLContext_Delete(self._ctxt_ref)


cdef class SyclContext(_SyclContext):
    """
    SyclContext(arg=None)
    Python class representing ``cl::sycl::context``. There are multiple
    ways to create a :class:`dpctl.SyclContext` object:

        - Invoking the constructor with no arguments creates a context using
          the default selector.

        :Example:
            .. code-block:: python

                import dpctl

                # Create a default SyclContext
                ctx = dpctl.SyclContext()
                print(ctx.get_devices())

        - Invoking the constuctor with a specific filter string that creates a
          context for the device corresponding to the filter string.

        :Example:
            .. code-block:: python

                import dpctl

                # Create SyclContext for a gpu device
                ctx = dpctl.SyclContext("gpu")
                d = ctx.get_devices()[0]
                assert(d.is_gpu)

        - Invoking the constuctor with a :class:`dpctl.SyclDevice` object
          creates a context for that device.

        :Example:
            .. code-block:: python

                import dpctl

                # Create a level zero gpu device
                d = dpctl.SyclDevice("level_zero:gpu")
                ctx = dpctl.SyclContext(d)
                d = ctx.get_devices()[0]
                assert(d.is_gpu)

        - Invoking the constuctor with a list of :class:`dpctl.SyclDevice`
          objects creates a common context for all the devices. This
          constructor call is especially useful when creation a context for
          multiple sub-devices.

        :Example:
            .. code-block:: python

                import dpctl

                # Create a CPU device using the opencl driver
                cpu_d = dpctl.SyclDevice("opencl:cpu")
                # Partition the CPU device into sub-devices with two cores each.
                sub_devices = cpu_d.create_sub_devices(partition=2)
                # Create a context common to all the sub-devices.
                ctx = dpctl.SyclContext(sub_devices)
                assert(len(ctx.get_devices) == len(sub_devices))

        - Invoking the constuctor with a named ``PyCapsule`` with name
          **"SyclContextRef"** that carries a pointer to a ``sycl::context``
          object. The capsule will be renamed upon successful consumption
          to ensure one-time use. A new named capsule can be constructed by
          using :func:`dpctl.SyclContext._get_capsule` method.

    Args:
        arg (optional): Defaults to ``None``.
            The argument can be a selector string, a :class:`dpctl.SyclDevice`
            instance, a :obj:`list` of :class:`dpctl.SyclDevice` objects, or a
            named ``PyCapsule`` called **"SyclContextRef"**.

    Raises:
        MemoryError: If the constructor could not allocate necessary
                     temporary memory.
        ValueError: If the :class:`dpctl.SyclContext` object creation failed.
        TypeError: If the list of :class:`dpctl.SyclDevice` objects was empty,
                   or the input capsule contained a null pointer or could not
                   be renamed.

    """

    @staticmethod
    cdef void _init_helper(_SyclContext context, DPCTLSyclContextRef CRef):
        context._ctxt_ref = CRef

    @staticmethod
    cdef SyclContext _create(DPCTLSyclContextRef ctxt):
        """
        Calls DPCTLContext_Delete(ctxt).

        Users should pass a copy if they intend to keep the argument ctxt alive.
        """
        cdef _SyclContext ret = <_SyclContext>_SyclContext.__new__(_SyclContext)
        SyclContext._init_helper(ret, ctxt)
        return SyclContext(ret)

    cdef int _init_context_from__SyclContext(self, _SyclContext other):
        self._ctxt_ref = DPCTLContext_Copy(other._ctxt_ref)
        if (self._ctxt_ref is NULL):
            return -1
        return 0

    cdef int _init_context_from_one_device(self, SyclDevice device, int props):
        cdef DPCTLSyclDeviceRef DRef = device.get_device_ref()
        cdef DPCTLSyclContextRef CRef = NULL
        cdef error_handler_callback * eh_callback = (
            <error_handler_callback *>&default_async_error_handler)
        # look up cached contexts for root devices first
        CRef = DPCTLDeviceMgr_GetCachedContext(DRef)
        if (CRef is NULL):
            # look-up failed, create a new one
            CRef = DPCTLContext_Create(DRef, eh_callback, props)
            if (CRef is NULL):
                return -1
        SyclContext._init_helper(<_SyclContext> self, CRef)
        return 0

    cdef int _init_context_from_devices(self, object devices, int props):
        cdef int num_devices = len(devices)
        cdef int i = 0
        cdef int j = 0
        cdef size_t num_bytes
        cdef DPCTLDeviceVectorRef DVRef = NULL
        cdef error_handler_callback * eh_callback = (
            <error_handler_callback *>&default_async_error_handler)
        cdef DPCTLSyclContextRef CRef = NULL
        cdef DPCTLSyclDeviceRef *elems

        if num_devices > 0:
            num_bytes = num_devices * sizeof(DPCTLSyclDeviceRef *)
            elems = <DPCTLSyclDeviceRef *>PyMem_Malloc(num_bytes)
            if (elems is NULL):
                return -3
            for dev in devices:
                if not isinstance(dev, SyclDevice):
                    elems[i] = NULL
                else:
                    elems[i] = (<SyclDevice>dev).get_device_ref()
                if (elems[i] is NULL):
                    PyMem_Free(elems)
                    return -4
                i = i + 1
            # CreateFromArray will make copies of devices referenced by elems
            DVRef = DPCTLDeviceVector_CreateFromArray(num_devices, elems)
            if (DVRef is NULL):
                PyMem_Free(elems)
                return -5
            PyMem_Free(elems)
        else:
            return -2
        CRef = DPCTLContext_CreateFromDevices(DVRef, eh_callback, props)
        DPCTLDeviceVector_Delete(DVRef)
        if (CRef is NULL):
            return -1
        SyclContext._init_helper(<_SyclContext> self, CRef)
        return 0

    cdef int _init_context_from_capsule(self, object cap):
        """
        For named ``PyCapsule`` with name **"SyclContextRef"**, which carries
        pointer to ``sycl::context`` object, interpreted as
        ``DPCTLSyclContextRef``, creates corresponding
        :class:`dpctl.SyclContext`.
        """
        cdef DPCTLSyclContextRef CRef = NULL
        cdef DPCTLSyclContextRef CRef_copy = NULL
        cdef int ret = 0
        if pycapsule.PyCapsule_IsValid(cap, "SyclContextRef"):
            CRef = <DPCTLSyclContextRef> pycapsule.PyCapsule_GetPointer(
                cap, "SyclContextRef"
            )
            if (CRef is NULL):
                return -6
            ret = pycapsule.PyCapsule_SetName(cap, "used_SyclContextRef")
            if (ret):
                return -6
            CRef_copy = DPCTLContext_Copy(CRef)
            if (CRef_copy is NULL):
                return -7
            self._ctxt_ref = CRef_copy
            return 0
        else:
            # __cinit__ checks that capsule is valid, so one can be here only
            # if call to `_init_context_from_capsule` was made outside of
            # __cinit__ and the capsule was not checked to be valid
            return -128

    def __cinit__(self, arg=None):
        cdef int ret = 0
        if isinstance(arg, _SyclContext):
            ret = self._init_context_from__SyclContext(<_SyclContext> arg)
        elif isinstance(arg, SyclDevice):
            ret = self._init_context_from_one_device(<SyclDevice> arg, 0)
        elif pycapsule.PyCapsule_IsValid(arg, "SyclContextRef"):
            status = self._init_context_from_capsule(arg)
        elif isinstance(arg, (list, tuple)) and all(
            [isinstance(argi, SyclDevice) for argi in arg]
        ):
            ret = self._init_context_from_devices(arg, 0)
        else:
            dev = SyclDevice(arg)
            ret = self._init_context_from_one_device(<SyclDevice> dev, 0)
        if (ret < 0):
            if (ret == -1):
                raise ValueError("Context failed to be created.")
            elif (ret == -2):
                raise TypeError(
                    "List of devices to create context from must be non-empty."
                )
            elif (ret == -3):
                raise MemoryError(
                    "Could not allocate necessary temporary memory."
                )
            elif (ret == -4) or (ret == -7):
                raise ValueError(
                    "Internal Error: Could not create a copy of a sycl device."
                )
            elif (ret == -5):
                raise ValueError(
                    "Internal Error: Creation of DeviceVector failed."
                )
            elif (ret == -6):
                raise TypeError(
                    "Input capsule {} contains a null pointer or could not be"
                    " renamed".format(arg)
                )
            raise ValueError(
                "Unrecognized error code ({}) encountered.".format(ret)
            )

    cdef bool equals(self, SyclContext ctxt):
        """
        Returns true if the :class:`dpctl.SyclContext` argument has the
        same underlying ``DPCTLSyclContextRef`` object as this
        :class:`dpctl.SyclContext` instance.

        Returns:
            :obj:`bool`: ``True`` if the two :class:`dpctl.SyclContext` objects
            point to the same ``DPCTLSyclContextRef`` object, otherwise
            ``False``.
        """
        return DPCTLContext_AreEq(self._ctxt_ref, ctxt.get_context_ref())

    def __eq__(self, other):
        """
        Returns True if the :class:`dpctl.SyclContext` argument has the
        same underlying ``DPCTLSyclContextRef`` object as this
        :class:`dpctl.SyclContext` instance.

        Returns:
            :obj:`bool`: ``True`` if the two :class:`dpctl.SyclContext` objects
            point to the same ``DPCTLSyclContextRef`` object, otherwise
            ``False``.
        """
        if isinstance(other, SyclContext):
            return self.equals(<SyclContext> other)
        else:
            return False

    cdef DPCTLSyclContextRef get_context_ref(self):
        return self._ctxt_ref

    def addressof_ref(self):
        """
        Returns the address of the ``DPCTLSyclContextRef`` pointer as a
        ``size_t``.

        Returns:
            :obj:`int`: The address of the ``DPCTLSyclContextRef`` object
            used to create this :class:`dpctl.SyclContext` cast to a
            ``size_t``.
        """
        return int(<size_t>self._ctx_ref)

    def get_devices(self):
        """
        Returns the list of :class:`dpctl.SyclDevice` objects associated with
        :class:`dpctl.SyclContext` instance.

        Returns:
            :obj:`list`: A :obj:`list` of :class:`dpctl.SyclDevice` objects
            that belong to this context.

        Raises:
            ValueError: If the ``DPCTLContext_GetDevices`` call returned
                        ``NULL`` instead of a ``DPCTLDeviceVectorRef`` object.
        """
        cdef DPCTLDeviceVectorRef DVRef = DPCTLContext_GetDevices(
            self.get_context_ref()
        )
        cdef size_t num_devs
        cdef size_t i
        cdef DPCTLSyclDeviceRef DRef
        if (DVRef is NULL):
            raise ValueError("Internal error: NULL device vector encountered")
        num_devs = DPCTLDeviceVector_Size(DVRef)
        devices = []
        for i in range(num_devs):
            DRef = DPCTLDeviceVector_GetAt(DVRef, i)
            devices.append(SyclDevice._create(DRef))
        DPCTLDeviceVector_Delete(DVRef)
        return devices

    @property
    def device_count(self):
        """
        The number of sycl devices associated with the
        :class:`dpctl.SyclContext` instance.

        Returns:
            :obj:`int`: Number of devices associated with the context.

        Raises:
            ValueError: If ``DPCTLContext_DeviceCount`` led to a
                        failure.
        """
        cdef size_t num_devs = DPCTLContext_DeviceCount(self.get_context_ref())
        if num_devs:
            return num_devs
        else:
            raise ValueError(
                "An error was encountered quering the number of devices "
                "associated with this context"
            )

    @property
    def __name__(self):
        return "SyclContext"

    def __repr__(self):
        """
        Returns the name of the class and number of devices in the context.

        :Example:
            .. code-block:: python

                import dpctl

                # Create a default SyclContext
                ctx = dpctl.SyclContext()
                print(ctx) # E.g : <dpctl.SyclContext at 0x7f154d8ab070>

                cpu_d = dpctl.SyclDevice("opencl:cpu")
                sub_devices = create_sub_devices(partition=2)
                ctx2 = dpctl.SyclContext(sub_devices)
                # prints: <dpctl.SyclContext for 4 devices at 0x7f154d8ab070>
                print(ctx2)

        Returns:
            :obj:`str`: A string representation of the
            :class:`dpctl.SyclContext` object.

        """
        cdef size_t n = self.device_count
        if n == 1:
            return ("<dpctl." + self.__name__ + " at {}>".format(hex(id(self))))
        else:
            return (
                "<dpctl."
                + self.__name__
                + " for {} devices at {}>".format(n, hex(id(self)))
            )

    def _get_capsule(self):
        """
        Returns a copy of the underlying ``sycl::context`` pointer as a void
        pointer inside a named ``PyCapsule`` that has the name
        **SyclContextRef**. The ownership of the pointer inside the capsule is
        passed to the caller, and pointer is deleted when the capsule goes out
        of scope.

        Returns:
            :class:`pycapsule`: A capsule object storing a copy of the
            ``sycl::context`` pointer belonging to thus
            :class:`dpctl.SyclContext` instance.

        Raises:
            ValueError: If the ``DPCTLContext_Copy`` fails to copy the
                        ``sycl::context`` pointer.
        """
        cdef DPCTLSyclContextRef CRef = NULL
        CRef = DPCTLContext_Copy(self._ctxt_ref)
        if (CRef is NULL):
            raise ValueError("SyclContext copy failed.")
        return pycapsule.PyCapsule_New(
            <void *>CRef,
            "SyclContextRef",
            &_context_capsule_deleter
        )
