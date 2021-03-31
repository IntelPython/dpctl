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

""" Implements SyclContext Cython extension type.
"""

from __future__ import print_function
import logging
from ._backend cimport (
    DPCTLSyclContextRef,
    DPCTLSyclDeviceRef,
    DPCTLContext_Create,
    DPCTLContext_CreateFromDevices,
    DPCTLContext_DeviceCount,
    DPCTLContext_GetDevices,
    DPCTLContext_Copy,
    DPCTLContext_Delete,
    DPCTLContext_AreEq,
    DPCTLDevice_Delete,
    DPCTLDevice_Copy,
    DPCTLDeviceVectorRef,
    DPCTLDeviceVector_CreateFromArray,
    DPCTLDeviceVector_GetAt,
    DPCTLDeviceVector_Size,
    DPCTLDeviceVector_Delete,
    error_handler_callback,
    DPCTLDeviceMgr_GetCachedContext,
)
from ._sycl_queue cimport default_async_error_handler
from ._sycl_device cimport SyclDevice
from cpython.mem cimport PyMem_Malloc, PyMem_Free

__all__ = [
    "SyclContext",
]

_logger = logging.getLogger(__name__)

cdef class _SyclContext:
    """ Data owner for SyclContext
    """

    def __dealloc__(self):
        DPCTLContext_Delete(self._ctxt_ref)


cdef class SyclContext(_SyclContext):
    """ Python wrapper class for cl::sycl::context.
    """
    
    @staticmethod
    cdef void _init_helper(_SyclContext context, DPCTLSyclContextRef CRef):
         context._ctxt_ref = CRef
    
    @staticmethod
    cdef SyclContext _create (DPCTLSyclContextRef ctxt):
        """
        Calls DPCTLContext_Delete(ctxt).

        Users should pass a copy if they intend to keep the argument ctxt alive.
        """
        cdef _SyclContext ret = <_SyclContext>_SyclContext.__new__(_SyclContext)
        SyclContext._init_helper(ret, ctxt)
        return SyclContext(ret)

    cdef int _init_from__SyclContext(self, _SyclContext other):
        self._ctxt_ref = DPCTLContext_Copy(other._ctxt_ref)
        if (self._ctxt_ref is NULL):
            return -1
        return 0

    cdef int _init_from_one_device(self, SyclDevice device, int props):
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

    cdef int _init_from_devices(self, object devices, int props):
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
        DPCTLContext_CreateFromDevices(DVRef, eh_callback, props)
        DPCTLDeviceVector_Delete(DVRef)
        if (CRef is NULL):
            return -1
        SyclContext._init_helper(<_SyclContext> self, CRef)
        return 0

    def __cinit__(self, arg=None):
        """ SyclContext() - create a context for a default device
            SyclContext(filter_selector_string) - create a context for specified device
            SyclContext(SyclDevice_instance) - create a context for the given device
            SyclContext((dev1, dev2, ...)) - create a context for given set of devices
        """
        cdef int ret = 0
        if isinstance(arg, _SyclContext):
            ret = self._init_from__SyclContext(<_SyclContext> arg)
        elif isinstance(arg, SyclDevice):
            ret = self._init_from_one_device(<SyclDevice> arg, 0)
        elif isinstance(arg, (list, tuple)) and all([isinstance(argi, SyclDevice) for argi in arg]):
            ret = self._init_from_devices(arg, 0)
        else:
            dev = SyclDevice(arg)
            ret = self._init_from_one_device(<SyclDevice> dev, 0)
        if (ret < 0):
            if (ret == -1):
                raise ValueError("Context failed to be created.")
            if (ret == -2):
                raise TypeError("List of devices to create context from must be non-empty.")
            if (ret == -3):
                raise MemoryError("Could not allocate necessary temporary memory.")
            if (ret == -4):
                raise ValueError("Internal Error: Could not create a copy of a sycl device.")
            if (ret == -5):
                raise ValueError("Internal Error: Creation of DeviceVector failed.")
            raise ValueError("Unrecognized error code ({}) encountered.".format(ret))

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

    def get_devices (self):
        """
        Returns the list of SyclDevice objects associated with SyclContext instance.
        """
        cdef DPCTLDeviceVectorRef DVRef = DPCTLContext_GetDevices(self.get_context_ref())
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
    def device_count (self):
        """
        The number of sycl devices associated with SyclContext instance.
        """
        cdef size_t num_devs = DPCTLContext_DeviceCount(self.get_context_ref())
        if num_devs:
            return num_devs
        else:
            raise ValueError("An error was encountered quering the number of devices "
                             "associated with this context")

    @property
    def __name__(self):
        return "SyclContext"

    def __repr__(self):
        cdef size_t n = self.device_count
        if n == 1:
            return ("<dpctl." + self.__name__ + " at {}>".format(hex(id(self))))
        else:
            return ("<dpctl." + self.__name__ + " for {} devices at {}>".format(n, hex(id(self))))
