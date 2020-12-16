# ===------------ sycl_core.pyx - dpctl module -------*- Cython -*----------===#
#
#                      Data Parallel Control (dpCtl)
#
# Copyright 2020 Intel Corporation
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
#
# ===-----------------------------------------------------------------------===#
#
# \file
# This file implements a sub-set of Sycl's interface using dpctl's CAPI.
#
# ===-----------------------------------------------------------------------===#

# distutils: language = c++
# cython: language_level=3

from __future__ import print_function
from enum import Enum, auto
import logging
from ._backend cimport *
from .memory._memory cimport _Memory
from libc.stdlib cimport malloc, free

__all__ = [
    "dump",
    "get_current_backend",
    "get_current_queue",
    "get_current_device_type",
    "get_num_platforms",
    "get_num_activated_queues",
    "get_num_queues",
    "has_cpu_queues",
    "has_gpu_queues",
    "has_sycl_platforms",
    "set_default_queue",
    "is_in_device_context",
    "device_type",
    "backend_type",
    "device_context",
    "SyclContext",
    "SyclDevice",
    "SyclEvent",
    "SyclQueue"
]

_logger = logging.getLogger(__name__)


class device_type(Enum):
    """
    Automatically numbers SYCL device types with values starting from 1.
    """
    gpu = auto()
    cpu = auto()
    accelerator = auto()
    host_device = auto()

class backend_type(Enum):
    """
    Automatically numbers SYCL backend types with values starting from 1.
    """
    opencl = auto()
    level_zero = auto()
    cuda = auto()
    host = auto()

cdef class UnsupportedBackendError (Exception):
    """This exception is raised when a device type other than CPU or GPU is
       encountered.
    """
    pass

cdef class UnsupportedDeviceError (Exception):
    """This exception is raised when a device type other than CPU or GPU is
       encountered.
    """
    pass

cdef class SyclKernelSubmitError (Exception):
    """This exception is raised when a sycl program could not be built from
       either a spirv binary file or a string source.
    """
    pass

cdef class SyclKernelInvalidRangeError (Exception):
    """This exception is raised when a range that has more than three
       dimensions or less than one dimension.
    """
    pass

cdef class SyclQueueCreationError (Exception):
    """This exception is raised when a range that has more than three
       dimensions or less than one dimension.
    """
    pass

cdef class SyclContext:

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
        """Returns the address of the DPCTLSyclContextRef pointer as a
        long.

        Returns:
            The address of the DPCTLSyclContextRef object used to create this
            SyclContext cast to a long.
        """
        return int(<size_t>self._ctx_ref)

cdef class SyclDevice:
    """ Wrapper class for a Sycl Device
    """

    @staticmethod
    cdef SyclDevice _create (DPCTLSyclDeviceRef dref):
        cdef SyclDevice ret = SyclDevice.__new__(SyclDevice)
        ret._device_ref = dref
        ret._vendor_name = DPCTLDevice_GetVendorName(dref)
        ret._device_name = DPCTLDevice_GetName(dref)
        ret._driver_version = DPCTLDevice_GetDriverInfo(dref)
        ret._max_compute_units = DPCTLDevice_GetMaxComputeUnits(dref)
        ret._max_work_item_dims = DPCTLDevice_GetMaxWorkItemDims(dref)
        ret._max_work_item_sizes = DPCTLDevice_GetMaxWorkItemSizes(dref)
        ret._max_work_group_size = DPCTLDevice_GetMaxWorkGroupSize(dref)
        ret._max_num_sub_groups = DPCTLDevice_GetMaxNumSubGroups(dref)
        ret._int64_base_atomics = DPCTLDevice_HasInt64BaseAtomics(dref)
        ret._int64_extended_atomics = DPCTLDevice_HasInt64ExtendedAtomics(dref)
        return ret

    def __dealloc__ (self):
        DPCTLDevice_Delete(self._device_ref)
        DPCTLCString_Delete(self._device_name)
        DPCTLCString_Delete(self._vendor_name)
        DPCTLCString_Delete(self._driver_version)
        DPCTLSize_t_Array_Delete(self._max_work_item_sizes)

    def dump_device_info (self):
        """ Print information about the SYCL device.
        """
        DPCTLDevice_DumpInfo(self._device_ref)

    cpdef get_device_name (self):
        """ Returns the name of the device as a string
        """
        return self._device_name.decode()

    cpdef get_device_type (self):
        """ Returns the type of the device as a `device_type` enum
        """
        if DPCTLDevice_IsGPU(self._device_ref):
            return device_type.gpu
        elif DPCTLDevice_IsCPU(self._device_ref):
            return device_type.cpu
        else:
            raise ValueError("Unknown device type.")

    cpdef get_vendor_name (self):
        """ Returns the device vendor name as a string
        """
        return self._vendor_name.decode()

    cpdef get_driver_version (self):
        """ Returns the OpenCL software driver version as a string
            in the form: major number.minor number, if this SYCL
            device is an OpenCL device. Returns a string class
            with the value "1.2" if this SYCL device is a host device.
        """
        return self._driver_version.decode()

    cpdef has_int64_base_atomics (self):
        """ Returns true if device has int64_base_atomics else returns false.
        """
        return self._int64_base_atomics

    cpdef has_int64_extended_atomics (self):
        """ Returns true if device has int64_extended_atomics else returns false.
        """
        return self._int64_extended_atomics

    cpdef get_max_compute_units (self):
        """ Returns the number of parallel compute units
            available to the device. The minimum value is 1.
        """
        return self._max_compute_units

    cpdef get_max_work_item_dims (self):
        """ Returns the maximum dimensions that specify
            the global and local work-item IDs used by the
            data parallel execution model. The minimum
            value is 3 if this SYCL device is not of device
            type ``info::device_type::custom``.
        """
        return self._max_work_item_dims

    cpdef get_max_work_item_sizes (self):
        """ Returns the maximum number of work-items
            that are permitted in each dimension of the
            work-group of the nd_range. The minimum
            value is (1; 1; 1) for devices that are not of
            device type ``info::device_type::custom``.
        """
        max_work_item_sizes = []
        for n in range(3):
            max_work_item_sizes.append(self._max_work_item_sizes[n])
        return tuple(max_work_item_sizes)

    cpdef get_max_work_group_size (self):
        """ Returns the maximum number of work-items
            that are permitted in a work-group executing a
            kernel on a single compute unit. The minimum
            value is 1.
        """
        return self._max_work_group_size

    cpdef get_max_num_sub_groups (self):
        """ Returns the maximum number of sub-groups
            in a work-group for any kernel executed on the
            device. The minimum value is 1.
        """
        return self._max_num_sub_groups

    cdef DPCTLSyclDeviceRef get_device_ref (self):
        """ Returns the DPCTLSyclDeviceRef pointer for this class.
        """
        return self._device_ref

    def addressof_ref (self):
        """Returns the address of the DPCTLSyclDeviceRef pointer as a
        long.

        Returns:
            The address of the DPCTLSyclDeviceRef object used to create this
            SyclDevice cast to a long.
        """
        return int(<size_t>self._device_ref)

cdef class SyclEvent:
    """ Wrapper class for a Sycl Event
    """

    @staticmethod
    cdef SyclEvent _create (DPCTLSyclEventRef eref, list args):
        cdef SyclEvent ret = SyclEvent.__new__(SyclEvent)
        ret._event_ref = eref
        ret._args = args
        return ret

    def __dealloc__ (self):
        self.wait()
        DPCTLEvent_Delete(self._event_ref)

    cdef DPCTLSyclEventRef get_event_ref (self):
        """ Returns the DPCTLSyclEventRef pointer for this class.
        """
        return self._event_ref

    cpdef void wait (self):
        DPCTLEvent_Wait(self._event_ref)

    def addressof_ref (self):
        """ Returns the address of the C API DPCTLSyclEventRef pointer as
        a long.

        Returns:
            The address of the DPCTLSyclEventRef object used to create this
            SyclEvent cast to a long.
        """
        return int(<size_t>self._event_ref)

import ctypes

cdef class SyclQueue:
    """ Wrapper class for a Sycl queue.
    """

    @staticmethod
    cdef SyclQueue _create(DPCTLSyclQueueRef qref):
        if qref is NULL:
            raise SyclQueueCreationError("Queue creation failed.")
        cdef SyclQueue ret = SyclQueue.__new__(SyclQueue)
        ret._context = SyclContext._create(DPCTLQueue_GetContext(qref))
        ret._device = SyclDevice._create(DPCTLQueue_GetDevice(qref))
        ret._queue_ref = qref
        return ret

    @staticmethod
    cdef SyclQueue _create_from_context_and_device(SyclContext ctx, SyclDevice dev):
        cdef SyclQueue ret = SyclQueue.__new__(SyclQueue)
        cdef DPCTLSyclContextRef cref = ctx.get_context_ref()
        cdef DPCTLSyclDeviceRef dref = dev.get_device_ref()
        cdef DPCTLSyclQueueRef qref = DPCTLQueueMgr_GetQueueFromContextAndDevice(
            cref, dref)

        if qref is NULL:
            raise SyclQueueCreationError("Queue creation failed.")
        ret._queue_ref = qref
        ret._context = ctx
        ret._device = dev
        return ret

    def __dealloc__ (self):
        DPCTLQueue_Delete(self._queue_ref)

    cdef _raise_queue_submit_error (self, fname, errcode):
        e = SyclKernelSubmitError("Kernel submission to Sycl queue failed.")
        e.fname = fname
        e.code = errcode
        raise e

    cdef _raise_invalid_range_error (self, fname, ndims, errcode):
        e = SyclKernelInvalidRangeError("Range with ", ndims, " not allowed. "
                                        "Range should have between one and "
                                        "three dimensions.")
        e.fname = fname
        e.code = errcode
        raise e

    cdef int _populate_args (self, list args, void **kargs,                    \
                             DPCTLKernelArgType *kargty):
        cdef int ret = 0
        for idx, arg in enumerate(args):
            if isinstance(arg, ctypes.c_char):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._CHAR
            elif isinstance(arg, ctypes.c_int):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._INT
            elif isinstance(arg, ctypes.c_uint):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._UNSIGNED_INT
            elif isinstance(arg, ctypes.c_uint8):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._UNSIGNED_INT8
            elif isinstance(arg, ctypes.c_long):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._LONG
            elif isinstance(arg, ctypes.c_ulong):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._UNSIGNED_LONG
            elif isinstance(arg, ctypes.c_longlong):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._LONG_LONG
            elif isinstance(arg, ctypes.c_ulonglong):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._UNSIGNED_LONG_LONG
            elif isinstance(arg, ctypes.c_short):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._SHORT
            elif isinstance(arg, ctypes.c_size_t):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._SIZE_T
            elif isinstance(arg, ctypes.c_float):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._FLOAT
            elif isinstance(arg, ctypes.c_double):
                kargs[idx] = <void*><size_t>(ctypes.addressof(arg))
                kargty[idx] = _arg_data_type._DOUBLE
            elif isinstance(arg, _Memory):
                kargs[idx]= <void*>(<size_t>arg._pointer)
                kargty[idx] = _arg_data_type._VOID_PTR
            else:
                ret = -1
        return ret


    cdef int _populate_range (self, size_t Range[3], list S, size_t nS):

        cdef int ret = 0

        if nS == 1:
            Range[0] = <size_t>S[0]
            Range[1] = 1
            Range[2] = 1
        elif nS == 2:
            Range[0] = <size_t>S[0]
            Range[1] = <size_t>S[1]
            Range[2] = 1
        elif nS == 3:
            Range[0] = <size_t>S[0]
            Range[1] = <size_t>S[1]
            Range[2] = <size_t>S[2]
        else:
            ret = -1

        return ret

    cpdef bool equals (self, SyclQueue q):
        """ Returns true if the SyclQueue argument has the same _queue_ref
            as this SycleQueue.
        """
        return DPCTLQueue_AreEq(self._queue_ref, q.get_queue_ref())

    def get_sycl_backend (self):
        """ Returns the Sycl bakend associated with the queue.
        """
        cdef DPCTLSyclBackendType BE = DPCTLQueue_GetBackend(self._queue_ref)
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

    cpdef SyclContext get_sycl_context (self):
        return self._context

    cpdef SyclDevice get_sycl_device (self):
        return self._device

    cdef DPCTLSyclQueueRef get_queue_ref (self):
        return self._queue_ref

    def addressof_ref (self):
        """ Returns the address of the C API DPCTLSyclQueueRef pointer as a long.

        Returns:
            The address of the DPCTLSyclQueueRef object used to create this
            SyclQueue cast to a long.
        """
        return int(<size_t>self._queue_ref)

    cpdef SyclEvent submit (self, SyclKernel kernel, list args, list gS,       \
                            list lS = None, list dEvents = None):

        cdef void **kargs = NULL
        cdef DPCTLKernelArgType *kargty = NULL
        cdef DPCTLSyclEventRef *depEvents = NULL
        cdef DPCTLSyclEventRef Eref = NULL
        cdef int ret
        cdef size_t gRange[3]
        cdef size_t lRange[3]
        cdef size_t nGS = len(gS)
        cdef size_t nLS = len(lS) if lS is not None else 0
        cdef size_t nDE = len(dEvents) if dEvents is not None else 0

        # Allocate the arrays to be sent to DPCTLQueue_Submit
        kargs = <void**>malloc(len(args) * sizeof(void*))
        if not kargs:
            raise MemoryError()
        kargty = <DPCTLKernelArgType*>malloc(len(args)*sizeof(DPCTLKernelArgType))
        if not kargty:
            free(kargs)
            raise MemoryError()
        # Create the array of dependent events if any
        if dEvents is not None and nDE > 0:
            depEvents = <DPCTLSyclEventRef*>malloc(nDE*sizeof(DPCTLSyclEventRef))
            if not depEvents:
                free(kargs)
                free(kargty)
                raise MemoryError()
            else:
                for idx, de in enumerate(dEvents):
                    depEvents[idx] = (<SyclEvent>de).get_event_ref()

        # populate the args and argstype arrays
        ret = self._populate_args(args, kargs, kargty)
        if ret == -1:
            free(kargs)
            free(kargty)
            free(depEvents)
            raise TypeError("Unsupported type for a kernel argument")

        if lS is None:
            ret = self._populate_range (gRange, gS, nGS)
            if ret == -1:
                free(kargs)
                free(kargty)
                free(depEvents)
                self._raise_invalid_range_error("SyclQueue.submit", nGS, -1)

            Eref = DPCTLQueue_SubmitRange(kernel.get_kernel_ref(),
                                         self.get_queue_ref(),
                                         kargs,
                                         kargty,
                                         len(args),
                                         gRange,
                                         nGS,
                                         depEvents,
                                         nDE)
        else:
            ret = self._populate_range (gRange, gS, nGS)
            if ret == -1:
                free(kargs)
                free(kargty)
                free(depEvents)
                self._raise_invalid_range_error("SyclQueue.submit", nGS, -1)
            ret = self._populate_range (lRange, lS, nLS)
            if ret == -1:
                free(kargs)
                free(kargty)
                free(depEvents)
                self._raise_invalid_range_error("SyclQueue.submit", nLS, -1)

            if nGS != nLS:
                free(kargs)
                free(kargty)
                free(depEvents)
                raise ValueError("Local and global ranges need to have same "
                                 "number of dimensions.")

            Eref = DPCTLQueue_SubmitNDRange(kernel.get_kernel_ref(),
                                           self.get_queue_ref(),
                                           kargs,
                                           kargty,
                                           len(args),
                                           gRange,
                                           lRange,
                                           nGS,
                                           depEvents,
                                           nDE)
        free(kargs)
        free(kargty)
        free(depEvents)

        if Eref is NULL:
            # \todo get the error number from dpctl-capi
            self._raise_queue_submit_error("DPCTLQueue_Submit", -1)

        return SyclEvent._create(Eref, args)

    cpdef void wait (self):
        DPCTLQueue_Wait(self._queue_ref)

    cpdef memcpy (self, dest, src, size_t count):
        cdef void *c_dest
        cdef void *c_src

        if isinstance(dest, _Memory):
            c_dest = <void*>(<_Memory>dest).memory_ptr
        else:
            raise TypeError("Parameter `dest` should have type _Memory.")

        if isinstance(src, _Memory):
            c_src = <void*>(<_Memory>src).memory_ptr
        else:
            raise TypeError("Parameter `src` should have type _Memory.")

        DPCTLQueue_Memcpy(self._queue_ref, c_dest, c_src, count)

    cpdef prefetch (self, mem, size_t count=0):
       cdef void *ptr

       if isinstance(mem, _Memory):
           ptr = <void*>(<_Memory>mem).memory_ptr
       else:
           raise TypeError("Parameter `mem` should have type _Memory")

       if (count <=0 or count > self.nbytes):
           count = self.nbytes

       DPCTLQueue_Prefetch(self._queue_ref, ptr, count)

    cpdef mem_advise (self, mem, size_t count, int advice):
       cdef void *ptr

       if isinstance(mem, _Memory):
           ptr = <void*>(<_Memory>mem).memory_ptr
       else:
           raise TypeError("Parameter `mem` should have type _Memory")

       if (count <=0 or count > self.nbytes):
           count = self.nbytes

       DPCTLQueue_MemAdvise(self._queue_ref, ptr, count, advice)


cdef class _SyclRTManager:
    """ Wrapper for the C API's sycl queue manager interface.
    """
    cdef dict _backend_str_ty_dict
    cdef dict _device_str_ty_dict
    cdef dict _backend_enum_ty_dict
    cdef dict _device_enum_ty_dict

    def __cinit__(self):

        self._backend_str_ty_dict = {
            "opencl" : _backend_type._OPENCL,
            "level0" : _backend_type._LEVEL_ZERO,
        }

        self._device_str_ty_dict = {
            "gpu" : _device_type._GPU,
            "cpu" : _device_type._CPU,
        }

        self._backend_enum_ty_dict = {
            backend_type.opencl : _backend_type._OPENCL,
            backend_type.level_zero : _backend_type._LEVEL_ZERO,
        }

        self._device_enum_ty_dict = {
            device_type.cpu : _device_type._CPU,
            device_type.gpu : _device_type._GPU,
        }

    def _set_as_current_queue(self, backend_ty, device_ty, device_id):
        cdef DPCTLSyclQueueRef queue_ref

        try :
            beTy = self._backend_str_ty_dict[backend_ty]
            try :
                devTy = self._device_str_ty_dict[device_ty]
                queue_ref = DPCTLQueueMgr_PushQueue(beTy, devTy, device_id)
                return SyclQueue._create(queue_ref)
            except KeyError:
                raise UnsupportedDeviceError("Device can only be gpu or cpu")
        except KeyError:
            raise UnsupportedBackendError("Backend can only be opencl or "
                                          "level-0")

    def _remove_current_queue(self):
        DPCTLQueueMgr_PopQueue()

    def dump(self):
        """ Prints information about the Runtime object.
        """
        DPCTLPlatform_DumpInfo()

    def print_available_backends(self):
        """ Prints the available SYCL backends.
        """
        print(self._backend_str_ty_dict.keys())

    cpdef get_current_backend(self):
        """ Returns the backend for the current queue as a `backend_type` enum
        """
        return self.get_current_queue().get_sycl_backend()

    cpdef get_current_device_type(self):
        """ Returns current device type as a `device_type` enum
        """
        return self.get_current_queue().get_sycl_device().get_device_type()

    cpdef SyclQueue get_current_queue(self):
        """ Returns the currently activate SYCL queue as a new SyclQueue object.
        If there are no active queues then a SyclQueueCreationError exception is
        raised.
        """
        return SyclQueue._create(DPCTLQueueMgr_GetCurrentQueue())

    def get_num_activated_queues(self):
        """ Returns the number of currently activated queues for this thread.
        """
        return DPCTLQueueMgr_GetNumActivatedQueues()

    def get_num_platforms(self):
        """ Returns the number of available non-host SYCL platforms.
        """
        return DPCTLPlatform_GetNumNonHostPlatforms()

    def get_num_queues(self, backend_ty, device_ty):
        cdef size_t num = 0
        try :
            beTy = self._backend_enum_ty_dict[backend_ty]
            try :
                devTy = self._device_enum_ty_dict[device_ty]
                num = DPCTLQueueMgr_GetNumQueues(beTy, devTy)
            except KeyError:
                raise UnsupportedDeviceError(
                        "Device can only be device_type.gpu or device_type.cpu"
                      )
        except KeyError:
            raise UnsupportedBackendError(
                      "Backend can only be backend_type.opencl or "
                      "backend_type.level_zero"
                  )

        return num

    def has_gpu_queues(self, backend_ty=backend_type.opencl):
        cdef size_t num = 0
        try :
            beTy = self._backend_enum_ty_dict[backend_ty]
            num = DPCTLQueueMgr_GetNumQueues(beTy, _device_type._GPU)
        except KeyError:
            raise UnsupportedBackendError(
                      "Backend can only be backend_type.opencl or "
                      "backend_type.level_zero"
                  )
        if num:
            return True
        else:
            return False

    def has_cpu_queues(self, backend_ty=backend_type.opencl):
        cdef size_t num = 0
        try :
            beTy = self._backend_enum_ty_dict[backend_ty]
            num = DPCTLQueueMgr_GetNumQueues(beTy, _device_type._CPU)
        except KeyError:
            raise UnsupportedBackendError(
                      "Backend can only be backend_type.opencl or "
                      "backend_type.level_zero"
                  )
        if num:
            return True
        else:
            return False

    def has_sycl_platforms(self):
        cdef size_t num_platforms = DPCTLPlatform_GetNumNonHostPlatforms()
        if num_platforms:
            return True
        else:
            return False

    def is_in_device_context(self):
        cdef size_t num = DPCTLQueueMgr_GetNumActivatedQueues()
        if num:
            return True
        else:
            return False

    def set_default_queue(self, backend_ty, device_ty, device_id):
        cdef DPCTLSyclQueueRef ret
        try :
            if isinstance(backend_ty, str):
                beTy = self._backend_str_ty_dict[backend_ty]
            else:
                beTy = self._backend_enum_ty_dict[backend_ty]
            try :
                if isinstance(device_ty, str):
                    devTy = self._device_str_ty_dict[device_ty]
                else:
                    devTyp = self._device_enum_ty_dist[device_ty]
                ret = DPCTLQueueMgr_SetAsDefaultQueue(beTy, devTy, device_id)
                if ret is NULL:
                    self._raise_queue_creation_error(
                        backend_ty, device_ty, device_id,
                        "DPCTLQueueMgr_PushQueue"
                    )

            except KeyError:
                raise UnsupportedDeviceError("Device can only be gpu or cpu")
        except KeyError:
            raise UnsupportedBackendError("Backend can only be opencl or "
                                          "level-0")


# This private instance of the _SyclQueueManager should not be directly
# accessed outside the module.
_mgr = _SyclRTManager()

# Global bound functions
dump                     = _mgr.dump
get_num_platforms        = _mgr.get_num_platforms
get_num_activated_queues = _mgr.get_num_activated_queues
get_num_queues           = _mgr.get_num_queues
has_cpu_queues           = _mgr.has_cpu_queues
has_gpu_queues           = _mgr.has_gpu_queues
has_sycl_platforms       = _mgr.has_sycl_platforms
set_default_queue        = _mgr.set_default_queue
is_in_device_context     = _mgr.is_in_device_context

cpdef SyclQueue get_current_queue():
    """ Returns the currently activate SYCL queue as a new SyclQueue object.

    Returns:
        SyclQueue: If there is a currently active SYCL queue that queue
        is returned wrapped in a SyclQueue object. The SyclQueue object
        owns a copy of the currently active SYCL queue as an opaque
        `DPCTLSyclQueueRef` pointer. The pointer is freed when the SyclQueue
        is garbage collected.

    Raises:
        SyclQueueCreationError: If no currently active SYCL queue found.
    """
    return _mgr.get_current_queue()

cpdef get_current_device_type():
    """
        Obtain current device type from Data Parallel Control package.
    """
    return _mgr.get_current_device_type()

cpdef get_current_backend():
    """
        Obtain current backend type from Data Parallel Control package.
    """
    return _mgr.get_current_backend()

from contextlib import contextmanager

@contextmanager
def device_context(str queue_str="opencl:gpu:0"):
    """
        The SYCL queue defined by the "backend:device type:device id" tuple is
        set as the currently active queue, *i.e.*, a subsequent call to
        :func:`dpctl.get_current_queue()` inside the context returns the queue.
        The active queue is also returned by the context manager and can be
        directly used without having to call :func:`dpctl.get_current_queue()`.

        If a the request queue is not found an exception is raised.
    """
    ctxt = None
    try:
        attrs = queue_str.split(':')
        nattrs = len(attrs)
        if (nattrs < 2 or nattrs > 3):
            raise ValueError("Invalid queue filter string. Should be "
                             "backend:device:device_number or "
                             "backend:device. In the later case the "
                             "device_number defaults to 0")
        if nattrs == 2:
            attrs.append("0")
        ctxt = _mgr._set_as_current_queue(attrs[0], attrs[1], int(attrs[2]))
        yield ctxt
    finally:
        # Code to release resource
        if ctxt:
            _logger.debug(
                "Removing the queue from the stack of active queues")
            _mgr._remove_current_queue()
        else:
            _logger.debug("No queue was created so nothing to do")
