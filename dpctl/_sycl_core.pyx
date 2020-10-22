##===------------- sycl_core.pyx - dpctl module -------*- Cython -*--------===##
##
##                      Data Parallel Control (dpCtl)
##
## Copyright 2020 Intel Corporation
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##
##===----------------------------------------------------------------------===##
##
## \file
## This file implements a sub-set of Sycl's interface using dpctl's CAPI.
##
##===----------------------------------------------------------------------===##

# distutils: language = c++
# cython: language_level=3

from __future__ import print_function
from enum import Enum, auto
import logging
from ._backend cimport *
from ._memory cimport Memory
from libc.stdlib cimport malloc, free


_logger = logging.getLogger(__name__)


class device_type(Enum):
    gpu = auto()
    cpu = auto()
    accelerator = auto()
    host_device = auto()

class backend_type(Enum):
    opencl = auto()
    level_zero = auto()
    cuda = auto()
    host = auto()

cdef class UnsupportedBackendError (Exception):
    '''This exception is raised when a device type other than CPU or GPU is
       encountered.
    '''
    pass

cdef class UnsupportedDeviceError (Exception):
    '''This exception is raised when a device type other than CPU or GPU is
       encountered.
    '''
    pass

cdef class SyclProgramCompilationError (Exception):
    '''This exception is raised when a sycl program could not be built from
       either a spirv binary file or a string source.
    '''
    pass

cdef class SyclKernelSubmitError (Exception):
    '''This exception is raised when a sycl program could not be built from
       either a spirv binary file or a string source.
    '''
    pass

cdef class SyclKernelInvalidRangeError (Exception):
    '''This exception is raised when a range that has more than three
       dimensions or less than one dimension.
    '''
    pass

cdef class SyclQueueCreationError (Exception):
    '''This exception is raised when a range that has more than three
       dimensions or less than one dimension.
    '''
    pass

cdef class SyclContext:

    @staticmethod
    cdef SyclContext _create (DPPLSyclContextRef ctxt):
        cdef SyclContext ret = SyclContext.__new__(SyclContext)
        ret._ctxt_ref = ctxt
        return ret

    def __dealloc__ (self):
        DPPLContext_Delete(self._ctxt_ref)

    cpdef bool equals (self, SyclContext ctxt):
        """ Returns true if the SyclContext argument has the same _context_ref
            as this SyclContext.
        """
        return DPPLContext_AreEq(self._ctxt_ref, ctxt.get_context_ref())

    cdef DPPLSyclContextRef get_context_ref (self):
        return self._ctxt_ref

    def addressof_ref (self):
        """Returns the address of the DPPLSyclContextRef pointer as a
        long.

        Returns:
            The address of the DPPLSyclContextRef object used to create this
            SyclContext cast to a long.
        """
        return int(<size_t>self._ctx_ref)

cdef class SyclDevice:
    ''' Wrapper class for a Sycl Device
    '''

    @staticmethod
    cdef SyclDevice _create (DPPLSyclDeviceRef dref):
        cdef SyclDevice ret = SyclDevice.__new__(SyclDevice)
        ret._device_ref = dref
        ret._vendor_name = DPPLDevice_GetVendorName(dref)
        ret._device_name = DPPLDevice_GetName(dref)
        ret._driver_version = DPPLDevice_GetDriverInfo(dref)
        return ret

    def __dealloc__ (self):
        DPPLDevice_Delete(self._device_ref)
        DPPLCString_Delete(self._device_name)
        DPPLCString_Delete(self._vendor_name)
        DPPLCString_Delete(self._driver_version)

    def dump_device_info (self):
        ''' Print information about the SYCL device.
        '''
        DPPLDevice_DumpInfo(self._device_ref)

    def get_device_name (self):
        ''' Returns the name of the device as a string
        '''
        return self._device_name.decode()

    def get_device_type (self):
        ''' Returns the type of the device as a `device_type` enum
        '''
        if DPPLDevice_IsGPU(self._device_ref):
            return device_type.gpu
        elif DPPLDevice_IsCPU(self._device_ref):
            return device_type.cpu
        else:
            raise ValueError("Unknown device type.")

    def get_vendor_name (self):
        ''' Returns the device vendor name as a string
        '''
        return self._vendor_name.decode()

    def get_driver_version (self):
        ''' Returns the OpenCL software driver version as a string
            in the form: major number.minor number, if this SYCL
            device is an OpenCL device. Returns a string class
            with the value "1.2" if this SYCL device is a host device.
        '''
        return self._driver_version.decode()

    cdef DPPLSyclDeviceRef get_device_ref (self):
        ''' Returns the DPPLSyclDeviceRef pointer for this class.
        '''
        return self._device_ref

    def addressof_ref (self):
        """Returns the address of the DPPLSyclDeviceRef pointer as a
        long.

        Returns:
            The address of the DPPLSyclDeviceRef object used to create this
            SyclDevice cast to a long.
        """
        return int(<size_t>self._device_ref)

cdef class SyclEvent:
    ''' Wrapper class for a Sycl Event
    '''

    @staticmethod
    cdef SyclEvent _create (DPPLSyclEventRef eref, list args):
        cdef SyclEvent ret = SyclEvent.__new__(SyclEvent)
        ret._event_ref = eref
        ret._args = args
        return ret

    def __dealloc__ (self):
        self.wait()
        DPPLEvent_Delete(self._event_ref)

    cdef DPPLSyclEventRef get_event_ref (self):
        ''' Returns the DPPLSyclEventRef pointer for this class.
        '''
        return self._event_ref

    cpdef void wait (self):
        DPPLEvent_Wait(self._event_ref)

    def addressof_ref (self):
        """Returns the address of the C API DPPLSyclEventRef pointer as
        a long.

        Returns:
            The address of the DPPLSyclEventRef object used to create this
            SyclEvent cast to a long.
        """
        return int(<size_t>self._event_ref)


cdef class SyclKernel:
    ''' Wraps a sycl::kernel object created from an OpenCL interoperability
        kernel.
    '''

    @staticmethod
    cdef SyclKernel _create (DPPLSyclKernelRef kref):
        cdef SyclKernel ret = SyclKernel.__new__(SyclKernel)
        ret._kernel_ref = kref
        ret._function_name = DPPLKernel_GetFunctionName(kref)
        return ret

    def __dealloc__ (self):
        DPPLKernel_Delete(self._kernel_ref)
        DPPLCString_Delete(self._function_name)

    def get_function_name (self):
        ''' Returns the name of the Kernel function.
        '''
        return self._function_name.decode()

    def get_num_args (self):
        ''' Returns the number of arguments for this kernel function.
        '''
        return DPPLKernel_GetNumArgs(self._kernel_ref)

    cdef DPPLSyclKernelRef get_kernel_ref (self):
        ''' Returns the DPPLSyclKernelRef pointer for this SyclKernel.
        '''
        return self._kernel_ref

    def addressof_ref (self):
        """Returns the address of the C API DPPLSyclKernelRef pointer
        as a long.

        Returns:
            The address of the DPPLSyclKernelRef object used to create this
            SyclKernel cast to a long.
        """
        return int(<size_t>self._kernel_ref)

cdef class SyclProgram:
    ''' Wraps a sycl::program object created from an OpenCL interoperability
        program.

        SyclProgram exposes the C API from dppl_sycl_program_interface.h. A
        SyclProgram can be created from either a source string or a SPIR-V
        binary file.
    '''

    @staticmethod
    cdef SyclProgram _create (DPPLSyclProgramRef pref):
        cdef SyclProgram ret = SyclProgram.__new__(SyclProgram)
        ret._program_ref = pref
        return ret

    def __dealloc__ (self):
        DPPLProgram_Delete(self._program_ref)

    cdef DPPLSyclProgramRef get_program_ref (self):
        return self._program_ref

    cpdef SyclKernel get_sycl_kernel(self, str kernel_name):
        name = kernel_name.encode('utf8')
        return SyclKernel._create(DPPLProgram_GetKernel(self._program_ref,
                                                        name))

    def has_sycl_kernel(self, str kernel_name):
        name = kernel_name.encode('utf8')
        return DPPLProgram_HasKernel(self._program_ref, name)

    def addressof_ref (self):
        """Returns the address of the C API DPPLSyclProgramRef pointer
        as a long.

        Returns:
            The address of the DPPLSyclProgramRef object used to create this
            SyclProgram cast to a long.
        """
        return int(<size_t>self._program_ref)

import ctypes

cdef class SyclQueue:
    ''' Wrapper class for a Sycl queue.
    '''

    @staticmethod
    cdef SyclQueue _create (DPPLSyclQueueRef qref):
        if qref is NULL:
            raise SyclQueueCreationError("Queue creation failed.")
        cdef SyclQueue ret = SyclQueue.__new__(SyclQueue)
        ret._context = SyclContext._create(DPPLQueue_GetContext(qref))
        ret._device = SyclDevice._create(DPPLQueue_GetDevice(qref))
        ret._queue_ref = qref
        return ret

    def __dealloc__ (self):
        DPPLQueue_Delete(self._queue_ref)

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
                             DPPLKernelArgType *kargty):
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
            elif isinstance(arg, Memory):
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
        return DPPLQueue_AreEq(self._queue_ref, q.get_queue_ref())

    def get_sycl_backend (self):
        """ Returns the Sycl bakend associated with the queue.
        """
        cdef DPPLSyclBackendType BE = DPPLQueue_GetBackend(self._queue_ref)
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

    cdef DPPLSyclQueueRef get_queue_ref (self):
        return self._queue_ref

    def addressof_ref (self):
        """Returns the address of the C API DPPLSyclQueueRef pointer as
        a long.

        Returns:
            The address of the DPPLSyclQueueRef object used to create this
            SyclQueue cast to a long.
        """
        return int(<size_t>self._queue_ref)

    cpdef SyclEvent submit (self, SyclKernel kernel, list args, list gS,       \
                            list lS = None, list dEvents = None):

        cdef void **kargs = NULL
        cdef DPPLKernelArgType *kargty = NULL
        cdef DPPLSyclEventRef *depEvents = NULL
        cdef DPPLSyclEventRef Eref = NULL
        cdef int ret
        cdef size_t gRange[3]
        cdef size_t lRange[3]
        cdef size_t nGS = len(gS)
        cdef size_t nLS = len(lS) if lS is not None else 0
        cdef size_t nDE = len(dEvents) if dEvents is not None else 0

        # Allocate the arrays to be sent to DPPLQueue_Submit
        kargs = <void**>malloc(len(args) * sizeof(void*))
        if not kargs:
            raise MemoryError()
        kargty = <DPPLKernelArgType*>malloc(len(args)*sizeof(DPPLKernelArgType))
        if not kargty:
            free(kargs)
            raise MemoryError()
        # Create the array of dependent events if any
        if dEvents is not None and nDE > 0:
            depEvents = <DPPLSyclEventRef*>malloc(nDE*sizeof(DPPLSyclEventRef))
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

            Eref = DPPLQueue_SubmitRange(kernel.get_kernel_ref(),
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

            Eref = DPPLQueue_SubmitNDRange(kernel.get_kernel_ref(),
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
            self._raise_queue_submit_error("DPPLQueue_Submit", -1)

        return SyclEvent._create(Eref, args)

    cpdef void wait (self):
        DPPLQueue_Wait(self._queue_ref)

    cpdef memcpy (self, dest, src, size_t count):
        cdef void *c_dest
        cdef void *c_src

        if isinstance(dest, Memory):
            c_dest = <void*>(<Memory>dest).memory_ptr
        else:
            raise TypeError("Parameter dest should have type Memory.")

        if isinstance(src, Memory):
            c_src = <void*>(<Memory>src).memory_ptr
        else:
            raise TypeError("Parameter src should have type Memory.")

        DPPLQueue_Memcpy(self._queue_ref, c_dest, c_src, count)

    cpdef prefetch (self, mem, size_t count=0):
       cdef void *ptr

       if isinstance(mem, Memory):
           ptr = <void*>(<Memory>mem).memory_ptr
       else:
           raise TypeError("Parameter mem should have type Memory")

       if (count <=0 or count > self.nbytes):
           count = self.nbytes

       DPPLQueue_Prefetch(self._queue_ref, ptr, count)

    cpdef mem_advise (self, mem, size_t count, int advice):
       cdef void *ptr

       if isinstance(mem, Memory):
           ptr = <void*>(<Memory>mem).memory_ptr
       else:
           raise TypeError("Parameter mem should have type Memory")

       if (count <=0 or count > self.nbytes):
           count = self.nbytes

       DPPLQueue_MemAdvise(self._queue_ref, ptr, count, advice)


cdef class _SyclRTManager:
    ''' Wrapper for the C API's sycl queue manager interface.
    '''
    cdef dict _backend_str_ty_dict
    cdef dict _device_str_ty_dict
    cdef dict _backend_enum_ty_dict
    cdef dict _device_enum_ty_dict

    def __cinit__ (self):

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

    def _set_as_current_queue (self, backend_ty, device_ty, device_id):
        cdef DPPLSyclQueueRef queue_ref

        try :
            beTy = self._backend_str_ty_dict[backend_ty]
            try :
                devTy = self._device_str_ty_dict[device_ty]
                queue_ref = DPPLQueueMgr_PushQueue(beTy, devTy, device_id)
                return SyclQueue._create(queue_ref)
            except KeyError:
                raise UnsupportedDeviceError("Device can only be gpu or cpu")
        except KeyError:
            raise UnsupportedBackendError("Backend can only be opencl or "
                                          "level-0")

    def _remove_current_queue (self):
        DPPLQueueMgr_PopQueue()

    def dump (self):
        ''' Prints information about the Runtime object.
        '''
        DPPLPlatform_DumpInfo()

    def print_available_backends (self):
        """ Prints the available backends.
        """
        print(self._backend_str_ty_dict.keys())

    def get_current_backend (self):
        """ Returns the backend for the current queue as `backend_type` enum
        """
        return self.get_current_queue().get_sycl_backend()

    def get_current_device_type (self):
        ''' Returns current device type as `device_type` enum
        '''
        return self.get_current_queue().get_sycl_device().get_device_type()

    cpdef SyclQueue get_current_queue (self):
        ''' Returns the activated SYCL queue as a PyCapsule.
        '''
        return SyclQueue._create(DPPLQueueMgr_GetCurrentQueue())

    def get_num_activated_queues (self):
        ''' Return the number of currently activated queues for this thread.
        '''
        return DPPLQueueMgr_GetNumActivatedQueues()

    def get_num_platforms (self):
        ''' Returns the number of available non-host SYCL platforms.
        '''
        return DPPLPlatform_GetNumNonHostPlatforms()

    def get_num_queues (self, backend_ty, device_ty):
        cdef size_t num = 0
        try :
            beTy = self._backend_enum_ty_dict[backend_ty]
            try :
                devTy = self._device_enum_ty_dict[device_ty]
                num = DPPLQueueMgr_GetNumQueues(beTy, devTy)
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

    def has_gpu_queues (self, backend_ty=backend_type.opencl):
        cdef size_t num = 0
        try :
            beTy = self._backend_enum_ty_dict[backend_ty]
            num = DPPLQueueMgr_GetNumQueues(beTy, _device_type._GPU)
        except KeyError:
            raise UnsupportedBackendError(
                      "Backend can only be backend_type.opencl or "
                      "backend_type.level_zero"
                  )
        if num:
            return True
        else:
            return False

    def has_cpu_queues (self, backend_ty=backend_type.opencl):
        cdef size_t num = 0
        try :
            beTy = self._backend_enum_ty_dict[backend_ty]
            num = DPPLQueueMgr_GetNumQueues(beTy, _device_type._CPU)
        except KeyError:
            raise UnsupportedBackendError(
                      "Backend can only be backend_type.opencl or "
                      "backend_type.level_zero"
                  )
        if num:
            return True
        else:
            return False

    def has_sycl_platforms (self):
        cdef size_t num_platforms = DPPLPlatform_GetNumNonHostPlatforms()
        if num_platforms:
            return True
        else:
            return False

    def is_in_device_context (self):
        cdef size_t num = DPPLQueueMgr_GetNumActivatedQueues()
        if num:
            return True
        else:
            return False

    def set_default_queue (self, backend_ty, device_ty, device_id):
        cdef DPPLSyclQueueRef ret
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
                ret = DPPLQueueMgr_SetAsDefaultQueue(beTy, devTy, device_id)
                if ret is NULL:
                    self._raise_queue_creation_error(
                        backend_ty, device_ty, device_id,
                        "DPPLQueueMgr_PushQueue"
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
get_current_device_type  = _mgr.get_current_device_type
get_num_platforms        = _mgr.get_num_platforms
get_num_activated_queues = _mgr.get_num_activated_queues
get_num_queues           = _mgr.get_num_queues
has_cpu_queues           = _mgr.has_cpu_queues
has_gpu_queues           = _mgr.has_gpu_queues
has_sycl_platforms       = _mgr.has_sycl_platforms
set_default_queue        = _mgr.set_default_queue
is_in_device_context     = _mgr.is_in_device_context

cpdef SyclQueue get_current_queue():
    ''' Obtain current Sycl Queue from Data Parallel Control package '''
    return _mgr.get_current_queue()


def create_program_from_source (SyclQueue q, unicode source, unicode copts=""):
    ''' Creates a Sycl interoperability program from an OpenCL source string.

        We use the DPPLProgram_CreateFromOCLSource() C API function to create
        a Sycl progrma from an OpenCL source program that can contain multiple
        kernels.

        Parameters:
                q (SyclQueue)   : The SyclQueue object wraps the Sycl device for
                                  which the program will be built.
                source (unicode): Source string for an OpenCL program.
                copts (unicode) : Optional compilation flags that will be used
                                  when compiling the program.

        Returns:
            program (SyclProgram): A SyclProgram object wrapping the
                                    syc::program returned by the C API.
    '''

    BE = q.get_sycl_backend()
    if BE != backend_type.opencl:
        raise ValueError(
            "Cannot create program for a ", BE, "type backend. Currently only "
            "OpenCL devices are supported for program creations."
        )

    cdef DPPLSyclProgramRef Pref
    cdef bytes bSrc = source.encode('utf8')
    cdef bytes bCOpts = copts.encode('utf8')
    cdef const char *Src = <const char*>bSrc
    cdef const char *COpts = <const char*>bCOpts
    cdef DPPLSyclContextRef CRef = q.get_sycl_context().get_context_ref()
    Pref = DPPLProgram_CreateFromOCLSource(CRef, Src, COpts)

    if Pref is NULL:
        raise SyclProgramCompilationError()

    return SyclProgram._create(Pref)

cimport cython.array

def create_program_from_spirv (SyclQueue q, const unsigned char[:] IL):
    ''' Creates a Sycl interoperability program from an SPIR-V binary.

        We use the DPPLProgram_CreateFromOCLSpirv() C API function to create
        a Sycl progrma from an compiled SPIR-V binary file.

        Parameters:
            q (SyclQueue): The SyclQueue object wraps the Sycl device for
                           which the program will be built.
            IL (const char[:]) : SPIR-V binary IL file for an OpenCL program.

        Returns:
            program (SyclProgram): A SyclProgram object wrapping the
                                   syc::program returned by the C API.
    '''
    BE = q.get_sycl_backend()
    if BE != backend_type.opencl:
        raise ValueError(
            "Cannot create program for a ", BE, "type backend. Currently only "
            "OpenCL devices are supported for program creations."
        )

    cdef DPPLSyclProgramRef Pref
    cdef const unsigned char *dIL = &IL[0]
    cdef DPPLSyclContextRef CRef = q.get_sycl_context().get_context_ref()
    cdef size_t length = IL.shape[0]
    Pref = DPPLProgram_CreateFromOCLSpirv(CRef, <const void*>dIL, length)
    if Pref is NULL:
        raise SyclProgramCompilationError()

    return SyclProgram._create(Pref)


from contextlib import contextmanager

@contextmanager
def device_context (str queue_str="opencl:gpu:0"):
    # Create a new device context and add it to the front of the runtime's
    # deque of active contexts (SyclQueueManager.active_contexts_).
    # Also return a reference to the context. The behavior allows consumers
    # of the context manager to either use the new context by indirectly
    # calling get_current_context, or use the returned context object directly.

    # If set_context is unable to create a new context an exception is raised.
    ctxt = None
    try:
        attrs = queue_str.split(':')
        nattrs = len(attrs)
        if (nattrs < 2 or nattrs > 3):
            raise ValueError("Invalid device context string. Should be "
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
                "Removing the context from the stack of active contexts")
            _mgr._remove_current_queue()
        else:
            _logger.debug("No context was created so nothing to do")
