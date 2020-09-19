##===------------- sycl_core.pyx - DPPL interface ------*- Cython -*-------===##
##
##               Python Data Parallel Processing Library (PyDPPL)
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
## This file implements the Cython interface for the Sycl API of PyDPPL.
##
##===----------------------------------------------------------------------===##

# distutils: language = c++
# cython: language_level=3

from __future__ import print_function
from enum import Enum, auto
import logging
from dppl.backend cimport *


_logger = logging.getLogger(__name__)


class device_type(Enum):
    gpu = auto()
    cpu = auto()


cdef class UnsupportedDeviceTypeError(Exception):
    '''This exception is raised when a device type other than CPU or GPU is
       encountered.
    '''
    pass


cdef class SyclContext:

    @staticmethod
    cdef SyclContext _create (DPPLSyclContextRef ctxt):
        cdef SyclContext ret = SyclContext.__new__(SyclContext)
        ret.ctxt_ptr = ctxt
        return ret

    def __dealloc__ (self):
        DPPLDeleteSyclContext(self.ctxt_ptr)

    cdef DPPLSyclContextRef get_context_ref (self):
        return self.ctxt_ptr


cdef class SyclDevice:
    ''' Wrapper class for a Sycl Device
    '''

    @staticmethod
    cdef SyclDevice _create (DPPLSyclDeviceRef dref):
        cdef SyclDevice ret = SyclDevice.__new__(SyclDevice)
        ret.device_ptr = dref
        ret.vendor_name = DPPLGetDeviceVendorName(dref)
        ret.device_name = DPPLGetDeviceName(dref)
        ret.driver_version = DPPLGetDeviceDriverInfo(dref)
        return ret

    def __dealloc__ (self):
        DPPLDeleteSyclDevice(self.device_ptr)
        DPPLDeleteCString(self.device_name)
        DPPLDeleteCString(self.vendor_name)
        DPPLDeleteCString(self.driver_version)

    def dump_device_info (self):
        ''' Print information about the SYCL device.
        '''
        DPPLDumpDeviceInfo(self.device_ptr)

    def get_device_name (self):
        ''' Returns the name of the device as a string
        '''
        return self.device_name

    def get_vendor_name (self):
        ''' Returns the device vendor name as a string
        '''
        return self.vendor_name

    def get_driver_version (self):
        ''' Returns the OpenCL software driver version as a string
            in the form: major number.minor number, if this SYCL
            device is an OpenCL device. Returns a string class
            with the value "1.2" if this SYCL device is a host device.
        '''
        return self.driver_version

    cdef DPPLSyclDeviceRef get_device_ptr (self):
        ''' Returns the DPPLSyclDeviceRef pointer for this class.
        '''
        return self.device_ptr


cdef class SyclQueue:
    ''' Wrapper class for a Sycl queue.
    '''

    @staticmethod
    cdef SyclQueue _create (DPPLSyclQueueRef qref):
        cdef SyclQueue ret = SyclQueue.__new__(SyclQueue)
        ret.queue_ptr = qref
        return ret

    def __dealloc__ (self):
        DPPLDeleteSyclQueue(self.queue_ptr)

    cpdef SyclContext get_sycl_context (self):
        return SyclContext._create(DPPLGetContextFromQueue(self.queue_ptr))

    cpdef SyclDevice get_sycl_device (self):
        return SyclDevice._create(DPPLGetDeviceFromQueue(self.queue_ptr))

    cdef DPPLSyclQueueRef get_queue_ref (self):
        return self.queue_ptr


cdef class _SyclQueueManager:
    def _set_as_current_queue (self, device_ty, device_id):
        cdef DPPLSyclQueueRef queue_ptr
        if device_ty == device_type.gpu:
            queue_ptr = DPPLPushSyclQueue(_device_type._GPU, device_id)
        elif device_ty == device_type.cpu:
            queue_ptr = DPPLPushSyclQueue(_device_type._CPU, device_id)
        else:
            e = UnsupportedDeviceTypeError("Device can only be cpu or gpu")
            raise e

        return SyclQueue._create(queue_ptr)

    def _remove_current_queue (self):
        DPPLPopSyclQueue()

    def has_sycl_platforms (self):
        cdef size_t num_platforms = DPPLPlatform_GetNumPlatforms()
        if num_platforms:
            return True
        else:
            return False

    def get_num_platforms (self):
        ''' Returns the number of available SYCL/OpenCL platforms.
        '''
        return DPPLPlatform_GetNumPlatforms()

    def get_num_activated_queues (self):
        ''' Return the number of currently activated queues for this thread.
        '''
        return DPPLGetNumActivatedQueues()

    def get_current_queue (self):
        ''' Returns the activated SYCL queue as a PyCapsule.
        '''
        return SyclQueue._create(DPPLGetCurrentQueue())

    def set_default_queue (self, device_ty, device_id):
        if device_ty == device_type.gpu:
            DPPLSetAsDefaultQueue(_device_type._GPU, device_id)
        elif device_ty == device_type.cpu:
            DPPLSetAsDefaultQueue(_device_type._CPU, device_id)
        else:
            e = UnsupportedDeviceTypeError("Device can only be cpu or gpu")
            raise e

    def has_gpu_queues (self):
        cdef size_t num = DPPLGetNumGPUQueues()
        if num:
            return True
        else:
            return False

    def has_cpu_queues (self):
        cdef size_t num = DPPLGetNumCPUQueues()
        if num:
            return True
        else:
            return False

    def dump (self):
        ''' Prints information about the Runtime object.
        '''
        DPPLPlatform_DumpInfo()

    def is_in_dppl_ctxt (self):
        cdef size_t num = DPPLGetNumActivatedQueues()
        if num:
            return True
        else:
            return False

# This private instance of the _SyclQueueManager should not be directly
# accessed outside the module.
_qmgr = _SyclQueueManager()

# Global bound functions
dump                     = _qmgr.dump
get_current_queue        = _qmgr.get_current_queue
get_num_platforms        = _qmgr.get_num_platforms
get_num_activated_queues = _qmgr.get_num_activated_queues
has_cpu_queues           = _qmgr.has_cpu_queues
has_gpu_queues           = _qmgr.has_gpu_queues
has_sycl_platforms       = _qmgr.has_sycl_platforms
set_default_queue        = _qmgr.set_default_queue
is_in_dppl_ctxt          = _qmgr.is_in_dppl_ctxt

from contextlib import contextmanager

@contextmanager
def device_context (dev=device_type.gpu, device_num=0):
    # Create a new device context and add it to the front of the runtime's
    # deque of active contexts (SyclQueueManager.active_contexts_).
    # Also return a reference to the context. The behavior allows consumers
    # of the context manager to either use the new context by indirectly
    # calling get_current_context, or use the returned context object directly.

    # If set_context is unable to create a new context an exception is raised.
    try:
        ctxt = None
        ctxt = _qmgr._set_as_current_queue(dev, device_num)
        yield ctxt
    finally:
        # Code to release resource
        if ctxt:
            _logger.debug(
                "Removing the context from the stack of active contexts")
            _qmgr._remove_current_queue()
        else:
            _logger.debug("No context was created so nothing to do")
