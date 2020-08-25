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
from cpython.pycapsule cimport (PyCapsule_New,
                                PyCapsule_IsValid,
                                PyCapsule_GetPointer)
from enum import Enum, auto
import logging

_logger = logging.getLogger(__name__)

class device_type(Enum):
    gpu = auto()
    cpu = auto()


cdef class UnsupportedDeviceTypeError(Exception):
    '''This exception is raised when a device type other than CPU or GPU is
       encountered.
    '''
    pass

cdef extern from "dppl_sycl_types.h":

    cdef struct DPPLOpaqueSyclQueue:
        pass
    ctypedef DPPLOpaqueSyclQueue* DPPLSyclQueueRef

cdef extern from "dppl_sycl_queue_interface.h":

    cdef enum _device_type 'DPPLSyclDeviceType':
        _GPU 'DPPL_GPU'
        _CPU 'DPPL_CPU'

    cdef void DPPLDumpPlatformInfo () except +
    cdef void DPPLDumpDeviceInfo (const DPPLSyclQueueRef Q) except +
    cdef DPPLSyclQueueRef DPPLGetCurrentQueue () except +
    cdef size_t DPPLGetNumCPUQueues () except +
    cdef size_t DPPLGetNumGPUQueues () except +
    cdef size_t DPPLGetNumActivatedQueues () except +
    cdef size_t DPPLGetNumPlatforms () except +
    cdef DPPLSyclQueueRef DPPLGetQueue (_device_type DTy,
                                        size_t device_num) except +
    cdef void DPPLRemoveCurrentQueue () except +
    cdef DPPLSyclQueueRef DPPLSetAsCurrentQueue (_device_type DTy,
                                                 size_t device_num) except +
    cdef void DPPLSetAsDefaultQueue (_device_type DTy,
                                     size_t device_num) except +

    cdef void DPPLDeleteQueue (DPPLSyclQueueRef Q) except +

# Destructor for a PyCapsule containing a SYCL queue
cdef void delete_queue (object cap):
    DPPLDeleteQueue(<DPPLSyclQueueRef>PyCapsule_GetPointer(cap, NULL))


cdef class _SyclQueueManager:

    def _set_as_current_queue (self, device_ty, device_id):
        cdef DPPLSyclQueueRef queue_ptr
        if device_ty == device_type.gpu:
            queue_ptr = DPPLSetAsCurrentQueue(_device_type._GPU, device_id)
        elif device_ty == device_type.cpu:
            queue_ptr = DPPLSetAsCurrentQueue(_device_type._CPU, device_id)
        else:
            e = UnsupportedDeviceTypeError("Device can only be cpu or gpu")
            raise e

        return PyCapsule_New(queue_ptr, NULL, &delete_queue)

    def _remove_current_queue (self):
        DPPLRemoveCurrentQueue()

    def has_sycl_platforms (self):
        cdef size_t num_platforms = DPPLGetNumPlatforms()
        if num_platforms:
            return True
        else:
            return False

    def get_num_platforms (self):
        ''' Returns the number of available SYCL/OpenCL platforms.
        '''
        return DPPLGetNumPlatforms()

    def get_num_activated_queues (self):
        ''' Return the number of currently activated queues for this thread.
        '''
        return DPPLGetNumActivatedQueues()

    def get_current_queue (self):
        ''' Returns the activated SYCL queue as a PyCapsule.
        '''
        cdef DPPLSyclQueueRef queue_ptr = DPPLGetCurrentQueue()
        return PyCapsule_New(queue_ptr, NULL, &delete_queue)

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
        DPPLDumpPlatformInfo()
        return 1

    def dump_device_info (self, queue_cap):
        ''' Prints information about the SYCL queue object.
        '''
        if PyCapsule_IsValid(queue_cap, NULL):
            DPPLDumpDeviceInfo(
                <DPPLSyclQueueRef>PyCapsule_GetPointer(queue_cap, NULL)
            )
            return 1
        else:
            raise ValueError("Expected a PyCapsule encapsulating a SYCL queue")

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
dump_device_info         = _qmgr.dump_device_info
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
