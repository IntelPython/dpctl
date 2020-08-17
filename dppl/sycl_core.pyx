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
###
### \file
### This file implements the Cython interface for the Sycl API of PyDPPL.
###
##===----------------------------------------------------------------------===##

# distutils: language = c++
# cython: language_level=2

from __future__ import print_function
from libc.stdint cimport int64_t
from libcpp cimport bool
from libcpp.memory cimport shared_ptr, make_shared
from cpython.pycapsule cimport (PyCapsule_New,
                                PyCapsule_IsValid,
                                PyCapsule_GetPointer)
from enum import Enum, auto

class device_type(Enum):
    gpu = auto()
    cpu = auto()


cdef class UnsupportedDeviceTypeError(Exception):
    """This exception is raised when a device type other than CPU or GPU is
       encountered.
    """
    pass


cdef extern from "dppl_sycl_interface.hpp" namespace "dppl":

    cdef enum _device_type 'sycl_device_type':
        _GPU 'dppl::sycl_device_type::gpu'
        _CPU 'dppl::sycl_device_type::cpu'

    cdef cppclass DpplSyclQueueManager:
        DpplSyclQueueManager () except +
        int64_t getNumPlatforms (size_t &num_platform) except -1
        int64_t getCurrentQueue (void **Q) except -1
        int64_t getQueue (void **Q, _device_type DTy,
                          size_t device_num) except -1
        int64_t setAsGlobalQueue (_device_type DTy,
                                  size_t device_num) except -1
        int64_t setAsCurrentQueue (void **Q, _device_type DTy,
                                   size_t device_num) except -1
        int64_t removeCurrentQueue () except -1
        int64_t dump () except -1
        int64_t dump_queue (const void *Q) except -1

    cdef int64_t deleteQueue (void *Q) except -1

# Destructor for a PyCapsule containing a SYCL queue
cdef void delete_queue (object cap):
    deleteQueue(PyCapsule_GetPointer(cap, NULL))


cdef class SyclQueueManager:
    cdef DpplSyclQueueManager rt

    def __cinit__ (self):
        self.rt = DpplSyclQueueManager()

    def get_num_platforms (self):
        ''' Returns the number of available SYCL/OpenCL platforms.
        '''
        cdef size_t num_platforms = 0
        self.rt.getNumPlatforms(num_platforms)
        return num_platforms

    def _set_as_current_queue (self, device_ty, device_id):
        cdef void *queue_ptr
        if device_ty == device_type.gpu:
            self.rt.setAsCurrentQueue(&queue_ptr, _device_type._GPU, device_id)
        elif device_ty == device_type.cpu:
            self.rt.setAsCurrentQueue(&queue_ptr, _device_type._CPU, device_id)
        else:
            e = UnsupportedDeviceTypeError("Device can only be cpu or gpu")
            raise e

        return PyCapsule_New(queue_ptr, NULL, &delete_queue)

    def _remove_current_queue (self):
        self.rt.removeCurrentQueue()

    def get_current_queue (self):
        ''' Returns the activated SYCL queue as a PyCapsule.
        '''
        cdef void* queue_ptr = NULL;
        self.rt.getCurrentQueue(&queue_ptr);
        return PyCapsule_New(queue_ptr, NULL, &delete_queue)

#    def set_global_queue (self, device_ty, device_id):
#        if device_ty == device_type.gpu:
#            self.rt.setGlobalContextWithGPU(device_id)
#        elif device_ty == device_type.cpu:
#            self.rt.setGlobalContextWithCPU(device_id)
#        else:
#            e = UnsupportedDeviceTypeError("Device can only be cpu or gpu")
#            raise e

    def dump (self):
        ''' Prints information about the Runtime object.
        '''
        return self.rt.dump()

    def dump_queue (self, queue_cap):
        ''' Prints information about the SYCL queue object.
        '''
        if PyCapsule_IsValid(queue_cap, NULL):
            self.rt.dump_queue(PyCapsule_GetPointer(queue_cap, NULL))
        else:
            raise ValueError("Expected a PyCapsule encapsulating a SYCL queue")

# Global runtime object
runtime = SyclQueueManager()

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
        ctxt = runtime._set_as_current_queue(dev, device_num)
        yield ctxt
    finally:
        # Code to release resource
        if ctxt:
            print("Removing the context from the deque of active contexts")
            runtime._remove_current_queue()
        else:
            print("No context was created so nothing to do")
