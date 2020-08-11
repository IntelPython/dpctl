##===---------- oneapi_interface.pyx - DPPL interface -----*- Cython -*----===##
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
### This file implements the Cython interface for the PyDPPL package.
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


cdef extern from "dppl_oneapi_interface.hpp" namespace "dppl":
    cdef cppclass DpplOneAPIRuntime:
        DpplOneAPIRuntime () except +
        int64_t getNumPlatforms (size_t *num_platform) except -1
        int64_t getCurrentQueue (void **Q) except -1
        int64_t getQueue (void **Q, _device_type DTy,
                          size_t device_num) except -1
        int64_t resetGlobalQueue (_device_type DTy,
                                  size_t device_num) except -1
        int64_t activateQueue (void **Q, _device_type DTy,
                               size_t device_num) except -1
        int64_t deactivateCurrentQueue () except -1
        int64_t number_of_activated_queues (size_t &num) except -1
        int64_t dump () except -1
        int64_t dump_queue (const void *Q) except -1

    cdef int64_t deleteQueue (void *Q) except -1

    cdef enum _device_type 'sycl_device_type':
        _GPU 'dppl::sycl_device_type::gpu'
        _CPU 'dppl::sycl_device_type::cpu'

# Destructor for a PyCapsule containing a SYCL queue
cdef void delete_queue (object cap):
    deleteQueue(PyCapsule_GetPointer(cap, NULL))


cdef class _DpplRuntime:
    cdef DpplOneAPIRuntime rt

    def __cinit__ (self):
        self.rt = DpplOneAPIRuntime()

    def _activate_queue (self, device_ty, device_id):
        cdef void *queue_ptr
        if device_ty == device_type.gpu:
            self.rt.activateQueue(&queue_ptr, _device_type._GPU, device_id)
        elif device_ty == device_type.cpu:
            self.rt.activateQueue(&queue_ptr, _device_type._CPU, device_id)
        else:
            e = UnsupportedDeviceTypeError("Device can only be cpu or gpu")
            raise e

        return PyCapsule_New(queue_ptr, NULL, &delete_queue)

    def _deactivate_current_queue (self):
        self.rt.deactivateCurrentQueue()

    def dump (self):
        ''' Prints information about the Runtime object.
        '''
        return self.rt.dump()

    def dump_queue_info (self, queue_cap):
        ''' Prints information about the SYCL queue object.
        '''
        if PyCapsule_IsValid(queue_cap, NULL):
            return self.rt.dump_queue(PyCapsule_GetPointer(queue_cap, NULL))
        else:
            raise ValueError("Expected a PyCapsule encapsulating a SYCL queue")

    def get_current_queue (self):
        ''' Returns the activated SYCL queue as a PyCapsule.
        '''
        cdef void* queue_ptr = NULL;
        self.rt.getCurrentQueue(&queue_ptr);
        return PyCapsule_New(queue_ptr, NULL, &delete_queue)

    def get_num_platforms (self):
        ''' Returns the number of available SYCL/OpenCL platforms.
        '''
        cdef size_t num_platforms = 0
        self.rt.getNumPlatforms(&num_platforms)
        return num_platforms

    def set_default_queue (self, device_ty, device_id):
        if device_ty == device_type.gpu:
            self.rt.resetGlobalQueue(_device_type._GPU, device_id)
        elif device_ty == device_type.cpu:
            self.rt.resetGlobalQueue(_device_type._CPU, device_id)
        else:
            e = UnsupportedDeviceTypeError("Device can only be cpu or gpu")
            raise e

    def is_in_dppl_ctxt (self):
        cdef size_t num = 0
        self.rt.number_of_activated_queues(num)
        if num:
            return True
        else:
            return False


# thread-local storage
from threading import local as threading_local

# Initialize a thread local instance of _DpplRuntime
_tls = threading_local()
_tls._runtime = _DpplRuntime()


################################################################################
#--------------------------------- Public API ---------------------------------#
################################################################################


dump              = _tls._runtime.dump
dump_queue_info   = _tls._runtime.dump_queue_info
get_current_queue = _tls._runtime.get_current_queue
get_num_platforms = _tls._runtime.get_num_platforms
set_default_queue = _tls._runtime.set_default_queue
is_in_dppl_ctxt   = _tls._runtime.is_in_dppl_ctxt

from contextlib import contextmanager

@contextmanager
def device_context (dev=device_type.gpu, device_num=0):
    # Create a new device context and add it to the front of the runtime's
    # deque of active contexts (DpplOneAPIRuntime.ctive_contexts_).
    # Also return a reference to the context. The behavior allows consumers
    # of the context manager to either use the new context by indirectly
    # calling get_current_context, or use the returned context object directly.

    # If set_context is unable to create a new context an exception is raised.
    try:
        ctxt = None
        ctxt = _tls._runtime._activate_queue(dev, device_num)
        yield ctxt
    finally:
        # Code to release resource
        if ctxt:
            _tls._runtime._deactivate_current_queue()
        else:
            print("No context was created so nothing to do")
