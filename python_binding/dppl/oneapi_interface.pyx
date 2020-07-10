#*******************************************************************************
# Copyright 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#******************************************************************************/

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
    """When expecting either a DeviceArray or numpy.ndarray object
    """
    pass


cdef extern from "dppl_oneapi_interface.hpp" namespace "dppl":
    cdef cppclass DppyOneAPIRuntime:
        DppyOneAPIRuntime () except +
        int64_t getNumPlatforms (size_t *num_platform) except -1
        int64_t getCurrentQueue (void **Q) except -1
        int64_t getQueue (void **Q, _device_type DTy,
                          size_t device_num) except -1
        int64_t resetGlobalQueue (_device_type DTy,
                                  size_t device_num) except -1
        int64_t activateQueue (void **Q, _device_type DTy,
                               size_t device_num) except -1
        int64_t deactivateCurrentQueue () except -1
        int64_t dump () except -1
        int64_t dump_queue (const void *Q) except -1

    cdef int64_t deleteQueue (void *Q) except -1

    cdef enum _device_type 'sycl_device_type':
        _GPU 'dppl::sycl_device_type::gpu'
        _CPU 'dppl::sycl_device_type::cpu'

# Destructor for a PyCapsule containing a SYCL queue
cdef void delete_queue (object cap):
    deleteQueue(PyCapsule_GetPointer(cap, NULL))


cdef class DppyRuntime:
    cdef DppyOneAPIRuntime rt

    def __cinit__ (self):
        self.rt = DppyOneAPIRuntime()

    def get_num_platforms (self):
        cdef size_t num_platforms = 0
        self.rt.getNumPlatforms(&num_platforms)
        return num_platforms

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

    def get_current_queue (self):
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
        return self.rt.dump()

    def dump_queue (self, queue_cap):
        if PyCapsule_IsValid(queue_cap, NULL):
            self.rt.dump_queue(PyCapsule_GetPointer(queue_cap, NULL))
        else:
            raise ValueError("Expected a PyCapsule encapsulating a SYCL queue")

# Global runtime object
runtime = DppyRuntime()

from contextlib import contextmanager

@contextmanager
def device_context (dev=device_type.gpu, device_num=0):
    # Create a new device context and add it to the front of the runtime's
    # deque of active contexts (DppyOneAPIRuntime.ctive_contexts_).
    # Also return a reference to the context. The behavior allows consumers
    # of the context manager to either use the new context by indirectly
    # calling get_current_context, or use the returned context object directly.

    # If set_context is unable to create a new context an exception is raised.
    try:
        ctxt = None
        ctxt = runtime._activate_queue(dev, device_num)
        yield ctxt
    finally:
        # Code to release resource
        if ctxt:
            print("Debug: Removing the context from the deque of active contexts")
            runtime._deactivate_current_queue()
        else:
            print("Debug: No context was created so nothing to do")
