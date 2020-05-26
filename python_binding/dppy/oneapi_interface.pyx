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
from cpython.ref cimport PyObject, Py_INCREF
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


cdef extern from "dppy_oneapi_interface.hpp" namespace "dppy":
    cdef cppclass DppyOneAPIContext:
        DppyOneAPIContext (const DppyOneAPIContext & C) except +
        int64_t dump () except -1


cdef extern from "dppy_oneapi_interface.hpp" namespace "dppy":
    cdef cppclass DppyOneAPIRuntime:
        DppyOneAPIRuntime () except +
        int64_t getNumPlatforms (size_t * num_platform) except -1
        int64_t getDefaultContext (shared_ptr[DppyOneAPIContext] * C) except -1
        int64_t pushCPUContext (shared_ptr[DppyOneAPIContext] * C,
                                size_t device_num) except -1
        int64_t pushGPUContext (shared_ptr[DppyOneAPIContext] * C,
                                   size_t device_num) except -1
        int64_t popContext () except -1
        int64_t dump () except -1


cdef class DppyContext:
    cdef shared_ptr[DppyOneAPIContext] *ctx;

    def __cinit__ (self, ctx):
        if PyCapsule_IsValid(ctx, NULL):
            self.ctx = (
                <shared_ptr[DppyOneAPIContext]*>PyCapsule_GetPointer(ctx, NULL)
            )
        else:
            raise ValueError("Expected a PyCapsule with a \
                              shared_ptr<DppyOneAPIContext>*")

    def __dealloc__ (self):
        del self.ctx

    def dump (self):
        (self.ctx).get().dump()

cdef class DppyRuntime:
    cdef DppyOneAPIRuntime rt

    def __cinit__ (self):
        self.rt = DppyOneAPIRuntime()

    def get_num_platforms (self):
        cdef size_t num_platforms = 0
        self.rt.getNumPlatforms(&num_platforms)
        return num_platforms

    def set_context (self, device_ty, device_id):
        # Create a dynamically allocated std::shared_ptr<DppyOneAPIContext>
        # instance. The shapred_ptr is initialized inside pushXXXContext. The
        # instantiated shared_ptr is then stored inside the encapsulating
        # DppyContext object that is returned to the caller. The shared_ptr
        # is freed when the DppyObject is destroyed.
        cdef shared_ptr[DppyOneAPIContext] *ctx = (
            new shared_ptr[DppyOneAPIContext]()
        )
        if device_ty == device_type.gpu:
            self.rt.pushGPUContext(ctx, device_id)
        elif device_ty == device_type.cpu:
            self.rt.pushCPUContext(ctx, device_id)
        else:
            e = UnsupportedDeviceTypeError("Device can only be cpu or gpu")
            raise e

        return DppyContext(PyCapsule_New(ctx, NULL, NULL))

    def pop_context (self):
        self.rt.popContext()

    def get_default_context (self):
        cdef shared_ptr[DppyOneAPIContext] *ctx = (
            new shared_ptr[DppyOneAPIContext]()
        )
        self.rt.getDefaultContext(ctx);
        return DppyContext(PyCapsule_New(ctx, NULL, NULL))

    def dump (self):
        self.rt.dump()

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
    #
    # If set_context is unable to create a new context an exception is raised.
    try:
        ctxt = None
        ctxt = runtime.set_context(dev, device_num)
        yield ctxt
    except:
        print("Context could not be created")
    finally:
        # Code to release resource
        if ctxt:
            print("Debug: Remove the context from the deque of active contexts")
            runtime.pop_context()
        else:
            print("Debug: No context was created so nothing to do")
