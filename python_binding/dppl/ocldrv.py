##===---------- ocldrv.py - dppl.ocldrv interface -----*- Python -*-----===##
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
### This file exposes Python classes for different OpenCL classes that are
### exposed by the _dppl_binding CFFI extension module.
##===----------------------------------------------------------------------===##
''' The dppl.ocldrv module contains a set of Python wrapper classes for
    OpenCL objects. The module has wrappers for cl_context, cl_device,
    cl_mem, cl_program, and cl_kernel objects.

    The two main user-visible API classes are Runtime, DeviceArray, and
    DeviceEnv and Runtime. The other classes are only used by the Numba
    JIT compiler.

    Global data members:
        runtime        - An instance of the Runtime class.
        has_cpu_device - A flag set to True when an OpenCL CPU device is found
                         on the system.
        has_cpu_device - A flag set to True when an OpenCL GPU device is found
                         on the system.

'''

from __future__ import absolute_import, division, print_function

from ._dppl_bindings import ffi, lib
from numpy import ndarray
from contextlib import contextmanager
import ctypes

import logging

__author__ = "Intel Corp."

_logger = logging.getLogger(__name__)

# create console handler and set level to debug
_ch = logging.StreamHandler()
_ch.setLevel(logging.WARNING)
# create formatter
_formatter = logging.Formatter('DPPL-%(levelname)s - %(message)s')
# add formatter to ch
_ch.setFormatter(_formatter)
# add ch to logger
_logger.addHandler(_ch)


##########################################################################
# Exception classes
##########################################################################


class DpplDriverError(Exception):
    """ The exception is raised when dppl.ocldrv cannot find an OpenCL Driver.
    """
    pass


class DeviceNotFoundError(Exception):
    """ The exception is raised when the requested type of OpenCL device is
        not available or not supported by dppl.ocldrv.
    """
    pass


class UnsupportedTypeError(Exception):
    """ The exception is raised when an unsupported type is encountered when
        creating an OpenCL KernelArg. Only DeviceArray or numpy.ndarray types
        are supported.
    """
    pass


##########################################################################
# Helper functions
##########################################################################


def _raise_driver_error(fname, errcode):
    e = DpplDriverError("Could not find an OpenCL Driver. Ensure OpenCL \
                         driver is installed.")
    e.fname = fname
    e.code = errcode
    raise e


def _raise_device_not_found_error(fname):
    e = DeviceNotFoundError("OpenCL device not available on the system.")
    e.fname = fname
    raise e


def _raise_unsupported_type_error(fname):
    e = UnsupportedTypeError("Type needs to be DeviceArray or numpy.ndarray.")
    e.fname = fname
    raise e


def _raise_unsupported_kernel_arg_error(fname):
    e = (UnsupportedTypeError("Type needs to be DeviceArray or a supported "
                              "ctypes type."))
    e.fname = fname
    raise e


def _is_supported_ctypes_raw_obj(obj):
    return isinstance(obj, (ctypes.c_ssize_t,
                            ctypes.c_double,
                            ctypes.c_float,
                            ctypes.c_uint8,
                            ctypes.c_size_t))

##########################################################################
# DeviceArray class
##########################################################################


class DeviceArray:
    ''' A Python wrapper for an OpenCL cl_men buffer with read-write access. A
        DeviceArray can only be created from a NumPy ndarray.
    '''
    _buffObj  = None
    _ndarray  = None
    _buffSize = None
    _dataPtr  = None

    def __init__(self, env_ptr, arr):
        ''' Creates a new DeviceArray from an ndarray.

            Note that DeviceArray creation only allocates the cl_mem buffer
            and does not actually move the data to the device. Data copy from
            host to device is done when the DeviceArray instance is passed as
            an argument to DeviceEnv.copy_array_to_device().
        '''

        # We only support device buffers for ndarray and ctypes (for basic
        # types like int, etc)
        if not isinstance(arr, ndarray):
            _raise_unsupported_type_error("DeviceArray constructor")

        # create a dp_buffer_t object
        self._buffObj  = ffi.new("buffer_t *")
        self._ndarray  = arr
        self._buffSize = arr.itemsize * arr.size
        self._dataPtr  = ffi.cast("void *", arr.ctypes.data)
        retval = (lib.create_dp_rw_mem_buffer(env_ptr,
                                              self._buffSize,
                                              self._buffObj))
        if retval == -1:
            _logger.warning("OpenCL Error Code  : %s", retval)
            _raise_driver_error("create_dp_rw_mem_buffer", -1)

    def __del__(self):
        ''' Destroy the DeviceArray and release the OpenCL buffer.'''

        retval = (lib.destroy_dp_rw_mem_buffer(self._buffObj))
        if retval == -1:
            _logger.warning("OpenCL Error Code  : %s", retval)
            _raise_driver_error("destroy_dp_rw_mem_buffer", -1)

    def get_buffer_obj(self):
        ''' Returns a cdata wrapper object encapsulating an OpenCL buffer.
        '''

        return self._buffObj

    def get_buffer_size(self):
        ''' Returns the size of the OpenCL buffer in bytes.
        '''

        return self._buffSize

    def get_buffer_ptr(self):
        ''' Returns a cdata wrapper over the actual OpenCL cl_mem pointer.
        '''

        return self.get_buffer_obj()[0].buffer_ptr

    def get_data_ptr(self):
        ''' Returns the data pointer for the NumPy ndarray used to create
            the DeviceArray object.
        '''

        return self._dataPtr

    def get_ndarray(self):
        ''' Returns the NumPy ndarray used to create the DeviceArray object.
        '''

        return self._ndarray

##########################################################################
# Program class
##########################################################################


class Program():

    def __init__(self, device_env, spirv_module):
        self._prog_t_obj = ffi.new("program_t *")
        retval = (lib.create_dp_program_from_spirv(device_env.get_env_ptr(),
                                                   spirv_module,
                                                   len(spirv_module),
                                                   self._prog_t_obj))
        if retval == -1:
            _logger.warning("OpenCL Error Code  : %s", retval)
            _raise_driver_error(
                "create_dp_program_from_spirv", -1)

        retval = (lib.build_dp_program(device_env.get_env_ptr(),
                                       self._prog_t_obj[0]))
        if retval == -1:
            _logger.warning("OpenCL Error Code  : %s", retval)
            _raise_driver_error("build_dp_program", -1)

    def __del__(self):
        retval = (lib.destroy_dp_program(self._prog_t_obj))
        if retval == -1:
            _logger.warning("OpenCL Error Code  : %s", retval)
            _raise_driver_error("destroy_dp_program", -1)

    def get_prog_t_obj(self):
        return self._prog_t_obj[0]


##########################################################################
# Kernel class
##########################################################################


class Kernel():

    def __init__(self, device_env, prog_t_obj, kernel_name):
        self._kernel_t_obj = ffi.new("kernel_t *")
        retval = (lib.create_dp_kernel(device_env.get_env_ptr(),
                                       prog_t_obj.get_prog_t_obj(),
                                       kernel_name.encode(),
                                       self._kernel_t_obj))
        if retval == -1:
            _logger.warning("OpenCL Error Code  : %s", retval)
            _raise_driver_error("create_dp_kernel", -1)

    def __del__(self):
        retval = (lib.destroy_dp_kernel(self._kernel_t_obj))
        if retval == -1:
            _logger.warning("OpenCL Error Code  : %s", retval)
            _raise_driver_error("destroy_dp_kernel", -1)

    def get_kernel_t_obj(self):
        return self._kernel_t_obj[0]

    def dump(self):
        retval = self._kernel_t_obj.dump_fn(self._kernel_t_obj)
        if retval == -1:
            _raise_driver_error("kernel dump_fn", -1)

##########################################################################
# KernelArg class
##########################################################################


class KernelArg():

    def __init__(self, arg, void_p_arg=False):
        self.arg = arg
        self.kernel_arg_t = ffi.new("kernel_arg_t *")
        if void_p_arg is True:
            self.ptr_to_arg_p = ffi.new("void **")
            self.ptr_to_arg_p[0] = ffi.cast("void *", 0)
            retval = (lib.create_dp_kernel_arg(self.ptr_to_arg_p,
                                               ffi.sizeof(self.ptr_to_arg_p),
                                               self.kernel_arg_t))
            if(retval):
                _raise_driver_error("create_dp_kernel_arg", -1)
        else:
            if isinstance(arg, DeviceArray):
                self.ptr_to_arg_p = ffi.new("void **")
                self.ptr_to_arg_p[0] = arg.get_buffer_obj()[0].buffer_ptr
                retval = (lib.create_dp_kernel_arg(
                                self.ptr_to_arg_p,
                                arg.get_buffer_obj()[0].sizeof_buffer_ptr,
                                self.kernel_arg_t))
                if(retval):
                    _raise_driver_error("create_dp_kernel_arg", -1)
            else:
                # it has to be of type ctypes
                if getattr(arg, '__module__', None) == "ctypes":
                    self.ptr_to_arg_p = ffi.cast("void *",
                                                 ctypes.addressof(arg))
                    retval = (lib.create_dp_kernel_arg(self.ptr_to_arg_p,
                                                       ctypes.sizeof(arg),
                                                       self.kernel_arg_t))
                    if(retval):
                        _raise_driver_error("create_dp_kernel_arg", -1)
                else:
                    _logger.warning("Unsupported Type %s", type(arg))
                    _raise_unsupported_kernel_arg_error("KernelArg init")

    def __del__(self):
        retval = (lib.destroy_dp_kernel_arg(self.kernel_arg_t))
        if retval == -1:
            _logger.warning("OpenCL Error Code  : %s", retval)
            _raise_driver_error("destroy_dp_kernel_arg", -1)

    def get_kernel_arg_obj(self):
        return self.kernel_arg_t[0]


##########################################################################
# DeviceEnv class
##########################################################################


class DeviceEnv():
    ''' A Python wrapper over an OpenCL cl_context object.
    '''

    def __init__(self, env_t_obj):
        self._env_ptr = env_t_obj

    def __del__(self):
        pass

    def retain_context(self):
        ''' Increment the refrence count of the OpenCL context object.
        '''

        retval = (lib.retain_dp_context(self._env_ptr.context))
        if(retval == -1):
            _raise_driver_error("retain_dp_context", -1)

        return (self._env_ptr.context)

    def release_context(self):
        ''' Increment the refrence count of the OpenCL context object.
        '''

        retval = (lib.release_dp_context(self._env_ptr.context))
        if retval == -1:
            _raise_driver_error("release_dp_context", -1)

    def copy_array_to_device(self, array):
        ''' Accepts either a DeviceArray or a NumPy ndarray and copies the
            data from host to an OpenCL device buffer. Returns either the
            DeviceArray that was passed in as an argument, or for the case of
            ndarrays returns a new DeviceArray.

            If the function is called with a DeviceArray argument, the
            function performs a blocking write of the data from the
            DeviceArray's ndarray member into its OpenCL device buffer member.
            When the function is called with an ndarray argument is, a new
            DeviceArray is first created. The data copy operation is then
            performed on the new DeviceArray.
        '''

        if isinstance(array, DeviceArray):
            retval = (lib.write_dp_mem_buffer_to_device(
                              self._env_ptr,
                              array.get_buffer_obj()[0],
                              True,
                              0,
                              array.get_buffer_size(),
                              array.get_data_ptr()))
            if retval == -1:
                _logger.warning("OpenCL Error Code  : %s", retval)
                _raise_driver_error("write_dp_mem_buffer_to_device", -1)
            return array
        elif (isinstance(array, ndarray) or getattr(array, '__module__', None)
              == "ctypes"):
            dArr = DeviceArray(self._env_ptr, array)
            retval = (lib.write_dp_mem_buffer_to_device(
                              self._env_ptr,
                              dArr.get_buffer_obj()[0],
                              True,
                              0,
                              dArr.get_buffer_size(),
                              dArr.get_data_ptr()))
            if retval == -1:
                _logger.warning("OpenCL Error Code  : %s", retval)
                _raise_driver_error("write_dp_mem_buffer_to_device", -1)
            return dArr
        else:
            _raise_unsupported_type_error("copy_array_to_device")

    def copy_array_from_device(self, array):
        ''' Copies data from a cl_mem buffer into a DeviceArray's host memory
            pointer. The function argument should be a DeviceArray object.
        '''

        if not isinstance(array, DeviceArray):
            _raise_unsupported_type_error("copy_array_to_device")
        retval = (lib.read_dp_mem_buffer_from_device(
                          self._env_ptr,
                          array.get_buffer_obj()[0],
                          True,
                          0,
                          array.get_buffer_size(),
                          array.get_data_ptr()))
        if retval == -1:
            _logger.warning("OpenCL Error Code  : %s", retval)
            _raise_driver_error("read_dp_mem_buffer_from_device", -1)

    def create_device_array(self, array):
        ''' Returns an new DeviceArray instance.
        '''

        if not ((isinstance(array, ndarray) or
                 getattr(array, '__module__', None)
              == "ctypes")):
            _raise_unsupported_type_error("alloc_array_in_device")

        return DeviceArray(self._env_ptr, array)

    def get_context_ptr(self):
        ''' Returns a cdata wrapper for the OpenCL cl_context object.
        '''

        return self._env_ptr.context

    def get_device_ptr(self):
        ''' Returns a cdata wrapper for the OpenCL cl_device object.
        '''

        return self._env_ptr.device

    def get_queue_ptr(self):
        ''' Returns a cdata wrapper for the OpenCL cl_command_queue object.
        '''

        return self._env_ptr.queue

    def get_env_ptr(self):
        ''' Returns a cdata wrapper for a C object encapsulating an OpenCL
            cl_device object, a cl_command_queue object,
            and a cl_context object.
        '''

        return self._env_ptr

    def get_max_work_item_dims(self):
        ''' Returns the maximum number of work items per work group for
            the OpenCL device.
        '''

        return self._env_ptr.max_work_item_dims

    def get_max_work_group_size(self):
        ''' Returns the max work group size for the OpenCL device.
        '''

        return self._env_ptr.max_work_group_size

    def dump(self):
        ''' Prints metadata for the underlying OpenCL device.
        '''

        retval = self._env_ptr[0].dump_fn(self._env_ptr)
        if retval == -1:
            _raise_driver_error("env dump_fn", -1)
        return retval

##########################################################################
# Runtime class
##########################################################################


class Runtime():
    '''Runtime is a singleton class that creates a C wrapper object storing
    available OpenCL contexts and corresponding OpenCL command queues. The
    context and the queue are stored only for the first available GPU and CPU
    OpenCL devices found on the system.
    '''

    _runtime     = None
    _cpu_device  = None
    _gpu_device  = None
    _curr_device = None

    def __new__(cls):
        obj = cls._runtime
        if obj is not None:
            return obj
        else:
            obj = object.__new__(cls)
            ffiobj = ffi.new("runtime_t *")
            retval = (lib.create_dp_runtime(ffiobj))
            if(retval):
                _logger.warning("OpenCL Error Code  : %s", retval)
                _raise_driver_error("create_dp_runtime", -1)

            cls._runtime = ffiobj

            if cls._runtime[0][0].has_cpu:
                cls._cpu_device = DeviceEnv(cls._runtime[0][0].first_cpu_env)
            else:
                _logger.warning("No CPU device")

            if cls._runtime[0][0].has_gpu:
                cls._gpu_device = DeviceEnv(cls._runtime[0][0].first_gpu_env)
            else:
                _logger.warning("No GPU device")

            cls._curr_device = DeviceEnv(cls._runtime[0][0].curr_env)

        return obj

    def __init__(self):
        pass

    def __del__(self):
        if Runtime._runtime:
            retval = (lib.destroy_dp_runtime(Runtime._runtime))
            if(retval):
                _raise_driver_error("destroy_dp_runtime", -1)

    def has_cpu_device(self):
        ''' Returns True is the system has an OpenCL driver for the CPU.'''

        return Runtime._cpu_device is not None

    def has_gpu_device(self):
        ''' Returns True is the system has an OpenCL driver for the GPU.'''

        return Runtime._gpu_device is not None

    def get_cpu_device(self):
        ''' Returns a cdata wrapper for the first available OpenCL
        CPU context.
        '''

        if(Runtime._cpu_device is None):
            _raise_device_not_found_error("get_cpu_device")

        return Runtime._cpu_device

    def get_gpu_device(self):
        ''' Returns a cdata wrapper for the first available OpenCL
        GPU context.
        '''

        if(Runtime._gpu_device is None):
            _raise_device_not_found_error("get_gpu_device")

        return Runtime._gpu_device

    def get_current_device(self):
        ''' Returns a cdata wrapper for the first available OpenCL
        CPU context.
        '''

        return Runtime._curr_device

    def get_runtime_ptr(self):
        ''' Returns a reference to the runtime object.
        '''

        return Runtime._runtime[0]

    def dump(self):
        ''' Prints OpenCL metadata about the available devices and contexts.
        '''

        retval = Runtime._runtime[0].dump_fn(Runtime._runtime[0])
        if retval == -1:
            _raise_driver_error("runtime dump_fn", -1)
        return retval

##########################################################################
# Public API
##########################################################################

#------- Global Data

runtime = Runtime()
has_cpu_device = runtime.has_cpu_device()
has_gpu_device = runtime.has_gpu_device()

#------- Global Functions

def enqueue_kernel (device_env, kernel, kernelargs, global_work_size,
                    local_work_size):
    ''' A single wrapper function over OpenCL clCreateKernelArgs and
        clEnqueueNDRangeKernel. The function blocks till the enqued kernel
        finishes execution.
    '''

    l_work_size_array = None
    kernel_arg_array = ffi.new("kernel_arg_t [" + str(len(kernelargs)) + "]")
    g_work_size_array = ffi.new("size_t [" + str(len(global_work_size)) + "]")
    if local_work_size:
        l_work_size_array = ffi.new(
                              "size_t [" + str(len(local_work_size)) + "]")
    else:
        l_work_size_array = ffi.NULL
    for i in range(len(kernelargs)):
        kernel_arg_array[i] = kernelargs[i].get_kernel_arg_obj()
    for i in range(len(global_work_size)):
        g_work_size_array[i] = global_work_size[i]
    for i in range(len(local_work_size)):
        l_work_size_array[i] = local_work_size[i]
    retval = (lib.set_args_and_enqueue_dp_kernel(device_env.get_env_ptr(),
                                                 kernel.get_kernel_t_obj(),
                                                 len(kernelargs),
                                                 kernel_arg_array,
                                                 len(global_work_size),
                                                 ffi.NULL,
                                                 g_work_size_array,
                                                 l_work_size_array))
    if(retval):
        _raise_driver_error("set_args_and_enqueue_dp_kernel", -1)


def is_available():
    ''' Return a Boolean to indicate the availability of a DPPL device.
    '''

    return runtime.has_cpu_device() or runtime.has_gpu_device()


def dppl_error():
    ''' Raised a DpplDriverError exception.
    '''

    _raise_driver_error()


##########################################################################
# Context Managers
##########################################################################


@contextmanager
def igpu_context(*args, **kwds):
    ''' A context manager sets the current DeviceEnv inside the global
        runtime object to the default GPU DeviceEnv. The GPU DeviceEnv is
        yeilded by the context manager.
    '''

    device_id = 0
    # some validation code
    if(args):
        assert(len(args) == 1 and args[0] == 0)
    _logger.debug("Set the current env to igpu device queue %s", device_id)
    lib.set_curr_env(runtime.get_runtime_ptr(),
                     runtime.get_gpu_device().get_env_ptr())
    device_env = runtime.get_current_device()
    yield device_env

    # After yield as the exit method
    #TODO : one exit reset the current env to previous value
    _logger.debug("Exit method called")


@contextmanager
def cpu_context(*args, **kwds):
    ''' A context manager sets the current DeviceEnv inside the global
        runtime object to the default CPU DeviceEnv. The CPU DeviceEnv is
        yeilded by the context manager.
    '''

    device_id = 0
    # some validation code
    if(args):
        assert(len(args) == 1 and args[0] == 0)
    _logger.debug("Set the current env to cpu device queue %s", device_id)
    lib.set_curr_env(runtime.get_runtime_ptr(),
                     runtime.get_cpu_device().get_env_ptr())
    device_env = runtime.get_current_device()
    yield device_env

    # After yield as the exit method
    _logger.debug("Exit method called")
