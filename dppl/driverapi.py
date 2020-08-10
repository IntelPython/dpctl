##===---------- driverapi.py - dppl.ocldrv interface -----*- Python -*-----===##
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
### This file implements a CFFI interface for dppl_opencl_interface.h
### functions.
##===----------------------------------------------------------------------===##

import os

from cffi import FFI


ffi = FFI()

dpglue_incldir = os.environ.get('DP_GLUE_INCLDIR', None)
dpglue_libdir  = os.environ.get('DP_GLUE_LIBDIR', None)
opencl_libdir  = os.environ.get('OpenCL_LIBDIR', None)

if opencl_libdir is None:
    raise ValueError("Abort! Set the OpenCL_LIBDIR envar to point to "
                     "an OpenCL ICD")

if dpglue_libdir is None:
    raise ValueError("Abort! Set the DP_GLUE_LIBDIR envar to point to "
                     "libdpglue.so")

if dpglue_incldir is None:
    raise ValueError("Abort! Set the DP_GLUE_INCLDIR envar to point to "
                     "dppl_opencl_interface.h")

glue_h = ''.join(list(filter(lambda x: len(x) > 0 and x[0] != "#",
                             open(dpglue_incldir +
                             '/dppl_opencl_interface.h', 'r')
                             .readlines())))

# cdef() expects a single string declaring the C types, functions and
# globals needed to use the shared object. It must be in valid C syntax.
ffi.cdef(glue_h)

ffi_lib_name = "dppl._dppl_bindings"

ffi.set_source(
    ffi_lib_name,
    """
         #include "dppl_opencl_interface.h"   // the C header of the library
    """,
    include_dirs=[dpglue_incldir],
    library_dirs=[dpglue_libdir, opencl_libdir],
    libraries=["DPPLOpenCLInterface", "OpenCL"],
)   # library name, for the linker


if __name__ == "__main__":
    ffi.compile(verbose=True)
