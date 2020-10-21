##===--------- opencl_core.py - dpctl.ocldrv interface -----*- Python -*---===##
##
##                      Data paraller Control (dpctl)
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

dppl_opencl_interface_incldir = os.environ.get("DPPL_OPENCL_INTERFACE_INCLDIR", None)
dppl_opencl_interface_libdir = os.environ.get("DPPL_OPENCL_INTERFACE_LIBDIR", None)
opencl_libdir = os.environ.get("OpenCL_LIBDIR", None)

if opencl_libdir is None:
    raise ValueError("Abort! Set the OpenCL_LIBDIR envar to point to " "an OpenCL ICD")

if dppl_opencl_interface_libdir is None:
    raise ValueError(
        "Abort! Set the DPPL_OPENCL_INTERFACE_LIBDIR envar to "
        "point to ibdplibdpglueglue.so"
    )

if dppl_opencl_interface_incldir is None:
    raise ValueError(
        "Abort! Set the DP_GLUE_INCLDIR envar to point to " "dppl_opencl_interface.h"
    )

glue_h = "".join(
    list(
        filter(
            lambda x: len(x) > 0 and x[0] != "#",
            open(
                dppl_opencl_interface_incldir + "/dppl_opencl_interface.h", "r"
            ).readlines(),
        )
    )
).replace("DPPL_API", "")

# cdef() expects a single string declaring the C types, functions and
# globals needed to use the shared object. It must be in valid C syntax.
ffi.cdef(glue_h)

ffi_lib_name = "dpctl._opencl_core"

import sys

IS_WIN = sys.platform in ["win32", "cygwin"]
del sys

ffi.set_source(
    ffi_lib_name,
    """
         #include "dppl_opencl_interface.h"   // the C header of the library
    """,
    include_dirs=[dppl_opencl_interface_incldir],
    library_dirs=[dppl_opencl_interface_libdir, opencl_libdir],
    extra_link_args=[] if IS_WIN else ["-Wl,-rpath=$ORIGIN"],
    libraries=["DPPLOpenCLInterface", "OpenCL"],
)  # library name, for the linker
del IS_WIN

if __name__ == "__main__":
    ffi.compile(verbose=True)
