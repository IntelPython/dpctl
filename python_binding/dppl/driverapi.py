from cffi import FFI
import os
import sys
from distutils import sysconfig

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
                     "dp_glue.h")

glue_h = ''.join(list(filter(lambda x: len(x) > 0 and x[0] != "#",
                             open(dpglue_incldir + '/dp_glue.h', 'r')
                             .readlines())))

# cdef() expects a single string declaring the C types, functions and
# globals needed to use the shared object. It must be in valid C syntax.
ffi.cdef(glue_h)

ffi_lib_name = "dppl._dppl_bindings"

ffi.set_source(
    ffi_lib_name,
    """
         #include "dp_glue.h"   // the C header of the library
    """,
    include_dirs=[dpglue_incldir],
    library_dirs=[dpglue_libdir, opencl_libdir],
    libraries=["dpglue", "OpenCL"],
)   # library name, for the linker


if __name__ == "__main__":
    ffi.compile(verbose=True)
