##===----------------- _memory.pyx - dpctl module -------*- Cython -*------===##
##
##                      Data Parallel Control (dpCtl)
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
## This top-level dpctl module.
##
##===----------------------------------------------------------------------===##
'''
    Data Parallel Control (dpCtl)

    dpCtl provides a lightweight Python abstraction over DPC++/SYCL and
    OpenCL runtime objects. The DPC++ runtime wrapper objects can be
    accessed by importing dpctl. The OpenCL runtime wrapper objects can be
    accessed by importing dpctl.ocldrv. The library is in an early-beta
    stage of development and not yet ready for production usage.

    dpCtl's intended usage is as a common SYCL interoperability layer for
    different Python libraries and applications. The OpenCL support inside
    PyDPPL is slated to be deprecated and then removed in future releases
    of the library.

    Currently, only a small subset of DPC++ runtime objects are exposed
    through the dpctl module. The main API classes are defined in the _sycl_core.pyx file.

    Please use `pydoc dpctl._sycl_core` to look at the current API for dpctl.

    Please use `pydoc dpctl.ocldrv` to look at the current API for dpctl.ocldrv.

'''
__author__ = "Intel Corp."

from ._sycl_core import *
from ._version import get_versions

def get_include():
    """
    Return the directory that contains the dpCtl *.h header files.

    Extension modules that need to be compiled against dpCtl should use
    this function to locate the appropriate include directory.
    """
    import os.path
    return os.path.join(__file__, 'include')

__version__ = get_versions()['version']
del get_versions
