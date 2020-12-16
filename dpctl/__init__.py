# ===-----------------------------------------------------------------------===#
#                      Data Parallel Control (dpCtl)
#
# Copyright 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===-----------------------------------------------------------------------===#

"""
    **Data Parallel Control (dpCtl)**

    DpCtl provides a lightweight Python wrapper over a subset of
    DPC++/SYCL's API. The goal of dpCtl is not (yet) to provide a
    abstraction for every SYCL function. DpCtl is intended to provide
    a common runtime to manage specific SYCL resources, such as devices
    and USM memory, for SYCL-based Python packages and extension modules.

    The main features presently provided by dpCtl are:

    * A SYCL queue manager exposed directly inside the top-level `dpctl`
      module.
    * A USM memory manager (`dpctl.memory`) that provides Python objects
      implementing the Python buffer protocol using USM shared and USM host
      allocators. The memory manager also exposes various utility functions
      to wrap SYCL's USM allocators, deallocators, `memcpy` functions, *etc.*
"""
__author__ = "Intel Corp."

from dpctl._sycl_core import *
from dpctl._sycl_core import __all__ as _sycl_core__all__
from ._version import get_versions


__all__ = _sycl_core__all__


def get_include():
    """
    Return the directory that contains the dpCtl \*.h header files.

    Extension modules that need to be compiled against dpCtl should use
    this function to locate the appropriate include directory.
    """
    import os.path

    return os.path.join(os.path.dirname(__file__), "include")


__version__ = get_versions()["version"]
del get_versions
