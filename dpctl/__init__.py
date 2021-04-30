#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2021 Intel Corporation
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

"""
    **Data Parallel Control (dpctl)**

    Dpctl's Python API implements Python wrappers for a subset of DPC++/SYCL's
    API. The Python API exposes wrappers for the SYCL runtime classes (expect
    ``device_selector``) described in Section 4.6 of the
    [SYCL 2020 spec](https://www.khronos.org/registry/SYCL/specs/sycl-2020/
html/sycl-2020.html#_sycl_runtime_classes).
    Apart from the main SYCL runtime classes, dpctl includes a `memory`
    sub-module that exposes the SYCL USM allocators and deallocators.
"""
__author__ = "Intel Corp."

from dpctl._sycl_context import *
from dpctl._sycl_context import __all__ as _sycl_context__all__
from dpctl._sycl_device import *
from dpctl._sycl_device import __all__ as _sycl_device__all__
from dpctl._sycl_device_factory import *
from dpctl._sycl_device_factory import __all__ as _sycl_device_factory__all__
from dpctl._sycl_event import *
from dpctl._sycl_event import __all__ as _sycl_event__all__
from dpctl._sycl_platform import *
from dpctl._sycl_platform import __all__ as _sycl_platform__all__
from dpctl._sycl_queue import *
from dpctl._sycl_queue import __all__ as _sycl_queue__all__
from dpctl._sycl_queue_manager import *
from dpctl._sycl_queue_manager import __all__ as _sycl_qm__all__

from ._version import get_versions
from .enum_types import *
from .enum_types import __all__ as _enum_types_all__

__all__ = (
    _sycl_context__all__
    + _sycl_device__all__
    + _sycl_device_factory__all__
    + _sycl_event__all__
    + _sycl_platform__all__
    + _sycl_queue__all__
    + _sycl_qm__all__
    + _enum_types_all__
)


def get_include():
    """
    Return the directory that contains the dpctl *.h header files.

    Extension modules that need to be compiled against dpctl should use
    this function to locate the appropriate include directory.
    """
    import os.path

    return os.path.join(os.path.dirname(__file__), "include")


__version__ = get_versions()["version"]
del get_versions
del _sycl_context__all__
del _sycl_device__all__
del _sycl_device_factory__all__
del _sycl_event__all__
del _sycl_queue__all__
del _sycl_qm__all__
del _sycl_platform__all__
del _enum_types_all__
