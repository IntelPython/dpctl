#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2024 Intel Corporation
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
    **Data Parallel Control (dpctl)** is a Python abstraction layer over SYCL.

    Dpctl implements a subset of SYCL's API providing wrappers for the
    SYCL runtime classes described in :sycl_runtime_classes:`Section 4.6 <>` of
    the :sycl_spec_2020:`SYCL 2020 spec <>`.
"""
__author__ = "Intel Corp."

import os
import os.path

from ._device_selection import select_device_with_aspects
from ._sycl_context import SyclContext, SyclContextCreationError
from ._sycl_device import (
    SyclDevice,
    SyclDeviceCreationError,
    SyclSubDeviceCreationError,
)
from ._sycl_device_factory import (
    get_devices,
    get_num_devices,
    has_accelerator_devices,
    has_cpu_devices,
    has_gpu_devices,
    select_accelerator_device,
    select_cpu_device,
    select_default_device,
    select_gpu_device,
)
from ._sycl_event import SyclEvent
from ._sycl_platform import SyclPlatform, get_platforms, lsplatform
from ._sycl_queue import (
    SyclKernelInvalidRangeError,
    SyclKernelSubmitError,
    SyclQueue,
    SyclQueueCreationError,
)
from ._sycl_queue_manager import get_device_cached_queue
from ._sycl_timer import SyclTimer
from ._version import get_versions
from .enum_types import (
    backend_type,
    device_type,
    event_status_type,
    global_mem_cache_type,
)

__all__ = [
    "SyclContext",
    "SyclContextCreationError",
]
__all__ += [
    "SyclDevice",
    "SyclDeviceCreationError",
    "SyclSubDeviceCreationError",
]
__all__ += [
    "get_devices",
    "select_accelerator_device",
    "select_cpu_device",
    "select_default_device",
    "select_gpu_device",
    "select_host_device",
    "select_device_with_aspects",
    "get_num_devices",
    "has_cpu_devices",
    "has_gpu_devices",
    "has_accelerator_devices",
    "has_host_device",
]
__all__ += [
    "SyclEvent",
    "SyclTimer",
]
__all__ += [
    "get_platforms",
    "lsplatform",
    "SyclPlatform",
]
__all__ += [
    "SyclQueue",
    "SyclKernelInvalidRangeError",
    "SyclKernelSubmitError",
    "SyclQueueCreationError",
]
__all__ += [
    "get_device_cached_queue",
]
__all__ += [
    "device_type",
    "backend_type",
    "event_status_type",
    "global_mem_cache_type",
]
__all__ += [
    "get_include",
]
# add submodules
__all__ += [
    "memory",
    "program",
    "tensor",
    "utils",
]

if hasattr(os, "add_dll_directory"):
    # Include folder containing DPCTLSyclInterface.dll to search path
    os.add_dll_directory(os.path.dirname(__file__))


def get_include():
    r"""
    Return the directory that contains the dpctl \*.h header files.

    Extension modules that need to be compiled against dpctl should use
    this function to locate the appropriate include directory.
    """
    return os.path.join(os.path.dirname(__file__), "include")


__version__ = get_versions()["version"]
del get_versions
