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
    `device_selector`) described in Section 4.6 of the SYCL 2020 spec (https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#_sycl_runtime_classes).
    Apart from the main SYCL runtime classes, dpctl includes a `memory` sub-module that exposes the SYCL USM allocators and deallocators.
"""
__author__ = "Intel Corp."

from dpctl._sycl_context import SyclContext
from dpctl._sycl_device import SyclDevice
from dpctl._sycl_device_factory import (
    get_devices,
    get_num_devices,
    has_accelerator_devices,
    has_cpu_devices,
    has_gpu_devices,
    has_host_device,
    select_accelerator_device,
    select_cpu_device,
    select_default_device,
    select_gpu_device,
    select_host_device,
)
from dpctl._sycl_event import SyclEvent
from dpctl._sycl_platform import SyclPlatform, get_platforms, lsplatform
from dpctl._sycl_queue import (
    SyclKernelInvalidRangeError,
    SyclKernelSubmitError,
    SyclQueue,
    SyclQueueCreationError,
)
from dpctl._sycl_queue_manager import (
    device_context,
    get_current_backend,
    get_current_device_type,
    get_current_queue,
    get_num_activated_queues,
    is_in_device_context,
    set_global_queue,
)
from ._version import get_versions
from .enum_types import backend_type, device_type

__all__ = [
    "SyclContext",
]
__all__ += [
    "SyclDevice",
]

__all__ += [
    "get_devices",
    "select_accelerator_device",
    "select_cpu_device",
    "select_default_device",
    "select_gpu_device",
    "select_host_device",
    "get_num_devices",
    "has_cpu_devices",
    "has_gpu_devices",
    "has_accelerator_devices",
    "has_host_device",
]
__all__ += [
    "SyclEvent",
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
    "device_context",
    "get_current_backend",
    "get_current_device_type",
    "get_current_queue",
    "get_num_activated_queues",
    "is_in_device_context",
    "set_global_queue",
]
__all__ += [
    "device_type",
    "backend_type",
]
__all__ += [
    "get_include",
]


def get_include():
    r"""
    Return the directory that contains the dpctl \*.h header files.

    Extension modules that need to be compiled against dpctl should use
    this function to locate the appropriate include directory.
    """
    import os.path

    return os.path.join(os.path.dirname(__file__), "include")


__version__ = get_versions()["version"]
del get_versions
