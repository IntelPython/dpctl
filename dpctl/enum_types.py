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

"""Defines Python enumeration types for SYCL enumerations.

This module provides two enumeration types corresponding to SYCL's
backend and device_type enumerations.

"""
from enum import Enum, auto

__all__ = ["device_type", "backend_type", "event_status_type"]


class device_type(Enum):
    """
    An enumeration of supported SYCL device types.

    :Example:
        .. code-block:: python

            import dpctl

            # filter GPU devices amongst available SYCL devices
            gpu_devs = [
                d for d in dpctl.get_devices() if (
                    d.device_type == dpctl.device_type.gpu
                ) ]

            # alternatively, get GPU devices directly
            gpu_devs2 = dpctl.get_devices(device_type=dpctl.device_type.gpu)
    """

    all = auto()
    accelerator = auto()
    automatic = auto()
    cpu = auto()
    custom = auto()
    gpu = auto()


class backend_type(Enum):
    """
    An enumeration of supported SYCL backends.

    :Example:
        .. code-block:: python

            import dpctl

            # create a SYCL device with OpenCL backend using filter selector
            d = dpctl.SyclDevice("opencl")
            print(d.backend)
            # Possible output: <backend_type.opencl: 5>
    """

    all = auto()
    cuda = auto()
    level_zero = auto()
    opencl = auto()


class event_status_type(Enum):
    """
    An enumeration of SYCL event states.

    :Example:
        .. code-block:: python

            import dpctl
            ev = dpctl.SyclEvent()
            print(ev.execution_status )
            # Possible output: <event_status_type.complete: 4>
    """

    unknown_status = auto()
    submitted = auto()
    running = auto()
    complete = auto()


class global_mem_cache_type(Enum):
    """
    An enumeration of global memory cache types for a device.

    :Example:
        .. code-block:: python

            import dpctl
            dev = dpctl.SyclDevice()
            print(dev.global_mem_cache_type)
            # Possible output: <global_mem_cache_type.read_write: 4>
    """

    indeterminate = auto()
    none = auto()
    read_only = auto()
    read_write = auto()
