#                      Data Parallel Control (dpctl)
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

"""Defines Python enumeration types for SYCL enumerations.

This module provides two enumeration types corresponding to SYCL's
backend and device_type enumerations.

"""
from enum import Enum, auto

__all__ = [
    "device_type",
    "backend_type",
    "event_status_type",
]


class device_type(Enum):
    """
    An enumeration of supported SYCL device types.

    ==================   ============
    Device type          Enum value
    ==================   ============
    gpu                  1
    cpu                  2
    accelerator          3
    host_device          4
    ==================   ============
    """

    all = auto()
    accelerator = auto()
    automatic = auto()
    cpu = auto()
    custom = auto()
    gpu = auto()
    host_device = auto()


class backend_type(Enum):
    """
    An enumeration of supported SYCL backends.

    ==================   ============
    Name of backend      Enum value
    ==================   ============
    opencl               1
    level_zero           2
    cuda                 3
    host                 4
    ==================   ============

    """

    all = auto()
    cuda = auto()
    host = auto()
    level_zero = auto()
    opencl = auto()


class event_status_type(Enum):

    unknown_status = auto()
    submitted = auto()
    running = auto()
    complete = auto()
