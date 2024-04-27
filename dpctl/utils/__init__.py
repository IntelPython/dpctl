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
A collection of utility functions.
"""

from .._sycl_device import SyclDevice
from ._compute_follows_data import (
    ExecutionPlacementError,
    get_coerced_usm_type,
    get_execution_queue,
    validate_usm_type,
)
from ._device_queries import (
    intel_device_info_device_id,
    intel_device_info_free_memory,
    intel_device_info_gpu_eu_count,
    intel_device_info_gpu_eu_count_per_subslice,
    intel_device_info_gpu_eu_simd_width,
    intel_device_info_gpu_hw_threads_per_eu,
    intel_device_info_gpu_slices,
    intel_device_info_gpu_subslices_per_slice,
    intel_device_info_max_mem_bandwidth,
    intel_device_info_memory_bus_width,
    intel_device_info_memory_clock_rate,
)
from ._onetrace_context import onetrace_enabled


def intel_device_info(dev, /):
    """intel_device_info(sycl_device)

    For Intel(R) GPU devices returns a dictionary
    with device architectural details, and an empty
    dictionary otherwise. The dictionary contains
    the following keys:

    device_id:
        32-bits device PCI identifier
    gpu_eu_count:
        Total number of execution units
    gpu_hw_threads_per_eu:
        Number of thread contexts in EU
    gpu_eu_simd_width:
        Physical SIMD width of EU
    gpu_slices:
        Total number of slices
    gpu_subslices_per_slice:
        Number of sub-slices per slice
    gpu_eu_count_per_subslice:
        Number of EUs in subslice
    max_mem_bandwidth:
        Maximum memory bandwidth in bytes/second
    free_memory:
        Global memory available on the device in units of bytes

    Unsupported descriptors are omitted from the dictionary.

    Descriptors other than the PCI identifier are supported only
    for :class:`.SyclDevices` with Level-Zero backend.

    .. note::
        Environment variable ``ZES_ENABLE_SYSMAN`` may need to be set
        to ``1`` for the ``"free_memory"`` key to be reported.
    """
    if not isinstance(dev, SyclDevice):
        raise TypeError(f"Expected dpctl.SyclDevice, got {type(dev)}")
    dev_id = intel_device_info_device_id(dev)
    if dev_id:
        res = {
            "device_id": dev_id,
        }
        if dev.has_aspect_gpu:
            eu_count = intel_device_info_gpu_eu_count(dev)
            if eu_count:
                res["gpu_eu_count"] = eu_count
            hw_threads = intel_device_info_gpu_hw_threads_per_eu(dev)
            if hw_threads:
                res["gpu_hw_threads_per_eu"] = hw_threads
            simd_w = intel_device_info_gpu_eu_simd_width(dev)
            if simd_w:
                res["gpu_eu_simd_width"] = simd_w
            n_slices = intel_device_info_gpu_slices(dev)
            if n_slices:
                res["gpu_slices"] = n_slices
            n_subslices = intel_device_info_gpu_subslices_per_slice(dev)
            if n_subslices:
                res["gpu_subslices_per_slice"] = n_subslices
            n_eu_per_subslice = intel_device_info_gpu_eu_count_per_subslice(dev)
            if n_eu_per_subslice:
                res["gpu_eu_count_per_subslice"] = n_eu_per_subslice
        bw = intel_device_info_max_mem_bandwidth(dev)
        if bw:
            res["max_mem_bandwidth"] = bw
        fm = intel_device_info_free_memory(dev)
        if fm:
            res["free_memory"] = fm
        mcr = intel_device_info_memory_clock_rate(dev)
        if mcr:
            res["memory_clock_rate"] = mcr
        mbw = intel_device_info_memory_bus_width(dev)
        if mbw:
            res["memory_bus_width"] = mbw
        return res
    return dict()


__all__ = [
    "get_execution_queue",
    "get_coerced_usm_type",
    "validate_usm_type",
    "onetrace_enabled",
    "intel_device_info",
    "ExecutionPlacementError",
]
