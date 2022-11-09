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

# coding: utf-8

from ._use_queue_device import (
    get_device_global_mem_size,
    get_device_local_mem_size,
    get_max_compute_units,
    get_sub_group_sizes,
    offloaded_array_mod,
)

__all__ = [
    "get_max_compute_units",
    "get_device_global_mem_size",
    "get_device_local_mem_size",
    "offloaded_array_mod",
    "get_sub_group_sizes",
]

__doc__ = """
Example pybind11 extension demonstrating binding of dpctl entities to
SYCL entities.

dpctl provides type casters that bind ``sycl::queue`` to `dpctl.SyclQueue`,
``sycl::device`` to `dpctl.SyclDevice`, etc.

Use of these type casters simplifies writing of Python extensions and compile
then using SYCL C++ compilers, such as Intel(R) oneAPI DPC++ compiler.
"""
