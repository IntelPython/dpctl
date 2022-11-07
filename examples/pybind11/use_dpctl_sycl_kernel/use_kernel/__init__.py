#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2022 Intel Corporation
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

from ._use_kernel import submit_custom_kernel

__all__ = [
    "submit_custom_kernel",
]

__doc__ = """
Example pybind11 extension demonstrating binding of dpctl entities to
SYCL entities.

dpctl provides type casters that bind ``sycl::kernel`` to
`dpctl.program.SyclKernel`, ``sycl::device`` to `dpctl.SyclDevice`, etc.

Use of these type casters simplifies writing of Python extensions and compile
then using SYCL C++ compilers, such as Intel(R) oneAPI DPC++ compiler.
"""
