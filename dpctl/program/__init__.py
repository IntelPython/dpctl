#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2025 Intel Corporation
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
**Data Parallel Control Program** provides a way to create a SYCL kernel
from either an OpenCL program represented as a string or a SPIR-V binary
file.

"""

from ._program import (
    SyclKernel,
    SyclKernelBundle,
    SyclKernelBundleCompilationError,
    create_kernel_bundle_from_source,
    create_kernel_bundle_from_spirv,
    create_program_from_source,
    create_program_from_spirv,
)

__all__ = [
    "create_kernel_bundle_from_source",
    "create_kernel_bundle_from_spirv",
    "create_program_from_source",
    "create_program_from_spirv",
    "SyclKernel",
    "SyclKernelBundle",
    "SyclKernelBundleCompilationError",
    "SyclProgram",
]


def __getattr__(name):
    if name == "SyclProgram":
        from warnings import warn

        warn(
            "dpctl.program.SyclProgram is deprecated and will be removed in a "
            "future release. Use dpctl.program.SyclKernelBundle instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return SyclKernelBundle
    raise AttributeError(f"module {__name__} has no attribute {name}")
