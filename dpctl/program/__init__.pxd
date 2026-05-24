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

# distutils: language = c++
# cython: language_level=3

"""
Declares the extension types and functions for the Cython API
implemented in dpctl.compiler._compiler.pyx (deprecated, use dpctl.compiler
instead).
"""


from dpctl.compiler._compiler cimport (
    SyclKernel,
    SyclKernelBundle,
    create_kernel_bundle_from_source,
    create_kernel_bundle_from_spirv,
    create_program_from_source,
    create_program_from_spirv,
)
