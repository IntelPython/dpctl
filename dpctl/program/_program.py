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

"""Deprecated wrapper functions for backward compatibility."""

import warnings

from dpctl.compiler import (
    SyclKernel,
    create_kernel_bundle_from_source,
    create_kernel_bundle_from_spirv,
)


def create_program_from_source(q, src, copts=""):
    """This function is a deprecated alias for
    :func:`dpctl.compiler.create_kernel_bundle_from_source`.
    New code should use :func:`dpctl.compiler.create_kernel_bundle_from_source`.
    """
    warnings.warn(
        "create_program_from_source is deprecated and will be removed in a "
        "future release. Use create_kernel_bundle_from_source instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_kernel_bundle_from_source(q, src, copts)


def create_program_from_spirv(q, IL, copts=""):
    """This function is a deprecated alias for
    :func:`dpctl.compiler.create_kernel_bundle_from_spirv`.
    New code should use :func:`dpctl.compiler.create_kernel_bundle_from_spirv`.
    """
    warnings.warn(
        "create_program_from_spirv is deprecated and will be removed in a "
        "future release. Use create_kernel_bundle_from_spirv instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_kernel_bundle_from_spirv(q, IL, copts)


__all__ = [
    "create_program_from_source",
    "create_program_from_spirv",
    "SyclKernel",
]
