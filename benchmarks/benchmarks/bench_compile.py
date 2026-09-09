#                      Data Parallel Control (dpctl)
#
# Copyright 2026 Intel Corporation
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

"""Benchmarks for kernel bundle creation.

Cold and warm compilation are separate benchmarks. Cold gives every call a
kernel name no compiler has seen, so neither the in-memory nor the
persistent cache (disabled in ``benchmarks/__init__.py``) can serve it; warm
re-submits identical source to measure the cache-hit path instead.
"""

import itertools

from asv_runner.benchmarks.mark import SkipNotImplemented

try:
    import dpctl.compiler as dpc
except ImportError:
    dpc = None

from ._utils import (
    _SELECTORS,
    ocl_axpy_source,
    opencl_queue_or_skip,
    queue_for,
    spirv_bytes,
    sycl_axpy_source,
    sycl_source_queue_or_skip,
)


class BundleFromSPIRV:
    """Kernel bundle from a pre-compiled SPIR-V module."""

    params = [_SELECTORS]
    param_names = ["selector"]

    def setup(self, selector):
        if dpc is None:
            raise SkipNotImplemented("dpctl.compiler is not available")
        self.queue = queue_for(selector)
        self.spirv = spirv_bytes()
        self.bundle = dpc.create_kernel_bundle_from_spirv(
            self.queue, self.spirv
        )

    def time_bundle_from_spirv(self, selector):
        dpc.create_kernel_bundle_from_spirv(self.queue, self.spirv)

    def time_get_sycl_kernel(self, selector):
        self.bundle.get_sycl_kernel("axpy")

    def time_has_sycl_kernel(self, selector):
        self.bundle.has_sycl_kernel("axpy")


class BundleFromOpenCLSource:
    """Kernel bundle built from OpenCL C source (OpenCL backend only)."""

    timeout = 300
    number = 1
    repeat = 3
    warmup_time = 0

    def setup(self):
        if dpc is None:
            raise SkipNotImplemented("dpctl.compiler is not available")
        self.queue = opencl_queue_or_skip()
        self.counter = itertools.count()
        self.warm_source = ocl_axpy_source()
        dpc.create_kernel_bundle_from_source(self.queue, self.warm_source)

    def time_bundle_from_source_cold(self):
        name = f"axpy_{next(self.counter)}"
        dpc.create_kernel_bundle_from_source(self.queue, ocl_axpy_source(name))

    def time_bundle_from_source_warm(self):
        dpc.create_kernel_bundle_from_source(self.queue, self.warm_source)


class BundleFromSYCLSource:
    """Kernel bundle built from SYCL source via the kernel_compiler
    extension.

    Skipped unless the extension is present and the device reports it can
    compile SYCL source.
    """

    params = [_SELECTORS]
    param_names = ["selector"]
    timeout = 600
    number = 1
    repeat = 2
    warmup_time = 0

    def setup(self, selector):
        if dpc is None:
            raise SkipNotImplemented("dpctl.compiler is not available")
        self.queue = sycl_source_queue_or_skip(selector)
        self.counter = itertools.count()
        self.warm_source = sycl_axpy_source()
        dpc.create_kernel_bundle_from_sycl_source(self.queue, self.warm_source)

    def time_bundle_from_sycl_source_cold(self, selector):
        name = f"axpy_{next(self.counter)}"
        dpc.create_kernel_bundle_from_sycl_source(
            self.queue, sycl_axpy_source(name)
        )

    def time_bundle_from_sycl_source_warm(self, selector):
        dpc.create_kernel_bundle_from_sycl_source(self.queue, self.warm_source)


class SourceCompilationProbe:
    """Cost of the availability probes themselves.

    Called by consumers before every compilation attempt.
    """

    params = [_SELECTORS]
    param_names = ["selector"]

    def setup(self, selector):
        self.device = queue_for(selector).sycl_device

    def time_can_compile(self, selector):
        self.device.can_compile("sycl")
