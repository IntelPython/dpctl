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

"""Shared utilities for dpctl benchmarks.

Every benchmark that needs a device goes through :func:`queue_for` or
:func:`device_for` so that benchmark names stay identical on every node in
the pool: a node without a given device reports the parameter as skipped
rather than failing the whole suite.
"""

import os

from asv_runner.benchmarks.mark import SkipNotImplemented

import dpctl

# Device selectors
_SELECTORS = ["opencl:cpu", "level_zero:gpu"]

# Allocation and transfer sizes, in bytes.
_SIZES = [4 * 1024, 1024**2, 16 * 1024**2, 256 * 1024**2]

# USM kinds.
_USM_TYPES = ["device", "host", "shared"]

# Max fraction of device memory a single allocation may claim.
_MEM_BUDGET = 0.25

_queues = {}
_devices = {}
_spirv = None


def queue_for(selector):
    """Return a memoized queue for *selector*, or skip when unavailable."""
    if selector not in _queues:
        try:
            _queues[selector] = dpctl.SyclQueue(selector)
        except dpctl.SyclQueueCreationError:
            _queues[selector] = None
    q = _queues[selector]
    if q is None:
        raise SkipNotImplemented(f"no {selector} device available")
    return q


def device_for(selector):
    """Return a memoized device for *selector*, or skip when unavailable."""
    if selector not in _devices:
        try:
            _devices[selector] = dpctl.SyclDevice(selector)
        except dpctl.SyclDeviceCreationError:
            _devices[selector] = None
    d = _devices[selector]
    if d is None:
        raise SkipNotImplemented(f"no {selector} device available")
    return d


def usm_class(usm_type):
    """Return the dpctl.memory class allocating *usm_type* memory."""
    import dpctl.memory as dpm

    return {
        "device": dpm.MemoryUSMDevice,
        "host": dpm.MemoryUSMHost,
        "shared": dpm.MemoryUSMShared,
    }[usm_type]


def skip_unless_fits(queue, nbytes):
    """Skip when *nbytes* exceeds this device's allocation budget."""
    budget = _MEM_BUDGET * queue.sycl_device.global_mem_size
    if nbytes > budget:
        raise SkipNotImplemented(
            f"{nbytes} bytes exceeds the device memory budget"
        )


def opencl_queue_or_skip():
    """Return a memoized OpenCL queue, or skip.

    ``create_kernel_bundle_from_source`` only supports the OpenCL backend.
    """
    return queue_for("opencl")


def sycl_source_queue_or_skip(selector):
    """Return a queue whose device can compile SYCL source, or skip."""
    try:
        import dpctl.compiler as dpc
    except ImportError:
        raise SkipNotImplemented("dpctl.compiler is not available")
    q = queue_for(selector)
    if not dpc.is_sycl_source_compilation_available():
        raise SkipNotImplemented("SYCL source compilation extension absent")
    if not q.sycl_device.can_compile("sycl"):
        raise SkipNotImplemented("device cannot compile SYCL source")
    return q


def spirv_bytes():
    """Return the SPIR-V module shipped with the installed dpctl, or skip.

    Defines ``add(int*, int*, int*)`` and ``axpy(int*, int*, int*, int)``.
    """
    global _spirv
    if _spirv is None:
        path = os.path.join(
            os.path.dirname(os.path.abspath(dpctl.__file__)),
            "tests",
            "input_files",
            "multi_kernel.spv",
        )
        if not os.path.exists(path):
            raise SkipNotImplemented(f"SPIR-V module not found at {path}")
        with open(path, "rb") as fh:
            _spirv = fh.read()
    return _spirv


def ocl_axpy_source(kernel_name="axpy"):
    """Return OpenCL C source for an axpy kernel called *kernel_name*."""
    return (
        f"kernel void {kernel_name}("
        "   global int *a, global int *b, global int *c, int d) {"
        "   size_t index = get_global_id(0);"
        "   c[index] = d * a[index] + b[index];"
        "}"
    )


def sycl_axpy_source(kernel_name="axpy"):
    """Return SYCL source for an axpy kernel called *kernel_name*."""
    return f"""
    #include <sycl/sycl.hpp>

    namespace syclext = sycl::ext::oneapi::experimental;

    extern "C" SYCL_EXTERNAL
    SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclext::nd_range_kernel<1>))
    void {kernel_name}(int* a, int* b, int* c, int d) {{
        sycl::nd_item<1> item =
                        sycl::ext::oneapi::this_work_item::get_nd_item<1>();
        size_t i = item.get_global_linear_id();
        c[i] = d * a[i] + b[i];
    }}
    """
