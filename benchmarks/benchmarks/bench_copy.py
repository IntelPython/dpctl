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

"""Benchmarks for data movement through a queue and through _Memory.

Every asynchronous benchmark waits before returning, so no benchmark leaves
work queued for the next iteration to absorb.
"""

import numpy as np
from asv_runner.benchmarks.mark import SkipNotImplemented

import dpctl.memory as dpm

from ._utils import _SELECTORS, _SIZES, queue_for, skip_unless_fits


class HostDeviceCopy:
    """Blocking and asynchronous host/device transfers."""

    params = [_SELECTORS, _SIZES]
    param_names = ["selector", "nbytes"]

    def setup(self, selector, nbytes):
        self.queue = queue_for(selector)
        skip_unless_fits(self.queue, 2 * nbytes)
        self.nbytes = nbytes
        self.host = np.zeros(nbytes, dtype="u1")
        self.dev = dpm.MemoryUSMDevice(nbytes, queue=self.queue)
        # page in both sides before measuring
        self.queue.memcpy(self.dev, self.host, nbytes)
        self.queue.memcpy(self.host, self.dev, nbytes)

    def time_memcpy_h2d(self, selector, nbytes):
        self.queue.memcpy(self.dev, self.host, self.nbytes)

    def time_memcpy_d2h(self, selector, nbytes):
        self.queue.memcpy(self.host, self.dev, self.nbytes)

    def time_memcpy_async_h2d_wait(self, selector, nbytes):
        self.queue.memcpy_async(self.dev, self.host, self.nbytes).wait()

    def time_copy_from_host(self, selector, nbytes):
        self.dev.copy_from_host(self.host)

    def time_copy_to_host(self, selector, nbytes):
        self.dev.copy_to_host(self.host)


class DeviceDeviceCopy:
    """Device-to-device transfers."""

    params = [_SELECTORS, _SIZES]
    param_names = ["selector", "nbytes"]

    def setup(self, selector, nbytes):
        self.queue = queue_for(selector)
        skip_unless_fits(self.queue, 2 * nbytes)
        self.nbytes = nbytes
        self.src = dpm.MemoryUSMDevice(nbytes, queue=self.queue)
        self.dst = dpm.MemoryUSMDevice(nbytes, queue=self.queue)
        self.src.memset()
        self.queue.memcpy(self.dst, self.src, nbytes)

    def time_memcpy_d2d(self, selector, nbytes):
        self.queue.memcpy(self.dst, self.src, self.nbytes)

    def time_copy_from_device(self, selector, nbytes):
        self.dst.copy_from_device(self.src)


class Fill:
    """Byte-wise and typed fills."""

    params = [_SELECTORS, _SIZES]
    param_names = ["selector", "nbytes"]

    def setup(self, selector, nbytes):
        self.queue = queue_for(selector)
        skip_unless_fits(self.queue, nbytes)
        self.nbytes = nbytes
        self.dev = dpm.MemoryUSMDevice(nbytes, queue=self.queue)
        self.dev.memset()

    def time_memset(self, selector, nbytes):
        if not hasattr(self.queue, "memset"):
            raise SkipNotImplemented("SyclQueue.memset not available")
        self.queue.memset(self.dev, 0, self.nbytes)

    def time_memory_memset(self, selector, nbytes):
        self.dev.memset()

    def time_fill_u1(self, selector, nbytes):
        if not hasattr(self.queue, "fill"):
            raise SkipNotImplemented("SyclQueue.fill not available")
        self.queue.fill(self.dev, 0, self.nbytes, "u1")

    def time_fill_f4(self, selector, nbytes):
        if not hasattr(self.queue, "fill"):
            raise SkipNotImplemented("SyclQueue.fill not available")
        self.queue.fill(self.dev, 0.0, self.nbytes // 4, "f4")

    def time_memset_async_wait(self, selector, nbytes):
        if not hasattr(self.queue, "memset_async"):
            raise SkipNotImplemented("SyclQueue.memset_async not available")
        self.queue.memset_async(self.dev, 0, self.nbytes).wait()
