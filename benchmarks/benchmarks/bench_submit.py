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

"""Benchmarks for kernel submission and synchronization overhead.

The kernel is deliberately trivial and the global range is tiny, so what is
measured is dpctl's submission path and the runtime round trip, not compute.
"""

from asv_runner.benchmarks.mark import SkipNotImplemented

import dpctl
import dpctl.memory as dpm

try:
    import dpctl.compiler as dpc
except ImportError:
    dpc = None

from ._utils import _SELECTORS, queue_for, spirv_bytes

_BATCH = 100


class Submit:
    """Submission overhead on a one-work-item range."""

    params = [_SELECTORS]
    param_names = ["selector"]

    def setup(self, selector):
        if dpc is None:
            raise SkipNotImplemented("dpctl.compiler is not available")
        self.queue = queue_for(selector)
        bundle = dpc.create_kernel_bundle_from_spirv(
            self.queue, spirv_bytes()
        )
        self.kernel = bundle.get_sycl_kernel("add")
        nbytes = 4 * 1024
        self.args = [
            dpm.MemoryUSMDevice(nbytes, queue=self.queue) for _ in range(3)
        ]
        for m in self.args:
            m.memset()
        self.range = [1]
        # first launch on this queue builds runtime state; do not charge it
        # to the first measured iteration
        self.queue.submit(self.kernel, self.args, self.range)

    def time_submit_roundtrip(self, selector):
        self.queue.submit(self.kernel, self.args, self.range)

    def time_submit_async_wait(self, selector):
        self.queue.submit_async(self.kernel, self.args, self.range).wait()

    def time_submit_async_batch(self, selector):
        q = self.queue
        for _ in range(_BATCH):
            ev = q.submit_async(self.kernel, self.args, self.range)
        ev.wait()


class Synchronize:
    """Barrier, queue wait, and event handling costs on an idle queue."""

    params = [_SELECTORS]
    param_names = ["selector"]

    def setup(self, selector):
        self.queue = queue_for(selector)
        self.queue.submit_barrier().wait()

    def time_submit_barrier_wait(self, selector):
        self.queue.submit_barrier().wait()

    def time_queue_wait_idle(self, selector):
        self.queue.wait()

    def time_default_event_wait(self, selector):
        dpctl.SyclEvent().wait()
