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

"""Benchmarks for construction of the core dpctl runtime objects."""

import dpctl

from ._utils import _SELECTORS, device_for, queue_for


class Construct:
    """Construction cost of SyclDevice, SyclContext, SyclQueue.

    ``SyclQueue(selector)`` also builds a context, while
    ``SyclQueue(context, device)`` reuses one. Both are measured so that a
    regression can be attributed to the queue or to the context.
    """

    params = [_SELECTORS]
    param_names = ["selector"]

    def setup(self, selector):
        self.device = device_for(selector)
        self.context = dpctl.SyclContext(self.device)

    def time_sycl_device(self, selector):
        dpctl.SyclDevice(selector)

    def time_sycl_context_from_device(self, selector):
        dpctl.SyclContext(self.device)

    def time_sycl_queue_from_selector(self, selector):
        dpctl.SyclQueue(selector)

    def time_sycl_queue_from_device(self, selector):
        dpctl.SyclQueue(self.device)

    def time_sycl_queue_from_context_device(self, selector):
        dpctl.SyclQueue(self.context, self.device)

    def time_sycl_queue_in_order(self, selector):
        dpctl.SyclQueue(self.device, property="in_order")


class ConstructPlatform:
    """Construction cost of SyclPlatform."""

    def time_sycl_platform_default(self):
        dpctl.SyclPlatform()


class DeviceProperties:
    """Hot device attribute reads.

    Catches an attribute that stops being cached and starts round-tripping
    to the SYCL runtime on every access.
    """

    params = [_SELECTORS]
    param_names = ["selector"]

    def setup(self, selector):
        self.device = device_for(selector)
        # touch every attribute once so a first-access cost is not charged
        # to the first measured iteration
        self.time_device_property_reads(selector)

    def time_device_property_reads(self, selector):
        d = self.device
        d.name
        d.driver_version
        d.max_compute_units
        d.max_work_group_size
        d.global_mem_size
        d.has_aspect_fp64


class QueueAccessors:
    """Accessor cost on an existing queue."""

    params = [_SELECTORS]
    param_names = ["selector"]

    def setup(self, selector):
        self.queue = queue_for(selector)

    def time_sycl_device_attr(self, selector):
        self.queue.sycl_device

    def time_sycl_context_attr(self, selector):
        self.queue.sycl_context
