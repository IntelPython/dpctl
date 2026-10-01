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

"""Benchmarks for the cached-queue lookup in dpctl._sycl_queue_manager.

``get_device_cached_queue`` accepts three key kinds and they do not cost the
same: only the ``(SyclContext, SyclDevice)`` tuple key reaches the map
without building a queue first, while the device key and the filter-string
key each construct a ``SyclQueue`` before the lookup. All four paths are
tracked so the asymmetry stays visible.
"""

import dpctl

from ._utils import _SELECTORS, device_for


class QueueCache:
    """Cached queue lookup, one benchmark per key kind."""

    params = [_SELECTORS]
    param_names = ["selector"]

    def setup(self, selector):
        self.selector = selector
        self.device = device_for(selector)
        # populate the cache so every measurement below is a hit
        q = dpctl.get_device_cached_queue(self.device)
        self.context = q.sycl_context
        self.ctx_dev = (self.context, self.device)

    def time_cached_queue_ctx_dev_key(self, selector):
        dpctl.get_device_cached_queue(self.ctx_dev)

    def time_cached_queue_device_key(self, selector):
        dpctl.get_device_cached_queue(self.device)

    def time_cached_queue_str_key(self, selector):
        dpctl.get_device_cached_queue(selector)

    def time_uncached_queue(self, selector):
        dpctl.SyclQueue(selector)
