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

"""Benchmarks for USM allocation and USM pointer queries.

Each timed call allocates and releases in the same call, so ASV iterations
cannot accumulate device memory. Sizes that do not fit the device budget are
skipped rather than left to fail the run.
"""

from ._utils import (
    _SELECTORS,
    _SIZES,
    _USM_TYPES,
    queue_for,
    skip_unless_fits,
    usm_class,
)


class USMAllocation:
    """Allocate and free one USM block per timed call."""

    params = [_SELECTORS, _USM_TYPES, _SIZES]
    param_names = ["selector", "usm_type", "nbytes"]

    def setup(self, selector, usm_type, nbytes):
        self.queue = queue_for(selector)
        skip_unless_fits(self.queue, nbytes)
        self.cls = usm_class(usm_type)
        # first allocation of a given kind may initialize a runtime pool
        self.cls(nbytes, queue=self.queue)

    def time_alloc_free(self, selector, usm_type, nbytes):
        self.cls(nbytes, queue=self.queue)

    def time_alloc_free_aligned(self, selector, usm_type, nbytes):
        self.cls(nbytes, alignment=4096, queue=self.queue)


class USMFirstTouch:
    """Allocate, write one byte, free.

    Separates lazy allocation from the page-in that the first write pays.
    """

    params = [_SELECTORS, _USM_TYPES, [1024**2, 16 * 1024**2]]
    param_names = ["selector", "usm_type", "nbytes"]

    def setup(self, selector, usm_type, nbytes):
        self.queue = queue_for(selector)
        skip_unless_fits(self.queue, nbytes)
        self.cls = usm_class(usm_type)
        m = self.cls(nbytes, queue=self.queue)
        m.memset()

    def time_alloc_touch_free(self, selector, usm_type, nbytes):
        m = self.cls(nbytes, queue=self.queue)
        m.memset()


class USMQueries:
    """Pointer and interface queries on an existing allocation."""

    params = [_SELECTORS, _USM_TYPES]
    param_names = ["selector", "usm_type"]

    def setup(self, selector, usm_type):
        self.queue = queue_for(selector)
        self.mem = usm_class(usm_type)(1024**2, queue=self.queue)
        self.mem.get_usm_type()

    def time_get_usm_type(self, selector, usm_type):
        self.mem.get_usm_type()

    def time_get_usm_type_enum(self, selector, usm_type):
        self.mem.get_usm_type_enum()

    def time_sycl_usm_array_interface(self, selector, usm_type):
        self.mem.__sycl_usm_array_interface__

    def time_sycl_queue_attr(self, selector, usm_type):
        self.mem.sycl_queue
