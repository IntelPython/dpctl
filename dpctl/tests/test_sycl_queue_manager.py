#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2024 Intel Corporation
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

"""Defines unit test cases for the dpctl._sycl_queue_manager module.
"""

import pytest

import dpctl


def test__DeviceDefaultQueueCache():
    import copy

    from dpctl._sycl_queue_manager import _global_device_queue_cache as cache
    from dpctl._sycl_queue_manager import get_device_cached_queue

    try:
        d = dpctl.SyclDevice()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("Could not create default device")

    q1 = get_device_cached_queue(d)
    cache_copy = copy.copy(cache.get())
    q2, changed = cache_copy.get_or_create(d)

    assert not changed
    assert q1 == q2
    q3 = get_device_cached_queue(d.filter_string)
    assert q3 == q1
