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
#
# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True
# cython: freethreading_compatible = True

import logging
import threading
from ._sycl_context cimport SyclContext
from ._sycl_device cimport SyclDevice

__all__ = [
    "_global_device_queue_cache",
    "get_device_cached_queue",
]

_logger = logging.getLogger(__name__)


cdef class _DeviceDefaultQueueCache:
    cdef dict __device_queue_map__
    cdef object _cache_lock

    def __cinit__(self):
        self.__device_queue_map__ = dict()
        self._cache_lock = threading.Lock()

    def get_or_create(self, key):
        """Return cached SyclQueue for given key, creating it if needed."""
        if (
            isinstance(key, tuple)
            and len(key) == 2
            and isinstance(key[0], SyclContext)
            and isinstance(key[1], SyclDevice)
        ):
            ctx_dev = key
            q = None
        elif isinstance(key, SyclDevice):
            q = SyclQueue(key)
            ctx_dev = q.sycl_context, key
        elif isinstance(key, str):
            q = SyclQueue(key)
            ctx_dev = q.sycl_context, q.sycl_device
        else:
            raise TypeError
        with self._cache_lock:
            if ctx_dev in self.__device_queue_map__:
                return self.__device_queue_map__[ctx_dev]
            if q is None:
                q = SyclQueue(*ctx_dev)
            self.__device_queue_map__[ctx_dev] = q
            return q

    def _update_map(self, dev_queue_map):
        self.__device_queue_map__.update(dev_queue_map)

    def __copy__(self):
        cdef _DeviceDefaultQueueCache _copy = _DeviceDefaultQueueCache.__new__(
            _DeviceDefaultQueueCache
        )
        _copy._update_map(self.__device_queue_map__)
        return _copy


# all threads share the same cached default
_global_device_queue_cache = _DeviceDefaultQueueCache()


cpdef object get_device_cached_queue(object key):
    """
    Returns a cached queue associated with given device.

    Args:
        key : Either a 2-tuple consisting of a :class:`dpctl.SyclContext` and
            a :class:`dpctl.SyclDevice`, or a :class:`dpctl.SyclDevice`
            instance, or a filter string identifying a device.

    Returns:
        :class:`dpctl.SyclQueue`: A cached SYCL queue associated with the
        input device.

    Raises:
        TypeError: If the input key is not one of the accepted types.

    """
    return _global_device_queue_cache.get_or_create(key)
