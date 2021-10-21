#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2021 Intel Corporation
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

# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True

"""This file implements Python buffer protocol using Sycl USM shared and host
allocators. The USM device allocator is also exposed through this module for
use in other Python modules.
"""


import dpctl

from .._sycl_queue cimport SyclQueue

__all__ = ["get_execution_queue", ]


cdef bint queue_equiv(SyclQueue q1, SyclQueue q2):
    """ Queues are equivalent if contexts are the same,
    devices are the same, and properties are the same."""
    return (
        (q1 is q2) or
        (
            (q1.sycl_context == q2.sycl_context) and
            (q1.sycl_device == q2.sycl_device) and
            (q1.is_in_order == q2.is_in_order) and
            (q1.has_enable_profiling == q2.has_enable_profiling)
        )
    )


def get_execution_queue(qs):
    """ Given a list of :class:`dpctl.SyclQueue` objects
    returns the execution queue under compute follows data paradigm,
    or returns `None` if queues are not equivalent.
    """
    if not isinstance(qs, (list, tuple)):
        raise TypeError(
            "Expected a list or a tuple, got {}".format(type(qs))
        )
    if len(qs) == 0:
        return None
    elif len(qs) == 1:
        return qs[0] if isinstance(qs[0], dpctl.SyclQueue) else None
    for q1, q2 in zip(qs, qs[1:]):
        if not isinstance(q1, dpctl.SyclQueue):
            return None
        elif not isinstance(q2, dpctl.SyclQueue):
            return None
        elif not queue_equiv(<SyclQueue> q1, <SyclQueue> q2):
             return None
    return qs[0]
