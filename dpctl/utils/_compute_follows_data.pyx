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

__all__ = ["get_execution_queue", "get_coerced_usm_type"]


cdef bint queue_equiv(SyclQueue q1, SyclQueue q2):
    """ Queues are equivalent if q1 == q2, that is they are copies
    of the same underlying SYCL object and hence are the same."""
    return q1.__eq__(q2)


def get_execution_queue(qs):
    """ Given a list of :class:`dpctl.SyclQueue` objects
    returns the execution queue under compute follows data paradigm,
    or returns `None` if queues are not equal.
    """
    if not isinstance(qs, (list, tuple)):
        raise TypeError(
            "Expected a list or a tuple, got {}".format(type(qs))
        )
    if len(qs) == 0:
        return None
    elif len(qs) == 1:
        return qs[0] if isinstance(qs[0], dpctl.SyclQueue) else None
    for q1, q2 in zip(qs[:-1], qs[1:]):
        if not isinstance(q1, dpctl.SyclQueue):
            return None
        elif not isinstance(q2, dpctl.SyclQueue):
            return None
        elif not queue_equiv(<SyclQueue> q1, <SyclQueue> q2):
             return None
    return qs[0]


def get_coerced_usm_type(usm_types):
    """ Given a list of strings denoting the types of USM allocations
    for input arrays returns the type of USM allocation for the output
    array(s) per compute follows data paradigm.
    Returns `None` if the type can not be deduced."""
    if not isinstance(usm_types, (list, tuple)):
        raise TypeError(
            "Expected a list or a tuple, got {}".format(type(usm_types))
        )
    if len(usm_types) == 0:
        return None
    _k = ["device", "shared", "host"]
    _m = {k:i for i, k in enumerate(_k)}
    res = len(_k)
    for t in usm_types:
        if not isinstance(t, str):
            return None
        if t not in _m:
            return None
        res = min(res, _m[t])
    return _k[res]


def _validate_usm_type_allow_none(usm_type):
    "Validates usm_type argument"
    if usm_type is not None:
        if isinstance(usm_type, str):
            if usm_type not in ["device", "shared", "host"]:
                raise ValueError(
                    f"Unrecognized value of usm_type={usm_type}, "
                    "expected 'device', 'shared', 'host', or None."
                )
        else:
            raise TypeError(
                f"Expected usm_type to be a str or None, got {type(usm_type)}"
            )

def _validate_usm_type_disallow_none(usm_type):
    "Validates usm_type argument"
    if isinstance(usm_type, str):
        if usm_type not in ["device", "shared", "host"]:
            raise ValueError(
                f"Unrecognized value of usm_type={usm_type}, "
                "expected 'device', 'shared', or 'host'."
            )
    else:
        raise TypeError(
            f"Expected usm_type to be a str, got {type(usm_type)}"
        )


def validate_usm_type(usm_type, allow_none=True):
    "Validates usm_type argument"
    if allow_none:
        _validate_usm_type_allow_none(usm_type)
    else:
        _validate_usm_type_disallow_none(usm_type)
