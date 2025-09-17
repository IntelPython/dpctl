#                       Data Parallel Control (dpctl)
#
#  Copyright 2020-2025 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numbers

import numpy as np

import dpctl.memory as dpm
import dpctl.tensor as dpt
from dpctl.tensor._usmarray import _is_object_with_buffer_protocol as _is_buffer

from ._type_utils import (
    WeakBooleanType,
    WeakComplexType,
    WeakFloatingType,
    WeakIntegralType,
    _to_device_supported_dtype,
)


def _get_queue_usm_type(o):
    """Return SYCL device where object `o` allocated memory, or None."""
    if isinstance(o, dpt.usm_ndarray):
        return o.sycl_queue, o.usm_type
    elif hasattr(o, "__sycl_usm_array_interface__"):
        try:
            m = dpm.as_usm_memory(o)
            return m.sycl_queue, m.get_usm_type()
        except Exception:
            return None, None
    return None, None


def _get_dtype(o, dev):
    if isinstance(o, dpt.usm_ndarray):
        return o.dtype
    if hasattr(o, "__sycl_usm_array_interface__"):
        return dpt.asarray(o).dtype
    if _is_buffer(o):
        host_dt = np.array(o).dtype
        dev_dt = _to_device_supported_dtype(host_dt, dev)
        return dev_dt
    if hasattr(o, "dtype"):
        dev_dt = _to_device_supported_dtype(o.dtype, dev)
        return dev_dt
    if isinstance(o, bool):
        return WeakBooleanType(o)
    if isinstance(o, int):
        return WeakIntegralType(o)
    if isinstance(o, float):
        return WeakFloatingType(o)
    if isinstance(o, complex):
        return WeakComplexType(o)
    return np.object_


def _validate_dtype(dt) -> bool:
    return isinstance(
        dt,
        (WeakBooleanType, WeakIntegralType, WeakFloatingType, WeakComplexType),
    ) or (
        isinstance(dt, dpt.dtype)
        and dt
        in [
            dpt.bool,
            dpt.int8,
            dpt.uint8,
            dpt.int16,
            dpt.uint16,
            dpt.int32,
            dpt.uint32,
            dpt.int64,
            dpt.uint64,
            dpt.float16,
            dpt.float32,
            dpt.float64,
            dpt.complex64,
            dpt.complex128,
        ]
    )


def _get_shape(o):
    if isinstance(o, dpt.usm_ndarray):
        return o.shape
    if _is_buffer(o):
        return memoryview(o).shape
    if isinstance(o, numbers.Number):
        return tuple()
    return getattr(o, "shape", tuple())


__all__ = [
    "_get_dtype",
    "_get_queue_usm_type",
    "_get_shape",
    "_validate_dtype",
]
