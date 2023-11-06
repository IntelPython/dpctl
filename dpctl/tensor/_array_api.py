#                       Data Parallel Control (dpctl)
#
#  Copyright 2020-2023 Intel Corporation
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

import dpctl
import dpctl.tensor as dpt
from dpctl.tensor._tensor_impl import (
    default_device_complex_type,
    default_device_fp_type,
    default_device_index_type,
    default_device_int_type,
)

__array_api_version__ = "2022.12"


def _isdtype_impl(dtype, kind):
    if isinstance(kind, dpt.dtype):
        return dtype == kind

    elif isinstance(kind, str):
        if kind == "bool":
            return dtype.kind == "b"
        elif kind == "signed integer":
            return dtype.kind == "i"
        elif kind == "unsigned integer":
            return dtype.kind == "u"
        elif kind == "integral":
            return dtype.kind in "iu"
        elif kind == "real floating":
            return dtype.kind == "f"
        elif kind == "complex floating":
            return dtype.kind == "c"
        elif kind == "numeric":
            return dtype.kind in "iufc"
        else:
            raise ValueError(f"Unrecognized data type kind: {kind}")

    elif isinstance(kind, tuple):
        return any(_isdtype_impl(dtype, k) for k in kind)
    else:
        raise TypeError(f"Unsupported data type kind: {kind}")


class __array_namespace_info__:
    def __init__(self):
        self._capabilities = {
            "boolean_indexing": True,
            "data_dependent_shapes": True,
        }
        self._all_dtypes = {
            "bool": dpt.bool,
            "float16": dpt.float16,
            "float32": dpt.float32,
            "float64": dpt.float64,
            "complex64": dpt.complex64,
            "complex128": dpt.complex128,
            "int8": dpt.int8,
            "int16": dpt.int16,
            "int32": dpt.int32,
            "int64": dpt.int64,
            "uint8": dpt.uint8,
            "uint16": dpt.uint16,
            "uint32": dpt.uint32,
            "uint64": dpt.uint64,
        }

    def capabilities(self):
        return self._capabilities.copy()

    def default_device(self):
        return dpctl.select_default_device()

    def default_dtypes(self, device=None):
        if device is None:
            device = dpctl.select_default_device()
        return {
            "real floating": default_device_fp_type(device),
            "complex floating": default_device_complex_type,
            "integral": default_device_int_type(device),
            "indexing": default_device_index_type(device),
        }

    def dtypes(self, device=None, kind=None):
        if device is None:
            device = dpctl.select_default_device()
        ignored_types = []
        if not device.has_aspect_fp16:
            ignored_types.append("float16")
        if not device.has_aspect_fp64:
            ignored_types.append("float64")
        if kind is None:
            return {
                key: val
                for key, val in self._all_dtypes.items()
                if key not in ignored_types
            }
        else:
            return {
                key: val
                for key, val in self._all_dtypes.items()
                if key not in ignored_types and _isdtype_impl(val, kind)
            }

    def devices(self):
        return dpctl.get_devices()
