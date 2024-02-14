#                       Data Parallel Control (dpctl)
#
#  Copyright 2020-2024 Intel Corporation
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
import dpctl.tensor._type_utils as tu

_integral_dtypes = [
    "i1",
    "u1",
    "i2",
    "u2",
    "i4",
    "u4",
    "i8",
    "u8",
]
_real_fp_dtypes = ["f2", "f4", "f8"]
_complex_fp_dtypes = [
    "c8",
    "c16",
]
_real_value_dtypes = _integral_dtypes + _real_fp_dtypes
_no_complex_dtypes = [
    "b1",
] + _real_value_dtypes
_all_dtypes = _no_complex_dtypes + _complex_fp_dtypes

_usm_types = ["device", "shared", "host"]


def _map_to_device_dtype(dt, dev):
    return tu._to_device_supported_dtype(dt, dev)


def _compare_dtypes(dt, ref_dt, sycl_queue=None):
    assert isinstance(sycl_queue, dpctl.SyclQueue)
    dev = sycl_queue.sycl_device
    expected_dt = _map_to_device_dtype(ref_dt, dev)
    return dt == expected_dt


__all__ = [
    "_no_complex_dtypes",
    "_all_dtypes",
    "_usm_types",
    "_map_to_device_dtype",
    "_compare_dtypes",
]
