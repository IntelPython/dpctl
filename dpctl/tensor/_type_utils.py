#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2022 Intel Corporation
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

import dpctl.tensor as dpt


def _all_data_types(_fp16, _fp64):
    if _fp64:
        if _fp16:
            return [
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
        else:
            return [
                dpt.bool,
                dpt.int8,
                dpt.uint8,
                dpt.int16,
                dpt.uint16,
                dpt.int32,
                dpt.uint32,
                dpt.int64,
                dpt.uint64,
                dpt.float32,
                dpt.float64,
                dpt.complex64,
                dpt.complex128,
            ]
    else:
        if _fp16:
            return [
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
                dpt.complex64,
            ]
        else:
            return [
                dpt.bool,
                dpt.int8,
                dpt.uint8,
                dpt.int16,
                dpt.uint16,
                dpt.int32,
                dpt.uint32,
                dpt.int64,
                dpt.uint64,
                dpt.float32,
                dpt.complex64,
            ]


def is_maximal_inexact_type(dt: dpt.dtype, _fp16: bool, _fp64: bool):
    """
    Return True if data type `dt` is the
    maximal size inexact data type
    """
    if _fp64:
        return dt in [dpt.float64, dpt.complex128]
    return dt in [dpt.float32, dpt.complex64]


def _can_cast(from_: dpt.dtype, to_: dpt.dtype, _fp16: bool, _fp64: bool):
    """
    Can `from_` be cast to `to_` safely on a device with
    fp16 and fp64 aspects as given?
    """
    can_cast_v = dpt.can_cast(from_, to_)  # ask NumPy
    if _fp16 and _fp64:
        return can_cast_v
    if not can_cast_v:
        if (
            from_.kind in "biu"
            and to_.kind in "fc"
            and is_maximal_inexact_type(to_, _fp16, _fp64)
        ):
            return True

    return can_cast_v
