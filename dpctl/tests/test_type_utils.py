#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2023 Intel Corporation
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

import pytest

import dpctl.tensor as dpt
from dpctl.tensor._type_utils import (
    _all_data_types,
    _can_cast,
    _is_maximal_inexact_type,
)


def test_all_data_types():
    fp16_fp64_types = set([dpt.float16, dpt.float64, dpt.complex128])
    fp64_types = set([dpt.float64, dpt.complex128])

    all_dts = _all_data_types(True, True)
    assert fp16_fp64_types.issubset(all_dts)

    all_dts = _all_data_types(True, False)
    assert dpt.float16 in all_dts
    assert not fp64_types.issubset(all_dts)

    all_dts = _all_data_types(False, True)
    assert dpt.float16 not in all_dts
    assert fp64_types.issubset(all_dts)

    all_dts = _all_data_types(False, False)
    assert not fp16_fp64_types.issubset(all_dts)


@pytest.mark.parametrize("fp16", [True, False])
@pytest.mark.parametrize("fp64", [True, False])
def test_maximal_inexact_types(fp16, fp64):
    assert not _is_maximal_inexact_type(dpt.int32, fp16, fp64)
    assert fp64 == _is_maximal_inexact_type(dpt.float64, fp16, fp64)
    assert fp64 == _is_maximal_inexact_type(dpt.complex128, fp16, fp64)
    assert fp64 != _is_maximal_inexact_type(dpt.float32, fp16, fp64)
    assert fp64 != _is_maximal_inexact_type(dpt.complex64, fp16, fp64)


def test_can_cast_device():
    assert _can_cast(dpt.int64, dpt.float64, True, True)
    # if f8 is available, can't cast i8 to f4
    assert not _can_cast(dpt.int64, dpt.float32, True, True)
    assert not _can_cast(dpt.int64, dpt.float32, False, True)
    # should be able to cast to f8 when f2 unavailable
    assert _can_cast(dpt.int64, dpt.float64, False, True)
    # casting to f4 acceptable when f8 unavailable
    assert _can_cast(dpt.int64, dpt.float32, True, False)
    assert _can_cast(dpt.int64, dpt.float32, False, False)
    # can't safely cast inexact type to inexact type of lesser precision
    assert not _can_cast(dpt.float32, dpt.float16, True, False)
    assert not _can_cast(dpt.float64, dpt.float32, False, True)
