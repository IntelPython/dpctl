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

import numpy as np
import pytest

import dpctl
import dpctl.tensor as dpt
import dpctl.tensor._copy_utils as cu
import dpctl.tensor._type_utils as tu

from .utils import _all_dtypes, _map_to_device_dtype


class MockDevice:
    def __init__(self, fp16: bool, fp64: bool):
        self.has_aspect_fp16 = fp16
        self.has_aspect_fp64 = fp64


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_type_utils_map_to_device_type(dtype):
    for fp64 in [
        True,
        False,
    ]:
        for fp16 in [True, False]:
            dev = MockDevice(fp16, fp64)
            dt_in = dpt.dtype(dtype)
            dt_out = _map_to_device_dtype(dt_in, dev)
            assert isinstance(dt_out, dpt.dtype)


def test_type_util_all_data_types():
    for fp64 in [
        True,
        False,
    ]:
        for fp16 in [True, False]:
            r = tu._all_data_types(fp16, fp64)
            assert isinstance(r, list)
            # 11: bool + 4 signed + 4 unsigned inegral + float32 + complex64
            assert len(r) == 11 + int(fp16) + 2 * int(fp64)


def test_type_util_can_cast():
    for fp64 in [
        True,
        False,
    ]:
        for fp16 in [True, False]:
            for from_ in _all_dtypes:
                for to_ in _all_dtypes:
                    r = tu._can_cast(
                        dpt.dtype(from_), dpt.dtype(to_), fp16, fp64
                    )
                    assert isinstance(r, bool)


def test_type_utils_empty_like_orderK():
    try:
        a = dpt.empty((10, 10), dtype=dpt.int32, order="F")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    X = cu._empty_like_orderK(a, dpt.int32, a.usm_type, a.device)
    assert X.flags["F"]


def test_type_utils_empty_like_orderK_invalid_args():
    with pytest.raises(TypeError):
        cu._empty_like_orderK([1, 2, 3], dpt.int32, "device", None)
    with pytest.raises(TypeError):
        cu._empty_like_pair_orderK(
            [1, 2, 3],
            (
                1,
                2,
                3,
            ),
            dpt.int32,
            (3,),
            "device",
            None,
        )
    try:
        a = dpt.empty(10, dtype=dpt.int32)
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    with pytest.raises(TypeError):
        cu._empty_like_pair_orderK(
            a,
            (
                1,
                2,
                3,
            ),
            dpt.int32,
            (10,),
            "device",
            None,
        )


def test_type_utils_find_buf_dtype():
    def _denier_fn(dt):
        return False

    for fp64 in [
        True,
        False,
    ]:
        for fp16 in [True, False]:
            dev = MockDevice(fp16, fp64)
            arg_dt = dpt.float64
            r = tu._find_buf_dtype(
                arg_dt, _denier_fn, dev, tu._acceptance_fn_default_unary
            )
            assert r == (
                None,
                None,
            )


def test_type_utils_get_device_default_type():
    with pytest.raises(RuntimeError):
        tu._get_device_default_dtype("-", MockDevice(True, True))
    try:
        dev = dpctl.SyclDevice()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    for k in ["b", "i", "u", "f", "c"]:
        dt = tu._get_device_default_dtype(k, dev)
        assert isinstance(dt, dpt.dtype)
        assert dt.kind == k


def test_type_utils_find_buf_dtype2():
    def _denier_fn(dt1, dt2):
        return False

    for fp64 in [
        True,
        False,
    ]:
        for fp16 in [True, False]:
            dev = MockDevice(fp16, fp64)
            arg1_dt = dpt.float64
            arg2_dt = dpt.complex64
            r = tu._find_buf_dtype2(
                arg1_dt,
                arg2_dt,
                _denier_fn,
                dev,
                tu._acceptance_fn_default_binary,
            )
            assert r == (
                None,
                None,
                None,
            )


def test_unary_func_arg_validation():
    with pytest.raises(TypeError):
        dpt.abs([1, 2, 3])
    try:
        a = dpt.arange(8)
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    dpt.abs(a, order="invalid")


def test_binary_func_arg_validation():
    with pytest.raises(dpctl.utils.ExecutionPlacementError):
        dpt.add([1, 2, 3], 1)
    try:
        a = dpt.arange(8)
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    with pytest.raises(ValueError):
        dpt.add(a, Ellipsis)
    dpt.add(a, a, order="invalid")


def test_all_data_types():
    fp16_fp64_types = set([dpt.float16, dpt.float64, dpt.complex128])
    fp64_types = set([dpt.float64, dpt.complex128])

    all_dts = tu._all_data_types(True, True)
    assert fp16_fp64_types.issubset(all_dts)

    all_dts = tu._all_data_types(True, False)
    assert dpt.float16 in all_dts
    assert not fp64_types.issubset(all_dts)

    all_dts = tu._all_data_types(False, True)
    assert dpt.float16 not in all_dts
    assert fp64_types.issubset(all_dts)

    all_dts = tu._all_data_types(False, False)
    assert not fp16_fp64_types.issubset(all_dts)


@pytest.mark.parametrize("fp16", [True, False])
@pytest.mark.parametrize("fp64", [True, False])
def test_maximal_inexact_types(fp16, fp64):
    assert not tu._is_maximal_inexact_type(dpt.int32, fp16, fp64)
    assert fp64 == tu._is_maximal_inexact_type(dpt.float64, fp16, fp64)
    assert fp64 == tu._is_maximal_inexact_type(dpt.complex128, fp16, fp64)
    assert fp64 != tu._is_maximal_inexact_type(dpt.float32, fp16, fp64)
    assert fp64 != tu._is_maximal_inexact_type(dpt.complex64, fp16, fp64)


def test_can_cast_device():
    assert tu._can_cast(dpt.int64, dpt.float64, True, True)
    # if f8 is available, can't cast i8 to f4
    assert not tu._can_cast(dpt.int64, dpt.float32, True, True)
    assert not tu._can_cast(dpt.int64, dpt.float32, False, True)
    # should be able to cast to f8 when f2 unavailable
    assert tu._can_cast(dpt.int64, dpt.float64, False, True)
    # casting to f4 acceptable when f8 unavailable
    assert tu._can_cast(dpt.int64, dpt.float32, True, False)
    assert tu._can_cast(dpt.int64, dpt.float32, False, False)
    # can't safely cast inexact type to inexact type of lesser precision
    assert not tu._can_cast(dpt.float32, dpt.float16, True, False)
    assert not tu._can_cast(dpt.float64, dpt.float32, False, True)


def test_acceptance_fns():
    """Check type promotion acceptance functions"""
    try:
        dev = dpctl.SyclDevice()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("Default device is not available")
    assert tu._acceptance_fn_reciprocal(
        dpt.float32, dpt.float32, dpt.float32, dev
    )
    assert tu._acceptance_fn_negative(dpt.int8, dpt.int16, dpt.int16, dev)


def test_weak_types():
    wbt = tu.WeakBooleanType(True)
    assert wbt.get()
    assert tu._weak_type_num_kind(wbt) == 0

    wit = tu.WeakIntegralType(7)
    assert wit.get() == 7
    assert tu._weak_type_num_kind(wit) == 1

    wft = tu.WeakFloatingType(3.1415926)
    assert wft.get() == 3.1415926
    assert tu._weak_type_num_kind(wft) == 2

    wct = tu.WeakComplexType(2.0 + 3.0j)
    assert wct.get() == 2 + 3j
    assert tu._weak_type_num_kind(wct) == 3


def test_arg_validation():
    with pytest.raises(TypeError):
        tu._weak_type_num_kind(dict())

    with pytest.raises(TypeError):
        tu._strong_dtype_num_kind(Ellipsis)

    with pytest.raises(ValueError):
        tu._strong_dtype_num_kind(np.dtype("O"))

    wt = tu.WeakFloatingType(2.0)
    with pytest.raises(ValueError):
        tu._resolve_weak_types(wt, wt, None)
