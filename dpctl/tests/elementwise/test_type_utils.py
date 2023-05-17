import pytest

import dpctl
import dpctl.tensor as dpt
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
    X = tu._empty_like_orderK(a, dpt.int32, a.usm_type, a.device)
    assert X.flags["F"]


def test_type_utils_empty_like_orderK_invalid_args():
    with pytest.raises(TypeError):
        tu._empty_like_orderK([1, 2, 3], dpt.int32, "device", None)
    with pytest.raises(TypeError):
        tu._empty_like_pair_orderK(
            [1, 2, 3],
            (
                1,
                2,
                3,
            ),
            dpt.int32,
            "device",
            None,
        )
    try:
        a = dpt.empty(10, dtype=dpt.int32)
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    with pytest.raises(TypeError):
        tu._empty_like_pair_orderK(
            a,
            (
                1,
                2,
                3,
            ),
            dpt.int32,
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
            r = tu._find_buf_dtype(arg_dt, _denier_fn, dev)
            assert r == (
                None,
                None,
            )


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
            r = tu._find_buf_dtype2(arg1_dt, arg2_dt, _denier_fn, dev)
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
