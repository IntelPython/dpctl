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

import builtins

import dpctl.tensor as dpt
import dpctl.tensor._tensor_impl as ti


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


def _is_maximal_inexact_type(dt: dpt.dtype, _fp16: bool, _fp64: bool):
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
            and _is_maximal_inexact_type(to_, _fp16, _fp64)
        ):
            return True

    return can_cast_v


def _empty_like_orderK(X, dt, usm_type=None, dev=None):
    """Returns empty array like `x`, using order='K'

    For an array `x` that was obtained by permutation of a contiguous
    array the returned array will have the same shape and the same
    strides as `x`.
    """
    if not isinstance(X, dpt.usm_ndarray):
        raise TypeError(f"Expected usm_ndarray, got {type(X)}")
    if usm_type is None:
        usm_type = X.usm_type
    if dev is None:
        dev = X.device
    fl = X.flags
    if fl["C"] or X.size <= 1:
        return dpt.empty_like(
            X, dtype=dt, usm_type=usm_type, device=dev, order="C"
        )
    elif fl["F"]:
        return dpt.empty_like(
            X, dtype=dt, usm_type=usm_type, device=dev, order="F"
        )
    st = list(X.strides)
    perm = sorted(
        range(X.ndim), key=lambda d: builtins.abs(st[d]), reverse=True
    )
    inv_perm = sorted(range(X.ndim), key=lambda i: perm[i])
    st_sorted = [st[i] for i in perm]
    sh = X.shape
    sh_sorted = tuple(sh[i] for i in perm)
    R = dpt.empty(sh_sorted, dtype=dt, usm_type=usm_type, device=dev, order="C")
    if min(st_sorted) < 0:
        sl = tuple(
            slice(None, None, -1)
            if st_sorted[i] < 0
            else slice(None, None, None)
            for i in range(X.ndim)
        )
        R = R[sl]
    return dpt.permute_dims(R, inv_perm)


def _empty_like_pair_orderK(X1, X2, dt, usm_type, dev):
    if not isinstance(X1, dpt.usm_ndarray):
        raise TypeError(f"Expected usm_ndarray, got {type(X1)}")
    if not isinstance(X2, dpt.usm_ndarray):
        raise TypeError(f"Expected usm_ndarray, got {type(X2)}")
    nd1 = X1.ndim
    nd2 = X2.ndim
    if nd1 > nd2:
        return _empty_like_orderK(X1, dt, usm_type, dev)
    elif nd1 < nd2:
        return _empty_like_orderK(X2, dt, usm_type, dev)
    fl1 = X1.flags
    fl2 = X2.flags
    if fl1["C"] or fl2["C"]:
        return dpt.empty_like(
            X1, dtype=dt, usm_type=usm_type, device=dev, order="C"
        )
    if fl1["F"] and fl2["F"]:
        return dpt.empty_like(
            X1, dtype=dt, usm_type=usm_type, device=dev, order="F"
        )
    st1 = list(X1.strides)
    st2 = list(X2.strides)
    perm = sorted(
        range(nd1),
        key=lambda d: (builtins.abs(st1[d]), builtins.abs(st2[d])),
        reverse=True,
    )
    inv_perm = sorted(range(nd1), key=lambda i: perm[i])
    st1_sorted = [st1[i] for i in perm]
    st2_sorted = [st2[i] for i in perm]
    sh = X1.shape
    sh_sorted = tuple(sh[i] for i in perm)
    R = dpt.empty(sh_sorted, dtype=dt, usm_type=usm_type, device=dev, order="C")
    if max(min(st1_sorted), min(st2_sorted)) < 0:
        sl = tuple(
            slice(None, None, -1)
            if (st1_sorted[i] < 0 and st2_sorted[i] < 0)
            else slice(None, None, None)
            for i in range(nd1)
        )
        R = R[sl]
    return dpt.permute_dims(R, inv_perm)


def _to_device_supported_dtype(dt, dev):
    has_fp16 = dev.has_aspect_fp16
    has_fp64 = dev.has_aspect_fp64

    if has_fp64:
        if not has_fp16:
            if dt is dpt.float16:
                return dpt.float32
    else:
        if dt is dpt.float64:
            return dpt.float32
        elif dt is dpt.complex128:
            return dpt.complex64
        if not has_fp16 and dt is dpt.float16:
            return dpt.float32
    return dt


def _find_buf_dtype(arg_dtype, query_fn, sycl_dev):
    res_dt = query_fn(arg_dtype)
    if res_dt:
        return None, res_dt

    _fp16 = sycl_dev.has_aspect_fp16
    _fp64 = sycl_dev.has_aspect_fp64
    all_dts = _all_data_types(_fp16, _fp64)
    for buf_dt in all_dts:
        if _can_cast(arg_dtype, buf_dt, _fp16, _fp64):
            res_dt = query_fn(buf_dt)
            if res_dt:
                return buf_dt, res_dt

    return None, None


def _get_device_default_dtype(dt_kind, sycl_dev):
    if dt_kind == "b":
        return dpt.dtype(ti.default_device_bool_type(sycl_dev))
    elif dt_kind == "i":
        return dpt.dtype(ti.default_device_int_type(sycl_dev))
    elif dt_kind == "u":
        return dpt.dtype(ti.default_device_int_type(sycl_dev).upper())
    elif dt_kind == "f":
        return dpt.dtype(ti.default_device_fp_type(sycl_dev))
    elif dt_kind == "c":
        return dpt.dtype(ti.default_device_complex_type(sycl_dev))
    raise RuntimeError


def _find_buf_dtype2(arg1_dtype, arg2_dtype, query_fn, sycl_dev):
    res_dt = query_fn(arg1_dtype, arg2_dtype)
    if res_dt:
        return None, None, res_dt

    _fp16 = sycl_dev.has_aspect_fp16
    _fp64 = sycl_dev.has_aspect_fp64
    all_dts = _all_data_types(_fp16, _fp64)
    for buf1_dt in all_dts:
        for buf2_dt in all_dts:
            if _can_cast(arg1_dtype, buf1_dt, _fp16, _fp64) and _can_cast(
                arg2_dtype, buf2_dt, _fp16, _fp64
            ):
                res_dt = query_fn(buf1_dt, buf2_dt)
                if res_dt:
                    ret_buf1_dt = None if buf1_dt == arg1_dtype else buf1_dt
                    ret_buf2_dt = None if buf2_dt == arg2_dtype else buf2_dt
                    if ret_buf1_dt is None or ret_buf2_dt is None:
                        return ret_buf1_dt, ret_buf2_dt, res_dt
                    else:
                        # both are being promoted, if the kind of result is
                        # different than the kind of original input dtypes,
                        # we must use default dtype for the resulting kind.
                        if (res_dt.kind != arg1_dtype.kind) and (
                            res_dt.kind != arg2_dtype.kind
                        ):
                            default_dt = _get_device_default_dtype(
                                res_dt.kind, sycl_dev
                            )
                            if res_dt == default_dt:
                                return ret_buf1_dt, ret_buf2_dt, res_dt
                            else:
                                continue
                        else:
                            return ret_buf1_dt, ret_buf2_dt, res_dt

    return None, None, None


__all__ = [
    "_find_buf_dtype",
    "_find_buf_dtype2",
    "_empty_like_orderK",
    "_empty_like_pair_orderK",
    "_to_device_supported_dtype",
]
