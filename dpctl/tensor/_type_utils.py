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


def _acceptance_fn_default(
    arg1_dtype, arg2_dtype, ret_buf1_dt, ret_buf2_dt, res_dt, sycl_dev
):
    return True


def _acceptance_fn_divide(
    arg1_dtype, arg2_dtype, ret_buf1_dt, ret_buf2_dt, res_dt, sycl_dev
):
    # both are being promoted, if the kind of result is
    # different than the kind of original input dtypes,
    # we use default dtype for the resulting kind.
    # This covers, e.g. (array_dtype_i1 / array_dtype_u1)
    # result of which in divide is double (in NumPy), but
    # regular type promotion rules peg at float16
    if (ret_buf1_dt.kind != arg1_dtype.kind) and (
        ret_buf2_dt.kind != arg2_dtype.kind
    ):
        default_dt = _get_device_default_dtype(res_dt.kind, sycl_dev)
        if res_dt == default_dt:
            return True
        else:
            return False
    else:
        return True


def _find_buf_dtype2(arg1_dtype, arg2_dtype, query_fn, sycl_dev, acceptance_fn):
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
                        acceptable = acceptance_fn(
                            arg1_dtype,
                            arg2_dtype,
                            ret_buf1_dt,
                            ret_buf2_dt,
                            res_dt,
                            sycl_dev,
                        )
                        if acceptable:
                            return ret_buf1_dt, ret_buf2_dt, res_dt
                        else:
                            continue

    return None, None, None


__all__ = [
    "_find_buf_dtype",
    "_find_buf_dtype2",
    "_to_device_supported_dtype",
    "_acceptance_fn_default",
    "_acceptance_fn_divide",
]
