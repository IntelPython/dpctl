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

import numpy as np

import dpctl.tensor as dpt
import dpctl.tensor._tensor_impl as ti


def _all_data_types(_fp16, _fp64):
    _non_fp_types = [
        dpt.bool,
        dpt.int8,
        dpt.uint8,
        dpt.int16,
        dpt.uint16,
        dpt.int32,
        dpt.uint32,
        dpt.int64,
        dpt.uint64,
    ]
    if _fp64:
        if _fp16:
            return _non_fp_types + [
                dpt.float16,
                dpt.float32,
                dpt.float64,
                dpt.complex64,
                dpt.complex128,
            ]
        else:
            return _non_fp_types + [
                dpt.float32,
                dpt.float64,
                dpt.complex64,
                dpt.complex128,
            ]
    else:
        if _fp16:
            return _non_fp_types + [
                dpt.float16,
                dpt.float32,
                dpt.complex64,
            ]
        else:
            return _non_fp_types + [
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


def _dtype_supported_by_device_impl(
    dt: dpt.dtype, has_fp16: bool, has_fp64: bool
) -> bool:
    if has_fp64:
        if not has_fp16:
            if dt is dpt.float16:
                return False
    else:
        if dt is dpt.float64:
            return False
        elif dt is dpt.complex128:
            return False
        if not has_fp16 and dt is dpt.float16:
            return False
    return True


def _can_cast(
    from_: dpt.dtype, to_: dpt.dtype, _fp16: bool, _fp64: bool, casting="safe"
) -> bool:
    """
    Can `from_` be cast to `to_` safely on a device with
    fp16 and fp64 aspects as given?
    """
    if not _dtype_supported_by_device_impl(to_, _fp16, _fp64):
        return False
    can_cast_v = np.can_cast(from_, to_, casting=casting)  # ask NumPy
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


def _to_device_supported_dtype_impl(dt, has_fp16, has_fp64):
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


def _to_device_supported_dtype(dt, dev):
    has_fp16 = dev.has_aspect_fp16
    has_fp64 = dev.has_aspect_fp64

    return _to_device_supported_dtype_impl(dt, has_fp16, has_fp64)


def _acceptance_fn_default_unary(arg_dtype, ret_buf_dt, res_dt, sycl_dev):
    return True


def _acceptance_fn_reciprocal(arg_dtype, buf_dt, res_dt, sycl_dev):
    # if the kind of result is different from
    # the kind of input, use the default data
    # we use default dtype for the resulting kind.
    # This guarantees alignment of reciprocal and
    # divide output types.
    if buf_dt.kind != arg_dtype.kind:
        default_dt = _get_device_default_dtype(res_dt.kind, sycl_dev)
        if res_dt == default_dt:
            return True
        else:
            return False
    else:
        return True


def _find_buf_dtype(arg_dtype, query_fn, sycl_dev, acceptance_fn):
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
                acceptable = acceptance_fn(arg_dtype, buf_dt, res_dt, sycl_dev)
                if acceptable:
                    return buf_dt, res_dt
                else:
                    continue

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


def _acceptance_fn_default_binary(
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


class finfo_object:
    """
    `numpy.finfo` subclass which returns Python floating-point scalars for
    `eps`, `max`, `min`, and `smallest_normal` attributes.
    """

    def __init__(self, dtype):
        _supported_dtype([dpt.dtype(dtype)])
        self._finfo = np.finfo(dtype)

    @property
    def bits(self):
        """
        number of bits occupied by the real-valued floating-point data type.
        """
        return int(self._finfo.bits)

    @property
    def smallest_normal(self):
        """
        smallest positive real-valued floating-point number with full
        precision.
        """
        return float(self._finfo.smallest_normal)

    @property
    def tiny(self):
        """an alias for `smallest_normal`"""
        return float(self._finfo.tiny)

    @property
    def eps(self):
        """
        difference between 1.0 and the next smallest representable real-valued
        floating-point number larger than 1.0 according to the IEEE-754
        standard.
        """
        return float(self._finfo.eps)

    @property
    def epsneg(self):
        """
        difference between 1.0 and the next smallest representable real-valued
        floating-point number smaller than 1.0 according to the IEEE-754
        standard.
        """
        return float(self._finfo.epsneg)

    @property
    def min(self):
        """smallest representable real-valued number."""
        return float(self._finfo.min)

    @property
    def max(self):
        "largest representable real-valued number."
        return float(self._finfo.max)

    @property
    def resolution(self):
        "the approximate decimal resolution of this type."
        return float(self._finfo.resolution)

    @property
    def precision(self):
        """
        the approximate number of decimal digits to which this kind of
        floating point type is precise.
        """
        return float(self._finfo.precision)

    @property
    def dtype(self):
        """
        the dtype for which finfo returns information. For complex input, the
        returned dtype is the associated floating point dtype for its real and
        complex components.
        """
        return self._finfo.dtype

    def __str__(self):
        return self._finfo.__str__()

    def __repr__(self):
        return self._finfo.__repr__()


def can_cast(from_, to, casting="safe"):
    """ can_cast(from, to, casting="safe")

    Determines if one data type can be cast to another data type according \
    to Type Promotion Rules.

    Args:
       from_ (Union[usm_ndarray, dtype]):
           source data type. If `from_` is an array, a device-specific type
           promotion rules apply.
       to (dtype):
           target data type
       casting (Optional[str]):
            controls what kind of data casting may occur.
                * "no" means data types should not be cast at all.
                * "safe" means only casts that preserve values are allowed.
                * "same_kind" means only safe casts and casts within a kind,
                  like `float64` to `float32`, are allowed.
                * "unsafe" means any data conversion can be done.
            Default: `"safe"`.

    Returns:
        bool:
            Gives `True` if cast can occur according to the casting rule.

    Device-specific type promotion rules take into account which data type are
    and are not supported by a specific device.
    """
    if isinstance(to, dpt.usm_ndarray):
        raise TypeError(f"Expected `dpt.dtype` type, got {type(to)}.")

    dtype_to = dpt.dtype(to)
    _supported_dtype([dtype_to])

    if isinstance(from_, dpt.usm_ndarray):
        dtype_from = from_.dtype
        return _can_cast(
            dtype_from,
            dtype_to,
            from_.sycl_device.has_aspect_fp16,
            from_.sycl_device.has_aspect_fp64,
            casting=casting,
        )
    else:
        dtype_from = dpt.dtype(from_)
        _supported_dtype([dtype_from])
        # query casting as if all dtypes are supported
        return _can_cast(dtype_from, dtype_to, True, True, casting=casting)


def result_type(*arrays_and_dtypes):
    """
    result_type(*arrays_and_dtypes)

    Returns the dtype that results from applying the Type Promotion Rules to \
        the arguments.

    Args:
        arrays_and_dtypes (Union[usm_ndarray, dtype]):
            An arbitrary length sequence of usm_ndarray objects or dtypes.

    Returns:
        dtype:
            The dtype resulting from an operation involving the
            input arrays and dtypes.
    """
    dtypes = []
    devices = []
    for arg_i in arrays_and_dtypes:
        if isinstance(arg_i, dpt.usm_ndarray):
            devices.append(arg_i.sycl_device)
            dtypes.append(arg_i.dtype)
        else:
            dt = dpt.dtype(arg_i)
            _supported_dtype([dt])
            dtypes.append(dt)

    has_fp16 = True
    has_fp64 = True
    if devices:
        inspected = False
        for d in devices:
            if inspected:
                unsame_fp16_support = d.has_aspect_fp16 != has_fp16
                unsame_fp64_support = d.has_aspect_fp64 != has_fp64
                if unsame_fp16_support or unsame_fp64_support:
                    raise ValueError(
                        "Input arrays reside on devices "
                        "with different device supports; "
                        "unable to determine which "
                        "device-specific type promotion rules "
                        "to use."
                    )
            else:
                has_fp16 = d.has_aspect_fp16
                has_fp64 = d.has_aspect_fp64
                inspected = True

    if not (has_fp16 and has_fp64):
        for dt in dtypes:
            if not _dtype_supported_by_device_impl(dt, has_fp16, has_fp64):
                raise ValueError(f"Argument {dt} is not supported by ")
        res_dt = np.result_type(*dtypes)
        res_dt = _to_device_supported_dtype_impl(res_dt, has_fp16, has_fp64)
        return res_dt

    return np.result_type(*dtypes)


def iinfo(dtype):
    """iinfo(dtype)

    Returns machine limits for integer data types.

    Args:
        dtype (dtype, usm_ndarray):
            integer dtype or
            an array with integer dtype.

    Returns:
        iinfo_object:
            An object with the following attributes
            * bits: int
                number of bits occupied by the data type
            * max: int
                largest representable number.
            * min: int
                smallest representable number.
            * dtype: dtype
                integer data type.
    """
    if isinstance(dtype, dpt.usm_ndarray):
        dtype = dtype.dtype
    _supported_dtype([dpt.dtype(dtype)])
    return np.iinfo(dtype)


def finfo(dtype):
    """finfo(type)

    Returns machine limits for floating-point data types.

    Args:
        dtype (dtype, usm_ndarray): floating-point dtype or
            an array with floating point data type.
            If complex, the information is about its component
            data type.

    Returns:
        finfo_object:
            an object have the following attributes
                * bits: int
                    number of bits occupied by dtype.
                * eps: float
                    difference between 1.0 and the next smallest representable
                    real-valued floating-point number larger than 1.0 according
                    to the IEEE-754 standard.
                * max: float
                    largest representable real-valued number.
                * min: float
                    smallest representable real-valued number.
                * smallest_normal: float
                    smallest positive real-valued floating-point number with
                    full precision.
                * dtype: dtype
                    real-valued floating-point data type.

    """
    if isinstance(dtype, dpt.usm_ndarray):
        dtype = dtype.dtype
    _supported_dtype([dpt.dtype(dtype)])
    return finfo_object(dtype)


def _supported_dtype(dtypes):
    for dtype in dtypes:
        if dtype.char not in "?bBhHiIlLqQefdFD":
            raise ValueError(f"Dpctl doesn't support dtype {dtype}.")
    return True


__all__ = [
    "_find_buf_dtype",
    "_find_buf_dtype2",
    "_to_device_supported_dtype",
    "_acceptance_fn_default_unary",
    "_acceptance_fn_reciprocal",
    "_acceptance_fn_default_binary",
    "_acceptance_fn_divide",
    "can_cast",
    "finfo",
    "iinfo",
    "result_type",
]