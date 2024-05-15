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
import dpctl.tensor as dpt
import dpctl.tensor._tensor_elementwise_impl as tei
import dpctl.tensor._tensor_impl as ti
from dpctl.tensor._copy_utils import (
    _empty_like_orderK,
    _empty_like_pair_orderK,
    _empty_like_triple_orderK,
)
from dpctl.tensor._elementwise_common import (
    _get_dtype,
    _get_queue_usm_type,
    _get_shape,
    _validate_dtype,
)
from dpctl.tensor._manipulation_functions import _broadcast_shape_impl
from dpctl.tensor._type_utils import _can_cast, _to_device_supported_dtype
from dpctl.utils import ExecutionPlacementError

from ._type_utils import (
    WeakComplexType,
    WeakIntegralType,
    _is_weak_dtype,
    _strong_dtype_num_kind,
    _weak_type_num_kind,
)


def _resolve_one_strong_two_weak_types(st_dtype, dtype1, dtype2, dev):
    "Resolves weak data types per NEP-0050,"
    "where the second and third arguments are"
    "permitted to be weak types"
    if _is_weak_dtype(st_dtype):
        raise ValueError
    if _is_weak_dtype(dtype1):
        if _is_weak_dtype(dtype2):
            kind_num1 = _weak_type_num_kind(dtype1)
            kind_num2 = _weak_type_num_kind(dtype2)
            st_kind_num = _strong_dtype_num_kind(st_dtype)

            if kind_num1 > st_kind_num:
                if isinstance(dtype1, WeakIntegralType):
                    ret_dtype1 = dpt.dtype(ti.default_device_int_type(dev))
                elif isinstance(dtype1, WeakComplexType):
                    if st_dtype is dpt.float16 or st_dtype is dpt.float32:
                        ret_dtype1 = dpt.complex64
                    ret_dtype1 = _to_device_supported_dtype(dpt.complex128, dev)
                else:
                    ret_dtype1 = _to_device_supported_dtype(dpt.float64, dev)
            else:
                ret_dtype1 = st_dtype

            if kind_num2 > st_kind_num:
                if isinstance(dtype2, WeakIntegralType):
                    ret_dtype2 = dpt.dtype(ti.default_device_int_type(dev))
                elif isinstance(dtype2, WeakComplexType):
                    if st_dtype is dpt.float16 or st_dtype is dpt.float32:
                        ret_dtype2 = dpt.complex64
                    ret_dtype2 = _to_device_supported_dtype(dpt.complex128, dev)
                else:
                    ret_dtype2 = _to_device_supported_dtype(dpt.float64, dev)
            else:
                ret_dtype2 = st_dtype

            return ret_dtype1, ret_dtype2

        max_dt_num_kind, max_dtype = max(
            [
                (_strong_dtype_num_kind(st_dtype), st_dtype),
                (_strong_dtype_num_kind(dtype2), dtype2),
            ]
        )
        dt1_kind_num = _weak_type_num_kind(dtype1)
        if dt1_kind_num > max_dt_num_kind:
            if isinstance(dtype1, WeakIntegralType):
                return dpt.dtype(ti.default_device_int_type(dev)), dtype2
            if isinstance(dtype1, WeakComplexType):
                if max_dtype is dpt.float16 or max_dtype is dpt.float32:
                    return dpt.complex64, dtype2
                return (
                    _to_device_supported_dtype(dpt.complex128, dev),
                    dtype2,
                )
            return _to_device_supported_dtype(dpt.float64, dev), dtype2
        else:
            return max_dtype, dtype2
    elif _is_weak_dtype(dtype2):
        max_dt_num_kind, max_dtype = max(
            [
                (_strong_dtype_num_kind(st_dtype), st_dtype),
                (_strong_dtype_num_kind(dtype1), dtype1),
            ]
        )
        dt2_kind_num = _weak_type_num_kind(dtype2)
        if dt2_kind_num > max_dt_num_kind:
            if isinstance(dtype2, WeakIntegralType):
                return dtype1, dpt.dtype(ti.default_device_int_type(dev))
            if isinstance(dtype2, WeakComplexType):
                if max_dtype is dpt.float16 or max_dtype is dpt.float32:
                    return dtype1, dpt.complex64
                return (
                    dtype1,
                    _to_device_supported_dtype(dpt.complex128, dev),
                )
            return dtype1, _to_device_supported_dtype(dpt.float64, dev)
        else:
            return dtype1, max_dtype
    else:
        # both are strong dtypes
        # return unmodified
        return dtype1, dtype2


def _resolve_one_strong_one_weak_types(st_dtype, dtype, dev):
    "Resolves one weak data type with one strong data type per NEP-0050"
    if _is_weak_dtype(st_dtype):
        raise ValueError
    if _is_weak_dtype(dtype):
        st_kind_num = _strong_dtype_num_kind(st_dtype)
        kind_num = _weak_type_num_kind(dtype)
        if kind_num > st_kind_num:
            if isinstance(dtype, WeakIntegralType):
                return dpt.dtype(ti.default_device_int_type(dev))
            if isinstance(dtype, WeakComplexType):
                if st_dtype is dpt.float16 or st_dtype is dpt.float32:
                    return dpt.complex64
                return _to_device_supported_dtype(dpt.complex128, dev)
            return _to_device_supported_dtype(dpt.float64, dev)
        else:
            return st_dtype
    else:
        return dtype


def _check_clip_dtypes(res_dtype, arg1_dtype, arg2_dtype, sycl_dev):
    "Checks if both types `arg1_dtype` and `arg2_dtype` can be"
    "cast to `res_dtype` according to the rule `safe`"
    if arg1_dtype == res_dtype and arg2_dtype == res_dtype:
        return None, None, res_dtype

    _fp16 = sycl_dev.has_aspect_fp16
    _fp64 = sycl_dev.has_aspect_fp64
    if _can_cast(arg1_dtype, res_dtype, _fp16, _fp64) and _can_cast(
        arg2_dtype, res_dtype, _fp16, _fp64
    ):
        # prevent unnecessary casting
        ret_buf1_dt = None if res_dtype == arg1_dtype else res_dtype
        ret_buf2_dt = None if res_dtype == arg2_dtype else res_dtype
        return ret_buf1_dt, ret_buf2_dt, res_dtype
    else:
        return None, None, None


def _clip_none(x, val, out, order, _binary_fn):
    q1, x_usm_type = x.sycl_queue, x.usm_type
    q2, val_usm_type = _get_queue_usm_type(val)
    if q2 is None:
        exec_q = q1
        res_usm_type = x_usm_type
    else:
        exec_q = dpctl.utils.get_execution_queue((q1, q2))
        if exec_q is None:
            raise ExecutionPlacementError(
                "Execution placement can not be unambiguously inferred "
                "from input arguments."
            )
        res_usm_type = dpctl.utils.get_coerced_usm_type(
            (
                x_usm_type,
                val_usm_type,
            )
        )
    dpctl.utils.validate_usm_type(res_usm_type, allow_none=False)
    x_shape = x.shape
    val_shape = _get_shape(val)
    if not isinstance(val_shape, (tuple, list)):
        raise TypeError(
            "Shape of arguments can not be inferred. "
            "Arguments are expected to be "
            "lists, tuples, or both"
        )
    try:
        res_shape = _broadcast_shape_impl(
            [
                x_shape,
                val_shape,
            ]
        )
    except ValueError:
        raise ValueError(
            "operands could not be broadcast together with shapes "
            f"{x_shape} and {val_shape}"
        )
    sycl_dev = exec_q.sycl_device
    x_dtype = x.dtype
    val_dtype = _get_dtype(val, sycl_dev)
    if not _validate_dtype(val_dtype):
        raise ValueError("Operands have unsupported data types")

    val_dtype = _resolve_one_strong_one_weak_types(x_dtype, val_dtype, sycl_dev)

    res_dt = x.dtype
    _fp16 = sycl_dev.has_aspect_fp16
    _fp64 = sycl_dev.has_aspect_fp64
    if not _can_cast(val_dtype, res_dt, _fp16, _fp64):
        raise ValueError(
            f"function 'clip' does not support input types "
            f"({x_dtype}, {val_dtype}), "
            "and the inputs could not be safely coerced to any "
            "supported types according to the casting rule ''safe''."
        )

    orig_out = out
    if out is not None:
        if not isinstance(out, dpt.usm_ndarray):
            raise TypeError(
                f"output array must be of usm_ndarray type, got {type(out)}"
            )

        if not out.flags.writable:
            raise ValueError("provided `out` array is read-only")

        if out.shape != res_shape:
            raise ValueError(
                "The shape of input and output arrays are inconsistent. "
                f"Expected output shape is {res_shape}, got {out.shape}"
            )

        if res_dt != out.dtype:
            raise ValueError(
                f"Output array of type {res_dt} is needed, got {out.dtype}"
            )

        if dpctl.utils.get_execution_queue((exec_q, out.sycl_queue)) is None:
            raise ExecutionPlacementError(
                "Input and output allocation queues are not compatible"
            )

        if ti._array_overlap(x, out):
            if not ti._same_logical_tensors(x, out):
                out = dpt.empty_like(out)

        if isinstance(val, dpt.usm_ndarray):
            if (
                ti._array_overlap(val, out)
                and not ti._same_logical_tensors(val, out)
                and val_dtype == res_dt
            ):
                out = dpt.empty_like(out)

    if isinstance(val, dpt.usm_ndarray):
        val_ary = val
    else:
        val_ary = dpt.asarray(val, dtype=val_dtype, sycl_queue=exec_q)

    if order == "A":
        order = (
            "F"
            if all(
                arr.flags.f_contiguous
                for arr in (
                    x,
                    val_ary,
                )
            )
            else "C"
        )
    if val_dtype == res_dt:
        if out is None:
            if order == "K":
                out = _empty_like_pair_orderK(
                    x, val_ary, res_dt, res_shape, res_usm_type, exec_q
                )
            else:
                out = dpt.empty(
                    res_shape,
                    dtype=res_dt,
                    usm_type=res_usm_type,
                    sycl_queue=exec_q,
                    order=order,
                )
        if x_shape != res_shape:
            x = dpt.broadcast_to(x, res_shape)
        if val_ary.shape != res_shape:
            val_ary = dpt.broadcast_to(val_ary, res_shape)
        ht_binary_ev, binary_ev = _binary_fn(
            src1=x, src2=val_ary, dst=out, sycl_queue=exec_q
        )
        if not (orig_out is None or orig_out is out):
            # Copy the out data from temporary buffer to original memory
            ht_copy_out_ev, _ = ti._copy_usm_ndarray_into_usm_ndarray(
                src=out,
                dst=orig_out,
                sycl_queue=exec_q,
                depends=[binary_ev],
            )
            ht_copy_out_ev.wait()
            out = orig_out
        ht_binary_ev.wait()
        return out
    else:
        if order == "K":
            buf = _empty_like_orderK(val_ary, res_dt)
        else:
            buf = dpt.empty_like(val_ary, dtype=res_dt, order=order)
        ht_copy_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=val_ary, dst=buf, sycl_queue=exec_q
        )
        if out is None:
            if order == "K":
                out = _empty_like_pair_orderK(
                    x, buf, res_dt, res_shape, res_usm_type, exec_q
                )
            else:
                out = dpt.empty(
                    res_shape,
                    dtype=res_dt,
                    usm_type=res_usm_type,
                    sycl_queue=exec_q,
                    order=order,
                )

        if x_shape != res_shape:
            x = dpt.broadcast_to(x, res_shape)
        buf = dpt.broadcast_to(buf, res_shape)
        ht_binary_ev, binary_ev = _binary_fn(
            src1=x,
            src2=buf,
            dst=out,
            sycl_queue=exec_q,
            depends=[copy_ev],
        )
        if not (orig_out is None or orig_out is out):
            # Copy the out data from temporary buffer to original memory
            ht_copy_out_ev, _ = ti._copy_usm_ndarray_into_usm_ndarray(
                src=out,
                dst=orig_out,
                sycl_queue=exec_q,
                depends=[binary_ev],
            )
            ht_copy_out_ev.wait()
            out = orig_out
        ht_copy_ev.wait()
        ht_binary_ev.wait()
        return out


def clip(x, /, min=None, max=None, out=None, order="K"):
    """clip(x, min=None, max=None, out=None, order="K")

    Clips to the range [`min_i`, `max_i`] for each element `x_i`
    in `x`.

    Args:
        x (usm_ndarray): Array containing elements to clip.
            Must be compatible with `min` and `max` according
            to broadcasting rules.
        min ({None, Union[usm_ndarray, bool, int, float, complex]}, optional):
            Array containing minimum values.
            Must be compatible with `x` and `max` according
            to broadcasting rules.
        max ({None, Union[usm_ndarray, bool, int, float, complex]}, optional):
            Array containing maximum values.
            Must be compatible with `x` and `min` according
            to broadcasting rules.
        out ({None, usm_ndarray}, optional):
            Output array to populate.
            Array must have the correct shape and the expected data type.
        order ("C","F","A","K", optional):
            Memory layout of the newly output array, if parameter `out` is
            `None`.
            Default: "K".

    Returns:
        usm_ndarray:
            An array with elements clipped to the range [`min`, `max`].
            The returned array has the same data type as `x`.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(
            "Expected `x` to be of dpctl.tensor.usm_ndarray type, got "
            f"{type(x)}"
        )
    if order not in ["K", "C", "F", "A"]:
        order = "K"
    if min is None and max is None:
        exec_q = x.sycl_queue
        orig_out = out
        if out is not None:
            if not isinstance(out, dpt.usm_ndarray):
                raise TypeError(
                    "output array must be of usm_ndarray type, got "
                    f"{type(out)}"
                )

            if not out.flags.writable:
                raise ValueError("provided `out` array is read-only")

            if out.shape != x.shape:
                raise ValueError(
                    "The shape of input and output arrays are "
                    f"inconsistent. Expected output shape is {x.shape}, "
                    f"got {out.shape}"
                )

            if x.dtype != out.dtype:
                raise ValueError(
                    f"Output array of type {x.dtype} is needed, "
                    f"got {out.dtype}"
                )

            if (
                dpctl.utils.get_execution_queue((exec_q, out.sycl_queue))
                is None
            ):
                raise ExecutionPlacementError(
                    "Input and output allocation queues are not compatible"
                )

            if ti._array_overlap(x, out):
                if not ti._same_logical_tensors(x, out):
                    out = dpt.empty_like(out)
                else:
                    return out
        else:
            if order == "K":
                out = _empty_like_orderK(x, x.dtype)
            else:
                out = dpt.empty_like(x, order=order)

        ht_copy_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=x, dst=out, sycl_queue=exec_q
        )
        if not (orig_out is None or orig_out is out):
            # Copy the out data from temporary buffer to original memory
            ht_copy_out_ev, _ = ti._copy_usm_ndarray_into_usm_ndarray(
                src=out,
                dst=orig_out,
                sycl_queue=exec_q,
                depends=[copy_ev],
            )
            ht_copy_out_ev.wait()
            out = orig_out
        ht_copy_ev.wait()
        return out
    elif max is None:
        return _clip_none(x, min, out, order, tei._maximum)
    elif min is None:
        return _clip_none(x, max, out, order, tei._minimum)
    else:
        q1, x_usm_type = x.sycl_queue, x.usm_type
        q2, min_usm_type = _get_queue_usm_type(min)
        q3, max_usm_type = _get_queue_usm_type(max)
        if q2 is None and q3 is None:
            exec_q = q1
            res_usm_type = x_usm_type
        elif q3 is None:
            exec_q = dpctl.utils.get_execution_queue((q1, q2))
            if exec_q is None:
                raise ExecutionPlacementError(
                    "Execution placement can not be unambiguously inferred "
                    "from input arguments."
                )
            res_usm_type = dpctl.utils.get_coerced_usm_type(
                (
                    x_usm_type,
                    min_usm_type,
                )
            )
        elif q2 is None:
            exec_q = dpctl.utils.get_execution_queue((q1, q3))
            if exec_q is None:
                raise ExecutionPlacementError(
                    "Execution placement can not be unambiguously inferred "
                    "from input arguments."
                )
            res_usm_type = dpctl.utils.get_coerced_usm_type(
                (
                    x_usm_type,
                    max_usm_type,
                )
            )
        else:
            exec_q = dpctl.utils.get_execution_queue((q1, q2, q3))
            if exec_q is None:
                raise ExecutionPlacementError(
                    "Execution placement can not be unambiguously inferred "
                    "from input arguments."
                )
            res_usm_type = dpctl.utils.get_coerced_usm_type(
                (
                    x_usm_type,
                    min_usm_type,
                    max_usm_type,
                )
            )
        dpctl.utils.validate_usm_type(res_usm_type, allow_none=False)
        x_shape = x.shape
        min_shape = _get_shape(min)
        max_shape = _get_shape(max)
        if not all(
            isinstance(s, (tuple, list))
            for s in (
                min_shape,
                max_shape,
            )
        ):
            raise TypeError(
                "Shape of arguments can not be inferred. "
                "Arguments are expected to be "
                "lists, tuples, or both"
            )
        try:
            res_shape = _broadcast_shape_impl(
                [
                    x_shape,
                    min_shape,
                    max_shape,
                ]
            )
        except ValueError:
            raise ValueError(
                "operands could not be broadcast together with shapes "
                f"{x_shape}, {min_shape}, and {max_shape}"
            )
        sycl_dev = exec_q.sycl_device
        x_dtype = x.dtype
        min_dtype = _get_dtype(min, sycl_dev)
        max_dtype = _get_dtype(max, sycl_dev)
        if not all(_validate_dtype(o) for o in (min_dtype, max_dtype)):
            raise ValueError("Operands have unsupported data types")

        min_dtype, max_dtype = _resolve_one_strong_two_weak_types(
            x_dtype, min_dtype, max_dtype, sycl_dev
        )

        buf1_dt, buf2_dt, res_dt = _check_clip_dtypes(
            x_dtype,
            min_dtype,
            max_dtype,
            sycl_dev,
        )

        if res_dt is None:
            raise ValueError(
                f"function '{clip}' does not support input types "
                f"({x_dtype}, {min_dtype}, {max_dtype}), "
                "and the inputs could not be safely coerced to any "
                "supported types according to the casting rule ''safe''."
            )

        orig_out = out
        if out is not None:
            if not isinstance(out, dpt.usm_ndarray):
                raise TypeError(
                    "output array must be of usm_ndarray type, got "
                    f"{type(out)}"
                )

            if not out.flags.writable:
                raise ValueError("provided `out` array is read-only")

            if out.shape != res_shape:
                raise ValueError(
                    "The shape of input and output arrays are "
                    f"inconsistent. Expected output shape is {res_shape}, "
                    f"got {out.shape}"
                )

            if res_dt != out.dtype:
                raise ValueError(
                    f"Output array of type {res_dt} is needed, "
                    f"got {out.dtype}"
                )

            if (
                dpctl.utils.get_execution_queue((exec_q, out.sycl_queue))
                is None
            ):
                raise ExecutionPlacementError(
                    "Input and output allocation queues are not compatible"
                )

            if ti._array_overlap(x, out):
                if not ti._same_logical_tensors(x, out):
                    out = dpt.empty_like(out)

            if isinstance(min, dpt.usm_ndarray):
                if (
                    ti._array_overlap(min, out)
                    and not ti._same_logical_tensors(min, out)
                    and buf1_dt is None
                ):
                    out = dpt.empty_like(out)

            if isinstance(max, dpt.usm_ndarray):
                if (
                    ti._array_overlap(max, out)
                    and not ti._same_logical_tensors(max, out)
                    and buf2_dt is None
                ):
                    out = dpt.empty_like(out)

        if isinstance(min, dpt.usm_ndarray):
            a_min = min
        else:
            a_min = dpt.asarray(min, dtype=min_dtype, sycl_queue=exec_q)
        if isinstance(max, dpt.usm_ndarray):
            a_max = max
        else:
            a_max = dpt.asarray(max, dtype=max_dtype, sycl_queue=exec_q)

        if order == "A":
            order = (
                "F"
                if all(
                    arr.flags.f_contiguous
                    for arr in (
                        x,
                        a_min,
                        a_max,
                    )
                )
                else "C"
            )
        if buf1_dt is None and buf2_dt is None:
            if out is None:
                if order == "K":
                    out = _empty_like_triple_orderK(
                        x,
                        a_min,
                        a_max,
                        res_dt,
                        res_shape,
                        res_usm_type,
                        exec_q,
                    )
                else:
                    out = dpt.empty(
                        res_shape,
                        dtype=res_dt,
                        usm_type=res_usm_type,
                        sycl_queue=exec_q,
                        order=order,
                    )
            if x_shape != res_shape:
                x = dpt.broadcast_to(x, res_shape)
            if a_min.shape != res_shape:
                a_min = dpt.broadcast_to(a_min, res_shape)
            if a_max.shape != res_shape:
                a_max = dpt.broadcast_to(a_max, res_shape)
            ht_binary_ev, binary_ev = ti._clip(
                src=x, min=a_min, max=a_max, dst=out, sycl_queue=exec_q
            )
            if not (orig_out is None or orig_out is out):
                # Copy the out data from temporary buffer to original memory
                ht_copy_out_ev, _ = ti._copy_usm_ndarray_into_usm_ndarray(
                    src=out,
                    dst=orig_out,
                    sycl_queue=exec_q,
                    depends=[binary_ev],
                )
                ht_copy_out_ev.wait()
                out = orig_out
            ht_binary_ev.wait()
            return out

        elif buf1_dt is None:
            if order == "K":
                buf2 = _empty_like_orderK(a_max, buf2_dt)
            else:
                buf2 = dpt.empty_like(a_max, dtype=buf2_dt, order=order)
            ht_copy_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
                src=a_max, dst=buf2, sycl_queue=exec_q
            )
            if out is None:
                if order == "K":
                    out = _empty_like_triple_orderK(
                        x,
                        a_min,
                        buf2,
                        res_dt,
                        res_shape,
                        res_usm_type,
                        exec_q,
                    )
                else:
                    out = dpt.empty(
                        res_shape,
                        dtype=res_dt,
                        usm_type=res_usm_type,
                        sycl_queue=exec_q,
                        order=order,
                    )

            x = dpt.broadcast_to(x, res_shape)
            if a_min.shape != res_shape:
                a_min = dpt.broadcast_to(a_min, res_shape)
            buf2 = dpt.broadcast_to(buf2, res_shape)
            ht_binary_ev, binary_ev = ti._clip(
                src=x,
                min=a_min,
                max=buf2,
                dst=out,
                sycl_queue=exec_q,
                depends=[copy_ev],
            )
            if not (orig_out is None or orig_out is out):
                # Copy the out data from temporary buffer to original memory
                ht_copy_out_ev, _ = ti._copy_usm_ndarray_into_usm_ndarray(
                    src=out,
                    dst=orig_out,
                    sycl_queue=exec_q,
                    depends=[binary_ev],
                )
                ht_copy_out_ev.wait()
                out = orig_out
            ht_copy_ev.wait()
            ht_binary_ev.wait()
            return out

        elif buf2_dt is None:
            if order == "K":
                buf1 = _empty_like_orderK(a_min, buf1_dt)
            else:
                buf1 = dpt.empty_like(a_min, dtype=buf1_dt, order=order)
            ht_copy_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
                src=a_min, dst=buf1, sycl_queue=exec_q
            )
            if out is None:
                if order == "K":
                    out = _empty_like_triple_orderK(
                        x,
                        buf1,
                        a_max,
                        res_dt,
                        res_shape,
                        res_usm_type,
                        exec_q,
                    )
                else:
                    out = dpt.empty(
                        res_shape,
                        dtype=res_dt,
                        usm_type=res_usm_type,
                        sycl_queue=exec_q,
                        order=order,
                    )

            x = dpt.broadcast_to(x, res_shape)
            buf1 = dpt.broadcast_to(buf1, res_shape)
            if a_max.shape != res_shape:
                a_max = dpt.broadcast_to(a_max, res_shape)
            ht_binary_ev, binary_ev = ti._clip(
                src=x,
                min=buf1,
                max=a_max,
                dst=out,
                sycl_queue=exec_q,
                depends=[copy_ev],
            )
            if not (orig_out is None or orig_out is out):
                # Copy the out data from temporary buffer to original memory
                ht_copy_out_ev, _ = ti._copy_usm_ndarray_into_usm_ndarray(
                    src=out,
                    dst=orig_out,
                    sycl_queue=exec_q,
                    depends=[binary_ev],
                )
                ht_copy_out_ev.wait()
                out = orig_out
            ht_copy_ev.wait()
            ht_binary_ev.wait()
            return out

        if order == "K":
            if (
                x.flags.c_contiguous
                and a_min.flags.c_contiguous
                and a_max.flags.c_contiguous
            ):
                order = "C"
            elif (
                x.flags.f_contiguous
                and a_min.flags.f_contiguous
                and a_max.flags.f_contiguous
            ):
                order = "F"
        if order == "K":
            buf1 = _empty_like_orderK(a_min, buf1_dt)
        else:
            buf1 = dpt.empty_like(a_min, dtype=buf1_dt, order=order)
        ht_copy1_ev, copy1_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=a_min, dst=buf1, sycl_queue=exec_q
        )
        if order == "K":
            buf2 = _empty_like_orderK(a_max, buf2_dt)
        else:
            buf2 = dpt.empty_like(a_max, dtype=buf2_dt, order=order)
        ht_copy2_ev, copy2_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=a_max, dst=buf2, sycl_queue=exec_q
        )
        if out is None:
            if order == "K":
                out = _empty_like_triple_orderK(
                    x, buf1, buf2, res_dt, res_shape, res_usm_type, exec_q
                )
            else:
                out = dpt.empty(
                    res_shape,
                    dtype=res_dt,
                    usm_type=res_usm_type,
                    sycl_queue=exec_q,
                    order=order,
                )

        x = dpt.broadcast_to(x, res_shape)
        buf1 = dpt.broadcast_to(buf1, res_shape)
        buf2 = dpt.broadcast_to(buf2, res_shape)
        ht_, _ = ti._clip(
            src=x,
            min=buf1,
            max=buf2,
            dst=out,
            sycl_queue=exec_q,
            depends=[copy1_ev, copy2_ev],
        )
        dpctl.SyclEvent.wait_for([ht_copy1_ev, ht_copy2_ev, ht_])
        return out
