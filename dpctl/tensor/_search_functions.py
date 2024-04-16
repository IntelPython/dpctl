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
import dpctl.tensor._tensor_impl as ti
from dpctl.tensor._manipulation_functions import _broadcast_shapes
from dpctl.utils import ExecutionPlacementError

from ._copy_utils import _empty_like_orderK, _empty_like_triple_orderK
from ._type_utils import _all_data_types, _can_cast


def _where_result_type(dt1, dt2, dev):
    res_dtype = dpt.result_type(dt1, dt2)
    fp16 = dev.has_aspect_fp16
    fp64 = dev.has_aspect_fp64

    all_dts = _all_data_types(fp16, fp64)
    if res_dtype in all_dts:
        return res_dtype
    else:
        for res_dtype_ in all_dts:
            if _can_cast(dt1, res_dtype_, fp16, fp64) and _can_cast(
                dt2, res_dtype_, fp16, fp64
            ):
                return res_dtype_
        return None


def where(condition, x1, x2, /, *, order="K", out=None):
    """
    Returns :class:`dpctl.tensor.usm_ndarray` with elements chosen
    from ``x1`` or ``x2`` depending on ``condition``.

    Args:
        condition (usm_ndarray): When ``True`` yields from ``x1``,
            and otherwise yields from ``x2``.
            Must be compatible with ``x1`` and ``x2`` according
            to broadcasting rules.
        x1 (usm_ndarray): Array from which values are chosen when
            ``condition`` is ``True``.
            Must be compatible with ``condition`` and ``x2`` according
            to broadcasting rules.
        x2 (usm_ndarray): Array from which values are chosen when
            ``condition`` is not ``True``.
            Must be compatible with ``condition`` and ``x2`` according
            to broadcasting rules.
        order (``"K"``, ``"C"``, ``"F"``, ``"A"``, optional):
            Memory layout of the new output arra,
            if parameter ``out`` is ``None``.
            Default: ``"K"``.
        out (Optional[usm_ndarray]):
            the array into which the result is written.
            The data type of `out` must match the expected shape and the
            expected data type of the result.
            If ``None`` then a new array is returned. Default: ``None``.

    Returns:
        usm_ndarray:
            An array with elements from ``x1`` where ``condition`` is ``True``,
            and elements from ``x2`` elsewhere.

    The data type of the returned array is determined by applying
    the Type Promotion Rules to ``x1`` and ``x2``.
    """
    if not isinstance(condition, dpt.usm_ndarray):
        raise TypeError(
            "Expecting dpctl.tensor.usm_ndarray type, " f"got {type(condition)}"
        )
    if not isinstance(x1, dpt.usm_ndarray):
        raise TypeError(
            "Expecting dpctl.tensor.usm_ndarray type, " f"got {type(x1)}"
        )
    if not isinstance(x2, dpt.usm_ndarray):
        raise TypeError(
            "Expecting dpctl.tensor.usm_ndarray type, " f"got {type(x2)}"
        )
    if order not in ["K", "C", "F", "A"]:
        order = "K"
    exec_q = dpctl.utils.get_execution_queue(
        (
            condition.sycl_queue,
            x1.sycl_queue,
            x2.sycl_queue,
        )
    )
    if exec_q is None:
        raise dpctl.utils.ExecutionPlacementError
    out_usm_type = dpctl.utils.get_coerced_usm_type(
        (
            condition.usm_type,
            x1.usm_type,
            x2.usm_type,
        )
    )

    x1_dtype = x1.dtype
    x2_dtype = x2.dtype
    out_dtype = _where_result_type(x1_dtype, x2_dtype, exec_q.sycl_device)
    if out_dtype is None:
        raise TypeError(
            "function 'where' does not support input "
            f"types ({x1_dtype}, {x2_dtype}), "
            "and the inputs could not be safely coerced "
            "to any supported types according to the casting rule ''safe''."
        )

    res_shape = _broadcast_shapes(condition, x1, x2)

    orig_out = out
    if out is not None:
        if not isinstance(out, dpt.usm_ndarray):
            raise TypeError(
                "output array must be of usm_ndarray type, got " f"{type(out)}"
            )

        if not out.flags.writable:
            raise ValueError("provided `out` array is read-only")

        if out.shape != res_shape:
            raise ValueError(
                "The shape of input and output arrays are "
                f"inconsistent. Expected output shape is {res_shape}, "
                f"got {out.shape}"
            )

        if out_dtype != out.dtype:
            raise ValueError(
                f"Output array of type {out_dtype} is needed, "
                f"got {out.dtype}"
            )

        if dpctl.utils.get_execution_queue((exec_q, out.sycl_queue)) is None:
            raise ExecutionPlacementError(
                "Input and output allocation queues are not compatible"
            )

        if ti._array_overlap(condition, out):
            if not ti._same_logical_tensors(condition, out):
                out = dpt.empty_like(out)

        if ti._array_overlap(x1, out):
            if not ti._same_logical_tensors(x1, out):
                out = dpt.empty_like(out)

        if ti._array_overlap(x2, out):
            if not ti._same_logical_tensors(x2, out):
                out = dpt.empty_like(out)

    if order == "A":
        order = (
            "F"
            if all(
                arr.flags.f_contiguous
                for arr in (
                    condition,
                    x1,
                    x2,
                )
            )
            else "C"
        )

    if condition.size == 0:
        if out is not None:
            return out
        else:
            if order == "K":
                return _empty_like_triple_orderK(
                    condition,
                    x1,
                    x2,
                    out_dtype,
                    res_shape,
                    out_usm_type,
                    exec_q,
                )
            else:
                return dpt.empty(
                    res_shape,
                    dtype=out_dtype,
                    order=order,
                    usm_type=out_usm_type,
                    sycl_queue=exec_q,
                )

    deps = []
    wait_list = []
    if x1_dtype != out_dtype:
        if order == "K":
            _x1 = _empty_like_orderK(x1, out_dtype)
        else:
            _x1 = dpt.empty_like(x1, dtype=out_dtype, order=order)
        ht_copy1_ev, copy1_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=x1, dst=_x1, sycl_queue=exec_q
        )
        x1 = _x1
        deps.append(copy1_ev)
        wait_list.append(ht_copy1_ev)

    if x2_dtype != out_dtype:
        if order == "K":
            _x2 = _empty_like_orderK(x2, out_dtype)
        else:
            _x2 = dpt.empty_like(x2, dtype=out_dtype, order=order)
        ht_copy2_ev, copy2_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=x2, dst=_x2, sycl_queue=exec_q
        )
        x2 = _x2
        deps.append(copy2_ev)
        wait_list.append(ht_copy2_ev)

    if out is None:
        if order == "K":
            out = _empty_like_triple_orderK(
                condition, x1, x2, out_dtype, res_shape, out_usm_type, exec_q
            )
        else:
            out = dpt.empty(
                res_shape,
                dtype=out_dtype,
                order=order,
                usm_type=out_usm_type,
                sycl_queue=exec_q,
            )

    condition = dpt.broadcast_to(condition, res_shape)
    x1 = dpt.broadcast_to(x1, res_shape)
    x2 = dpt.broadcast_to(x2, res_shape)

    hev, where_ev = ti._where(
        condition=condition,
        x1=x1,
        x2=x2,
        dst=out,
        sycl_queue=exec_q,
        depends=deps,
    )
    if not (orig_out is None or orig_out is out):
        # Copy the out data from temporary buffer to original memory
        ht_copy_out_ev, _ = ti._copy_usm_ndarray_into_usm_ndarray(
            src=out,
            dst=orig_out,
            sycl_queue=exec_q,
            depends=[where_ev],
        )
        ht_copy_out_ev.wait()
        out = orig_out
    dpctl.SyclEvent.wait_for(wait_list)
    hev.wait()

    return out
