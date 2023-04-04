#                       Data Parallel Control (dpctl)
#
#  Copyright 2020-2022 Intel Corporation
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


def where(condition, x1, x2):
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
    exec_q = dpctl.utils.get_execution_queue(
        (
            condition.sycl_queue,
            x1.sycl_queue,
            x2.sycl_queue,
        )
    )
    if exec_q is None:
        raise dpctl.utils.ExecutionPlacementError
    dst_usm_type = dpctl.utils.get_coerced_usm_type(
        (
            condition.usm_type,
            x1.usm_type,
            x2.usm_type,
        )
    )

    x1_dtype = x1.dtype
    x2_dtype = x2.dtype
    dst_dtype = _where_result_type(x1_dtype, x2_dtype, exec_q.sycl_device)
    if dst_dtype is None:
        raise TypeError(
            "function 'where' does not support input "
            f"types ({x1_dtype}, {x2_dtype}), "
            "and the inputs could not be safely coerced "
            "to any supported types according to the casting rule ''safe''."
        )

    res_shape = _broadcast_shapes(condition, x1, x2)

    if condition.size == 0:
        return dpt.empty(
            res_shape, dtype=dst_dtype, usm_type=dst_usm_type, sycl_queue=exec_q
        )

    deps = []
    wait_list = []
    if x1_dtype is not dst_dtype:
        _x1 = dpt.empty_like(x1, dtype=dst_dtype)
        ht_copy1_ev, copy1_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=x1, dst=_x1, sycl_queue=exec_q
        )
        x1 = _x1
        deps.append(copy1_ev)
        wait_list.append(ht_copy1_ev)

    if x2_dtype is not dst_dtype:
        _x2 = dpt.empty_like(x2, dtype=dst_dtype)
        ht_copy2_ev, copy2_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=x2, dst=_x2, sycl_queue=exec_q
        )
        x2 = _x2
        deps.append(copy2_ev)
        wait_list.append(ht_copy2_ev)

    condition = dpt.broadcast_to(condition, res_shape)
    x1 = dpt.broadcast_to(x1, res_shape)
    x2 = dpt.broadcast_to(x2, res_shape)

    # dst is F-contiguous when all inputs are F contiguous
    # otherwise, defaults to C-contiguous
    if all(
        (
            condition.flags.fnc,
            x1.flags.fnc,
            x2.flags.fnc,
        )
    ):
        order = "F"
    else:
        order = "C"

    dst = dpt.empty(
        res_shape,
        dtype=dst_dtype,
        order=order,
        usm_type=dst_usm_type,
        sycl_queue=exec_q,
    )

    hev, _ = ti._where(
        condition=condition,
        x1=x1,
        x2=x2,
        dst=dst,
        sycl_queue=exec_q,
        depends=deps,
    )
    wait_list.append(hev)
    dpctl.SyclEvent.wait_for(wait_list)

    return dst
