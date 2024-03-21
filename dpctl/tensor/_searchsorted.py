from typing import Literal, Union

import dpctl
import dpctl.utils as du

from ._copy_utils import _empty_like_orderK
from ._ctors import empty
from ._tensor_impl import _copy_usm_ndarray_into_usm_ndarray as ti_copy
from ._tensor_impl import _take as ti_take
from ._tensor_impl import (
    default_device_index_type as ti_default_device_index_type,
)
from ._tensor_sorting_impl import _searchsorted_left, _searchsorted_right
from ._type_utils import isdtype, result_type
from ._usmarray import usm_ndarray


def searchsorted(
    x1: usm_ndarray,
    x2: usm_ndarray,
    /,
    *,
    side: Literal["left", "right"] = "left",
    sorter: Union[usm_ndarray, None] = None,
) -> usm_ndarray:
    """searchsorted(x1, x2, side='left', sorter=None)

    Finds the indices into `x1` such that, if the corresponding elements
    in `x2` were inserted before the indices, the order of `x1`, when sorted
    in ascending order, would be preserved.

    Args:
        x1 (usm_ndarray):
            input array. Must be a one-dimensional array. If `sorter` is
            `None`, must be sorted in ascending order; otherwise, `sorter` must
            be an array of indices that sort `x1` in ascending order.
        x2 (usm_ndarray):
            array containing search values.
        side (Literal["left", "right]):
            argument controlling which index is returned if a value lands
            exactly on an edge. If `x2` is an array of rank `N` where
            `v = x2[n, m, ..., j]`, the element `ret[n, m, ..., j]` in the
            return array `ret` contains the position `i` such that
            if `side="left"`, it is the first index such that
            `x1[i-1] < v <= x1[i]`, `0` if `v <= x1[0]`, and `x1.size`
            if `v > x1[-1]`;
            and if `side="right"`, it is the first position `i` such that
            `x1[i-1] <= v < x1[i]`, `0` if `v < x1[0]`, and `x1.size`
            if `v >= x1[-1]`. Default: `"left"`.
        sorter (Optional[usm_ndarray]):
            array of indices that sort `x1` in ascending order. The array must
            have the same shape as `x1` and have an integral data type.
            Out of bound index values of `sorter` array are treated using
            `"wrap"` mode documented in :py:func:`dpctl.tensor.take`.
            Default: `None`.
    """
    if not isinstance(x1, usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(x1)}")
    if not isinstance(x2, usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(x2)}")
    if sorter is not None and not isinstance(sorter, usm_ndarray):
        raise TypeError(
            f"Expected dpctl.tensor.usm_ndarray, got {type(sorter)}"
        )

    if side not in ["left", "right"]:
        raise ValueError(
            "Unrecognized value of 'side' keyword argument. "
            "Expected either 'left' or 'right'"
        )

    if sorter is None:
        q = du.get_execution_queue([x1.sycl_queue, x2.sycl_queue])
    else:
        q = du.get_execution_queue(
            [x1.sycl_queue, x2.sycl_queue, sorter.sycl_queue]
        )
    if q is None:
        raise du.ExecutionPlacementError(
            "Execution placement can not be unambiguously "
            "inferred from input arguments."
        )

    if x1.ndim != 1:
        raise ValueError("First argument array must be one-dimensional")

    x1_dt = x1.dtype
    x2_dt = x2.dtype

    host_evs = []
    ev = dpctl.SyclEvent()
    if sorter is not None:
        if not isdtype(sorter.dtype, "integral"):
            raise ValueError(
                f"Sorter array must have integral data type, got {sorter.dtype}"
            )
        if x1.shape != sorter.shape:
            raise ValueError(
                "Sorter array must be one-dimension with the same "
                "shape as the first argument array"
            )
        res = empty(x1.shape, dtype=x1_dt, usm_type=x1.usm_type, sycl_queue=q)
        ind = (sorter,)
        axis = 0
        wrap_out_of_bound_indices_mode = 0
        ht_ev, ev = ti_take(
            x1,
            ind,
            res,
            axis,
            wrap_out_of_bound_indices_mode,
            sycl_queue=q,
            depends=[
                ev,
            ],
        )
        x1 = res
        host_evs.append(ht_ev)

    if x1_dt != x2_dt:
        dt = result_type(x1, x2)
        if x1_dt != dt:
            x1_buf = _empty_like_orderK(x1, dt)
            ht_ev, ev = ti_copy(
                src=x1,
                dst=x1_buf,
                sycl_queue=q,
                depends=[
                    ev,
                ],
            )
            host_evs.append(ht_ev)
            x1 = x1_buf
        if x2_dt != dt:
            x2_buf = _empty_like_orderK(x2, dt)
            ht_ev, ev = ti_copy(
                src=x2,
                dst=x2_buf,
                sycl_queue=q,
                depends=[
                    ev,
                ],
            )
            host_evs.append(ht_ev)
            x2 = x2_buf

    dst_usm_type = du.get_coerced_usm_type([x1.usm_type, x2.usm_type])
    index_dt = ti_default_device_index_type(q)

    dst = _empty_like_orderK(x2, index_dt, usm_type=dst_usm_type)

    if side == "left":
        ht_ev, _ = _searchsorted_left(
            hay=x1,
            needles=x2,
            positions=dst,
            sycl_queue=q,
            depends=[
                ev,
            ],
        )
    else:
        ht_ev, _ = _searchsorted_right(
            hay=x1,
            needles=x2,
            positions=dst,
            sycl_queue=q,
            depends=[
                ev,
            ],
        )

    host_evs.append(ht_ev)
    dpctl.SyclEvent.wait_for(host_evs)

    return dst
