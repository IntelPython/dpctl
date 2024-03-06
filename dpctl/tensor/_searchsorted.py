from typing import Literal, Union

import dpctl.utils as du

from ._copy_utils import astype
from ._ctors import empty
from ._data_types import int32, int64
from ._tensor_sorting_impl import _searchsorted_left, _searchsorted_right
from ._type_utils import iinfo, isdtype, result_type
from ._usmarray import usm_ndarray


def searchsorted(
    x1: usm_ndarray,
    x2: usm_ndarray,
    /,
    *,
    side: Literal["left", "right"] = "left",
    sorter: Union[usm_ndarray, None] = None,
):
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

    if sorter is not None:
        if not isdtype(sorter.dtype, "integral"):
            raise ValueError
        if x1.shape != sorter.shape:
            raise ValueError
        x1 = x1[sorter]

    if x1.ndim != 1:
        raise ValueError("First argument array must be one-dimensional")

    if x1.dtype != x2.dtype:
        dt = result_type(x1, x2)
        x1 = astype(x1, dt, copy=None)
        x2 = astype(x2, dt, copy=None)

    dst_usm_type = du.get_coerced_usm_type([x1.usm_type, x2.usm_type])
    dst_dt = int32 if x2.size <= iinfo(int32).max else int64

    dst = empty(x2.shape, dtype=dst_dt, usm_type=dst_usm_type, sycl_queue=q)

    if side == "left":
        ht_ev, _ = _searchsorted_left(
            hay=x1, needles=x2, positions=dst, sycl_queue=q
        )
    else:
        ht_ev, _ = _searchsorted_right(
            hay=x1, needles=x2, positions=dst, sycl_queue=q
        )
    ht_ev.wait()

    return dst
