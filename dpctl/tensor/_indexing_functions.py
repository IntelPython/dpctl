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

import operator

import numpy as np
from numpy.core.numeric import normalize_axis_index

import dpctl
import dpctl.tensor as dpt
import dpctl.tensor._tensor_impl as ti

from ._copy_utils import _extract_impl, _nonzero_impl


def _get_indexing_mode(name):
    modes = {"wrap": 0, "clip": 1}
    try:
        return modes[name]
    except KeyError:
        raise ValueError(
            "`mode` must be `wrap` or `clip`." "Got `{}`.".format(name)
        )


def take(x, indices, /, *, axis=None, mode="wrap"):
    """take(x, indices, axis=None, mode="wrap")

    Takes elements from array along a given axis.

    Args:
       x (usm_ndarray):
          The array that elements will be taken from.
       indices (usm_ndarray):
          One-dimensional array of indices.
       axis:
          The axis over which the values will be selected.
          If x is one-dimensional, this argument is optional.
          Default: `None`.
       mode:
          How out-of-bounds indices will be handled.
          "wrap" - clamps indices to (-n <= i < n), then wraps
          negative indices.
          "clip" - clips indices to (0 <= i < n)
          Default: `"wrap"`.

    Returns:
       usm_ndarray:
          Array with shape x.shape[:axis] + indices.shape + x.shape[axis + 1:]
          filled with elements from x.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(
            "Expected instance of `dpt.usm_ndarray`, got `{}`.".format(type(x))
        )

    if not isinstance(indices, dpt.usm_ndarray):
        raise TypeError(
            "`indices` expected `dpt.usm_ndarray`, got `{}`.".format(
                type(indices)
            )
        )
    if not np.issubdtype(indices.dtype, np.integer):
        raise IndexError(
            "`indices` expected integer data type, got `{}`".format(
                indices.dtype
            )
        )
    if indices.ndim != 1:
        raise ValueError(
            "`indices` expected a 1D array, got `{}`".format(indices.ndim)
        )
    exec_q = dpctl.utils.get_execution_queue([x.sycl_queue, indices.sycl_queue])
    if exec_q is None:
        raise dpctl.utils.ExecutionPlacementError
    res_usm_type = dpctl.utils.get_coerced_usm_type(
        [x.usm_type, indices.usm_type]
    )

    mode = _get_indexing_mode(mode)

    x_ndim = x.ndim
    if axis is None:
        if x_ndim > 1:
            raise ValueError(
                "`axis` cannot be `None` for array of dimension `{}`".format(
                    x_ndim
                )
            )
        axis = 0

    if x_ndim > 0:
        axis = normalize_axis_index(operator.index(axis), x_ndim)
        res_shape = x.shape[:axis] + indices.shape + x.shape[axis + 1 :]
    else:
        if axis != 0:
            raise ValueError("`axis` must be 0 for an array of dimension 0.")
        res_shape = indices.shape

    res = dpt.empty(
        res_shape, dtype=x.dtype, usm_type=res_usm_type, sycl_queue=exec_q
    )

    hev, _ = ti._take(x, (indices,), res, axis, mode, sycl_queue=exec_q)
    hev.wait()

    return res


def put(x, indices, vals, /, *, axis=None, mode="wrap"):
    """put(x, indices, vals, axis=None, mode="wrap")

    Puts values of an array into another array
    along a given axis.

    Args:
       x (usm_ndarray):
          The array the values will be put into.
       indices (usm_ndarray)
          One-dimensional array of indices.
       vals:
          Array of values to be put into `x`.
          Must be broadcastable to the shape of `indices`.
       axis:
          The axis over which the values will be placed.
          If x is one-dimensional, this argument is optional.
          Default: `None`.
       mode:
          How out-of-bounds indices will be handled.
          "wrap" - clamps indices to (-n <= i < n), then wraps
          negative indices.
          "clip" - clips indices to (0 <= i < n)
          Default: `"wrap"`.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(
            "Expected instance of `dpt.usm_ndarray`, got `{}`.".format(type(x))
        )
    if isinstance(vals, dpt.usm_ndarray):
        queues_ = [x.sycl_queue, vals.sycl_queue]
        usm_types_ = [x.usm_type, vals.usm_type]
    else:
        queues_ = [
            x.sycl_queue,
        ]
        usm_types_ = [
            x.usm_type,
        ]
    if not isinstance(indices, dpt.usm_ndarray):
        raise TypeError(
            "`indices` expected `dpt.usm_ndarray`, got `{}`.".format(
                type(indices)
            )
        )
    if indices.ndim != 1:
        raise ValueError(
            "`indices` expected a 1D array, got `{}`".format(indices.ndim)
        )
    if not np.issubdtype(indices.dtype, np.integer):
        raise IndexError(
            "`indices` expected integer data type, got `{}`".format(
                indices.dtype
            )
        )
    queues_.append(indices.sycl_queue)
    usm_types_.append(indices.usm_type)
    exec_q = dpctl.utils.get_execution_queue(queues_)
    if exec_q is None:
        raise dpctl.utils.ExecutionPlacementError
    vals_usm_type = dpctl.utils.get_coerced_usm_type(usm_types_)

    mode = _get_indexing_mode(mode)

    x_ndim = x.ndim
    if axis is None:
        if x_ndim > 1:
            raise ValueError(
                "`axis` cannot be `None` for array of dimension `{}`".format(
                    x_ndim
                )
            )
        axis = 0

    if x_ndim > 0:
        axis = normalize_axis_index(operator.index(axis), x_ndim)

        val_shape = x.shape[:axis] + indices.shape + x.shape[axis + 1 :]
    else:
        if axis != 0:
            raise ValueError("`axis` must be 0 for an array of dimension 0.")
        val_shape = indices.shape

    if not isinstance(vals, dpt.usm_ndarray):
        vals = dpt.asarray(
            vals, dtype=x.dtype, usm_type=vals_usm_type, sycl_queue=exec_q
        )

    vals = dpt.broadcast_to(vals, val_shape)

    hev, _ = ti._put(x, (indices,), vals, axis, mode, sycl_queue=exec_q)
    hev.wait()


def extract(condition, arr):
    """extract(condition, arr)

    Returns the elements of an array that satisfies the condition.

    If `condition` is boolean ``dpctl.tensor.extract`` is
    equivalent to ``arr[condition]``.

    Note that ``dpctl.tensor.place`` does the opposite of
    ``dpctl.tensor.extract``.

    Args:
       conditions (usm_ndarray):
            An array whose non-zero or True entries indicate the element
            of `arr` to extract.
       arr (usm_ndarray):
            Input array of the same size as `condition`.

    Returns:
        usm_ndarray:
            Rank 1 array of values from `arr` where `condition` is True.
    """
    if not isinstance(condition, dpt.usm_ndarray):
        raise TypeError(
            "Expecting dpctl.tensor.usm_ndarray type, " f"got {type(condition)}"
        )
    if not isinstance(arr, dpt.usm_ndarray):
        raise TypeError(
            "Expecting dpctl.tensor.usm_ndarray type, " f"got {type(arr)}"
        )
    exec_q = dpctl.utils.get_execution_queue(
        (
            condition.sycl_queue,
            arr.sycl_queue,
        )
    )
    if exec_q is None:
        raise dpctl.utils.ExecutionPlacementError
    if condition.shape != arr.shape:
        raise ValueError("Arrays are not of the same size")
    return _extract_impl(arr, condition)


def place(arr, mask, vals):
    """place(arr, mask, vals)

    Change elements of an array based on conditional and input values.

    If `mask` is boolean ``dpctl.tensor.place`` is
    equivalent to ``arr[condition] = vals``.

    Args:
        arr (usm_ndarray):
            Array to put data into.
        mask (usm_ndarray):
            Boolean mask array. Must have the same size as `arr`.
        vals (usm_ndarray, sequence):
            Values to put into `arr`. Only the first N elements are
            used, where N is the number of True values in `mask`. If
            `vals` is smaller than N, it will be repeated, and if
            elements of `arr` are to be masked, this sequence must be
            non-empty. Array `vals` must be one dimensional.
    """
    if not isinstance(arr, dpt.usm_ndarray):
        raise TypeError(
            "Expecting dpctl.tensor.usm_ndarray type, " f"got {type(arr)}"
        )
    if not isinstance(mask, dpt.usm_ndarray):
        raise TypeError(
            "Expecting dpctl.tensor.usm_ndarray type, " f"got {type(mask)}"
        )
    if not isinstance(vals, dpt.usm_ndarray):
        raise TypeError(
            "Expecting dpctl.tensor.usm_ndarray type, " f"got {type(vals)}"
        )
    exec_q = dpctl.utils.get_execution_queue(
        (
            arr.sycl_queue,
            mask.sycl_queue,
            vals.sycl_queue,
        )
    )
    if exec_q is None:
        raise dpctl.utils.ExecutionPlacementError
    if arr.shape != mask.shape or vals.ndim != 1:
        raise ValueError("Array sizes are not as required")
    cumsum = dpt.empty(mask.size, dtype="i8", sycl_queue=exec_q)
    nz_count = ti.mask_positions(mask, cumsum, sycl_queue=exec_q)
    if nz_count == 0:
        return
    if vals.size == 0:
        raise ValueError("Cannot insert from an empty array!")
    if vals.dtype == arr.dtype:
        rhs = vals
    else:
        rhs = dpt.astype(vals, arr.dtype)
    hev, _ = ti._place(
        dst=arr,
        cumsum=cumsum,
        axis_start=0,
        axis_end=mask.ndim,
        rhs=rhs,
        sycl_queue=exec_q,
    )
    hev.wait()


def nonzero(arr):
    """nonzero(arr)

    Return the indices of non-zero elements.

    Returns the tuple of usm_narrays, one for each dimension
    of `arr`, containing the indices of the non-zero elements
    in that dimension. The values of `arr` are always tested in
    row-major, C-style order.

    Args:
        arr (usm_ndarray):
            Input array, which has non-zero array rank.
    Returns:
        Tuple[usm_ndarray, ...]:
            Indices of non-zero array elements.
    """
    if not isinstance(arr, dpt.usm_ndarray):
        raise TypeError(
            "Expecting dpctl.tensor.usm_ndarray type, " f"got {type(arr)}"
        )
    if arr.ndim == 0:
        raise ValueError("Array of positive rank is exepcted")
    return _nonzero_impl(arr)
