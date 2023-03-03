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
from dpctl.tensor._tensor_impl import _put, _take

from ._copy_utils import _extract_impl, _nonzero_impl, _place_impl


def take(x, indices, /, *, axis=None, mode="clip"):
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(
            "Expected instance of `dpt.usm_ndarray`, got `{}`.".format(type(x))
        )

    if not isinstance(indices, list) and not isinstance(indices, tuple):
        indices = (indices,)

    queues_ = [
        x.sycl_queue,
    ]
    usm_types_ = [
        x.usm_type,
    ]

    for i in indices:
        if not isinstance(i, dpt.usm_ndarray):
            raise TypeError(
                "`indices` expected `dpt.usm_ndarray`, got `{}`.".format(
                    type(i)
                )
            )
        if not np.issubdtype(i.dtype, np.integer):
            raise IndexError(
                "`indices` expected integer data type, got `{}`".format(i.dtype)
            )
        queues_.append(i.sycl_queue)
        usm_types_.append(i.usm_type)
    exec_q = dpctl.utils.get_execution_queue(queues_)
    if exec_q is None:
        raise dpctl.utils.ExecutionPlacementError(
            "Can not automatically determine where to allocate the "
            "result or performance execution. "
            "Use `usm_ndarray.to_device` method to migrate data to "
            "be associated with the same queue."
        )
    res_usm_type = dpctl.utils.get_coerced_usm_type(usm_types_)

    modes = {"clip": 0, "wrap": 1}
    try:
        mode = modes[mode]
    except KeyError:
        raise ValueError("`mode` must be `clip` or `wrap`.")

    x_ndim = x.ndim
    if axis is None:
        if x_ndim > 1:
            raise ValueError(
                "`axis` cannot be `None` for array of dimension `{}`".format(
                    x_ndim
                )
            )
        axis = 0

    if len(indices) > 1:
        indices = dpt.broadcast_arrays(*indices)
    if x_ndim > 0:
        axis = normalize_axis_index(operator.index(axis), x_ndim)
        res_shape = (
            x.shape[:axis] + indices[0].shape + x.shape[axis + len(indices) :]
        )
    else:
        res_shape = indices[0].shape

    res = dpt.empty(
        res_shape, dtype=x.dtype, usm_type=res_usm_type, sycl_queue=exec_q
    )

    hev, _ = _take(x, indices, res, axis, mode, sycl_queue=exec_q)
    hev.wait()

    return res


def put(x, indices, vals, /, *, axis=None, mode="clip"):
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

    if not isinstance(indices, list) and not isinstance(indices, tuple):
        indices = (indices,)

    for i in indices:
        if not isinstance(i, dpt.usm_ndarray):
            raise TypeError(
                "`indices` expected `dpt.usm_ndarray`, got `{}`.".format(
                    type(i)
                )
            )
        if not np.issubdtype(i.dtype, np.integer):
            raise IndexError(
                "`indices` expected integer data type, got `{}`".format(i.dtype)
            )
        queues_.append(i.sycl_queue)
        usm_types_.append(i.usm_type)
    exec_q = dpctl.utils.get_execution_queue(queues_)
    if exec_q is None:
        raise dpctl.utils.ExecutionPlacementError(
            "Can not automatically determine where to allocate the "
            "result or performance execution. "
            "Use `usm_ndarray.to_device` method to migrate data to "
            "be associated with the same queue."
        )
    val_usm_type = dpctl.utils.get_coerced_usm_type(usm_types_)

    modes = {"clip": 0, "wrap": 1}
    try:
        mode = modes[mode]
    except KeyError:
        raise ValueError("`mode` must be `wrap`, or `clip`.")

    # when axis is none, array is treated as 1D
    if axis is None:
        try:
            x = dpt.reshape(x, (x.size,), copy=False)
            axis = 0
        except ValueError:
            raise ValueError("Cannot create 1D view of input array")
    if len(indices) > 1:
        indices = dpt.broadcast_arrays(*indices)
    x_ndim = x.ndim
    if x_ndim > 0:
        axis = normalize_axis_index(operator.index(axis), x_ndim)

        val_shape = (
            x.shape[:axis] + indices[0].shape + x.shape[axis + len(indices) :]
        )
    else:
        val_shape = indices[0].shape

    if not isinstance(vals, dpt.usm_ndarray):
        vals = dpt.asarray(
            vals, dtype=x.dtype, usm_type=val_usm_type, sycl_queue=exec_q
        )

    vals = dpt.broadcast_to(vals, val_shape)

    hev, _ = _put(x, indices, vals, axis, mode, sycl_queue=exec_q)
    hev.wait()


def extract(condition, arr):
    """extract(condition, arr)

    Returns the elements of an array that satisfies the condition.

    If `condition` is boolean :func:``dpctl.tensor.extract`` is
    equivalent to ``arr[condition]``.

    Note that :func:``dpctl.tensor.place`` does the opposite of
    :func:``dpctl.tensor.extract``.

    Args:
       conditions: usm_ndarray
          An array whose non-zero or True entries indicate the element
          of `arr` to extract.
       arr: usm_ndarray
          Input array of the same size as `condition`.

    Returns:
       extract: usm_ndarray
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

    If `mask` is boolean :func:``dpctl.tensor.place`` is
    equivalent to ``arr[condition] = vals``.

    Args:
       arr: usm_ndarray
          Array to put data into.
       mask: usm_ndarray
          Boolean mask array. Must have the same size as `arr`.
       vals: usm_ndarray
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
    # FIXME
    _place_impl(arr, mask, vals, axis=0)


def nonzero(arr):
    """nonzero(arr)

    Return the indices of non-zero elements.

    Returns the tuple of usm_narrays, one for each dimension
    of `arr`, containing the indices of the non-zero elements
    in that dimension. The values of `arr` are always tested in
    row-major, C-style order.

    Args:
       arr: usm_ndarray
          Input array, which has non-zero array rank.
    Returns:
       tuple_of_usm_ndarrays: tuple
          Indices of non-zero array elements.
    """
    if not isinstance(arr, dpt.usm_ndarray):
        raise TypeError(
            "Expecting dpctl.tensor.usm_ndarray type, " f"got {type(arr)}"
        )
    if arr.ndim == 0:
        raise ValueError("Array of positive rank is exepcted")
    return _nonzero_impl(arr)
