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

    indices = dpt.broadcast_arrays(*indices)
    if x_ndim > 0:
        axis = operator.index(axis)
        axis = normalize_axis_index(axis, x_ndim)
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
        x = dpt.reshape(x, (x.size,), copy=False)
        axis = 0

    indices = dpt.broadcast_arrays(*indices)
    x_ndim = x.ndim
    if x_ndim > 0:
        axis = operator.index(axis)
        axis = normalize_axis_index(axis, x_ndim)

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
