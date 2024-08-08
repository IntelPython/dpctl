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

import builtins
import operator

import dpctl.tensor as dpt
import dpctl.tensor._tensor_impl as ti
import dpctl.tensor._tensor_reductions_impl as tri
import dpctl.utils as du
from dpctl.tensor._elementwise_common import (
    _get_dtype,
    _get_queue_usm_type,
    _get_shape,
    _validate_dtype,
)

from ._numpy_helper import normalize_axis_index, normalize_axis_tuple
from ._type_utils import (
    _resolve_one_strong_one_weak_types,
    _resolve_one_strong_two_weak_types,
)


def _boolean_reduction(x, axis, keepdims, func):
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(x)}")

    nd = x.ndim
    if axis is None:
        red_nd = nd
        # case of a scalar
        if red_nd == 0:
            return dpt.astype(x, dpt.bool)
        x_tmp = x
        res_shape = tuple()
        perm = list(range(nd))
    else:
        if not isinstance(axis, (tuple, list)):
            axis = (axis,)
        axis = normalize_axis_tuple(axis, nd, "axis")

        red_nd = len(axis)
        # check for axis=()
        if red_nd == 0:
            return dpt.astype(x, dpt.bool)
        perm = [i for i in range(nd) if i not in axis] + list(axis)
        x_tmp = dpt.permute_dims(x, perm)
        res_shape = x_tmp.shape[: nd - red_nd]

    exec_q = x.sycl_queue
    res_usm_type = x.usm_type

    _manager = du.SequentialOrderManager[exec_q]
    dep_evs = _manager.submitted_events
    # always allocate the temporary as
    # int32 and usm-device  to ensure that atomic updates
    # are supported
    res_tmp = dpt.empty(
        res_shape,
        dtype=dpt.int32,
        usm_type="device",
        sycl_queue=exec_q,
    )
    hev0, ev0 = func(
        src=x_tmp,
        trailing_dims_to_reduce=red_nd,
        dst=res_tmp,
        sycl_queue=exec_q,
        depends=dep_evs,
    )
    _manager.add_event_pair(hev0, ev0)

    # copy to boolean result array
    res = dpt.empty(
        res_shape,
        dtype=dpt.bool,
        usm_type=res_usm_type,
        sycl_queue=exec_q,
    )
    hev1, ev1 = ti._copy_usm_ndarray_into_usm_ndarray(
        src=res_tmp, dst=res, sycl_queue=exec_q, depends=[ev0]
    )
    _manager.add_event_pair(hev1, ev1)

    if keepdims:
        res_shape = res_shape + (1,) * red_nd
        inv_perm = sorted(range(nd), key=lambda d: perm[d])
        res = dpt.permute_dims(dpt.reshape(res, res_shape), inv_perm)
    return res


def all(x, /, *, axis=None, keepdims=False):
    """all(x, axis=None, keepdims=False)

    Tests whether all input array elements evaluate to True along a given axis.

    Args:
        x (usm_ndarray): Input array.
        axis (Optional[Union[int, Tuple[int,...]]]): Axis (or axes)
            along which to perform a logical AND reduction.
            When `axis` is `None`, a logical AND reduction
            is performed over all dimensions of `x`.
            If `axis` is negative, the axis is counted from
            the last dimension to the first.
            Default: `None`.
        keepdims (bool, optional): If `True`, the reduced axes are included
            in the result as singleton dimensions, and the result is
            broadcastable to the input array shape.
            If `False`, the reduced axes are not included in the result.
            Default: `False`.

    Returns:
        usm_ndarray:
            An array with a data type of `bool`
            containing the results of the logical AND reduction.
    """
    return _boolean_reduction(x, axis, keepdims, tri._all)


def any(x, /, *, axis=None, keepdims=False):
    """any(x, axis=None, keepdims=False)

    Tests whether any input array elements evaluate to True along a given axis.

    Args:
        x (usm_ndarray): Input array.
        axis (Optional[Union[int, Tuple[int,...]]]): Axis (or axes)
            along which to perform a logical OR reduction.
            When `axis` is `None`, a logical OR reduction
            is performed over all dimensions of `x`.
            If `axis` is negative, the axis is counted from
            the last dimension to the first.
            Default: `None`.
        keepdims (bool, optional): If `True`, the reduced axes are included
            in the result as singleton dimensions, and the result is
            broadcastable to the input array shape.
            If `False`, the reduced axes are not included in the result.
            Default: `False`.

    Returns:
        usm_ndarray:
            An array with a data type of `bool`
            containing the results of the logical OR reduction.
    """
    return _boolean_reduction(x, axis, keepdims, tri._any)


def _validate_diff_shape(sh1, sh2, axis):
    """Utility for validating that two shapes `sh1` and `sh2`
    are possible to concatenate along `axis`."""
    if not sh2:
        # scalars will always be accepted
        return True
    else:
        sh1_ndim = len(sh1)
        if sh1_ndim == len(sh2) and builtins.all(
            sh1[i] == sh2[i] for i in range(sh1_ndim) if i != axis
        ):
            return True
        else:
            return False


def _concat_diff_input(arr, axis, prepend, append):
    """Concatenates `arr`, `prepend` and, `append` along `axis`,
    where `arr` is an array and `prepend` and `append` are
    any mixture of arrays and scalars."""
    if prepend is not None and append is not None:
        q1, x_usm_type = arr.sycl_queue, arr.usm_type
        q2, prepend_usm_type = _get_queue_usm_type(prepend)
        q3, append_usm_type = _get_queue_usm_type(append)
        if q2 is None and q3 is None:
            exec_q = q1
            coerced_usm_type = x_usm_type
        elif q3 is None:
            exec_q = du.get_execution_queue((q1, q2))
            if exec_q is None:
                raise du.ExecutionPlacementError(
                    "Execution placement can not be unambiguously inferred "
                    "from input arguments."
                )
            coerced_usm_type = du.get_coerced_usm_type(
                (
                    x_usm_type,
                    prepend_usm_type,
                )
            )
        elif q2 is None:
            exec_q = du.get_execution_queue((q1, q3))
            if exec_q is None:
                raise du.ExecutionPlacementError(
                    "Execution placement can not be unambiguously inferred "
                    "from input arguments."
                )
            coerced_usm_type = du.get_coerced_usm_type(
                (
                    x_usm_type,
                    append_usm_type,
                )
            )
        else:
            exec_q = du.get_execution_queue((q1, q2, q3))
            if exec_q is None:
                raise du.ExecutionPlacementError(
                    "Execution placement can not be unambiguously inferred "
                    "from input arguments."
                )
            coerced_usm_type = du.get_coerced_usm_type(
                (
                    x_usm_type,
                    prepend_usm_type,
                    append_usm_type,
                )
            )
        du.validate_usm_type(coerced_usm_type, allow_none=False)
        arr_shape = arr.shape
        prepend_shape = _get_shape(prepend)
        append_shape = _get_shape(append)
        if not builtins.all(
            isinstance(s, (tuple, list))
            for s in (
                prepend_shape,
                append_shape,
            )
        ):
            raise TypeError(
                "Shape of arguments can not be inferred. "
                "Arguments are expected to be "
                "lists, tuples, or both"
            )
        valid_prepend_shape = _validate_diff_shape(
            arr_shape, prepend_shape, axis
        )
        if not valid_prepend_shape:
            raise ValueError(
                f"`diff` argument `prepend` with shape {prepend_shape} is "
                f"invalid for first input with shape {arr_shape}"
            )
        valid_append_shape = _validate_diff_shape(arr_shape, append_shape, axis)
        if not valid_append_shape:
            raise ValueError(
                f"`diff` argument `append` with shape {append_shape} is invalid"
                f" for first input with shape {arr_shape}"
            )
        sycl_dev = exec_q.sycl_device
        arr_dtype = arr.dtype
        prepend_dtype = _get_dtype(prepend, sycl_dev)
        append_dtype = _get_dtype(append, sycl_dev)
        if not builtins.all(
            _validate_dtype(o) for o in (prepend_dtype, append_dtype)
        ):
            raise ValueError("Operands have unsupported data types")
        prepend_dtype, append_dtype = _resolve_one_strong_two_weak_types(
            arr_dtype, prepend_dtype, append_dtype, sycl_dev
        )
        if isinstance(prepend, dpt.usm_ndarray):
            a_prepend = prepend
        else:
            a_prepend = dpt.asarray(
                prepend,
                dtype=prepend_dtype,
                usm_type=coerced_usm_type,
                sycl_queue=exec_q,
            )
        if isinstance(append, dpt.usm_ndarray):
            a_append = append
        else:
            a_append = dpt.asarray(
                append,
                dtype=append_dtype,
                usm_type=coerced_usm_type,
                sycl_queue=exec_q,
            )
        if not prepend_shape:
            prepend_shape = arr_shape[:axis] + (1,) + arr_shape[axis + 1 :]
            a_prepend = dpt.broadcast_to(a_prepend, prepend_shape)
        if not append_shape:
            append_shape = arr_shape[:axis] + (1,) + arr_shape[axis + 1 :]
            a_append = dpt.broadcast_to(a_append, append_shape)
        return dpt.concat((a_prepend, arr, a_append), axis=axis)
    elif prepend is not None:
        q1, x_usm_type = arr.sycl_queue, arr.usm_type
        q2, prepend_usm_type = _get_queue_usm_type(prepend)
        if q2 is None:
            exec_q = q1
            coerced_usm_type = x_usm_type
        else:
            exec_q = du.get_execution_queue((q1, q2))
            if exec_q is None:
                raise du.ExecutionPlacementError(
                    "Execution placement can not be unambiguously inferred "
                    "from input arguments."
                )
            coerced_usm_type = du.get_coerced_usm_type(
                (
                    x_usm_type,
                    prepend_usm_type,
                )
            )
        du.validate_usm_type(coerced_usm_type, allow_none=False)
        arr_shape = arr.shape
        prepend_shape = _get_shape(prepend)
        if not isinstance(prepend_shape, (tuple, list)):
            raise TypeError(
                "Shape of argument can not be inferred. "
                "Argument is expected to be a "
                "list or tuple"
            )
        valid_prepend_shape = _validate_diff_shape(
            arr_shape, prepend_shape, axis
        )
        if not valid_prepend_shape:
            raise ValueError(
                f"`diff` argument `prepend` with shape {prepend_shape} is "
                f"invalid for first input with shape {arr_shape}"
            )
        sycl_dev = exec_q.sycl_device
        arr_dtype = arr.dtype
        prepend_dtype = _get_dtype(prepend, sycl_dev)
        if not _validate_dtype(prepend_dtype):
            raise ValueError("Operand has unsupported data type")
        prepend_dtype = _resolve_one_strong_one_weak_types(
            arr_dtype, prepend_dtype, sycl_dev
        )
        if isinstance(prepend, dpt.usm_ndarray):
            a_prepend = prepend
        else:
            a_prepend = dpt.asarray(
                prepend,
                dtype=prepend_dtype,
                usm_type=coerced_usm_type,
                sycl_queue=exec_q,
            )
        if not prepend_shape:
            prepend_shape = arr_shape[:axis] + (1,) + arr_shape[axis + 1 :]
            a_prepend = dpt.broadcast_to(a_prepend, prepend_shape)
        return dpt.concat((a_prepend, arr), axis=axis)
    elif append is not None:
        q1, x_usm_type = arr.sycl_queue, arr.usm_type
        q2, append_usm_type = _get_queue_usm_type(append)
        if q2 is None:
            exec_q = q1
            coerced_usm_type = x_usm_type
        else:
            exec_q = du.get_execution_queue((q1, q2))
            if exec_q is None:
                raise du.ExecutionPlacementError(
                    "Execution placement can not be unambiguously inferred "
                    "from input arguments."
                )
            coerced_usm_type = du.get_coerced_usm_type(
                (
                    x_usm_type,
                    append_usm_type,
                )
            )
        du.validate_usm_type(coerced_usm_type, allow_none=False)
        arr_shape = arr.shape
        append_shape = _get_shape(append)
        if not isinstance(append_shape, (tuple, list)):
            raise TypeError(
                "Shape of argument can not be inferred. "
                "Argument is expected to be a "
                "list or tuple"
            )
        valid_append_shape = _validate_diff_shape(arr_shape, append_shape, axis)
        if not valid_append_shape:
            raise ValueError(
                f"`diff` argument `append` with shape {append_shape} is invalid"
                f" for first input with shape {arr_shape}"
            )
        sycl_dev = exec_q.sycl_device
        arr_dtype = arr.dtype
        append_dtype = _get_dtype(append, sycl_dev)
        if not _validate_dtype(append_dtype):
            raise ValueError("Operand has unsupported data type")
        append_dtype = _resolve_one_strong_one_weak_types(
            arr_dtype, append_dtype, sycl_dev
        )
        if isinstance(append, dpt.usm_ndarray):
            a_append = append
        else:
            a_append = dpt.asarray(
                append,
                dtype=append_dtype,
                usm_type=coerced_usm_type,
                sycl_queue=exec_q,
            )
        if not append_shape:
            append_shape = arr_shape[:axis] + (1,) + arr_shape[axis + 1 :]
            a_append = dpt.broadcast_to(a_append, append_shape)
        return dpt.concat((arr, a_append), axis=axis)
    else:
        arr1 = arr
    return arr1


def diff(x, /, *, axis=-1, n=1, prepend=None, append=None):
    """
    Calculates the `n`-th discrete forward difference of `x` along `axis`.

    Args:
        x (usm_ndarray):
            input array.
        axis (int):
            axis along which to compute the difference. A valid axis must be on
            the interval `[-N, N)`, where `N` is the rank (number of
            dimensions) of `x`.
            Default: `-1`
        n (int):
            number of times to recursively compute the difference.
            Default: `1`.
        prepend (Union[usm_ndarray, bool, int, float, complex]):
            value or values to prepend to the specified axis before taking the
            difference.
            Must have the same shape as `x` except along `axis`, which can have
            any shape.
            Default: `None`.
        append (Union[usm_ndarray, bool, int, float, complex]):
            value or values to append to the specified axis before taking the
            difference.
            Must have the same shape as `x` except along `axis`, which can have
            any shape.
            Default: `None`.

    Returns:
        usm_ndarray:
            an array containing the `n`-th differences. The array will have the
            same shape as `x`, except along `axis`, which will have shape:
                prepend.shape[axis] + x.shape[axis] + append.shape[axis] - n
            The data type of the returned array is determined by the Type
            Promotion Rules.
    """

    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(
            "Expecting dpctl.tensor.usm_ndarray type, " f"got {type(x)}"
        )
    x_nd = x.ndim
    axis = normalize_axis_index(operator.index(axis), x_nd)
    n = operator.index(n)
    if n < 0:
        raise ValueError(f"`n` must be positive, got {n}")
    arr = _concat_diff_input(x, axis, prepend, append)
    if n == 0:
        return arr
    # form slices and recurse
    sl0 = tuple(
        slice(None) if i != axis else slice(1, None) for i in range(x_nd)
    )
    sl1 = tuple(
        slice(None) if i != axis else slice(None, -1) for i in range(x_nd)
    )

    diff_op = dpt.not_equal if x.dtype == dpt.bool else dpt.subtract
    if n > 1:
        arr_tmp0 = diff_op(arr[sl0], arr[sl1])
        arr_tmp1 = diff_op(arr_tmp0[sl0], arr_tmp0[sl1])
        n = n - 2
        if n > 0:
            sl3 = tuple(
                slice(None) if i != axis else slice(None, -2)
                for i in range(x_nd)
            )
            for _ in range(n):
                arr_tmp0_sliced = arr_tmp0[sl3]
                diff_op(arr_tmp1[sl0], arr_tmp1[sl1], out=arr_tmp0_sliced)
                arr_tmp0, arr_tmp1 = arr_tmp1, arr_tmp0_sliced
        arr = arr_tmp1
    else:
        arr = diff_op(arr[sl0], arr[sl1])
    return arr
