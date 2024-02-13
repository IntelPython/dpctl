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

import operator

from numpy.core.numeric import normalize_axis_index, normalize_axis_tuple

import dpctl
import dpctl.tensor as dpt
import dpctl.tensor._tensor_elementwise_impl as tei
import dpctl.tensor._tensor_impl as ti
import dpctl.tensor._tensor_linalg_impl as tli
from dpctl.tensor._copy_utils import _empty_like_orderK, _empty_like_pair_orderK
from dpctl.tensor._manipulation_functions import _broadcast_shape_impl
from dpctl.tensor._type_utils import (
    _acceptance_fn_default_binary,
    _find_buf_dtype2,
    _to_device_supported_dtype,
)
from dpctl.utils import ExecutionPlacementError


def matrix_transpose(x):
    """matrix_transpose(x)

    Transposes the innermost two dimensions of `x`, where `x` is a
    2-dimensional matrix or a stack of 2-dimensional matrices.

    To convert from a 1-dimensional array to a 2-dimensional column
    vector, use x[:, dpt.newaxis].

    Args:
       x (usm_ndarray):
          Input array with shape (..., m, n).

    Returns:
       usm_ndarray:
          Array with shape (..., n, m).
    """

    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(
            "Expected instance of `dpt.usm_ndarray`, got `{}`.".format(type(x))
        )
    if x.ndim < 2:
        raise ValueError(
            "dpctl.tensor.matrix_transpose requires array to have"
            "at least 2 dimensions"
        )

    return x.mT


def tensordot(x1, x2, axes=2):
    """tensordot(x1, x2, axes=2)

    Returns a tensor contraction of `x1` and `x2` over specific axes.

    Args:
        x1 (usm_ndarray):
            first input array, expected to have numeric data type.
        x2 (usm_ndarray):
            second input array, expected to have numeric data type.
            Corresponding contracted axes of `x1` and `x2` must be equal.
        axes (Union[int, Tuple[Sequence[int], Sequence[int]]):
            number of axes to contract or explicit sequences of axes for
            `x1` and `x2`, respectively. If `axes` is an integer equal to `N`,
            then the contraction is performed over last `N` axes of `x1` and
            the first `N` axis of `x2` in order. The size of each corresponding
            axis must match and must be non-negative.
               * if `N` equals `0`, the result is the tensor outer product
               * if `N` equals `1`, the result is the tensor dot product
               * if `N` equals `2`, the result is the tensor double
                 contraction (default).
            If `axes` is a tuple of two sequences `(x1_axes, x2_axes)`, the
            first sequence applies to `x1` and the second sequence applies
            to `x2`. Both sequences must have equal length, and each axis
            `x1_axes[i]` for `x1` must have the same size as the respective
            axis `x2_axes[i]` for `x2`. Each sequence must consist of unique
            non-negative integers that specify valid axes for each respective
            array.
    Returns:
        usm_ndarray:
            an array containing the tensor contraction whose shape consists of
            the non-contracted axes of the first array `x1`, followed by the
            non-contracted axes of the second array `x2`. The returned array
            must have a data type determined by Type Promotion Rules.
    """
    if not isinstance(x1, dpt.usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(x1)}")
    if not isinstance(x2, dpt.usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(x2)}")
    q1, x1_usm_type = x1.sycl_queue, x1.usm_type
    q2, x2_usm_type = x2.sycl_queue, x2.usm_type
    exec_q = dpctl.utils.get_execution_queue((q1, q2))
    if exec_q is None:
        raise ExecutionPlacementError(
            "Execution placement can not be unambiguously inferred "
            "from input arguments."
        )
    res_usm_type = dpctl.utils.get_coerced_usm_type(
        (
            x1_usm_type,
            x2_usm_type,
        )
    )
    dpctl.utils.validate_usm_type(res_usm_type, allow_none=False)
    # handle axes and shapes validation
    x1_nd = x1.ndim
    x2_nd = x2.ndim
    x1_shape = x1.shape
    x2_shape = x2.shape
    if isinstance(axes, int):
        if axes < 0:
            raise ValueError("`axes` integer is expected to be non-negative")
        n_axes1 = axes
        n_axes2 = axes
        axes1 = normalize_axis_tuple(tuple(range(-axes, 0)), x1_nd)
        axes2 = tuple(range(0, axes))
    elif isinstance(axes, tuple):
        if len(axes) != 2:
            raise ValueError(
                "`axes` tuple is expected to contain two sequences"
            )
        axes1 = tuple(axes[0])
        axes2 = tuple(axes[1])
        n_axes1 = len(axes1)
        n_axes2 = len(axes2)
    else:
        raise TypeError("`axes` must be an integer or a tuple of sequences")
    if n_axes1 != n_axes2:
        raise ValueError(
            "number of axes contracted must be the same for each array"
        )
    if n_axes1 == 0:
        arr1 = x1[..., dpt.newaxis]
        arr2 = x2[dpt.newaxis, ...]
        n_axes1 = 1
        n_axes2 = 1
    else:
        same_shapes = True
        for i in range(n_axes1):
            axis1 = axes1[i]
            if axis1 < 0:
                raise ValueError("`axes` must be non-negative")
            axis2 = axes2[i]
            if axis2 < 0:
                raise ValueError("`axes` must be non-negative")
            same_shapes = same_shapes and (x1_shape[axis1] == x2_shape[axis2])
        if not same_shapes:
            raise ValueError("shape mismatch in contracted `tensordot` axes")
        axes1 = normalize_axis_tuple(axes1, x1_nd)
        axes2 = normalize_axis_tuple(axes2, x2_nd)
        perm1 = [i for i in range(x1_nd) if i not in axes1] + list(axes1)
        perm2 = list(axes2) + [i for i in range(x2_nd) if i not in axes2]
        arr1 = dpt.permute_dims(x1, perm1)
        arr2 = dpt.permute_dims(x2, perm2)
    arr1_outer_nd = arr1.ndim - n_axes1
    arr2_outer_nd = arr2.ndim - n_axes2
    res_shape = arr1.shape[:arr1_outer_nd] + arr2.shape[n_axes2:]
    # type validation
    sycl_dev = exec_q.sycl_device
    x1_dtype = x1.dtype
    x2_dtype = x2.dtype
    buf1_dt, buf2_dt, res_dt = _find_buf_dtype2(
        x1_dtype,
        x2_dtype,
        tli._dot_result_type,
        sycl_dev,
        acceptance_fn=_acceptance_fn_default_binary,
    )
    if res_dt is None:
        raise TypeError(
            "function 'tensordot' does not support input types "
            f"({x1_dtype}, {x2_dtype}), "
            "and the inputs could not be safely coerced to any "
            "supported types according to the casting rule ''safe''."
        )

    if buf1_dt is None and buf2_dt is None:
        out = dpt.empty(
            res_shape,
            dtype=res_dt,
            usm_type=res_usm_type,
            sycl_queue=exec_q,
            order="C",
        )
        ht_dot_ev, _ = tli._dot(
            x1=arr1,
            x2=arr2,
            batch_dims=0,
            x1_outer_dims=arr1_outer_nd,
            x2_outer_dims=arr2_outer_nd,
            inner_dims=n_axes1,
            dst=out,
            sycl_queue=exec_q,
        )
        ht_dot_ev.wait()

        return out

    elif buf1_dt is None:
        buf2 = _empty_like_orderK(arr2, buf2_dt)
        ht_copy_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=arr2, dst=buf2, sycl_queue=exec_q
        )
        out = dpt.empty(
            res_shape,
            dtype=res_dt,
            usm_type=res_usm_type,
            sycl_queue=exec_q,
            order="C",
        )
        ht_dot_ev, _ = tli._dot(
            x1=arr1,
            x2=buf2,
            batch_dims=0,
            x1_outer_dims=arr1_outer_nd,
            x2_outer_dims=arr2_outer_nd,
            inner_dims=n_axes1,
            dst=out,
            sycl_queue=exec_q,
            depends=[copy_ev],
        )
        ht_copy_ev.wait()
        ht_dot_ev.wait()

        return out

    elif buf2_dt is None:
        buf1 = _empty_like_orderK(arr1, buf1_dt)
        ht_copy_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=arr1, dst=buf1, sycl_queue=exec_q
        )
        out = dpt.empty(
            res_shape,
            dtype=res_dt,
            usm_type=res_usm_type,
            sycl_queue=exec_q,
            order="C",
        )
        ht_dot_ev, _ = tli._dot(
            x1=buf1,
            x2=arr2,
            batch_dims=0,
            x1_outer_dims=arr1_outer_nd,
            x2_outer_dims=arr2_outer_nd,
            inner_dims=n_axes1,
            dst=out,
            sycl_queue=exec_q,
            depends=[copy_ev],
        )
        ht_copy_ev.wait()
        ht_dot_ev.wait()

        return out

    buf1 = _empty_like_orderK(arr1, buf1_dt)
    ht_copy1_ev, copy1_ev = ti._copy_usm_ndarray_into_usm_ndarray(
        src=arr1, dst=buf1, sycl_queue=exec_q
    )
    buf2 = _empty_like_orderK(arr2, buf2_dt)
    ht_copy2_ev, copy2_ev = ti._copy_usm_ndarray_into_usm_ndarray(
        src=arr2, dst=buf2, sycl_queue=exec_q
    )
    out = dpt.empty(
        res_shape,
        dtype=res_dt,
        usm_type=res_usm_type,
        sycl_queue=exec_q,
        order="C",
    )
    ht_, _ = tli._dot(
        x1=buf1,
        x2=buf2,
        batch_dims=0,
        x1_outer_dims=arr1_outer_nd,
        x2_outer_dims=arr2_outer_nd,
        inner_dims=n_axes1,
        dst=out,
        sycl_queue=exec_q,
        depends=[copy1_ev, copy2_ev],
    )
    dpctl.SyclEvent.wait_for([ht_copy1_ev, ht_copy2_ev, ht_])

    return out


def vecdot(x1, x2, axis=-1):
    """vecdot(x1, x2, axis=-1)

    Computes the (vector) dot product of two arrays.

    Args:
        x1 (usm_ndarray):
            first input array.
        x2 (usm_ndarray):
            second input array. Input arrays must have compatible
            shapes along non-contract axes according to broadcasting
            rules, and must have the same size along the contracted
            axis. Input arrays should be of numeric type.
        axis (Optional[int]):
            axis over which to compute the dot product. The axis must
            be an integer on the interval `[-N, N)`, where `N` is the
            array rank of input arrays after broadcasting rules are
            applied. If specified as a negative integer, the axis along
            which dot product is performed is counted backward from
            the last axes (that is `-1` refers to the last axis). By
            default, dot product is computed over the last axis.
            Default: `-1`.

    Returns:
        usm_ndarray:
            if `x1` and `x2` are both one-dimensional arrays, a
            zero-dimensional array containing the dot product value
            is returned; otherwise, a non-zero-dimensional array containing
            the dot products and having rank `N-1`, where `N` is the rank
            of the shape of input arrays after broadcasting rules are applied
            to non-contracted axes.
    """
    if not isinstance(x1, dpt.usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(x1)}")
    if not isinstance(x2, dpt.usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(x2)}")
    q1, x1_usm_type = x1.sycl_queue, x1.usm_type
    q2, x2_usm_type = x2.sycl_queue, x2.usm_type
    exec_q = dpctl.utils.get_execution_queue((q1, q2))
    if exec_q is None:
        raise ExecutionPlacementError(
            "Execution placement can not be unambiguously inferred "
            "from input arguments."
        )
    res_usm_type = dpctl.utils.get_coerced_usm_type(
        (
            x1_usm_type,
            x2_usm_type,
        )
    )
    dpctl.utils.validate_usm_type(res_usm_type, allow_none=False)
    # axis and shape validation
    x1_nd = x1.ndim
    x2_nd = x2.ndim
    x1_shape = x1.shape
    x2_shape = x2.shape
    if x1_nd > x2_nd:
        x2_shape = (1,) * (x1_nd - x2_nd) + x2_shape
        x2_nd = len(x2_shape)
    elif x2_nd > x1_nd:
        x1_shape = (1,) * (x2_nd - x1_nd) + x1_shape
        x1_nd = len(x1_shape)
    axis = normalize_axis_index(operator.index(axis), x1_nd)
    if x1_shape[axis] != x2_shape[axis]:
        raise ValueError(
            "given axis must have the same shape for `x1` and `x2`"
        )
    try:
        broadcast_sh = _broadcast_shape_impl(
            [
                x1_shape,
                x2_shape,
            ]
        )
    except ValueError:
        raise ValueError("mismatch in `vecdot` dimensions")
    res_sh = tuple(
        [broadcast_sh[i] for i in range(len(broadcast_sh)) if i != axis]
    )
    # type validation
    sycl_dev = exec_q.sycl_device
    x1_dtype = x1.dtype
    x2_dtype = x2.dtype
    buf1_dt, buf2_dt, res_dt = _find_buf_dtype2(
        x1_dtype,
        x2_dtype,
        tli._dot_result_type,
        sycl_dev,
        acceptance_fn=_acceptance_fn_default_binary,
    )
    if res_dt is None:
        raise TypeError(
            "function 'vecdot' does not support input types "
            f"({x1_dtype}, {x2_dtype}), "
            "and the inputs could not be safely coerced to any "
            "supported types according to the casting rule ''safe''."
        )

    ht_list = []
    deps = []
    if buf1_dt is None and buf2_dt is None:
        if x1.dtype.kind == "c":
            x1_tmp = _empty_like_orderK(x1, x1.dtype)
            ht_conj_ev, conj_ev = tei._conj(
                src=x1,
                dst=x1_tmp,
                sycl_queue=exec_q,
            )
            ht_list.append(ht_conj_ev)
            deps.append(conj_ev)
            x1 = x1_tmp
        if x1.shape != broadcast_sh:
            x1 = dpt.broadcast_to(x1, broadcast_sh)
        if x2.shape != broadcast_sh:
            x2 = dpt.broadcast_to(x2, broadcast_sh)
        x1 = dpt.moveaxis(x1, axis, -1)
        x2 = dpt.moveaxis(x2, axis, -1)

        out = dpt.empty(
            res_sh,
            dtype=res_dt,
            usm_type=res_usm_type,
            sycl_queue=exec_q,
            order="C",
        )
        ht_dot_ev, _ = tli._dot(
            x1=x1,
            x2=x2,
            batch_dims=len(x1.shape[:-1]),
            x1_outer_dims=0,
            x2_outer_dims=0,
            inner_dims=1,
            dst=out,
            sycl_queue=exec_q,
            depends=deps,
        )
        ht_list.append(ht_dot_ev)
        dpctl.SyclEvent.wait_for(ht_list)

        return dpt.reshape(out, res_sh)

    elif buf1_dt is None:
        if x1.dtype.kind == "c":
            x1_tmp = _empty_like_orderK(x1, x1.dtype)
            ht_conj_ev, conj_e = tei._conj(
                src=x1, dst=x1_tmp, sycl_queue=exec_q
            )
            ht_list.append(ht_conj_ev)
            deps.append(conj_e)
            x1 = x1_tmp
        buf2 = _empty_like_orderK(x2, buf2_dt)
        ht_copy_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=x2, dst=buf2, sycl_queue=exec_q
        )
        ht_list.append(ht_copy_ev)
        deps.append(copy_ev)
        if x1.shape != broadcast_sh:
            x1 = dpt.broadcast_to(x1, broadcast_sh)
        if buf2.shape != broadcast_sh:
            buf2 = dpt.broadcast_to(buf2, broadcast_sh)
        x1 = dpt.moveaxis(x1, axis, -1)
        buf2 = dpt.moveaxis(buf2, axis, -1)
        out = dpt.empty(
            res_sh,
            dtype=res_dt,
            usm_type=res_usm_type,
            sycl_queue=exec_q,
            order="C",
        )
        ht_dot_ev, _ = tli._dot(
            x1=x1,
            x2=buf2,
            batch_dims=len(x1.shape[:-1]),
            x1_outer_dims=0,
            x2_outer_dims=0,
            inner_dims=1,
            dst=out,
            sycl_queue=exec_q,
            depends=deps,
        )
        ht_list.append(ht_dot_ev)
        dpctl.SyclEvent.wait_for(ht_list)

        return dpt.reshape(out, res_sh)

    elif buf2_dt is None:
        buf1 = _empty_like_orderK(x1, buf1_dt)
        ht_copy_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=x1, dst=buf1, sycl_queue=exec_q
        )
        ht_list.append(ht_copy_ev)
        deps.append(copy_ev)
        if buf1.dtype.kind == "c":
            ht_conj_ev, conj_ev = tei._conj(
                src=buf1, dst=buf1, sycl_queue=exec_q, depends=[copy_ev]
            )
            ht_list.append(ht_conj_ev)
            deps.append(conj_ev)
        if buf1.shape != broadcast_sh:
            buf1 = dpt.broadcast_to(buf1, broadcast_sh)
        if x2.shape != broadcast_sh:
            x2 = dpt.broadcast_to(x2, broadcast_sh)
        buf1 = dpt.moveaxis(buf1, axis, -1)
        x2 = dpt.moveaxis(x2, axis, -1)
        out = dpt.empty(
            res_sh,
            dtype=res_dt,
            usm_type=res_usm_type,
            sycl_queue=exec_q,
            order="C",
        )
        ht_dot_ev, _ = tli._dot(
            x1=buf1,
            x2=x2,
            batch_dims=len(x1.shape[:-1]),
            x1_outer_dims=0,
            x2_outer_dims=0,
            inner_dims=1,
            dst=out,
            sycl_queue=exec_q,
            depends=deps,
        )
        ht_list.append(ht_dot_ev)
        dpctl.SyclEvent.wait_for(ht_list)

        return dpt.reshape(out, res_sh)

    buf1 = _empty_like_orderK(x1, buf1_dt)
    ht_copy1_ev, copy1_ev = ti._copy_usm_ndarray_into_usm_ndarray(
        src=x1, dst=buf1, sycl_queue=exec_q
    )
    ht_list.append(ht_copy1_ev)
    deps.append(copy1_ev)
    if buf1.dtype.kind == "c":
        ht_conj_ev, conj_ev = tei._conj(
            src=buf1, dst=buf1, sycl_queue=exec_q, depends=[copy1_ev]
        )
        ht_list.append(ht_conj_ev)
        deps.append(conj_ev)
    buf2 = _empty_like_orderK(x2, buf2_dt)
    ht_copy2_ev, copy2_ev = ti._copy_usm_ndarray_into_usm_ndarray(
        src=x2, dst=buf2, sycl_queue=exec_q
    )
    ht_list.append(ht_copy2_ev)
    deps.append(copy2_ev)
    if buf1.shape != broadcast_sh:
        buf1 = dpt.broadcast_to(buf1, broadcast_sh)
    if buf2.shape != broadcast_sh:
        buf2 = dpt.broadcast_to(buf2, broadcast_sh)
    buf1 = dpt.moveaxis(buf1, axis, -1)
    buf2 = dpt.moveaxis(buf2, axis, -1)
    out = dpt.empty(
        res_sh,
        dtype=res_dt,
        usm_type=res_usm_type,
        sycl_queue=exec_q,
        order="C",
    )
    ht_dot_ev, _ = tli._dot(
        x1=buf1,
        x2=buf2,
        batch_dims=len(x1.shape[:-1]),
        x1_outer_dims=0,
        x2_outer_dims=0,
        inner_dims=1,
        dst=out,
        sycl_queue=exec_q,
        depends=deps,
    )
    ht_list.append(ht_dot_ev)
    dpctl.SyclEvent.wait_for(ht_list)

    return out


def matmul(x1, x2, out=None, dtype=None, order="K"):
    """matmul(x1, x2, out=None, order="K")

    Computes the matrix product. Implements the same semantics
    as the built-in operator `@`.

    Args:
        x1 (usm_ndarray):
            first input array. Expected to have numeric data type, and
            at least one dimension. If `x1` is one-dimensional having
            shape `(M,)`, and `x2` has more than one dimension, `x1` is
            effectively treated as a two-dimensional array with shape `(1, M)`,
            although the prepended dimension is removed from the output array.
            If `x1` has shape `(..., M, K)`, the innermost two dimensions form
            matrices on which to perform matrix multiplication.
        x2 (usm_ndarray):
            second input array. Expected to have numeric data type, and
            at least one dimension. If `x2` is one-dimensional having
            shape `(N,)`, and `x1` has more than one dimension, `x2` is
            effectively treated as a two-dimensional array with shape `(N, 1)`,
            although the appended dimension is removed from the output array.
            If `x2` has shape `(..., K, N)`, the innermost two dimensions form
            matrices on which to perform matrix multiplication.
        out (Optional[usm_ndarray]):
            the array into which the result of the matrix product is written.
            If `None` then a new array is returned.
        order (["K", "C", "F", "A"]):
            memory layout of the output array, if `out` is `None`, otherwise
            the `order` parameter value is not used.

    Returns:
        usm_ndarray:
            * if both `x1` and `x2` are one-dimensional arrays with shape
              `(N,)`, returned array is a zero-dimensional array containing
              inner product as its only element.
            * if `x1` is two-dimensional array with shape `(M, K)` and `x2` is
              a two-dimensional array with shape `(K, N)`, returned array is a
              two-dimensional array with shape `(M, N)` and contains the
              conventional matrix product.
            * if `x1` is a one-dimensinal array with shape `(K,)` and `x2` is an
              array with shape `(..., K, N)`, returned array contains the
              conventional matrix product and has shape `(..., N)`.
            * if `x1` is an array with shape `(..., M, K)` and `x2` is a
              one-dimensional array with shape `(K,)`, returned array has shape
              `(..., M)` and contains the conventional matrix product.
            * if `x1` is a two-dimensional array with shape `(M, K)` and `x2`
              is an array with shape `(..., K, N)`, returned array contains
              conventional matrix product for each stacked matrix and has shape
              `(..., M, N)`.
            * if `x1` has shape `(..., M, K)` and `x2` is a two-dimensional
              array with shape `(K, N)`, returned array contains conventional
              matrix product for each stacked matrix and has shape
              `(..., M, N)`.
            * if both `x1` and `x2` have more than two dimensions, returned
              array contains conventional matrix product for each stacked
              matrix and has shape determined by broadcasting rules for
              `x1.shape[:-2]` and `x2.shape[:-2]`.

            The data type of the returned array is determined by the Type
            Promotion Rules. If either `x1` or `x2` has a complex floating
            point type, neither argument is complex conjugated or transposed.
    """
    if not isinstance(x1, dpt.usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(x1)}")
    if not isinstance(x2, dpt.usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(x2)}")
    if order not in ["K", "C", "F", "A"]:
        order = "K"
    q1, x1_usm_type = x1.sycl_queue, x1.usm_type
    q2, x2_usm_type = x2.sycl_queue, x2.usm_type
    exec_q = dpctl.utils.get_execution_queue((q1, q2))
    if exec_q is None:
        raise ExecutionPlacementError(
            "Execution placement can not be unambiguously inferred "
            "from input arguments."
        )
    res_usm_type = dpctl.utils.get_coerced_usm_type(
        (
            x1_usm_type,
            x2_usm_type,
        )
    )
    dpctl.utils.validate_usm_type(res_usm_type, allow_none=False)

    x1_nd = x1.ndim
    x2_nd = x2.ndim
    if x1_nd == 0 or x2_nd == 0:
        raise ValueError("one or more operands to `matmul` is 0 dimensional")
    x1_shape = x1.shape
    x2_shape = x2.shape
    appended_axes = []
    if x1_nd == 1:
        x1 = x1[dpt.newaxis, :]
        x1_shape = x1.shape
        appended_axes.append(-2)
    if x2_nd == 1:
        x2 = x2[:, dpt.newaxis]
        x2_shape = x2.shape
        appended_axes.append(-1)
    if x1_shape[-1] != x2_shape[-2]:
        raise ValueError("mismatch in `matmul` inner dimension")
    x1_outer_sh = x1_shape[:-2]
    x2_outer_sh = x2_shape[:-2]
    try:
        res_outer_sh = _broadcast_shape_impl(
            [
                x1_outer_sh,
                x2_outer_sh,
            ]
        )
    except ValueError:
        raise ValueError("mismatch in `matmul` batching dimensions")
    x1_broadcast_shape = res_outer_sh + x1_shape[-2:]
    x2_broadcast_shape = res_outer_sh + x2_shape[-2:]
    res_shape = res_outer_sh + x1_shape[-2:-1] + x2_shape[-1:]

    sycl_dev = exec_q.sycl_device
    x1_dtype = x1.dtype
    x2_dtype = x2.dtype
    if dtype is None:
        buf1_dt, buf2_dt, res_dt = _find_buf_dtype2(
            x1_dtype,
            x2_dtype,
            tli._dot_result_type,
            sycl_dev,
            acceptance_fn=_acceptance_fn_default_binary,
        )
        if res_dt is None:
            raise ValueError(
                "function 'matmul' does not support input types "
                f"({x1_dtype}, {x2_dtype}), "
                "and the inputs could not be safely coerced to any "
                "supported types according to the casting rule ''safe''."
            )
    else:
        res_dt = dpt.dtype(dtype)
        res_dt = _to_device_supported_dtype(res_dt, sycl_dev)
        buf1_dt, buf2_dt = None, None
        if x1_dtype != res_dt:
            if dpt.can_cast(x1_dtype, res_dt, casting="same_kind"):
                buf1_dt = res_dt
            else:
                raise ValueError(
                    f"`matmul` input `x1` cannot be cast from {x1_dtype} to "
                    f"requested type {res_dt} according to the casting rule "
                    "''same_kind''."
                )
        if x2_dtype != res_dt:
            if dpt.can_cast(x2_dtype, res_dt, casting="same_kind"):
                buf2_dt = res_dt
            else:
                raise ValueError(
                    f"`matmul` input `x2` cannot be cast from {x2_dtype} to "
                    f"requested type {res_dt} according to the casting rule "
                    "''same_kind''."
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
                f"Output array of type {res_dt} is needed," f"got {out.dtype}"
            )

        if dpctl.utils.get_execution_queue((exec_q, out.sycl_queue)) is None:
            raise ExecutionPlacementError(
                "Input and output allocation queues are not compatible"
            )

        if ti._array_overlap(x1, out) and buf1_dt is None:
            out = dpt.empty_like(out)

        if ti._array_overlap(x2, out) and buf2_dt is None:
            # should not reach if out is reallocated
            # after being checked against x1
            out = dpt.empty_like(out)

    if order == "A":
        order = (
            "F"
            if all(
                arr.flags.f_contiguous
                for arr in (
                    x1,
                    x2,
                )
            )
            else "C"
        )

    if buf1_dt is None and buf2_dt is None:
        if out is None:
            if order == "K":
                out = _empty_like_pair_orderK(
                    x1, x2, res_dt, res_shape, res_usm_type, exec_q
                )
            else:
                out = dpt.empty(
                    res_shape,
                    dtype=res_dt,
                    usm_type=res_usm_type,
                    sycl_queue=exec_q,
                    order=order,
                )
        if x1.shape != x1_broadcast_shape:
            x1 = dpt.broadcast_to(x1, x1_broadcast_shape)
        if x2.shape != x2_broadcast_shape:
            x2 = dpt.broadcast_to(x2, x2_broadcast_shape)
        ht_dot_ev, dot_ev = tli._dot(
            x1=x1,
            x2=x2,
            batch_dims=len(res_shape[:-2]),
            x1_outer_dims=1,
            x2_outer_dims=1,
            inner_dims=1,
            dst=out,
            sycl_queue=exec_q,
        )
        if not (orig_out is None or orig_out is out):
            # Copy the out data from temporary buffer to original memory
            ht_copy_out_ev, _ = ti._copy_usm_ndarray_into_usm_ndarray(
                src=out,
                dst=orig_out,
                sycl_queue=exec_q,
                depends=[dot_ev],
            )
            ht_copy_out_ev.wait()
            out = orig_out
        ht_dot_ev.wait()
        if appended_axes:
            out = dpt.squeeze(out, tuple(appended_axes))
        return out
    elif buf1_dt is None:
        if order == "K":
            buf2 = _empty_like_orderK(x2, buf2_dt)
        else:
            buf2 = dpt.empty_like(x2, dtype=buf2_dt, order=order)
        ht_copy_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=x2, dst=buf2, sycl_queue=exec_q
        )
        if out is None:
            if order == "K":
                out = _empty_like_pair_orderK(
                    x1, buf2, res_dt, res_shape, res_usm_type, exec_q
                )
            else:
                out = dpt.empty(
                    res_shape,
                    dtype=res_dt,
                    usm_type=res_usm_type,
                    sycl_queue=exec_q,
                    order=order,
                )

        if x1.shape != x1_broadcast_shape:
            x1 = dpt.broadcast_to(x1, x1_broadcast_shape)
        if buf2.shape != x2_broadcast_shape:
            buf2 = dpt.broadcast_to(buf2, x2_broadcast_shape)
        ht_dot_ev, dot_ev = tli._dot(
            x1=x1,
            x2=buf2,
            batch_dims=len(res_shape[:-2]),
            x1_outer_dims=1,
            x2_outer_dims=1,
            inner_dims=1,
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
                depends=[dot_ev],
            )
            ht_copy_out_ev.wait()
            out = orig_out
        ht_copy_ev.wait()
        ht_dot_ev.wait()
        if appended_axes:
            out = dpt.squeeze(out, tuple(appended_axes))
        return out

    elif buf2_dt is None:
        if order == "K":
            buf1 = _empty_like_orderK(x1, buf1_dt)
        else:
            buf1 = dpt.empty_like(x1, dtype=buf1_dt, order=order)
        ht_copy_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=x1, dst=buf1, sycl_queue=exec_q
        )
        if out is None:
            if order == "K":
                out = _empty_like_pair_orderK(
                    buf1, x2, res_dt, res_shape, res_usm_type, exec_q
                )
            else:
                out = dpt.empty(
                    res_shape,
                    dtype=res_dt,
                    usm_type=res_usm_type,
                    sycl_queue=exec_q,
                    order=order,
                )

        if buf1.shape != x1_broadcast_shape:
            buf1 = dpt.broadcast_to(buf1, x1_broadcast_shape)
        if x2.shape != x2_broadcast_shape:
            x2 = dpt.broadcast_to(x2, x2_broadcast_shape)
        ht_dot_ev, dot_ev = tli._dot(
            x1=buf1,
            x2=x2,
            batch_dims=len(res_shape[:-2]),
            x1_outer_dims=1,
            x2_outer_dims=1,
            inner_dims=1,
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
                depends=[dot_ev],
            )
            ht_copy_out_ev.wait()
            out = orig_out
        ht_copy_ev.wait()
        ht_dot_ev.wait()
        if appended_axes:
            out = dpt.squeeze(out, tuple(appended_axes))
        return out

    if order == "K":
        if x1.flags.f_contiguous and x2.flags.f_contiguous:
            order = "F"
        elif x1.flags.c_contiguous and x2.flags.c_contiguous:
            order = "C"
    if order == "K":
        buf1 = _empty_like_orderK(x1, buf1_dt)
    else:
        buf1 = dpt.empty_like(x1, dtype=buf1_dt, order=order)
    ht_copy1_ev, copy1_ev = ti._copy_usm_ndarray_into_usm_ndarray(
        src=x1, dst=buf1, sycl_queue=exec_q
    )
    if order == "K":
        buf2 = _empty_like_orderK(x2, buf2_dt)
    else:
        buf2 = dpt.empty_like(x2, dtype=buf2_dt, order=order)
    ht_copy2_ev, copy2_ev = ti._copy_usm_ndarray_into_usm_ndarray(
        src=x2, dst=buf2, sycl_queue=exec_q
    )
    if out is None:
        if order == "K":
            out = _empty_like_pair_orderK(
                buf1, buf2, res_dt, res_shape, res_usm_type, exec_q
            )
        else:
            out = dpt.empty(
                res_shape,
                dtype=res_dt,
                usm_type=res_usm_type,
                sycl_queue=exec_q,
                order=order,
            )

    if buf1.shape != x1_broadcast_shape:
        buf1 = dpt.broadcast_to(buf1, x1_broadcast_shape)
    if buf2.shape != x2_broadcast_shape:
        buf2 = dpt.broadcast_to(buf2, x2_broadcast_shape)
    ht_, _ = tli._dot(
        x1=buf1,
        x2=buf2,
        batch_dims=len(res_shape[:-2]),
        x1_outer_dims=1,
        x2_outer_dims=1,
        inner_dims=1,
        dst=out,
        sycl_queue=exec_q,
        depends=[copy1_ev, copy2_ev],
    )
    dpctl.SyclEvent.wait_for([ht_copy1_ev, ht_copy2_ev, ht_])
    if appended_axes:
        out = dpt.squeeze(out, tuple(appended_axes))
    return out
