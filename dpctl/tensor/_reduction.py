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

from numpy.core.numeric import normalize_axis_tuple

import dpctl
import dpctl.tensor as dpt
import dpctl.tensor._tensor_impl as ti
import dpctl.tensor._tensor_reductions_impl as tri
from dpctl.utils import ExecutionPlacementError

from ._type_utils import (
    _default_accumulation_dtype,
    _default_accumulation_dtype_fp_types,
    _to_device_supported_dtype,
)


def _reduction_over_axis(
    x,
    axis,
    dtype,
    keepdims,
    out,
    _reduction_fn,
    _dtype_supported,
    _default_reduction_type_fn,
):
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(x)}")
    nd = x.ndim
    if axis is None:
        axis = tuple(range(nd))
        perm = list(axis)
        arr = x
    else:
        if not isinstance(axis, (tuple, list)):
            axis = (axis,)
        axis = normalize_axis_tuple(axis, nd, "axis")
        perm = [i for i in range(nd) if i not in axis] + list(axis)
        arr = dpt.permute_dims(x, perm)
    red_nd = len(axis)
    res_shape = arr.shape[: nd - red_nd]
    q = x.sycl_queue
    inp_dt = x.dtype
    if dtype is None:
        res_dt = _default_reduction_type_fn(inp_dt, q)
    else:
        res_dt = dpt.dtype(dtype)
        res_dt = _to_device_supported_dtype(res_dt, q.sycl_device)

    res_usm_type = x.usm_type

    implemented_types = _dtype_supported(inp_dt, res_dt, res_usm_type, q)
    if dtype is None and not implemented_types:
        raise RuntimeError(
            "Automatically determined reduction data type does not "
            "have direct implementation"
        )
    orig_out = out
    if out is not None:
        if not isinstance(out, dpt.usm_ndarray):
            raise TypeError(
                f"output array must be of usm_ndarray type, got {type(out)}"
            )
        if not out.flags.writable:
            raise ValueError("provided `out` array is read-only")
        if not keepdims:
            final_res_shape = res_shape
        else:
            inp_shape = x.shape
            final_res_shape = tuple(
                inp_shape[i] if i not in axis else 1 for i in range(nd)
            )
        if not out.shape == final_res_shape:
            raise ValueError(
                "The shape of input and output arrays are inconsistent. "
                f"Expected output shape is {final_res_shape}, got {out.shape}"
            )
        if res_dt != out.dtype:
            raise ValueError(
                f"Output array of type {res_dt} is needed, got {out.dtype}"
            )
        if dpctl.utils.get_execution_queue((q, out.sycl_queue)) is None:
            raise ExecutionPlacementError(
                "Input and output allocation queues are not compatible"
            )
        if keepdims:
            out = dpt.squeeze(out, axis=axis)
            orig_out = out
        if ti._array_overlap(x, out) and implemented_types:
            out = dpt.empty_like(out)
    else:
        out = dpt.empty(
            res_shape, dtype=res_dt, usm_type=res_usm_type, sycl_queue=q
        )

    host_tasks_list = []
    if red_nd == 0:
        ht_e_cpy, cpy_e = ti._copy_usm_ndarray_into_usm_ndarray(
            src=arr, dst=out, sycl_queue=q
        )
        host_tasks_list.append(ht_e_cpy)
        if not (orig_out is None or orig_out is out):
            ht_e_cpy2, _ = ti._copy_usm_ndarray_into_usm_ndarray(
                src=out, dst=orig_out, sycl_queue=q, depends=[cpy_e]
            )
            host_tasks_list.append(ht_e_cpy2)
            out = orig_out
        dpctl.SyclEvent.wait_for(host_tasks_list)
        return out

    if implemented_types:
        ht_e, red_e = _reduction_fn(
            src=arr, trailing_dims_to_reduce=red_nd, dst=out, sycl_queue=q
        )
        host_tasks_list.append(ht_e)
        if not (orig_out is None or orig_out is out):
            ht_e_cpy, _ = ti._copy_usm_ndarray_into_usm_ndarray(
                src=out, dst=orig_out, sycl_queue=q, depends=[red_e]
            )
            host_tasks_list.append(ht_e_cpy)
            out = orig_out
    else:
        if _dtype_supported(res_dt, res_dt, res_usm_type, q):
            tmp = dpt.empty(
                arr.shape, dtype=res_dt, usm_type=res_usm_type, sycl_queue=q
            )
            ht_e_cpy, cpy_e = ti._copy_usm_ndarray_into_usm_ndarray(
                src=arr, dst=tmp, sycl_queue=q
            )
            host_tasks_list.append(ht_e_cpy)
            ht_e_red, _ = _reduction_fn(
                src=tmp,
                trailing_dims_to_reduce=red_nd,
                dst=out,
                sycl_queue=q,
                depends=[cpy_e],
            )
            host_tasks_list.append(ht_e_red)
        else:
            buf_dt = _default_reduction_type_fn(inp_dt, q)
            tmp = dpt.empty(
                arr.shape, dtype=buf_dt, usm_type=res_usm_type, sycl_queue=q
            )
            ht_e_cpy, cpy_e = ti._copy_usm_ndarray_into_usm_ndarray(
                src=arr, dst=tmp, sycl_queue=q
            )
            tmp_res = dpt.empty(
                res_shape, dtype=buf_dt, usm_type=res_usm_type, sycl_queue=q
            )
            host_tasks_list.append(ht_e_cpy)
            ht_e_red, r_e = _reduction_fn(
                src=tmp,
                trailing_dims_to_reduce=red_nd,
                dst=tmp_res,
                sycl_queue=q,
                depends=[cpy_e],
            )
            host_tasks_list.append(ht_e_red)
            ht_e_cpy2, _ = ti._copy_usm_ndarray_into_usm_ndarray(
                src=tmp_res, dst=out, sycl_queue=q, depends=[r_e]
            )
            host_tasks_list.append(ht_e_cpy2)

    if keepdims:
        res_shape = res_shape + (1,) * red_nd
        inv_perm = sorted(range(nd), key=lambda d: perm[d])
        out = dpt.permute_dims(dpt.reshape(out, res_shape), inv_perm)
    dpctl.SyclEvent.wait_for(host_tasks_list)

    return out


def sum(x, /, *, axis=None, dtype=None, keepdims=False, out=None):
    """
    Calculates the sum of elements in the input array ``x``.

    Args:
        x (usm_ndarray):
            input array.
        axis (Optional[int, Tuple[int, ...]]):
            axis or axes along which sums must be computed. If a tuple
            of unique integers, sums are computed over multiple axes.
            If ``None``, the sum is computed over the entire array.
            Default: ``None``.
        dtype (Optional[dtype]):
            data type of the returned array. If ``None``, the default data
            type is inferred from the "kind" of the input array data type.

            * If ``x`` has a real- or complex-valued floating-point data
              type, the returned array will have the same data type as
              ``x``.
            * If ``x`` has signed integral data type, the returned array
              will have the default signed integral type for the device
              where input array ``x`` is allocated.
            * If ``x`` has unsigned integral data type, the returned array
              will have the default unsigned integral type for the device
              where input array ``x`` is allocated.
              array ``x`` is allocated.
            * If ``x`` has a boolean data type, the returned array will
              have the default signed integral type for the device
              where input array ``x`` is allocated.

            If the data type (either specified or resolved) differs from the
            data type of ``x``, the input array elements are cast to the
            specified data type before computing the sum.
            Default: ``None``.
        keepdims (Optional[bool]):
            if ``True``, the reduced axes (dimensions) are included in the
            result as singleton dimensions, so that the returned array remains
            compatible with the input arrays according to Array Broadcasting
            rules. Otherwise, if ``False``, the reduced axes are not included
            in the returned array. Default: ``False``.
        out (Optional[usm_ndarray]):
            the array into which the result is written.
            The data type of ``out`` must match the expected shape and the
            expected data type of the result or (if provided) ``dtype``.
            If ``None`` then a new array is returned. Default: ``None``.

    Returns:
        usm_ndarray:
            an array containing the sums. If the sum was computed over the
            entire array, a zero-dimensional array is returned. The returned
            array has the data type as described in the ``dtype`` parameter
            description above.
    """
    return _reduction_over_axis(
        x,
        axis,
        dtype,
        keepdims,
        out,
        tri._sum_over_axis,
        tri._sum_over_axis_dtype_supported,
        _default_accumulation_dtype,
    )


def prod(x, /, *, axis=None, dtype=None, keepdims=False, out=None):
    """
    Calculates the product of elements in the input array ``x``.

    Args:
        x (usm_ndarray):
            input array.
        axis (Optional[int, Tuple[int, ...]]):
            axis or axes along which products must be computed. If a tuple
            of unique integers, products are computed over multiple axes.
            If ``None``, the product is computed over the entire array.
            Default: ``None``.
        dtype (Optional[dtype]):
            data type of the returned array. If ``None``, the default data
            type is inferred from the "kind" of the input array data type.

            * If ``x`` has a real- or complex-valued floating-point data
              type, the returned array will have the same data type as
              ``x``.
            * If ``x`` has signed integral data type, the returned array
              will have the default signed integral type for the device
              where input array ``x`` is allocated.
            * If ``x`` has unsigned integral data type, the returned array
              will have the default unsigned integral type for the device
              where input array ``x`` is allocated.
            * If ``x`` has a boolean data type, the returned array will
              have the default signed integral type for the device
              where input array ``x`` is allocated.

            If the data type (either specified or resolved) differs from the
            data type of ``x``, the input array elements are cast to the
            specified data type before computing the product.
            Default: ``None``.
        keepdims (Optional[bool]):
            if ``True``, the reduced axes (dimensions) are included in the
            result as singleton dimensions, so that the returned array remains
            compatible with the input arrays according to Array Broadcasting
            rules. Otherwise, if ``False``, the reduced axes are not included
            in the returned array. Default: ``False``.
        out (Optional[usm_ndarray]):
            the array into which the result is written.
            The data type of ``out`` must match the expected shape and the
            expected data type of the result or (if provided) ``dtype``.
            If ``None`` then a new array is returned. Default: ``None``.

    Returns:
        usm_ndarray:
            an array containing the products. If the product was computed over
            the entire array, a zero-dimensional array is returned. The
            returned array has the data type as described in the ``dtype``
            parameter description above.
    """
    return _reduction_over_axis(
        x,
        axis,
        dtype,
        keepdims,
        out,
        tri._prod_over_axis,
        tri._prod_over_axis_dtype_supported,
        _default_accumulation_dtype,
    )


def logsumexp(x, /, *, axis=None, dtype=None, keepdims=False, out=None):
    """
    Calculates the logarithm of the sum of exponentials of elements in the
    input array ``x``.

    Args:
        x (usm_ndarray):
            input array.
        axis (Optional[int, Tuple[int, ...]]):
            axis or axes along which values must be computed. If a tuple
            of unique integers, values are computed over multiple axes.
            If ``None``, the result is computed over the entire array.
            Default: ``None``.
        dtype (Optional[dtype]):
            data type of the returned array. If ``None``, the default data
            type is inferred from the "kind" of the input array data type.

            * If ``x`` has a real-valued floating-point data type, the
              returned array will have the same data type as ``x``.
            * If ``x`` has a boolean or integral data type, the returned array
              will have the default floating point data type for the device
              where input array ``x`` is allocated.
            * If ``x`` has a complex-valued floating-point data type,
              an error is raised.

            If the data type (either specified or resolved) differs from the
            data type of ``x``, the input array elements are cast to the
            specified data type before computing the result.
            Default: ``None``.
        keepdims (Optional[bool]):
            if ``True``, the reduced axes (dimensions) are included in the
            result as singleton dimensions, so that the returned array remains
            compatible with the input arrays according to Array Broadcasting
            rules. Otherwise, if ``False``, the reduced axes are not included
            in the returned array. Default: ``False``.
        out (Optional[usm_ndarray]):
            the array into which the result is written.
            The data type of ``out`` must match the expected shape and the
            expected data type of the result or (if provided) ``dtype``.
            If ``None`` then a new array is returned. Default: ``None``.

    Returns:
        usm_ndarray:
            an array containing the results. If the result was computed over
            the entire array, a zero-dimensional array is returned.
            The returned array has the data type as described in the
            ``dtype`` parameter description above.
    """
    return _reduction_over_axis(
        x,
        axis,
        dtype,
        keepdims,
        out,
        tri._logsumexp_over_axis,
        lambda inp_dt, res_dt, *_: tri._logsumexp_over_axis_dtype_supported(
            inp_dt, res_dt
        ),
        _default_accumulation_dtype_fp_types,
    )


def reduce_hypot(x, /, *, axis=None, dtype=None, keepdims=False, out=None):
    """
    Calculates the square root of the sum of squares of elements in the input
    array ``x``.

    Args:
        x (usm_ndarray):
            input array.
        axis (Optional[int, Tuple[int, ...]]):
            axis or axes along which values must be computed. If a tuple
            of unique integers, values are computed over multiple axes.
            If ``None``, the result is computed over the entire array.
            Default: ``None``.
        dtype (Optional[dtype]):
            data type of the returned array. If ``None``, the default data
            type is inferred from the "kind" of the input array data type.

            * If ``x`` has a real-valued floating-point data type, the
              returned array will have the same data type as ``x``.
            * If ``x`` has a boolean or integral data type, the returned array
              will have the default floating point data type for the device
              where input array ``x`` is allocated.
            * If ``x`` has a complex-valued floating-point data type,
              an error is raised.

            If the data type (either specified or resolved) differs from the
            data type of ``x``, the input array elements are cast to the
            specified data type before computing the result. Default: ``None``.
        keepdims (Optional[bool]):
            if ``True``, the reduced axes (dimensions) are included in the
            result as singleton dimensions, so that the returned array remains
            compatible with the input arrays according to Array Broadcasting
            rules. Otherwise, if ``False``, the reduced axes are not included
            in the returned array. Default: ``False``.
        out (Optional[usm_ndarray]):
            the array into which the result is written.
            The data type of ``out`` must match the expected shape and the
            expected data type of the result or (if provided) ``dtype``.
            If ``None`` then a new array is returned. Default: ``None``.

    Returns:
        usm_ndarray:
            an array containing the results. If the result was computed over
            the entire array, a zero-dimensional array is returned. The
            returned array has the data type as described in the ``dtype``
            parameter description above.
    """
    return _reduction_over_axis(
        x,
        axis,
        dtype,
        keepdims,
        out,
        tri._hypot_over_axis,
        lambda inp_dt, res_dt, *_: tri._hypot_over_axis_dtype_supported(
            inp_dt, res_dt
        ),
        _default_accumulation_dtype_fp_types,
    )


def _comparison_over_axis(x, axis, keepdims, out, _reduction_fn):
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(x)}")

    nd = x.ndim
    if axis is None:
        axis = tuple(range(nd))
        perm = list(axis)
        x_tmp = x
    else:
        if not isinstance(axis, (tuple, list)):
            axis = (axis,)
        axis = normalize_axis_tuple(axis, nd, "axis")
        perm = [i for i in range(nd) if i not in axis] + list(axis)
        x_tmp = dpt.permute_dims(x, perm)
    red_nd = len(axis)
    if any([x_tmp.shape[i] == 0 for i in range(-red_nd, 0)]):
        raise ValueError("reduction cannot be performed over zero-size axes")
    res_shape = x_tmp.shape[: nd - red_nd]
    exec_q = x.sycl_queue
    res_dt = x.dtype
    res_usm_type = x.usm_type

    orig_out = out
    if out is not None:
        if not isinstance(out, dpt.usm_ndarray):
            raise TypeError(
                f"output array must be of usm_ndarray type, got {type(out)}"
            )
        if not out.flags.writable:
            raise ValueError("provided `out` array is read-only")
        if not keepdims:
            final_res_shape = res_shape
        else:
            inp_shape = x.shape
            final_res_shape = tuple(
                inp_shape[i] if i not in axis else 1 for i in range(nd)
            )
        if not out.shape == final_res_shape:
            raise ValueError(
                "The shape of input and output arrays are inconsistent. "
                f"Expected output shape is {final_res_shape}, got {out.shape}"
            )
        if res_dt != out.dtype:
            raise ValueError(
                f"Output array of type {res_dt} is needed, got {out.dtype}"
            )
        if dpctl.utils.get_execution_queue((exec_q, out.sycl_queue)) is None:
            raise ExecutionPlacementError(
                "Input and output allocation queues are not compatible"
            )
        if keepdims:
            out = dpt.squeeze(out, axis=axis)
            orig_out = out
        if ti._array_overlap(x, out):
            out = dpt.empty_like(out)
    else:
        out = dpt.empty(
            res_shape, dtype=res_dt, usm_type=res_usm_type, sycl_queue=exec_q
        )

    host_tasks_list = []
    if red_nd == 0:
        ht_e_cpy, cpy_e = ti._copy_usm_ndarray_into_usm_ndarray(
            src=x_tmp, dst=out, sycl_queue=exec_q
        )
        host_tasks_list.append(ht_e_cpy)
        if not (orig_out is None or orig_out is out):
            ht_e_cpy2, _ = ti._copy_usm_ndarray_into_usm_ndarray(
                src=out, dst=orig_out, sycl_queue=exec_q, depends=[cpy_e]
            )
            host_tasks_list.append(ht_e_cpy2)
            out = orig_out
        dpctl.SyclEvent.wait_for(host_tasks_list)
        return out

    hev, red_ev = _reduction_fn(
        src=x_tmp,
        trailing_dims_to_reduce=red_nd,
        dst=out,
        sycl_queue=exec_q,
    )
    host_tasks_list.append(hev)
    if not (orig_out is None or orig_out is out):
        ht_e_cpy2, _ = ti._copy_usm_ndarray_into_usm_ndarray(
            src=out, dst=orig_out, sycl_queue=exec_q, depends=[red_ev]
        )
        host_tasks_list.append(ht_e_cpy2)
        out = orig_out

    if keepdims:
        res_shape = res_shape + (1,) * red_nd
        inv_perm = sorted(range(nd), key=lambda d: perm[d])
        out = dpt.permute_dims(dpt.reshape(out, res_shape), inv_perm)
    dpctl.SyclEvent.wait_for(host_tasks_list)
    return out


def max(x, /, *, axis=None, keepdims=False, out=None):
    """
    Calculates the maximum value of the input array ``x``.

    Args:
        x (usm_ndarray):
            input array.
        axis (Optional[int, Tuple[int, ...]]):
            axis or axes along which maxima must be computed. If a tuple
            of unique integers, the maxima are computed over multiple axes.
            If ``None``, the max is computed over the entire array.
            Default: ``None``.
        keepdims (Optional[bool]):
            if ``True``, the reduced axes (dimensions) are included in the
            result as singleton dimensions, so that the returned array remains
            compatible with the input arrays according to Array Broadcasting
            rules. Otherwise, if ``False``, the reduced axes are not included
            in the returned array. Default: ``False``.
        out (Optional[usm_ndarray]):
            the array into which the result is written.
            The data type of ``out`` must match the expected shape and the
            expected data type of the result.
            If ``None`` then a new array is returned. Default: ``None``.

    Returns:
        usm_ndarray:
            an array containing the maxima. If the max was computed over the
            entire array, a zero-dimensional array is returned. The returned
            array has the same data type as ``x``.
    """
    return _comparison_over_axis(x, axis, keepdims, out, tri._max_over_axis)


def min(x, /, *, axis=None, keepdims=False, out=None):
    """
    Calculates the minimum value of the input array ``x``.

    Args:
        x (usm_ndarray):
            input array.
        axis (Optional[int, Tuple[int, ...]]):
            axis or axes along which minima must be computed. If a tuple
            of unique integers, the minima are computed over multiple axes.
            If ``None``, the min is computed over the entire array.
            Default: ``None``.
        keepdims (Optional[bool]):
            if ``True``, the reduced axes (dimensions) are included in the
            result as singleton dimensions, so that the returned array remains
            compatible with the input arrays according to Array Broadcasting
            rules. Otherwise, if ``False``, the reduced axes are not included
            in the returned array. Default: ``False``.
        out (Optional[usm_ndarray]):
            the array into which the result is written.
            The data type of ``out`` must match the expected shape and the
            expected data type of the result.
            If ``None`` then a new array is returned. Default: ``None``.

    Returns:
        usm_ndarray:
            an array containing the minima. If the min was computed over the
            entire array, a zero-dimensional array is returned. The returned
            array has the same data type as ``x``.
    """
    return _comparison_over_axis(x, axis, keepdims, out, tri._min_over_axis)


def _search_over_axis(x, axis, keepdims, out, _reduction_fn):
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(x)}")

    nd = x.ndim
    if axis is None:
        axis = tuple(range(nd))
        perm = list(axis)
        x_tmp = x
    else:
        if isinstance(axis, int):
            axis = (axis,)
        else:
            raise TypeError(
                f"`axis` argument expected `int` or `None`, got {type(axis)}"
            )
        axis = normalize_axis_tuple(axis, nd, "axis")
        perm = [i for i in range(nd) if i not in axis] + list(axis)
        x_tmp = dpt.permute_dims(x, perm)
    axis = normalize_axis_tuple(axis, nd, "axis")
    red_nd = len(axis)
    if any([x_tmp.shape[i] == 0 for i in range(-red_nd, 0)]):
        raise ValueError("reduction cannot be performed over zero-size axes")
    res_shape = x_tmp.shape[: nd - red_nd]
    exec_q = x.sycl_queue
    res_dt = ti.default_device_index_type(exec_q.sycl_device)
    res_usm_type = x.usm_type

    orig_out = out
    if out is not None:
        if not isinstance(out, dpt.usm_ndarray):
            raise TypeError(
                f"output array must be of usm_ndarray type, got {type(out)}"
            )
        if not out.flags.writable:
            raise ValueError("provided `out` array is read-only")
        if not keepdims:
            final_res_shape = res_shape
        else:
            inp_shape = x.shape
            final_res_shape = tuple(
                inp_shape[i] if i not in axis else 1 for i in range(nd)
            )
        if not out.shape == final_res_shape:
            raise ValueError(
                "The shape of input and output arrays are inconsistent. "
                f"Expected output shape is {final_res_shape}, got {out.shape}"
            )
        if res_dt != out.dtype:
            raise ValueError(
                f"Output array of type {res_dt} is needed, got {out.dtype}"
            )
        if dpctl.utils.get_execution_queue((exec_q, out.sycl_queue)) is None:
            raise ExecutionPlacementError(
                "Input and output allocation queues are not compatible"
            )
        if keepdims:
            out = dpt.squeeze(out, axis=axis)
            orig_out = out
        if ti._array_overlap(x, out) and red_nd > 0:
            out = dpt.empty_like(out)
    else:
        out = dpt.empty(
            res_shape, dtype=res_dt, usm_type=res_usm_type, sycl_queue=exec_q
        )

    if red_nd == 0:
        ht_e_fill, _ = ti._full_usm_ndarray(
            fill_value=0, dst=out, sycl_queue=exec_q
        )
        ht_e_fill.wait()
        return out

    host_tasks_list = []
    hev, red_ev = _reduction_fn(
        src=x_tmp,
        trailing_dims_to_reduce=red_nd,
        dst=out,
        sycl_queue=exec_q,
    )
    host_tasks_list.append(hev)
    if not (orig_out is None or orig_out is out):
        ht_e_cpy2, _ = ti._copy_usm_ndarray_into_usm_ndarray(
            src=out, dst=orig_out, sycl_queue=exec_q, depends=[red_ev]
        )
        host_tasks_list.append(ht_e_cpy2)
        out = orig_out

    if keepdims:
        res_shape = res_shape + (1,) * red_nd
        inv_perm = sorted(range(nd), key=lambda d: perm[d])
        out = dpt.permute_dims(dpt.reshape(out, res_shape), inv_perm)
    dpctl.SyclEvent.wait_for(host_tasks_list)
    return out


def argmax(x, /, *, axis=None, keepdims=False, out=None):
    """
    Returns the indices of the maximum values of the input array ``x`` along a
    specified axis.

    When the maximum value occurs multiple times, the indices corresponding to
    the first occurrence are returned.

    Args:
        x (usm_ndarray):
            input array.
        axis (Optional[int]):
            axis along which to search. If ``None``, returns the index of the
            maximum value of the flattened array.
            Default: ``None``.
        keepdims (Optional[bool]):
            if ``True``, the reduced axes (dimensions) are included in the
            result as singleton dimensions, so that the returned array remains
            compatible with the input arrays according to Array Broadcasting
            rules. Otherwise, if ``False``, the reduced axes are not included
            in the returned array. Default: ``False``.
        out (Optional[usm_ndarray]):
            the array into which the result is written.
            The data type of ``out`` must match the expected shape and the
            expected data type of the result.
            If ``None`` then a new array is returned. Default: ``None``.

    Returns:
        usm_ndarray:
            an array containing the indices of the first occurrence of the
            maximum values. If the entire array was searched, a
            zero-dimensional array is returned. The returned array has the
            default array index data type for the device of ``x``.
    """
    return _search_over_axis(x, axis, keepdims, out, tri._argmax_over_axis)


def argmin(x, /, *, axis=None, keepdims=False, out=None):
    """
    Returns the indices of the minimum values of the input array ``x`` along a
    specified axis.

    When the minimum value occurs multiple times, the indices corresponding to
    the first occurrence are returned.

    Args:
        x (usm_ndarray):
            input array.
        axis (Optional[int]):
            axis along which to search. If ``None``, returns the index of the
            minimum value of the flattened array.
            Default: ``None``.
        keepdims (Optional[bool]):
            if ``True``, the reduced axes (dimensions) are included in the
            result as singleton dimensions, so that the returned array remains
            compatible with the input arrays according to Array Broadcasting
            rules. Otherwise, if ``False``, the reduced axes are not included
            in the returned array. Default: ``False``.
        out (Optional[usm_ndarray]):
            the array into which the result is written.
            The data type of ``out`` must match the expected shape and the
            expected data type of the result.
            If ``None`` then a new array is returned. Default: ``None``.

    Returns:
        usm_ndarray:
            an array containing the indices of the first occurrence of the
            minimum values. If the entire array was searched, a
            zero-dimensional array is returned. The returned array has the
            default array index data type for the device of ``x``.
    """
    return _search_over_axis(x, axis, keepdims, out, tri._argmin_over_axis)
