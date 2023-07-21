#                       Data Parallel Control (dpctl)
#
#  Copyright 2020-2023 Intel Corporation
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

from ._type_utils import _to_device_supported_dtype


def _default_reduction_dtype(inp_dt, q):
    """Gives default output data type for given input data
    type `inp_dt` when reduction is performed on queue `q`
    """
    inp_kind = inp_dt.kind
    if inp_kind in "bi":
        res_dt = dpt.dtype(ti.default_device_int_type(q))
        if inp_dt.itemsize > res_dt.itemsize:
            res_dt = inp_dt
    elif inp_kind in "u":
        res_dt = dpt.dtype(ti.default_device_int_type(q).upper())
        res_ii = dpt.iinfo(res_dt)
        inp_ii = dpt.iinfo(inp_dt)
        if inp_ii.min >= res_ii.min and inp_ii.max <= res_ii.max:
            pass
        else:
            res_dt = inp_dt
    elif inp_kind in "f":
        res_dt = dpt.dtype(ti.default_device_fp_type(q))
        if res_dt.itemsize < inp_dt.itemsize:
            res_dt = inp_dt
    elif inp_kind in "c":
        res_dt = dpt.dtype(ti.default_device_complex_type(q))
        if res_dt.itemsize < inp_dt.itemsize:
            res_dt = inp_dt

    return res_dt


def sum(arr, axis=None, dtype=None, keepdims=False):
    """sum(x, axis=None, dtype=None, keepdims=False)

    Calculates the sum of the input array `x`.

    Args:
        x (usm_ndarray):
            input array.
        axis (Optional[int, Tuple[int,...]]):
            axis or axes along which sums must be computed. If a tuple
            of unique integers, sums are computed over multiple axes.
            If `None`, the sum if computed over the entire array.
            Default: `None`.
        dtype (Optional[dtype]):
            data type of the returned array. If `None`, the default data
            type is inferred from the "kind" of the input array data type.
                * If `x` has a real-valued floating-point data type,
                  the returned array will have the default real-valued
                  floating-point data type for the device where input
                  array `x` is allocated.
                * If x` has signed integral data type, the returned array
                  will have the default signed integral type for the device
                  where input array `x` is allocated.
                * If `x` has unsigned integral data type, the returned array
                  will have the default unsigned integral type for the device
                  where input array `x` is allocated.
                * If `x` has a complex-valued floating-point data typee,
                  the returned array will have the default complex-valued
                  floating-pointer data type for the device where input
                  array `x` is allocated.
                * If `x` has a boolean data type, the returned array will
                  have the default signed integral type for the device
                  where input array `x` is allocated.
            If the data type (either specified or resolved) differs from the
            data type of `x`, the input array elements are cast to the
            specified data type before computing the sum. Default: `None`.
        keepdims (Optional[bool]):
            if `True`, the reduced axes (dimensions) are included in the result
            as singleton dimensions, so that the returned array remains
            compatible with the input arrays according to Array Broadcasting
            rules. Otherwise, if `False`, the reduced axes are not included in
            the returned array. Default: `False`.
    Returns:
        usm_ndarray:
            an array containing the sums. If the sum was computed over the
            entire array, a zero-dimensional array is returned. The returned
            array has the data type as described in the `dtype` parameter
            description above.
    """
    if not isinstance(arr, dpt.usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(arr)}")
    nd = arr.ndim
    if axis is None:
        axis = tuple(range(nd))
    if not isinstance(axis, (tuple, list)):
        axis = (axis,)
    axis = normalize_axis_tuple(axis, nd, "axis")
    red_nd = len(axis)
    perm = [i for i in range(nd) if i not in axis] + list(axis)
    arr2 = dpt.permute_dims(arr, perm)
    res_shape = arr2.shape[: nd - red_nd]
    q = arr.sycl_queue
    inp_dt = arr.dtype
    if dtype is None:
        res_dt = _default_reduction_dtype(inp_dt, q)
    else:
        res_dt = dpt.dtype(dtype)
        res_dt = _to_device_supported_dtype(res_dt, q.sycl_device)

    res_usm_type = arr.usm_type
    if arr.size == 0:
        if keepdims:
            res_shape = res_shape + (1,) * red_nd
            inv_perm = sorted(range(nd), key=lambda d: perm[d])
            res_shape = tuple(res_shape[i] for i in inv_perm)
        return dpt.zeros(
            res_shape, dtype=res_dt, usm_type=res_usm_type, sycl_queue=q
        )
    if red_nd == 0:
        return dpt.astype(arr, res_dt, copy=False)

    host_tasks_list = []
    if ti._sum_over_axis_dtype_supported(inp_dt, res_dt, res_usm_type, q):
        res = dpt.empty(
            res_shape, dtype=res_dt, usm_type=res_usm_type, sycl_queue=q
        )
        ht_e, _ = ti._sum_over_axis(
            src=arr2, trailing_dims_to_reduce=red_nd, dst=res, sycl_queue=q
        )
        host_tasks_list.append(ht_e)
    else:
        if dtype is None:
            raise RuntimeError(
                "Automatically determined reduction data type does not "
                "have direct implementation"
            )
        tmp_dt = _default_reduction_dtype(inp_dt, q)
        tmp = dpt.empty(
            res_shape, dtype=tmp_dt, usm_type=res_usm_type, sycl_queue=q
        )
        ht_e_tmp, r_e = ti._sum_over_axis(
            src=arr2, trailing_dims_to_reduce=red_nd, dst=tmp, sycl_queue=q
        )
        host_tasks_list.append(ht_e_tmp)
        res = dpt.empty(
            res_shape, dtype=res_dt, usm_type=res_usm_type, sycl_queue=q
        )
        ht_e, _ = ti._copy_usm_ndarray_into_usm_ndarray(
            src=tmp, dst=res, sycl_queue=q, depends=[r_e]
        )
        host_tasks_list.append(ht_e)

    if keepdims:
        res_shape = res_shape + (1,) * red_nd
        inv_perm = sorted(range(nd), key=lambda d: perm[d])
        res = dpt.permute_dims(dpt.reshape(res, res_shape), inv_perm)
    dpctl.SyclEvent.wait_for(host_tasks_list)

    return res
