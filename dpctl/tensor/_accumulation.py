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

from numpy.core.numeric import normalize_axis_index

import dpctl
import dpctl.tensor as dpt
import dpctl.tensor._tensor_accumulation_impl as tai
import dpctl.tensor._tensor_impl as ti
from dpctl.tensor._type_utils import _to_device_supported_dtype


def _default_accumulation_dtype(inp_dt, q):
    """Gives default output data type for given input data
    type `inp_dt` when accumulation is performed on queue `q`
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
        res_dt = inp_dt
    elif inp_kind in "c":
        res_dt = inp_dt

    return res_dt


def _default_accumulation_dtype_fp_types(inp_dt, q):
    """Gives default output data type for given input data
    type `inp_dt` when accumulation is performed on queue `q`
    and the accumulation supports only floating-point data types
    """
    inp_kind = inp_dt.kind
    if inp_kind in "biu":
        res_dt = dpt.dtype(ti.default_device_fp_type(q))
        can_cast_v = dpt.can_cast(inp_dt, res_dt)
        if not can_cast_v:
            _fp64 = q.sycl_device.has_aspect_fp64
            res_dt = dpt.float64 if _fp64 else dpt.float32
    elif inp_kind in "f":
        res_dt = inp_dt
    elif inp_kind in "c":
        raise TypeError("reduction not defined for complex types")

    return res_dt


def _accumulate_over_axis(
    x,
    axis,
    dtype,
    include_initial,
    _accumulate_fn,
    _accumulate_include_initial_fn,
    _dtype_supported,
    _default_accumulation_type_fn,
):
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(x)}")
    nd = x.ndim
    if axis is None:
        if nd > 1:
            raise ValueError
        axis = 0
    else:
        axis = operator.index(axis)
    axis = normalize_axis_index(axis, nd, "axis")
    a1 = axis + 1
    if a1 == nd:
        perm = list(range(nd))
        arr = x
    else:
        perm = [i for i in range(nd) if i != axis] + [
            axis,
        ]
        arr = dpt.permute_dims(x, perm)
    q = x.sycl_queue
    inp_dt = x.dtype
    if dtype is None:
        res_dt = _default_accumulation_type_fn(inp_dt, q)
    else:
        res_dt = dpt.dtype(dtype)
        res_dt = _to_device_supported_dtype(res_dt, q.sycl_device)
    sh = arr.shape
    res_sh = sh[:-1] + (sh[-1] + 1,) if include_initial else sh
    res_usm_type = x.usm_type

    host_tasks_list = []
    if _dtype_supported(inp_dt, res_dt):
        res = dpt.empty(
            res_sh, dtype=res_dt, usm_type=res_usm_type, sycl_queue=q
        )
        if not include_initial:
            ht_e, _ = _accumulate_fn(
                src=arr,
                trailing_dims_to_accumulate=1,
                dst=res,
                sycl_queue=q,
            )
        else:
            ht_e, _ = _accumulate_include_initial_fn(
                src=arr,
                dst=res,
                sycl_queue=q,
            )
        host_tasks_list.append(ht_e)
    else:
        if dtype is None:
            raise RuntimeError(
                "Automatically determined accumulation data type does not "
                "have direct implementation"
            )
        if _dtype_supported(res_dt, res_dt):
            tmp = dpt.empty(
                arr.shape, dtype=res_dt, usm_type=res_usm_type, sycl_queue=q
            )
            ht_e_cpy, cpy_e = ti._copy_usm_ndarray_into_usm_ndarray(
                src=arr, dst=tmp, sycl_queue=q
            )
            host_tasks_list.append(ht_e_cpy)
            res = dpt.empty(
                res_sh, dtype=res_dt, usm_type=res_usm_type, sycl_queue=q
            )
            if not include_initial:
                ht_e, _ = _accumulate_fn(
                    src=tmp,
                    trailing_dims_to_accumulate=1,
                    dst=res,
                    sycl_queue=q,
                    depends=[cpy_e],
                )
            else:
                ht_e, _ = _accumulate_include_initial_fn(
                    src=tmp,
                    dst=res,
                    sycl_queue=q,
                    depends=[cpy_e],
                )
            host_tasks_list.append(ht_e)
        else:
            buf_dt = _default_accumulation_dtype(inp_dt, q)
            tmp = dpt.empty(
                arr.shape, dtype=buf_dt, usm_type=res_usm_type, sycl_queue=q
            )
            ht_e_cpy, cpy_e = ti._copy_usm_ndarray_into_usm_ndarray(
                src=arr, dst=tmp, sycl_queue=q
            )
            tmp_res = dpt.empty(
                res_sh, dtype=buf_dt, usm_type=res_usm_type, sycl_queue=q
            )
            host_tasks_list.append(ht_e_cpy)
            res = dpt.empty(
                res_sh, dtype=res_dt, usm_type=res_usm_type, sycl_queue=q
            )
            if not include_initial:
                ht_e, a_e = _accumulate_fn(
                    src=arr,
                    trailing_dims_to_accumulate=1,
                    dst=tmp_res,
                    sycl_queue=q,
                    depends=[cpy_e],
                )
            else:
                ht_e, a_e = _accumulate_include_initial_fn(
                    src=arr,
                    dst=tmp_res,
                    sycl_queue=q,
                    depends=[cpy_e],
                )
            host_tasks_list.append(ht_e)
            ht_e_cpy2, _ = ti._copy_usm_ndarray_into_usm_ndarray(
                src=tmp_res, dst=res, sycl_queue=q, depends=[a_e]
            )
            host_tasks_list.append(ht_e_cpy2)
    if a1 != nd:
        inv_perm = sorted(range(nd), key=lambda d: perm[d])
        res = dpt.permute_dims(res, inv_perm)
    dpctl.SyclEvent.wait_for(host_tasks_list)

    return res


def cumulative_sum(x, axis=None, dtype=None, include_initial=False):
    return _accumulate_over_axis(
        x,
        axis,
        dtype,
        include_initial,
        tai._cumsum_over_axis,
        tai._cumsum_final_axis_include_initial,
        tai._cumsum_dtype_supported,
        _default_accumulation_dtype,
    )


def cumulative_prod(x, axis=None, dtype=None, include_initial=False):
    return _accumulate_over_axis(
        x,
        axis,
        dtype,
        include_initial,
        tai._cumprod_over_axis,
        tai._cumprod_final_axis_include_initial,
        tai._cumprod_dtype_supported,
        _default_accumulation_dtype,
    )


def cumulative_logsumexp(x, axis=None, dtype=None, include_initial=False):
    return _accumulate_over_axis(
        x,
        axis,
        dtype,
        include_initial,
        tai._cumlogsumexp_over_axis,
        tai._cumlogsumexp_final_axis_include_initial,
        tai._cumlogsumexp_dtype_supported,
        _default_accumulation_dtype_fp_types,
    )
