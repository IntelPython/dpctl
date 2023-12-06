from numpy.core.numeric import normalize_axis_index

import dpctl
import dpctl.tensor as dpt
import dpctl.tensor._tensor_impl as ti

from ._tensor_sorting_impl import (
    _argsort_ascending,
    _argsort_descending,
    _sort_ascending,
    _sort_descending,
)


def sort(x, axis=-1, descending=False, stable=False):
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(
            f"Expected type dpctl.tensor.usm_ndarray, got {type(x)}"
        )
    nd = x.ndim
    axis = normalize_axis_index(axis, ndim=nd, msg_prefix="axis")
    a1 = axis + 1
    if a1 == nd:
        perm = list(range(nd))
        arr = x
    else:
        perm = [i for i in range(nd) if i != axis] + [
            axis,
        ]
        arr = dpt.permute_dims(x, perm)
    exec_q = x.sycl_queue
    host_tasks_list = []
    impl_fn = _sort_descending if descending else _sort_ascending
    if arr.flags.c_contiguous:
        res = dpt.empty_like(arr, order="C")
        ht_ev, _ = impl_fn(
            src=arr, trailing_dims_to_sort=1, dst=res, sycl_queue=exec_q
        )
        host_tasks_list.append(ht_ev)
    else:
        tmp = dpt.empty_like(arr, order="C")
        ht_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=arr, dst=tmp, sycl_queue=exec_q
        )
        host_tasks_list.append(ht_ev)
        res = dpt.empty_like(arr, order="C")
        ht_ev, _ = impl_fn(
            src=tmp,
            trailing_dims_to_sort=1,
            dst=res,
            sycl_queue=exec_q,
            depends=[copy_ev],
        )
        host_tasks_list.append(ht_ev)
    if a1 != nd:
        inv_perm = sorted(range(nd), key=lambda d: perm[d])
        res = dpt.permute_dims(res, inv_perm)
    dpctl.SyclEvent.wait_for(host_tasks_list)
    return res


def argsort(x, axis=-1, descending=False, stable=False):
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(
            f"Expected type dpctl.tensor.usm_ndarray, got {type(x)}"
        )
    nd = x.ndim
    axis = normalize_axis_index(axis, ndim=nd, msg_prefix="axis")
    a1 = axis + 1
    if a1 == nd:
        perm = list(range(nd))
        arr = x
    else:
        perm = [i for i in range(nd) if i != axis] + [
            axis,
        ]
        arr = dpt.permute_dims(x, perm)
    exec_q = x.sycl_queue
    host_tasks_list = []
    impl_fn = _argsort_descending if descending else _argsort_ascending
    index_dt = ti.default_device_index_type(exec_q)
    if arr.flags.c_contiguous:
        res = dpt.empty_like(arr, dtype=index_dt, order="C")
        ht_ev, _ = impl_fn(
            src=arr, trailing_dims_to_sort=1, dst=res, sycl_queue=exec_q
        )
        host_tasks_list.append(ht_ev)
    else:
        tmp = dpt.empty_like(arr, order="C")
        ht_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=arr, dst=tmp, sycl_queue=exec_q
        )
        host_tasks_list.append(ht_ev)
        res = dpt.empty_like(arr, dtype=index_dt, order="C")
        ht_ev, _ = impl_fn(
            src=tmp,
            trailing_dims_to_sort=1,
            dst=res,
            sycl_queue=exec_q,
            depends=[copy_ev],
        )
        host_tasks_list.append(ht_ev)
    if a1 != nd:
        inv_perm = sorted(range(nd), key=lambda d: perm[d])
        res = dpt.permute_dims(res, inv_perm)
    dpctl.SyclEvent.wait_for(host_tasks_list)
    return res
