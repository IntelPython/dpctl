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

from typing import NamedTuple

import dpctl
import dpctl.tensor as dpt

from ._tensor_elementwise_impl import _not_equal, _subtract
from ._tensor_impl import (
    _copy_usm_ndarray_into_usm_ndarray,
    _extract,
    _full_usm_ndarray,
    _linspace_step,
    _take,
    default_device_index_type,
    mask_positions,
)
from ._tensor_sorting_impl import (
    _argsort_ascending,
    _searchsorted_left,
    _sort_ascending,
)

__all__ = [
    "unique_values",
    "unique_counts",
    "unique_inverse",
    "unique_all",
    "UniqueAllResult",
    "UniqueCountsResult",
    "UniqueInverseResult",
]


class UniqueAllResult(NamedTuple):
    values: dpt.usm_ndarray
    indices: dpt.usm_ndarray
    inverse_indices: dpt.usm_ndarray
    counts: dpt.usm_ndarray


class UniqueCountsResult(NamedTuple):
    values: dpt.usm_ndarray
    counts: dpt.usm_ndarray


class UniqueInverseResult(NamedTuple):
    values: dpt.usm_ndarray
    inverse_indices: dpt.usm_ndarray


def unique_values(x: dpt.usm_ndarray) -> dpt.usm_ndarray:
    """unique_values(x)

    Returns the unique elements of an input array `x`.

    Args:
        x (usm_ndarray):
            input array. Inputs with more than one dimension are flattened.
    Returns:
        usm_ndarray
            an array containing the set of unique elements in `x`. The
            returned array has the same data type as `x`.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(x)}")
    array_api_dev = x.device
    exec_q = array_api_dev.sycl_queue
    if x.ndim == 1:
        fx = x
    else:
        fx = dpt.reshape(x, (x.size,), order="C")
    if fx.size == 0:
        return fx
    s = dpt.empty_like(fx, order="C")
    host_tasks = []
    if fx.flags.c_contiguous:
        ht_ev, sort_ev = _sort_ascending(
            src=fx, trailing_dims_to_sort=1, dst=s, sycl_queue=exec_q
        )
        host_tasks.append(ht_ev)
    else:
        tmp = dpt.empty_like(fx, order="C")
        ht_ev, copy_ev = _copy_usm_ndarray_into_usm_ndarray(
            src=fx, dst=tmp, sycl_queue=exec_q
        )
        host_tasks.append(ht_ev)
        ht_ev, sort_ev = _sort_ascending(
            src=tmp,
            trailing_dims_to_sort=1,
            dst=s,
            sycl_queue=exec_q,
            depends=[copy_ev],
        )
        host_tasks.append(ht_ev)
    unique_mask = dpt.empty(fx.shape, dtype="?", sycl_queue=exec_q)
    ht_ev, uneq_ev = _not_equal(
        src1=s[:-1],
        src2=s[1:],
        dst=unique_mask[1:],
        sycl_queue=exec_q,
        depends=[sort_ev],
    )
    host_tasks.append(ht_ev)
    ht_ev, one_ev = _full_usm_ndarray(
        fill_value=True, dst=unique_mask[0], sycl_queue=exec_q
    )
    host_tasks.append(ht_ev)
    cumsum = dpt.empty(s.shape, dtype=dpt.int64, sycl_queue=exec_q)
    # synchronizing call
    n_uniques = mask_positions(
        unique_mask, cumsum, sycl_queue=exec_q, depends=[one_ev, uneq_ev]
    )
    if n_uniques == fx.size:
        dpctl.SyclEvent.wait_for(host_tasks)
        return s
    unique_vals = dpt.empty(
        n_uniques, dtype=x.dtype, usm_type=x.usm_type, sycl_queue=exec_q
    )
    ht_ev, _ = _extract(
        src=s,
        cumsum=cumsum,
        axis_start=0,
        axis_end=1,
        dst=unique_vals,
        sycl_queue=exec_q,
    )
    host_tasks.append(ht_ev)
    dpctl.SyclEvent.wait_for(host_tasks)
    return unique_vals


def unique_counts(x: dpt.usm_ndarray) -> UniqueCountsResult:
    """unique_counts(x)

    Returns the unique elements of an input array `x` and the corresponding
    counts for each unique element in `x`.

    Args:
        x (usm_ndarray):
            input array. Inputs with more than one dimension are flattened.
    Returns:
        tuple[usm_ndarray, usm_ndarray]
            a namedtuple `(values, counts)` whose

            * first element is the field name `values` and is an array
               containing the unique elements of `x`. This array has the
               same data type as `x`.
            * second element has the field name `counts` and is an array
              containing the number of times each unique element occurs in `x`.
              This array has the same shape as `values` and has the default
              array index data type.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(x)}")
    array_api_dev = x.device
    exec_q = array_api_dev.sycl_queue
    x_usm_type = x.usm_type
    if x.ndim == 1:
        fx = x
    else:
        fx = dpt.reshape(x, (x.size,), order="C")
    ind_dt = default_device_index_type(exec_q)
    if fx.size == 0:
        return UniqueCountsResult(fx, dpt.empty_like(fx, dtype=ind_dt))
    s = dpt.empty_like(fx, order="C")
    host_tasks = []
    if fx.flags.c_contiguous:
        ht_ev, sort_ev = _sort_ascending(
            src=fx, trailing_dims_to_sort=1, dst=s, sycl_queue=exec_q
        )
        host_tasks.append(ht_ev)
    else:
        tmp = dpt.empty_like(fx, order="C")
        ht_ev, copy_ev = _copy_usm_ndarray_into_usm_ndarray(
            src=fx, dst=tmp, sycl_queue=exec_q
        )
        host_tasks.append(ht_ev)
        ht_ev, sort_ev = _sort_ascending(
            src=tmp,
            dst=s,
            trailing_dims_to_sort=1,
            sycl_queue=exec_q,
            depends=[copy_ev],
        )
        host_tasks.append(ht_ev)
    unique_mask = dpt.empty(s.shape, dtype="?", sycl_queue=exec_q)
    ht_ev, uneq_ev = _not_equal(
        src1=s[:-1],
        src2=s[1:],
        dst=unique_mask[1:],
        sycl_queue=exec_q,
        depends=[sort_ev],
    )
    host_tasks.append(ht_ev)
    ht_ev, one_ev = _full_usm_ndarray(
        fill_value=True, dst=unique_mask[0], sycl_queue=exec_q
    )
    host_tasks.append(ht_ev)
    cumsum = dpt.empty(unique_mask.shape, dtype=dpt.int64, sycl_queue=exec_q)
    # synchronizing call
    n_uniques = mask_positions(
        unique_mask, cumsum, sycl_queue=exec_q, depends=[one_ev, uneq_ev]
    )
    if n_uniques == fx.size:
        dpctl.SyclEvent.wait_for(host_tasks)
        return UniqueCountsResult(
            s,
            dpt.ones(
                n_uniques, dtype=ind_dt, usm_type=x_usm_type, sycl_queue=exec_q
            ),
        )
    unique_vals = dpt.empty(
        n_uniques, dtype=x.dtype, usm_type=x_usm_type, sycl_queue=exec_q
    )
    # populate unique values
    ht_ev, _ = _extract(
        src=s,
        cumsum=cumsum,
        axis_start=0,
        axis_end=1,
        dst=unique_vals,
        sycl_queue=exec_q,
    )
    host_tasks.append(ht_ev)
    unique_counts = dpt.empty(
        n_uniques + 1, dtype=ind_dt, usm_type=x_usm_type, sycl_queue=exec_q
    )
    idx = dpt.empty(x.size, dtype=ind_dt, sycl_queue=exec_q)
    ht_ev, id_ev = _linspace_step(start=0, dt=1, dst=idx, sycl_queue=exec_q)
    host_tasks.append(ht_ev)
    ht_ev, extr_ev = _extract(
        src=idx,
        cumsum=cumsum,
        axis_start=0,
        axis_end=1,
        dst=unique_counts[:-1],
        sycl_queue=exec_q,
        depends=[id_ev],
    )
    host_tasks.append(ht_ev)
    ht_ev, set_ev = _full_usm_ndarray(
        x.size, dst=unique_counts[-1], sycl_queue=exec_q
    )
    host_tasks.append(ht_ev)
    _counts = dpt.empty_like(unique_counts[1:])
    ht_ev, _ = _subtract(
        src1=unique_counts[1:],
        src2=unique_counts[:-1],
        dst=_counts,
        sycl_queue=exec_q,
        depends=[set_ev, extr_ev],
    )
    host_tasks.append(ht_ev)
    dpctl.SyclEvent.wait_for(host_tasks)
    return UniqueCountsResult(unique_vals, _counts)


def unique_inverse(x):
    """unique_inverse

    Returns the unique elements of an input array x and the indices from the
    set of unique elements that reconstruct `x`.

    Args:
        x (usm_ndarray):
            input array. Inputs with more than one dimension are flattened.
    Returns:
        tuple[usm_ndarray, usm_ndarray]
            a namedtuple `(values, inverse_indices)` whose

            * first element has the field name `values` and is an array
              containing the unique elements of `x`. The array has the same
              data type as `x`.
            * second element has the field name `inverse_indices` and is an
              array containing the indices of values that reconstruct `x`.
              The array has the same shape as `x` and has the default array
              index data type.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(x)}")
    array_api_dev = x.device
    exec_q = array_api_dev.sycl_queue
    x_usm_type = x.usm_type
    ind_dt = default_device_index_type(exec_q)
    if x.ndim == 1:
        fx = x
    else:
        fx = dpt.reshape(x, (x.size,), order="C")
    sorting_ids = dpt.empty_like(fx, dtype=ind_dt, order="C")
    unsorting_ids = dpt.empty_like(sorting_ids, dtype=ind_dt, order="C")
    if fx.size == 0:
        return UniqueInverseResult(fx, dpt.reshape(unsorting_ids, x.shape))
    host_tasks = []
    if fx.flags.c_contiguous:
        ht_ev, sort_ev = _argsort_ascending(
            src=fx, trailing_dims_to_sort=1, dst=sorting_ids, sycl_queue=exec_q
        )
        host_tasks.append(ht_ev)
    else:
        tmp = dpt.empty_like(fx, order="C")
        ht_ev, copy_ev = _copy_usm_ndarray_into_usm_ndarray(
            src=fx, dst=tmp, sycl_queue=exec_q
        )
        host_tasks.append(ht_ev)
        ht_ev, sort_ev = _argsort_ascending(
            src=tmp,
            trailing_dims_to_sort=1,
            dst=sorting_ids,
            sycl_queue=exec_q,
            depends=[copy_ev],
        )
        host_tasks.append(ht_ev)
    ht_ev, _ = _argsort_ascending(
        src=sorting_ids,
        trailing_dims_to_sort=1,
        dst=unsorting_ids,
        sycl_queue=exec_q,
        depends=[sort_ev],
    )
    host_tasks.append(ht_ev)
    s = dpt.empty_like(fx)
    # s = fx[sorting_ids]
    ht_ev, take_ev = _take(
        src=fx,
        ind=(sorting_ids,),
        dst=s,
        axis_start=0,
        mode=0,
        sycl_queue=exec_q,
        depends=[sort_ev],
    )
    host_tasks.append(ht_ev)
    unique_mask = dpt.empty(fx.shape, dtype="?", sycl_queue=exec_q)
    ht_ev, uneq_ev = _not_equal(
        src1=s[:-1],
        src2=s[1:],
        dst=unique_mask[1:],
        sycl_queue=exec_q,
        depends=[take_ev],
    )
    host_tasks.append(ht_ev)
    ht_ev, one_ev = _full_usm_ndarray(
        fill_value=True, dst=unique_mask[0], sycl_queue=exec_q
    )
    host_tasks.append(ht_ev)
    cumsum = dpt.empty(unique_mask.shape, dtype=dpt.int64, sycl_queue=exec_q)
    # synchronizing call
    n_uniques = mask_positions(
        unique_mask, cumsum, sycl_queue=exec_q, depends=[uneq_ev, one_ev]
    )
    if n_uniques == fx.size:
        dpctl.SyclEvent.wait_for(host_tasks)
        return UniqueInverseResult(s, dpt.reshape(unsorting_ids, x.shape))
    unique_vals = dpt.empty(
        n_uniques, dtype=x.dtype, usm_type=x_usm_type, sycl_queue=exec_q
    )
    ht_ev, uv_ev = _extract(
        src=s,
        cumsum=cumsum,
        axis_start=0,
        axis_end=1,
        dst=unique_vals,
        sycl_queue=exec_q,
    )
    host_tasks.append(ht_ev)
    cum_unique_counts = dpt.empty(
        n_uniques + 1, dtype=ind_dt, usm_type=x_usm_type, sycl_queue=exec_q
    )
    idx = dpt.empty(x.size, dtype=ind_dt, sycl_queue=exec_q)
    ht_ev, id_ev = _linspace_step(start=0, dt=1, dst=idx, sycl_queue=exec_q)
    host_tasks.append(ht_ev)
    ht_ev, extr_ev = _extract(
        src=idx,
        cumsum=cumsum,
        axis_start=0,
        axis_end=1,
        dst=cum_unique_counts[:-1],
        sycl_queue=exec_q,
        depends=[id_ev],
    )
    host_tasks.append(ht_ev)
    ht_ev, set_ev = _full_usm_ndarray(
        x.size, dst=cum_unique_counts[-1], sycl_queue=exec_q
    )
    host_tasks.append(ht_ev)
    _counts = dpt.empty_like(cum_unique_counts[1:])
    ht_ev, _ = _subtract(
        src1=cum_unique_counts[1:],
        src2=cum_unique_counts[:-1],
        dst=_counts,
        sycl_queue=exec_q,
        depends=[set_ev, extr_ev],
    )
    host_tasks.append(ht_ev)

    inv_dt = dpt.int64 if x.size > dpt.iinfo(dpt.int32).max else dpt.int32
    inv = dpt.empty_like(x, dtype=inv_dt, order="C")
    ht_ev, _ = _searchsorted_left(
        hay=unique_vals,
        needles=x,
        positions=inv,
        sycl_queue=exec_q,
        depends=[
            uv_ev,
        ],
    )
    host_tasks.append(ht_ev)

    dpctl.SyclEvent.wait_for(host_tasks)
    return UniqueInverseResult(unique_vals, inv)


def unique_all(x: dpt.usm_ndarray) -> UniqueAllResult:
    """unique_all(x)

    Returns the unique elements of an input array `x`, the first occurring
    indices for each unique element in `x`, the indices from the set of unique
    elements that reconstruct `x`, and the corresponding counts for each
    unique element in `x`.

    Args:
        x (usm_ndarray):
            input array. Inputs with more than one dimension are flattened.
    Returns:
        tuple[usm_ndarray, usm_ndarray, usm_ndarray, usm_ndarray]
            a namedtuple `(values, indices, inverse_indices, counts)` whose

            * first element has the field name `values` and is an array
              containing the unique elements of `x`. The array has the same
              data type as `x`.
            * second element has the field name `indices` and is an array
              the indices (of first occurrences) of `x` that result in
              `values`. The array has the same shape as `values` and has the
              default array index data type.
            * third element has the field name `inverse_indices` and is an
              array containing the indices of values that reconstruct `x`.
              The array has the same shape as `x` and has the default array
              index data type.
            * fourth element has the field name `counts` and is an array
              containing the number of times each unique element occurs in `x`.
              This array has the same shape as `values` and has the default
              array index data type.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(x)}")
    array_api_dev = x.device
    exec_q = array_api_dev.sycl_queue
    x_usm_type = x.usm_type
    ind_dt = default_device_index_type(exec_q)
    if x.ndim == 1:
        fx = x
    else:
        fx = dpt.reshape(x, (x.size,), order="C")
    sorting_ids = dpt.empty_like(fx, dtype=ind_dt, order="C")
    unsorting_ids = dpt.empty_like(sorting_ids, dtype=ind_dt, order="C")
    if fx.size == 0:
        # original array contains no data
        # so it can be safely returned as values
        return UniqueAllResult(
            fx,
            sorting_ids,
            dpt.reshape(unsorting_ids, x.shape),
            dpt.empty_like(fx, dtype=ind_dt),
        )
    host_tasks = []
    if fx.flags.c_contiguous:
        ht_ev, sort_ev = _argsort_ascending(
            src=fx, trailing_dims_to_sort=1, dst=sorting_ids, sycl_queue=exec_q
        )
        host_tasks.append(ht_ev)
    else:
        tmp = dpt.empty_like(fx, order="C")
        ht_ev, copy_ev = _copy_usm_ndarray_into_usm_ndarray(
            src=fx, dst=tmp, sycl_queue=exec_q
        )
        host_tasks.append(ht_ev)
        ht_ev, sort_ev = _argsort_ascending(
            src=tmp,
            trailing_dims_to_sort=1,
            dst=sorting_ids,
            sycl_queue=exec_q,
            depends=[copy_ev],
        )
        host_tasks.append(ht_ev)
    ht_ev, _ = _argsort_ascending(
        src=sorting_ids,
        trailing_dims_to_sort=1,
        dst=unsorting_ids,
        sycl_queue=exec_q,
        depends=[sort_ev],
    )
    host_tasks.append(ht_ev)
    s = dpt.empty_like(fx)
    # s = fx[sorting_ids]
    ht_ev, take_ev = _take(
        src=fx,
        ind=(sorting_ids,),
        dst=s,
        axis_start=0,
        mode=0,
        sycl_queue=exec_q,
        depends=[sort_ev],
    )
    host_tasks.append(ht_ev)
    unique_mask = dpt.empty(fx.shape, dtype="?", sycl_queue=exec_q)
    ht_ev, uneq_ev = _not_equal(
        src1=s[:-1],
        src2=s[1:],
        dst=unique_mask[1:],
        sycl_queue=exec_q,
        depends=[take_ev],
    )
    host_tasks.append(ht_ev)
    ht_ev, one_ev = _full_usm_ndarray(
        fill_value=True, dst=unique_mask[0], sycl_queue=exec_q
    )
    host_tasks.append(ht_ev)
    cumsum = dpt.empty(unique_mask.shape, dtype=dpt.int64, sycl_queue=exec_q)
    # synchronizing call
    n_uniques = mask_positions(
        unique_mask, cumsum, sycl_queue=exec_q, depends=[uneq_ev, one_ev]
    )
    if n_uniques == fx.size:
        dpctl.SyclEvent.wait_for(host_tasks)
        _counts = dpt.ones(
            n_uniques, dtype=ind_dt, usm_type=x_usm_type, sycl_queue=exec_q
        )
        return UniqueAllResult(
            s,
            sorting_ids,
            dpt.reshape(unsorting_ids, x.shape),
            _counts,
        )
    unique_vals = dpt.empty(
        n_uniques, dtype=x.dtype, usm_type=x_usm_type, sycl_queue=exec_q
    )
    ht_ev, uv_ev = _extract(
        src=s,
        cumsum=cumsum,
        axis_start=0,
        axis_end=1,
        dst=unique_vals,
        sycl_queue=exec_q,
    )
    host_tasks.append(ht_ev)
    cum_unique_counts = dpt.empty(
        n_uniques + 1, dtype=ind_dt, usm_type=x_usm_type, sycl_queue=exec_q
    )
    idx = dpt.empty(x.size, dtype=ind_dt, sycl_queue=exec_q)
    ht_ev, id_ev = _linspace_step(start=0, dt=1, dst=idx, sycl_queue=exec_q)
    host_tasks.append(ht_ev)
    ht_ev, extr_ev = _extract(
        src=idx,
        cumsum=cumsum,
        axis_start=0,
        axis_end=1,
        dst=cum_unique_counts[:-1],
        sycl_queue=exec_q,
        depends=[id_ev],
    )
    host_tasks.append(ht_ev)
    ht_ev, set_ev = _full_usm_ndarray(
        x.size, dst=cum_unique_counts[-1], sycl_queue=exec_q
    )
    host_tasks.append(ht_ev)
    _counts = dpt.empty_like(cum_unique_counts[1:])
    ht_ev, sub_ev = _subtract(
        src1=cum_unique_counts[1:],
        src2=cum_unique_counts[:-1],
        dst=_counts,
        sycl_queue=exec_q,
        depends=[set_ev, extr_ev],
    )
    host_tasks.append(ht_ev)

    inv_dt = dpt.int64 if x.size > dpt.iinfo(dpt.int32).max else dpt.int32
    inv = dpt.empty_like(x, dtype=inv_dt, order="C")
    ht_ev, _ = _searchsorted_left(
        hay=unique_vals,
        needles=x,
        positions=inv,
        sycl_queue=exec_q,
        depends=[
            uv_ev,
        ],
    )
    host_tasks.append(ht_ev)

    dpctl.SyclEvent.wait_for(host_tasks)
    return UniqueAllResult(
        unique_vals,
        sorting_ids[cum_unique_counts[:-1]],
        inv,
        _counts,
    )
