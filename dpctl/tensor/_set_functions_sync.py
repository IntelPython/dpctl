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


from typing import NamedTuple

import dpctl.tensor as dpt

from ._tensor_impl import (
    _extract,
    _full_usm_ndarray,
    _linspace_step,
    default_device_index_type,
    mask_positions,
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

    Returns the unique elements of an input array x.

    Args:
        x (usm_ndarray):
            input array. The input with more than one dimension is flattened.
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
    s = dpt.sort(fx)
    unique_mask = dpt.empty(fx.shape, dtype="?", sycl_queue=exec_q)
    dpt.not_equal(s[:-1], s[1:], out=unique_mask[1:])
    unique_mask[0] = True
    cumsum = dpt.empty(s.shape, dtype=dpt.int64, sycl_queue=exec_q)
    n_uniques = mask_positions(unique_mask, cumsum, sycl_queue=exec_q)
    if n_uniques == fx.size:
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
    ht_ev.wait()
    return unique_vals


def unique_counts(x: dpt.usm_ndarray) -> UniqueCountsResult:
    """unique_counts(x)

    Returns the unique elements of an input array `x` and the corresponding
    counts for each unique element in `x`.

    Args:
        x (usm_ndarray):
            input array. The input with more than one dimension is flattened.
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
    if x.ndim == 1:
        fx = x
    else:
        fx = dpt.reshape(x, (x.size,), order="C")
    ind_dt = default_device_index_type(exec_q)
    if fx.size == 0:
        return UniqueCountsResult(fx, dpt.empty_like(fx, dtype=ind_dt))
    s = dpt.sort(fx)
    unique_mask = dpt.empty(s.shape, dtype="?", sycl_queue=exec_q)
    dpt.not_equal(s[:-1], s[1:], out=unique_mask[1:])
    unique_mask[0] = True
    cumsum = dpt.empty(unique_mask.shape, dtype=dpt.int64, sycl_queue=exec_q)
    # synchronizing call
    n_uniques = mask_positions(unique_mask, cumsum, sycl_queue=exec_q)
    if n_uniques == fx.size:
        return UniqueCountsResult(s, dpt.ones(n_uniques, dtype=ind_dt))
    unique_vals = dpt.empty(
        n_uniques, dtype=x.dtype, usm_type=x.usm_type, sycl_queue=exec_q
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
    ht_ev.wait()
    unique_counts = dpt.empty(
        n_uniques + 1, dtype=ind_dt, usm_type=x.usm_type, sycl_queue=exec_q
    )
    idx = dpt.arange(x.size, dtype=ind_dt, sycl_queue=exec_q)
    ht_ev, _ = _extract(
        src=idx,
        cumsum=cumsum,
        axis_start=0,
        axis_end=1,
        dst=unique_counts[:-1],
        sycl_queue=exec_q,
    )
    unique_counts[-1] = fx.size
    ht_ev.wait()
    _counts = dpt.empty_like(unique_counts[1:])
    dpt.subtract(unique_counts[1:], unique_counts[:-1], out=_counts)
    return UniqueCountsResult(unique_vals, _counts)


def unique_inverse(x):
    """unique_inverse

    Returns the unique elements of an input array x and the indices from the
    set of unique elements that reconstruct `x`.

    Args:
        x (usm_ndarray):
            input array. The input with more than one dimension is flattened.
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
    ind_dt = default_device_index_type(exec_q)
    if x.ndim == 1:
        fx = x
    else:
        fx = dpt.reshape(x, (x.size,), order="C")
    sorting_ids = dpt.argsort(fx)
    unsorting_ids = dpt.argsort(sorting_ids)
    if fx.size == 0:
        return UniqueInverseResult(fx, dpt.reshape(unsorting_ids, x.shape))
    s = fx[sorting_ids]
    unique_mask = dpt.empty(fx.shape, dtype="?", sycl_queue=exec_q)
    unique_mask[0] = True
    dpt.not_equal(s[:-1], s[1:], out=unique_mask[1:])
    cumsum = dpt.empty(unique_mask.shape, dtype=dpt.int64, sycl_queue=exec_q)
    # synchronizing call
    n_uniques = mask_positions(unique_mask, cumsum, sycl_queue=exec_q)
    if n_uniques == fx.size:
        return UniqueInverseResult(s, unsorting_ids)
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
    ht_ev.wait()
    cum_unique_counts = dpt.empty(
        n_uniques + 1, dtype=ind_dt, usm_type=x.usm_type, sycl_queue=exec_q
    )
    idx = dpt.empty(x.size, dtype=ind_dt, sycl_queue=exec_q)
    ht_ev, id_ev = _linspace_step(start=0, dt=1, dst=idx, sycl_queue=exec_q)
    ht_ev.wait()
    ht_ev, _ = _extract(
        src=idx,
        cumsum=cumsum,
        axis_start=0,
        axis_end=1,
        dst=cum_unique_counts[:-1],
        sycl_queue=exec_q,
    )
    ht_ev.wait()
    cum_unique_counts[-1] = fx.size
    _counts = dpt.subtract(cum_unique_counts[1:], cum_unique_counts[:-1])
    # TODO: when searchsorted is available,
    #   inv = searchsorted(unique_vals, fx)
    counts = dpt.asnumpy(_counts).tolist()
    inv = dpt.empty_like(fx, dtype=ind_dt)
    pos = 0
    for i in range(len(counts)):
        pos_next = pos + counts[i]
        _dst = inv[pos:pos_next]
        ht_ev, _ = _full_usm_ndarray(fill_value=i, dst=_dst, sycl_queue=exec_q)
        ht_ev.wait()
        pos = pos_next
    return UniqueInverseResult(
        unique_vals, dpt.reshape(inv[unsorting_ids], x.shape)
    )


def unique_all(x: dpt.usm_ndarray) -> UniqueAllResult:
    """unique_all(x)

    Returns the unique elements of an input array `x`, the first occurring
    indices for each unique element in `x`, the indices from the set of unique
    elements that reconstruct `x`, and the corresponding counts for each
    unique element in `x`.

    Args:
        x (usm_ndarray):
            input array. The input with more than one dimension is flattened.
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
    ind_dt = default_device_index_type(exec_q)
    if x.ndim == 1:
        fx = x
    else:
        fx = dpt.reshape(x, (x.size,), order="C")
    sorting_ids = dpt.argsort(fx)
    unsorting_ids = dpt.argsort(sorting_ids)
    if fx.size == 0:
        # original array contains no data
        # so it can be safely returned as values
        return UniqueAllResult(
            fx,
            sorting_ids,
            dpt.reshape(unsorting_ids, x.shape),
            dpt.empty_like(fx, dtype=ind_dt),
        )
    s = fx[sorting_ids]
    unique_mask = dpt.empty(fx.shape, dtype="?", sycl_queue=exec_q)
    dpt.not_equal(s[:-1], s[1:], out=unique_mask[1:])
    unique_mask[0] = True
    cumsum = dpt.empty(unique_mask.shape, dtype=dpt.int64, sycl_queue=exec_q)
    # synchronizing call
    n_uniques = mask_positions(unique_mask, cumsum, sycl_queue=exec_q)
    if n_uniques == fx.size:
        _counts = dpt.ones(
            n_uniques, dtype=ind_dt, usm_type=x.usm_type, sycl_queue=exec_q
        )
        return UniqueAllResult(
            s,
            sorting_ids,
            dpt.reshape(unsorting_ids, x.shape),
            _counts,
        )
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
    ht_ev.wait()
    cum_unique_counts = dpt.empty(
        n_uniques + 1, dtype=ind_dt, usm_type=x.usm_type, sycl_queue=exec_q
    )
    idx = dpt.arange(fx.size, dtype=ind_dt, sycl_queue=exec_q)
    ht_ev, extr_ev = _extract(
        src=idx,
        cumsum=cumsum,
        axis_start=0,
        axis_end=1,
        dst=cum_unique_counts[:-1],
        sycl_queue=exec_q,
    )
    ht_ev.wait()
    cum_unique_counts[-1] = fx.size
    _counts = cum_unique_counts[1:] - cum_unique_counts[:-1]
    # TODO: when searchsorted is available,
    #   inv = searchsorted(unique_vals, fx)
    counts = dpt.asnumpy(_counts).tolist()
    inv = dpt.empty_like(fx, dtype=ind_dt)
    pos = 0
    for i in range(len(counts)):
        pos_next = pos + counts[i]
        _dst = inv[pos:pos_next]
        ht_ev, _ = _full_usm_ndarray(fill_value=i, dst=_dst, sycl_queue=exec_q)
        ht_ev.wait()
        pos = pos_next
    return UniqueAllResult(
        unique_vals,
        sorting_ids[cum_unique_counts[:-1]],
        dpt.reshape(inv[unsorting_ids], x.shape),
        _counts,
    )
