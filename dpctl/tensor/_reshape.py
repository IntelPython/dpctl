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

import numpy as np

import dpctl.tensor as dpt
from dpctl.tensor._tensor_impl import (
    _copy_usm_ndarray_for_reshape,
    _ravel_multi_index,
    _unravel_index,
)

__doc__ = "Implementation module for :func:`dpctl.tensor.reshape`."


def _make_unit_indexes(shape):
    """
    Construct a diagonal matrix with with one on the diagonal
    except if the corresponding element of shape is 1.
    """
    nd = len(shape)
    mi = np.zeros((nd, nd), dtype="u4")
    for i, dim in enumerate(shape):
        mi[i, i] = 1 if dim > 1 else 0
    return mi


def ti_unravel_index(flat_index, shape, order="C"):
    return _unravel_index(flat_index, shape, order)


def ti_ravel_multi_index(multi_index, shape, order="C"):
    return _ravel_multi_index(multi_index, shape, order)


def reshaped_strides(old_sh, old_sts, new_sh, order="C"):
    """
    When reshaping array with `old_sh` shape and `old_sts` strides
    into the new shape `new_sh`, returns the new stride if the reshape
    can be a view, otherwise returns `None`.
    """
    eye_new_mi = _make_unit_indexes(new_sh)
    new_sts = [
        sum(
            st_i * ind_i
            for st_i, ind_i in zip(
                old_sts, ti_unravel_index(flat_index, old_sh, order=order)
            )
        )
        for flat_index in [
            ti_ravel_multi_index(unitvec, new_sh, order=order)
            for unitvec in eye_new_mi
        ]
    ]
    eye_old_mi = _make_unit_indexes(old_sh)
    check_sts = [
        sum(
            st_i * ind_i
            for st_i, ind_i in zip(
                new_sts, ti_unravel_index(flat_index, new_sh, order=order)
            )
        )
        for flat_index in [
            ti_ravel_multi_index(unitvec, old_sh, order=order)
            for unitvec in eye_old_mi
        ]
    ]
    valid = all(
        check_st == old_st or old_dim == 1
        for check_st, old_st, old_dim in zip(check_sts, old_sts, old_sh)
    )
    return new_sts if valid else None


def reshape(X, /, shape, *, order="C", copy=None):
    """reshape(x, shape, order="C")

    Reshapes array ``x`` into new shape.

    Args:
        x (usm_ndarray):
            input array
        shape (Tuple[int]):
            the desired shape of the resulting array.
        order ("C", "F", optional):
            memory layout of the resulting array
            if a copy is found to be necessary. Supported
            choices are ``"C"`` for C-contiguous, or row-major layout;
            and ``"F"`` for F-contiguous, or column-major layout.

    Returns:
        usm_ndarray:
            Reshaped array is a view, if possible,
            and a copy otherwise with memory layout as indicated
            by ``order`` keyword.
    """
    if not isinstance(X, dpt.usm_ndarray):
        raise TypeError
    if not isinstance(shape, (list, tuple)):
        shape = (shape,)
    if order in "cfCF":
        order = order.upper()
    else:
        raise ValueError(
            f"Keyword 'order' not recognized. Expecting 'C' or 'F', got {order}"
        )
    if copy not in (True, False, None):
        raise ValueError(
            f"Keyword 'copy' not recognized. Expecting True, False, "
            f"or None, got {copy}"
        )
    shape = [operator.index(d) for d in shape]
    negative_ones_count = 0
    for nshi in shape:
        if nshi == -1:
            negative_ones_count = negative_ones_count + 1
        if (nshi < -1) or negative_ones_count > 1:
            raise ValueError(
                "Target shape should have at most 1 negative "
                "value which can only be -1"
            )
    if negative_ones_count:
        sz = -np.prod(shape)
        if sz == 0:
            raise ValueError(
                f"Can not reshape array of size {X.size} into "
                f"shape {tuple(i for i in shape if i >= 0)}"
            )
        v = X.size // sz
        shape = [v if d == -1 else d for d in shape]
    if X.size != np.prod(shape):
        raise ValueError(f"Can not reshape into {shape}")
    if X.size:
        newsts = reshaped_strides(X.shape, X.strides, shape, order=order)
    else:
        newsts = (1,) * len(shape)
    copy_required = newsts is None
    if copy_required and (copy is False):
        raise ValueError(
            "Reshaping the array requires a copy, but no copying was "
            "requested by using copy=False"
        )
    copy_q = X.sycl_queue
    if copy_required or (copy is True):
        # must perform a copy
        flat_res = dpt.usm_ndarray(
            (X.size,),
            dtype=X.dtype,
            buffer=X.usm_type,
            buffer_ctor_kwargs={"queue": copy_q},
        )
        if order == "C":
            hev, _ = _copy_usm_ndarray_for_reshape(
                src=X, dst=flat_res, sycl_queue=copy_q
            )
        else:
            X_t = dpt.permute_dims(X, range(X.ndim - 1, -1, -1))
            hev, _ = _copy_usm_ndarray_for_reshape(
                src=X_t, dst=flat_res, sycl_queue=copy_q
            )
        hev.wait()
        return dpt.usm_ndarray(
            tuple(shape), dtype=X.dtype, buffer=flat_res, order=order
        )
    # can form a view
    if (len(shape) == X.ndim) and all(
        s1 == s2 for s1, s2 in zip(shape, X.shape)
    ):
        return X
    return dpt.usm_ndarray(
        shape,
        dtype=X.dtype,
        buffer=X,
        strides=tuple(newsts),
        offset=X._element_offset,
    )
