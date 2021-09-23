import operator

import numpy as np

import dpctl.tensor as dpt
from dpctl.tensor._copy_utils import copy_same_dtype


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
                old_sts, np.unravel_index(flat_index, old_sh, order=order)
            )
        )
        for flat_index in [
            np.ravel_multi_index(unitvec, new_sh, order=order)
            for unitvec in eye_new_mi
        ]
    ]
    eye_old_mi = _make_unit_indexes(old_sh)
    check_sts = [
        sum(
            st_i * ind_i
            for st_i, ind_i in zip(
                new_sts, np.unravel_index(flat_index, new_sh, order=order)
            )
        )
        for flat_index in [
            np.ravel_multi_index(unitvec, old_sh, order=order)
            for unitvec in eye_old_mi
        ]
    ]
    valid = all(
        [check_st == old_st for check_st, old_st in zip(check_sts, old_sts)]
    )
    return new_sts if valid else None


def reshape(X, newshape, order="C"):
    """
    reshape(X: usm_ndarray, newshape: tuple, order="C") -> usm_ndarray

    Reshapes given usm_ndarray into new shape. Returns a view, if possible,
    a copy otherwise. Memory layout of the copy is controlled by order keyword.
    """
    if type(X) is not dpt.usm_ndarray:
        raise TypeError
    if not isinstance(newshape, (list, tuple)):
        newshape = (newshape,)
    if order not in ["C", "F"]:
        raise ValueError(
            f"Keyword 'order' not recognized. Expecting 'C' or 'F', got {order}"
        )
    newshape = [operator.index(d) for d in newshape]
    negative_ones_count = 0
    for i in range(len(newshape)):
        if newshape[i] == -1:
            negative_ones_count = negative_ones_count + 1
        if (newshape[i] < -1) or negative_ones_count > 1:
            raise ValueError(
                "Target shape should have at most 1 negative "
                "value which can only be -1"
            )
    if negative_ones_count:
        v = X.size // (-np.prod(newshape))
        newshape = [v if d == -1 else d for d in newshape]
    if X.size != np.prod(newshape):
        raise ValueError("Can not reshape into {}".format(newshape))
    newsts = reshaped_strides(X.shape, X.strides, newshape, order=order)
    if newsts is None:
        # must perform a copy
        flat_res = dpt.usm_ndarray(
            (X.size,),
            dtype=X.dtype,
            buffer=X.usm_type,
            buffer_ctor_kwargs={"queue": X.sycl_queue},
            order=order,
        )
        for i in range(X.size):
            copy_same_dtype(
                flat_res[i], X[np.unravel_index(i, X.shape, order=order)]
            )
        return dpt.usm_ndarray(
            tuple(newshape), dtype=X.dtype, buffer=flat_res, order=order
        )
    else:
        # can form a view
        return dpt.usm_ndarray(
            newshape,
            dtype=X.dtype,
            buffer=X,
            strides=tuple(newsts),
            offset=X.__sycl_usm_array_interface__.get("offset", 0),
        )
