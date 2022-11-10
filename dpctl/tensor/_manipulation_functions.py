#                       Data Parallel Control (dpctl)
#
#  Copyright 2020-2022 Intel Corporation
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


from itertools import chain, product, repeat

import numpy as np
from numpy.core.numeric import normalize_axis_index, normalize_axis_tuple

import dpctl
import dpctl.tensor as dpt
import dpctl.tensor._tensor_impl as ti
import dpctl.utils as dputils

__doc__ = (
    "Implementation module for array manipulation "
    "functions in :module:`dpctl.tensor`"
)


def _broadcast_strides(X_shape, X_strides, res_ndim):
    """
    Broadcasts strides to match the given dimensions;
    returns tuple type strides.
    """
    out_strides = [0] * res_ndim
    X_shape_len = len(X_shape)
    str_dim = -X_shape_len
    for i in range(X_shape_len):
        shape_value = X_shape[i]
        if not shape_value == 1:
            out_strides[str_dim] = X_strides[i]
        str_dim += 1

    return tuple(out_strides)


def _broadcast_shapes(*args):
    """
    Broadcast the input shapes into a single shape;
    returns tuple broadcasted shape.
    """
    shapes = [array.shape for array in args]
    if len(set(shapes)) == 1:
        return shapes[0]
    mutable_shapes = False
    nds = [len(s) for s in shapes]
    biggest = max(nds)
    for i in range(len(args)):
        diff = biggest - nds[i]
        if diff > 0:
            ty = type(shapes[i])
            shapes[i] = ty(chain(repeat(1, diff), shapes[i]))
    common_shape = []
    for axis in range(biggest):
        lengths = [s[axis] for s in shapes]
        unique = set(lengths + [1])
        if len(unique) > 2:
            raise ValueError(
                "Shape mismatch: two or more arrays have "
                f"incompatible dimensions on axis ({axis},)"
            )
        elif len(unique) == 2:
            unique.remove(1)
            new_length = unique.pop()
            common_shape.append(new_length)
            for i in range(len(args)):
                if shapes[i][axis] == 1:
                    if not mutable_shapes:
                        shapes = [list(s) for s in shapes]
                        mutable_shapes = True
                    shapes[i][axis] = new_length
        else:
            common_shape.append(1)

    return tuple(common_shape)


def permute_dims(X, axes):
    """
    permute_dims(X: usm_ndarray, axes: tuple or list) -> usm_ndarray

    Permute the axes (dimensions) of an array; returns the permuted
    array as a view.
    """
    if not isinstance(X, dpt.usm_ndarray):
        raise TypeError(f"Expected usm_ndarray type, got {type(X)}.")
    if not isinstance(axes, (tuple, list)):
        axes = (axes,)
    if not X.ndim == len(axes):
        raise ValueError(
            "The length of the passed axes does not match "
            "to the number of usm_ndarray dimensions."
        )
    axes = normalize_axis_tuple(axes, X.ndim, "axes")
    newstrides = tuple(X.strides[i] for i in axes)
    newshape = tuple(X.shape[i] for i in axes)
    return dpt.usm_ndarray(
        shape=newshape,
        dtype=X.dtype,
        buffer=X,
        strides=newstrides,
        offset=X.__sycl_usm_array_interface__.get("offset", 0),
    )


def expand_dims(X, axes):
    """
    expand_dims(X: usm_ndarray, axes: int or tuple or list) -> usm_ndarray

    Expands the shape of an array by inserting a new axis (dimension)
    of size one at the position specified by axes; returns a view, if possible,
    a copy otherwise with the number of dimensions increased.
    """
    if not isinstance(X, dpt.usm_ndarray):
        raise TypeError(f"Expected usm_ndarray type, got {type(X)}.")
    if not isinstance(axes, (tuple, list)):
        axes = (axes,)

    out_ndim = len(axes) + X.ndim
    axes = normalize_axis_tuple(axes, out_ndim)

    shape_it = iter(X.shape)
    shape = tuple(1 if ax in axes else next(shape_it) for ax in range(out_ndim))

    return dpt.reshape(X, shape)


def squeeze(X, axes=None):
    """
    squeeze(X: usm_ndarray, axes: int or tuple or list) -> usm_ndarray

    Removes singleton dimensions (axes) from X; returns a view, if possible,
    a copy otherwise, but with all or a subset of the dimensions
    of length 1 removed.
    """
    if not isinstance(X, dpt.usm_ndarray):
        raise TypeError(f"Expected usm_ndarray type, got {type(X)}.")
    X_shape = X.shape
    if axes is not None:
        if not isinstance(axes, (tuple, list)):
            axes = (axes,)
        axes = normalize_axis_tuple(axes, X.ndim if X.ndim != 0 else X.ndim + 1)
        new_shape = []
        for i, x in enumerate(X_shape):
            if i not in axes:
                new_shape.append(x)
            else:
                if x != 1:
                    raise ValueError(
                        "Cannot select an axis to squeeze out "
                        "which has size not equal to one."
                    )
        new_shape = tuple(new_shape)
    else:
        new_shape = tuple(axis for axis in X_shape if axis != 1)
    if new_shape == X.shape:
        return X
    else:
        return dpt.reshape(X, new_shape)


def broadcast_to(X, shape):
    """
    broadcast_to(X: usm_ndarray, shape: tuple or list) -> usm_ndarray

    Broadcast an array to a new shape; returns the broadcasted
    array as a view.
    """
    if not isinstance(X, dpt.usm_ndarray):
        raise TypeError(f"Expected usm_ndarray type, got {type(X)}.")

    # Use numpy.broadcast_to to check the validity of the input
    # parametr 'shape'. Raise ValueError if 'X' is not compatible
    # with 'shape' according to NumPy's broadcasting rules.
    new_array = np.broadcast_to(
        np.broadcast_to(np.empty(tuple(), dtype="u1"), X.shape), shape
    )
    new_sts = _broadcast_strides(X.shape, X.strides, new_array.ndim)
    return dpt.usm_ndarray(
        shape=new_array.shape,
        dtype=X.dtype,
        buffer=X,
        strides=new_sts,
        offset=X.__sycl_usm_array_interface__.get("offset", 0),
    )


def broadcast_arrays(*args):
    """
    broadcast_arrays(*args: usm_ndarrays) -> list of usm_ndarrays

    Broadcasts one or more usm_ndarrays against one another.
    """
    for X in args:
        if not isinstance(X, dpt.usm_ndarray):
            raise TypeError(f"Expected usm_ndarray type, got {type(X)}.")

    shape = _broadcast_shapes(*args)

    if all(X.shape == shape for X in args):
        return args

    return [broadcast_to(X, shape) for X in args]


def flip(X, axes=None):
    """
    flip(X: usm_ndarray, axes: int or tuple or list) -> usm_ndarray

    Reverses the order of elements in an array along the given axis.
    The shape of the array is preserved, but the elements are reordered;
    returns a view of X with the entries of axis reversed.
    """
    if not isinstance(X, dpt.usm_ndarray):
        raise TypeError(f"Expected usm_ndarray type, got {type(X)}.")
    X_ndim = X.ndim
    if axes is None:
        indexer = (np.s_[::-1],) * X_ndim
    else:
        axes = normalize_axis_tuple(axes, X_ndim)
        indexer = tuple(
            np.s_[::-1] if i in axes else np.s_[:] for i in range(X.ndim)
        )
    return X[indexer]


def roll(X, shift, axes=None):
    """
    roll(X: usm_ndarray, shift: int or tuple or list,\
         axes: int or tuple or list) -> usm_ndarray

    Rolls array elements along a specified axis.
    Array elements that roll beyond the last position are re-introduced
    at the first position. Array elements that roll beyond the first position
    are re-introduced at the last position.
    returns an output array having the same data type as X and whose elements,
    relative to X, are shifted.
    """
    if not isinstance(X, dpt.usm_ndarray):
        raise TypeError(f"Expected usm_ndarray type, got {type(X)}.")
    if axes is None:
        res = dpt.empty(
            X.shape, dtype=X.dtype, usm_type=X.usm_type, sycl_queue=X.sycl_queue
        )
        hev, _ = ti._copy_usm_ndarray_for_reshape(
            src=X, dst=res, shift=shift, sycl_queue=X.sycl_queue
        )
        hev.wait()
        return res
    axes = normalize_axis_tuple(axes, X.ndim, allow_duplicate=True)
    broadcasted = np.broadcast(shift, axes)
    if broadcasted.ndim > 1:
        raise ValueError("'shift' and 'axis' should be scalars or 1D sequences")
    shifts = {ax: 0 for ax in range(X.ndim)}
    for sh, ax in broadcasted:
        shifts[ax] += sh
    rolls = [((np.s_[:], np.s_[:]),)] * X.ndim
    for ax, offset in shifts.items():
        offset %= X.shape[ax] or 1
        if offset:
            # (original, result), (original, result)
            rolls[ax] = (
                (np.s_[:-offset], np.s_[offset:]),
                (np.s_[-offset:], np.s_[:offset]),
            )

    res = dpt.empty(
        X.shape, dtype=X.dtype, usm_type=X.usm_type, sycl_queue=X.sycl_queue
    )
    hev_list = []
    for indices in product(*rolls):
        arr_index, res_index = zip(*indices)
        hev, _ = ti._copy_usm_ndarray_into_usm_ndarray(
            src=X[arr_index], dst=res[res_index], sycl_queue=X.sycl_queue
        )
        hev_list.append(hev)

    dpctl.SyclEvent.wait_for(hev_list)
    return res


def _arrays_validation(arrays):
    n = len(arrays)
    if n == 0:
        raise TypeError("Missing 1 required positional argument: 'arrays'")

    if not isinstance(arrays, (list, tuple)):
        raise TypeError(f"Expected tuple or list type, got {type(arrays)}.")

    for X in arrays:
        if not isinstance(X, dpt.usm_ndarray):
            raise TypeError(f"Expected usm_ndarray type, got {type(X)}.")

    exec_q = dputils.get_execution_queue([X.sycl_queue for X in arrays])
    if exec_q is None:
        raise ValueError("All the input arrays must have same sycl queue")

    res_usm_type = dputils.get_coerced_usm_type([X.usm_type for X in arrays])
    if res_usm_type is None:
        raise ValueError("All the input arrays must have usm_type")

    X0 = arrays[0]
    _supported_dtype(Xi.dtype for Xi in arrays)

    res_dtype = X0.dtype
    for i in range(1, n):
        res_dtype = np.promote_types(res_dtype, arrays[i])

    for i in range(1, n):
        if X0.ndim != arrays[i].ndim:
            raise ValueError(
                "All the input arrays must have same number of dimensions, "
                f"but the array at index 0 has {X0.ndim} dimension(s) and the "
                f"array at index {i} has {arrays[i].ndim} dimension(s)"
            )
    return res_dtype, res_usm_type, exec_q


def _check_same_shapes(X0_shape, axis, n, arrays):
    for i in range(1, n):
        Xi_shape = arrays[i].shape
        for j, X0j in enumerate(X0_shape):
            if X0j != Xi_shape[j] and j != axis:
                raise ValueError(
                    "All the input array dimensions for the concatenation "
                    f"axis must match exactly, but along dimension {j}, the "
                    f"array at index 0 has size {X0j} and the array "
                    f"at index {i} has size {Xi_shape[j]}"
                )


def concat(arrays, axis=0):
    """
    concat(arrays: tuple or list of usm_ndarrays, axis: int) -> usm_ndarray

    Joins a sequence of arrays along an existing axis.
    """
    res_dtype, res_usm_type, exec_q = _arrays_validation(arrays)

    n = len(arrays)
    X0 = arrays[0]

    axis = normalize_axis_index(axis, X0.ndim)
    X0_shape = X0.shape
    _check_same_shapes(X0_shape, axis, n, arrays)

    res_shape_axis = 0
    for X in arrays:
        res_shape_axis = res_shape_axis + X.shape[axis]

    res_shape = tuple(
        X0_shape[i] if i != axis else res_shape_axis for i in range(X0.ndim)
    )

    res = dpt.empty(
        res_shape, dtype=res_dtype, usm_type=res_usm_type, sycl_queue=exec_q
    )

    hev_list = []
    fill_start = 0
    for i in range(n):
        fill_end = fill_start + arrays[i].shape[axis]
        c_shapes_copy = tuple(
            np.s_[fill_start:fill_end] if j == axis else np.s_[:]
            for j in range(X0.ndim)
        )
        hev, _ = ti._copy_usm_ndarray_into_usm_ndarray(
            src=arrays[i], dst=res[c_shapes_copy], sycl_queue=exec_q
        )
        fill_start = fill_end
        hev_list.append(hev)

    dpctl.SyclEvent.wait_for(hev_list)

    return res


def stack(arrays, axis=0):
    """
    stack(arrays: tuple or list of usm_ndarrays, axis: int) -> usm_ndarray

    Joins a sequence of arrays along a new axis.
    """
    res_dtype, res_usm_type, exec_q = _arrays_validation(arrays)

    n = len(arrays)
    X0 = arrays[0]
    res_ndim = X0.ndim + 1
    axis = normalize_axis_index(axis, res_ndim)
    X0_shape = X0.shape

    for i in range(1, n):
        if X0_shape != arrays[i].shape:
            raise ValueError("All input arrays must have the same shape")

    res_shape = tuple(
        X0_shape[i - 1 * (i >= axis)] if i != axis else n
        for i in range(res_ndim)
    )

    res = dpt.empty(
        res_shape, dtype=res_dtype, usm_type=res_usm_type, sycl_queue=exec_q
    )

    hev_list = []
    for i in range(n):
        c_shapes_copy = tuple(
            i if j == axis else np.s_[:] for j in range(res_ndim)
        )
        hev, _ = ti._copy_usm_ndarray_into_usm_ndarray(
            src=arrays[i], dst=res[c_shapes_copy], sycl_queue=exec_q
        )
        hev_list.append(hev)

    dpctl.SyclEvent.wait_for(hev_list)

    return res


def can_cast(from_, to, casting="safe"):
    """
    can_cast(from: usm_ndarray or dtype, to: dtype) -> bool

    Determines if one data type can be cast to another data type according \
        to Type Promotion Rules rules.
    """
    if isinstance(to, dpt.usm_ndarray):
        raise TypeError("Expected dtype type.")

    dtype_to = dpt.dtype(to)

    dtype_from = (
        from_.dtype if isinstance(from_, dpt.usm_ndarray) else dpt.dtype(from_)
    )

    _supported_dtype([dtype_from, dtype_to])

    return np.can_cast(dtype_from, dtype_to, casting)


def result_type(*arrays_and_dtypes):
    """
    result_type(arrays_and_dtypes: an arbitrary number usm_ndarrays or dtypes)\
         -> dtype

    Returns the dtype that results from applying the Type Promotion Rules to \
        the arguments.
    """
    dtypes = [
        X.dtype if isinstance(X, dpt.usm_ndarray) else dpt.dtype(X)
        for X in arrays_and_dtypes
    ]

    _supported_dtype(dtypes)

    return np.result_type(*dtypes)


def iinfo(dtype):
    """
    iinfo(dtype: integer data-type) -> iinfo_object

    Returns machine limits for integer data types.
    """
    if isinstance(dtype, dpt.usm_ndarray):
        raise TypeError("Expected dtype type, got {to}.")
    _supported_dtype([dpt.dtype(dtype)])
    return np.iinfo(dtype)


def finfo(dtype):
    """
    finfo(type: float data-type) -> finfo_object

    Returns machine limits for float data types.
    """
    if isinstance(dtype, dpt.usm_ndarray):
        raise TypeError("Expected dtype type, got {to}.")
    _supported_dtype([dpt.dtype(dtype)])
    return np.finfo(dtype)


def _supported_dtype(dtypes):
    for dtype in dtypes:
        if dtype.char not in "?bBhHiIlLqQefdFD":
            raise ValueError(f"Dpctl doesn't support dtype {dtype}.")
    return True
