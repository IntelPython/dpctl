#                       Data Parallel Control (dpctl)
#
#  Copyright 2020-2021 Intel Corporation
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


from itertools import chain, repeat

import numpy as np
from numpy.core.numeric import normalize_axis_tuple

import dpctl.tensor as dpt


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
