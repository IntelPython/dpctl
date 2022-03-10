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


from numpy.core.numeric import normalize_axis_tuple

import dpctl.tensor as dpt


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
