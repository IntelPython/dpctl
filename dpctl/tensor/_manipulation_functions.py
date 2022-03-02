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


import numpy as np

import dpctl.tensor as dpt


def _check_value_of_axes(axes):
    check_array = np.zeros(len(axes))
    for i in axes:
        ii = i.__index__()
        if ii < 0 or ii > len(axes) or check_array[ii] != 0:
            return False
        check_array[ii] = 1
    del check_array
    return True


def permute_dims(X, axes):
    """
    permute_dims(X: usm_ndarray, axes: tuple or list) -> usm_ndarray

    Permutes the axes(dimensions) of the given usm_ndarray. Returns a view
    containing the axes permutation.
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
    if not _check_value_of_axes(axes):
        raise ValueError(
            "The values of the axes must be in the range "
            f"from 0 to {X.ndim} and have no duplicates."
        )
    newstrides = [X.strides[i] for i in axes]
    newshape = [X.shape[i] for i in axes]
    return dpt.usm_ndarray(
        shape=tuple(newshape),
        dtype=X.dtype,
        buffer=X,
        strides=tuple(newstrides),
        offset=X.__sycl_usm_array_interface__.get("offset", 0),
    )
