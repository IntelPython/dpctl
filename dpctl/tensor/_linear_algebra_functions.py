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

import dpctl.tensor as dpt


def matrix_transpose(x):
    """matrix_transpose(x)

    Transposes the innermost two dimensions of `x`, where `x` is a
    2-dimensional matrix or a stack of 2-dimensional matrices.

    To convert from a 1-dimensional array to a 2-dimensional column
    vector, use x[:, dpt.newaxis].

    Args:
       x (usm_ndarray):
          Input array with shape (..., m, n).

    Returns:
       usm_ndarray:
          Array with shape (..., n, m).
    """

    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(
            "Expected instance of `dpt.usm_ndarray`, got `{}`.".format(type(x))
        )
    if x.ndim < 2:
        raise ValueError(
            "dpctl.tensor.matrix_transpose requires array to have"
            "at least 2 dimensions"
        )

    return x.mT
