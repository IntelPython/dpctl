#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
    **Data Parallel Tensor Collection** is a collection of tensor
    implementations that implement Python data API
    (https://data-apis.github.io/array-api/latest/) standard.

"""

from dpctl.tensor._copy_utils import asnumpy, astype, copy, from_numpy, to_numpy
from dpctl.tensor._ctors import (
    arange,
    asarray,
    empty,
    empty_like,
    eye,
    full,
    full_like,
    linspace,
    ones,
    ones_like,
    zeros,
    zeros_like,
)
from dpctl.tensor._device import Device
from dpctl.tensor._dlpack import from_dlpack
from dpctl.tensor._manipulation_functions import (
    broadcast_arrays,
    broadcast_to,
    concat,
    expand_dims,
    flip,
    permute_dims,
    roll,
    squeeze,
    stack,
)
from dpctl.tensor._reshape import reshape
from dpctl.tensor._usmarray import usm_ndarray

__all__ = [
    "Device",
    "usm_ndarray",
    "arange",
    "asarray",
    "astype",
    "copy",
    "empty",
    "zeros",
    "ones",
    "full",
    "eye",
    "linspace",
    "empty_like",
    "zeros_like",
    "ones_like",
    "full_like",
    "flip",
    "reshape",
    "roll",
    "concat",
    "stack",
    "broadcast_arrays",
    "broadcast_to",
    "expand_dims",
    "permute_dims",
    "squeeze",
    "from_numpy",
    "to_numpy",
    "asnumpy",
    "from_dlpack",
]
