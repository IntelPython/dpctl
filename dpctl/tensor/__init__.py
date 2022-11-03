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

from numpy import dtype

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
    meshgrid,
    ones,
    ones_like,
    tril,
    triu,
    zeros,
    zeros_like,
)
from dpctl.tensor._device import Device
from dpctl.tensor._dlpack import from_dlpack
from dpctl.tensor._manipulation_functions import (
    broadcast_arrays,
    broadcast_to,
    can_cast,
    concat,
    expand_dims,
    finfo,
    flip,
    iinfo,
    permute_dims,
    result_type,
    roll,
    squeeze,
    stack,
)
from dpctl.tensor._reshape import reshape
from dpctl.tensor._usmarray import usm_ndarray

bool = dtype("bool")
int8 = dtype("int8")
int16 = dtype("int16")
int32 = dtype("int32")
int64 = dtype("int64")
uint8 = dtype("uint8")
uint16 = dtype("uint16")
uint32 = dtype("uint32")
uint64 = dtype("uint64")
float16 = dtype("float16")
float32 = dtype("float32")
float64 = dtype("float64")
complex64 = dtype("complex64")
complex128 = dtype("complex128")

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
    "tril",
    "triu",
    "dtype",
    "bool",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "iinfo",
    "finfo",
    "can_cast",
    "result_type",
    "meshgrid",
]
