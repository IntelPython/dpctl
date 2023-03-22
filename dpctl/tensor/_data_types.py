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

from numpy import dtype

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


def isdtype(dtype_, kind):
    """isdtype(dtype, kind)

    Returns a boolean indicating whether a provided `dtype` is
    of a specified data type `kind`.

    See [array API](array_api) for more information.

    [array_api]: https://data-apis.org/array-api/latest/
    """

    if not isinstance(dtype_, dtype):
        raise TypeError("Expected instance of `dpt.dtype`, got {dtype_}")

    if isinstance(kind, dtype):
        return dtype_ == kind

    elif isinstance(kind, str):
        if kind == "bool":
            return dtype_ == dtype("bool")
        elif kind == "signed integer":
            return dtype_.kind == "i"
        elif kind == "unsigned integer":
            return dtype_.kind == "u"
        elif kind == "integral":
            return dtype_.kind in ("u", "i")
        elif kind == "real floating":
            return dtype_.kind == "f"
        elif kind == "complex floating":
            return dtype_.kind == "c"
        elif kind == "numeric":
            return isdtype(
                dtype_, ("integral", "real floating", "complex floating")
            )
        else:
            raise ValueError(f"Unrecognized data type kind: {kind}")

    elif isinstance(kind, tuple):
        return any(isdtype(dtype_, k) for k in kind)

    else:
        raise TypeError(f"Unsupported data type kind: {kind}")


__all__ = [
    "dtype",
    "isdtype",
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
]
