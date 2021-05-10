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


# these typenum values are aligned to values in NumPy
cdef int UAR_BOOL = 0
cdef int UAR_BYTE = 1
cdef int UAR_UBYTE = 2
cdef int UAR_SHORT = 3
cdef int UAR_USHORT = 4
cdef int UAR_INT = 5
cdef int UAR_UINT = 6
cdef int UAR_LONG = 7
cdef int UAR_ULONG = 8
cdef int UAR_LONGLONG = 9
cdef int UAR_ULONGLONG = 10
cdef int UAR_FLOAT = 11
cdef int UAR_DOUBLE = 12
cdef int UAR_CFLOAT = 14
cdef int UAR_CDOUBLE = 15
cdef int UAR_TYPE_SENTINEL = 17
cdef int UAR_HALF = 23

cdef str _make_typestr(int typenum):
    """
    Make typestring from type number
    """
    cdef type_to_str = ['|b1', '|i1', '|u1', '|i2', '|u2',
                        '|i4', '|u4', '', '', '|i8', '|u8',
                        '|f4', '|f8', '', '|c8', '|c16', '']

    if (typenum < 0):
        return ""
    if (typenum > 16):
        if (typenum == 23):
            return "|f2"
        return ""

    return type_to_str[typenum]


cdef int type_bytesize(int typenum):
    """
    NPY_BOOL=0         : 1
    NPY_BYTE=1         : 1
    NPY_UBYTE=2        : 1
    NPY_SHORT=3        : 2
    NPY_USHORT=4       : 2
    NPY_INT=5          : 4
    NPY_UINT=6         : 4
    NPY_LONG=7         :
    NPY_ULONG=8        :
    NPY_LONGLONG=9     : 8
    NPY_ULONGLONG=10   : 8
    NPY_FLOAT=11       : 4
    NPY_DOUBLE=12      : 8
    NPY_LONGDOUBLE=13  : N/A
    NPY_CFLOAT=14      : 8
    NPY_CDOUBLE=15     : 16
    NPY_CLONGDOUBLE=16 : N/A
    NPY_HALF=23        : 2
    """
    cdef int *type_to_bytesize = [
        1, 1, 1, 2, 2, 4, 4, 8, 8, 8, 8, 4, 8, -1, 8, 16, -1]

    if typenum < 0:
        return -1
    if typenum > 16:
        if typenum == 23:
            return 2
        return -1

    return type_to_bytesize[typenum]


cdef int typenum_from_format(str s) except *:
    """
    Internal utility to convert string describing type format

    Format is [<|=>][biufc]#
    Shortcuts for formats are i, u, d, D
    """
    if not s:
        raise TypeError("Format string '" + s + "' cannot be empty.")
    try:
        dt = np.dtype(s)
    except Exception as e:
        raise TypeError("Format '" + s + "' is not understood.") from e
    if (dt.byteorder == ">"):
        raise TypeError("Format '" + s + "' can only have native byteorder.")
    return dt.num


cdef int dtype_to_typenum(dtype) except *:
    if isinstance(dtype, str):
        return typenum_from_format(dtype)
    elif isinstance(dtype, bytes):
        return typenum_from_format(dtype.decode("UTF-8"))
    elif hasattr(dtype, 'descr'):
        obj = getattr(dtype, 'descr')
        if (not isinstance(obj, list) or len(obj) != 1):
            return -1
        obj = obj[0]
        if (not isinstance(obj, tuple) or len(obj) != 2 or obj[0]):
            return -1
        obj = obj[1]
        if not isinstance(obj, str):
            return -1
        return typenum_from_format(obj)
    else:
        return -1
