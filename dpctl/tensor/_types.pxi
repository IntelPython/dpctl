#                       Data Parallel Control (dpctl)
#
#  Copyright 2020-2025 Intel Corporation
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

# these typenum values are aligned to values in NumPy
cdef:
    int UAR_BOOL = 0  # pragma: no cover
    int UAR_BYTE = 1  # pragma: no cover
    int UAR_UBYTE = 2  # pragma: no cover
    int UAR_SHORT = 3  # pragma: no cover
    int UAR_USHORT = 4  # pragma: no cover
    int UAR_INT = 5  # pragma: no cover
    int UAR_UINT = 6  # pragma: no cover
    int UAR_LONG = 7  # pragma: no cover
    int UAR_ULONG = 8  # pragma: no cover
    int UAR_LONGLONG = 9  # pragma: no cover
    int UAR_ULONGLONG = 10  # pragma: no cover
    int UAR_FLOAT = 11  # pragma: no cover
    int UAR_DOUBLE = 12  # pragma: no cover
    int UAR_CFLOAT = 14  # pragma: no cover
    int UAR_CDOUBLE = 15  # pragma: no cover
    int UAR_TYPE_SENTINEL = 17  # pragma: no cover
    int UAR_HALF = 23  # pragma: no cover

cdef int type_bytesize(int typenum):
    """
    NPY_BOOL=0         : 1
    NPY_BYTE=1         : 1
    NPY_UBYTE=2        : 1
    NPY_SHORT=3        : 2
    NPY_USHORT=4       : 2
    NPY_INT=5          : sizeof(int)
    NPY_UINT=6         : sizeof(unsigned int)
    NPY_LONG=7         : sizeof(long)
    NPY_ULONG=8        : sizeof(unsigned long)
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
        1,
        sizeof(char),
        sizeof(unsigned char),
        sizeof(short),
        sizeof(unsigned short),
        sizeof(int),
        sizeof(unsigned int),
        sizeof(long),
        sizeof(unsigned long),
        sizeof(long long),
        sizeof(unsigned long long),
        sizeof(float),
        sizeof(double), -1,
        sizeof(float complex),
        sizeof(double complex), -1]

    if typenum < 0:  # pragma: no cover
        return -1
    if typenum > 16:
        if typenum == 23:
            return 2
        return -1

    return type_to_bytesize[typenum]


cdef str _make_typestr(int typenum):
    """
    Make typestring from type number
    """
    cdef type_to_str = ["|b", "|i", "|u", "|i", "|u",
                        "|i", "|u", "|i", "|u", "|i", "|u",
                        "|f", "|f", "", "|c", "|c", ""]

    if (typenum < 0):  # pragma: no cover
        return ""
    if (typenum > 16):
        if (typenum == 23):
            return "|f2"
        return ""  # pragma: no cover

    return type_to_str[typenum] + str(type_bytesize(typenum))


cdef int typenum_from_format(str s):
    """
    Internal utility to convert string describing type format

    Format is [<|=>][biufc]#
    Shortcuts for formats are i, u, d, D
    """
    if not s:
        return -1
    try:
        dt = np.dtype(s)
    except Exception:
        return -1
    if (dt.byteorder == ">"):
        return -2
    return dt.num


cdef int descr_to_typenum(object dtype):
    """
    Returns typenum for argumentd dtype that has attribute descr,
    assumed numpy.dtype
    """
    obj = getattr(dtype, "descr")
    if (not isinstance(obj, list) or len(obj) != 1):
        return -1    # token for ValueError
    obj = obj[0]
    if (
        not isinstance(obj, tuple) or len(obj) != 2 or obj[0]
    ):  # pragma: no cover
        return -1
    obj = obj[1]
    if not isinstance(obj, str):  # pragma: no cover
        return -1
    return typenum_from_format(obj)


cdef int dtype_to_typenum(dtype):
    if isinstance(dtype, str):
        return typenum_from_format(dtype)
    elif isinstance(dtype, bytes):
        return typenum_from_format(dtype.decode("UTF-8"))
    elif hasattr(dtype, "descr"):
        return descr_to_typenum(dtype)
    else:
        try:
            dt = np.dtype(dtype)
        except TypeError:
            return -3
        except Exception:  # pragma: no cover
            return -1
        if hasattr(dt, "descr"):
            return descr_to_typenum(dt)
        else:  # pragma: no cover
            return -3  # token for TypeError
