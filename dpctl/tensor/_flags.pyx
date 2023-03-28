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

# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True

from libcpp cimport bool as cpp_bool

from dpctl.tensor._usmarray cimport (
    USM_ARRAY_C_CONTIGUOUS,
    USM_ARRAY_F_CONTIGUOUS,
    USM_ARRAY_WRITABLE,
    usm_ndarray,
)


cdef cpp_bool _check_bit(int flag, int mask):
    return (flag & mask) == mask


cdef class Flags:
    """
    Helper class to represent memory layout flags of
    :class:`dpctl.tensor.usm_ndarray`.
    """
    cdef int flags_
    cdef usm_ndarray arr_

    def __cinit__(self, usm_ndarray arr, int flags):
        self.arr_ = arr
        self.flags_ = flags

    @property
    def flags(self):
        """
        Integer representation of the memory layout flags of
        :class:`dpctl.tensor.usm_ndarray` instance.
        """
        return self.flags_

    @property
    def c_contiguous(self):
        """
        True if the memory layout of the
        :class:`dpctl.tensor.usm_ndarray` instance is C-contiguous.
        """
        return _check_bit(self.flags_, USM_ARRAY_C_CONTIGUOUS)

    @property
    def f_contiguous(self):
        """
        True if the memory layout of the
        :class:`dpctl.tensor.usm_ndarray` instance is F-contiguous.
        """
        return _check_bit(self.flags_, USM_ARRAY_F_CONTIGUOUS)

    @property
    def writable(self):
        """
        True if :class:`dpctl.tensor.usm_ndarray` instance is writable.
        """
        return _check_bit(self.flags_, USM_ARRAY_WRITABLE)

    @writable.setter
    def writable(self, new_val):
        if not isinstance(new_val, bool):
            raise TypeError("Expecting a boolean value")
        self.arr_._set_writable_flag(new_val)

    @property
    def fc(self):
        """
        True if the memory layout of the :class:`dpctl.tensor.usm_ndarray`
        instance is C-contiguous and F-contiguous.
        """
        return (
           _check_bit(self.flags_, USM_ARRAY_C_CONTIGUOUS)
           and _check_bit(self.flags_, USM_ARRAY_F_CONTIGUOUS)
        )

    @property
    def forc(self):
        """
        True if the memory layout of the :class:`dpctl.tensor.usm_ndarray`
        instance is C-contiguous or F-contiguous.
        """
        return (
           _check_bit(self.flags_, USM_ARRAY_C_CONTIGUOUS)
           or _check_bit(self.flags_, USM_ARRAY_F_CONTIGUOUS)
        )

    @property
    def fnc(self):
        """
        True if the memory layout of the :class:`dpctl.tensor.usm_ndarray`
        instance is F-contiguous and not C-contiguous.
        """
        return (
           _check_bit(self.flags_, USM_ARRAY_C_CONTIGUOUS)
           and not _check_bit(self.flags_, USM_ARRAY_F_CONTIGUOUS)
        )

    @property
    def contiguous(self):
        """
        True if the memory layout of the :class:`dpctl.tensor.usm_ndarray`
        instance is C-contiguous and F-contiguous.
        Equivalent to `forc.`
        """
        return self.forc

    def __getitem__(self, name):
        if name in ["C_CONTIGUOUS", "C"]:
            return self.c_contiguous
        elif name in ["F_CONTIGUOUS", "F"]:
            return self.f_contiguous
        elif name in ["WRITABLE", "W"]:
            return self.writable
        elif name == "FC":
            return self.fc
        elif name == "CONTIGUOUS":
            return self.forc

    def __setitem__(self, name, val):
        if name in ["WRITABLE", "W"]:
            self.writable = val
        else:
            raise ValueError(
                "Only writable ('W' or 'WRITABLE') flag can be set"
            )

    def __repr__(self):
        out = []
        for name in "C_CONTIGUOUS", "F_CONTIGUOUS", "WRITABLE":
            out.append("  {} : {}".format(name, self[name]))
        return '\n'.join(out)

    def __eq__(self, other):
        cdef Flags other_
        if isinstance(other, self.__class__):
           other_ = <Flags>other
           return self.flags_ == other_.flags_
        elif isinstance(other, int):
           return self.flags_ == <int>other
        else:
           return False
