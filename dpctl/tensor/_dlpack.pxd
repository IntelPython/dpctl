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

# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True

from ._usmarray cimport usm_ndarray


cdef extern from './include/dlpack/dlpack.h' nogil:
    int device_CPU 'kDLCPU'
    int device_oneAPI 'kDLOneAPI'
    int device_OpenCL 'kDLOpenCL'


cpdef object to_dlpack_capsule(usm_ndarray array) except +
cpdef usm_ndarray from_dlpack_capsule(object dltensor) except +

cpdef from_dlpack(array)

cdef class DLPackCreationError(Exception):
    """
    A DLPackCreateError exception is raised when constructing
    DLPack capsule from `usm_ndarray` based on a USM allocation
    on a partitioned SYCL device.
    """
    pass
