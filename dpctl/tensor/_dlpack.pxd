#                       Data Parallel Control (dpctl)
#
#  Copyright 2020-2024 Intel Corporation
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

from .._sycl_device cimport SyclDevice
from ._usmarray cimport usm_ndarray


cdef extern from 'dlpack/dlpack.h' nogil:
    int device_CPU 'kDLCPU'
    int device_CUDA 'kDLCUDA'
    int device_CUDAHost 'kDLCUDAHost'
    int device_CUDAManaged 'kDLCUDAManaged'
    int device_DLROCM 'kDLROCM'
    int device_ROCMHost 'kDLROCMHost'
    int device_OpenCL 'kDLOpenCL'
    int device_Vulkan 'kDLVulkan'
    int device_Metal 'kDLMetal'
    int device_VPI 'kDLVPI'
    int device_OneAPI 'kDLOneAPI'
    int device_WebGPU 'kDLWebGPU'
    int device_Hexagon 'kDLHexagon'
    int device_MAIA 'kDLMAIA'

cpdef object to_dlpack_capsule(usm_ndarray array) except +
cpdef object to_dlpack_versioned_capsule(usm_ndarray array, bint copied) except +
cpdef usm_ndarray from_dlpack_capsule(object dltensor) except +

cdef int get_parent_device_ordinal_id(SyclDevice dev) except *

cdef class DLPackCreationError(Exception):
    """
    A DLPackCreateError exception is raised when constructing
    DLPack capsule from `usm_ndarray` based on a USM allocation
    on a partitioned SYCL device.
    """
    pass
