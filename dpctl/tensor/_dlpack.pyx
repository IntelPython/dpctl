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

cimport cpython
from libc cimport stdlib
from libc.stdint cimport int32_t, int64_t, uint8_t, uint16_t, uint64_t

cimport dpctl as c_dpctl
cimport dpctl.memory as c_dpmem

from .._backend cimport (
    DPCTLDevice_Delete,
    DPCTLDevice_GetParentDevice,
    DPCTLSyclDeviceRef,
    DPCTLSyclUSMRef,
)
from ._usmarray cimport usm_ndarray

import numpy as np

import dpctl


cdef extern from './include/dlpack/dlpack.h' nogil:
    cdef int DLPACK_VERSION

    cdef enum DLDeviceType:
        kDLCPU
        kDLCUDA
        kDLCUDAHost
        kDLCUDAManaged
        kDLROCM
        kDLROCMHost
        kDLOpenCL
        kDLVulkan
        kDLMetal
        kDLVPI
        kDLOneAPI

    ctypedef struct DLDevice:
        DLDeviceType device_type
        int device_id

    cdef enum DLDataTypeCode:
        kDLInt
        kDLUInt
        kDLFloat
        kDLBfloat
        kDLComplex

    ctypedef struct DLDataType:
        uint8_t code
        uint8_t bits
        uint16_t lanes

    ctypedef struct DLTensor:
        void* data
        DLDevice device
        int ndim
        DLDataType dtype
        int64_t* shape
        int64_t* strides
        uint64_t byte_offset

    ctypedef struct DLManagedTensor:
        DLTensor dl_tensor
        void* manager_ctx
        void (*deleter)(DLManagedTensor*)  # noqa: E211


def get_build_dlpack_version():
    return str(DLPACK_VERSION)


cdef void pycapsule_deleter(object dlt_capsule):
    cdef DLManagedTensor *dlm_tensor = NULL
    if cpython.PyCapsule_IsValid(dlt_capsule, 'dltensor'):
        dlm_tensor = <DLManagedTensor*>cpython.PyCapsule_GetPointer(
            dlt_capsule, 'dltensor')
        dlm_tensor.deleter(dlm_tensor)
    elif cpython.PyCapsule_IsValid(dlt_capsule, 'used_dltensor'):
        dlm_tensor = <DLManagedTensor*>cpython.PyCapsule_GetPointer(
            dlt_capsule, 'used_dltensor')
        dlm_tensor.deleter(dlm_tensor)


cdef void managed_tensor_deleter(DLManagedTensor *dlm_tensor) with gil:
    if dlm_tensor is not NULL:
        stdlib.free(dlm_tensor.dl_tensor.shape)
        cpython.Py_DECREF(<usm_ndarray>dlm_tensor.manager_ctx)
        dlm_tensor.manager_ctx = NULL
        stdlib.free(dlm_tensor)


cdef class DLPackCreationError(Exception):
    """
    A DLPackCreateError exception is raised when constructing
    DLPack capsule from `usm_ndarray` based on a USM allocation
    on a partitioned SYCL device.
    """
    pass


cpdef to_dlpack_capsule(usm_ndarray usm_ary) except+:
    """Constructs named Python capsule object referencing
    instance of `DLManagerTensor` from `usm_ndarray` instance"""
    cdef c_dpctl.SyclQueue ary_sycl_queue
    cdef c_dpctl.SyclDevice ary_sycl_device
    cdef DPCTLSyclDeviceRef pDRef = NULL
    cdef DLManagedTensor *dlm_tensor = NULL
    cdef DLTensor* dl_tensor = NULL
    cdef int nd = usm_ary.get_ndim()
    cdef char* data_ptr = usm_ary.get_data()
    cdef Py_ssize_t *shape_ptr = NULL
    cdef Py_ssize_t *strides_ptr = NULL
    cdef int64_t *shape_strides_ptr = NULL
    cdef int i = 0
    cdef int device_id = -1

    ary_sycl_queue = usm_ary.get_sycl_queue()
    ary_sycl_device = ary_sycl_queue.get_sycl_device()

    # check that ary_sycl_device is a non-partitioned device
    pDRef = DPCTLDevice_GetParentDevice(ary_sycl_device.get_device_ref())
    if pDRef is not NULL:
        DPCTLDevice_Delete(pDRef)
        raise DLPackCreationError(
            "to_dlpack_capsule: DLPack can only export arrays allocated on "
            "non-partitioned SYCL devices."
        )

    dlm_tensor = <DLManagedTensor *> stdlib.malloc(
        sizeof(DLManagedTensor))
    if dlm_tensor is NULL:
        raise MemoryError(
            "to_dlpack_capsule: Could not allocate memory for DLManagedTensor"
        )
    shape_strides_ptr = <int64_t *>stdlib.malloc((sizeof(int64_t) * 2) * nd)
    if shape_strides_ptr is NULL:
        stdlib.free(dlm_tensor)
        raise MemoryError(
            "to_dlpack_capsule: Could not allocate memory for shape/strides"
        )
    shape_ptr = usm_ary.get_shape()
    for i in range(nd):
        shape_strides_ptr[i] = shape_ptr[i]
    strides_ptr = usm_ary.get_strides()
    if strides_ptr:
        for i in range(nd):
            shape_strides_ptr[nd + i] = strides_ptr[i]

    device_id = ary_sycl_device.get_overall_ordinal()
    if device_id < 0:
        stdlib.free(shape_strides_ptr)
        stdlib.free(dlm_tensor)
        raise DLPackCreationError(
            "to_dlpack_capsule: failed to determine device_id"
        )

    ary_dt = usm_ary.dtype
    ary_dtk = ary_dt.kind

    dl_tensor = &dlm_tensor.dl_tensor
    dl_tensor.data = <void*>data_ptr
    dl_tensor.ndim = nd
    dl_tensor.byte_offset = <uint64_t>0
    dl_tensor.shape = &shape_strides_ptr[0]
    if strides_ptr is NULL:
        dl_tensor.strides = NULL
    else:
        dl_tensor.strides = &shape_strides_ptr[nd]
    dl_tensor.device.device_type = kDLOneAPI
    dl_tensor.device.device_id = device_id
    dl_tensor.dtype.lanes = <uint16_t>1
    dl_tensor.dtype.bits = <uint8_t>(ary_dt.itemsize * 8)
    if (ary_dtk == "b"):
        dl_tensor.dtype.code = <uint8_t>kDLUInt
    elif (ary_dtk == "u"):
        dl_tensor.dtype.code = <uint8_t>kDLUInt
    elif (ary_dtk == "i"):
        dl_tensor.dtype.code = <uint8_t>kDLInt
    elif (ary_dtk == "f"):
        dl_tensor.dtype.code = <uint8_t>kDLFloat
    elif (ary_dtk == "c"):
        dl_tensor.dtype.code = <uint8_t>kDLComplex
    else:
        stdlib.free(shape_strides_ptr)
        stdlib.free(dlm_tensor)
        raise ValueError("Unrecognized array data type")

    dlm_tensor.manager_ctx = <void*>usm_ary
    cpython.Py_INCREF(usm_ary)
    dlm_tensor.deleter = managed_tensor_deleter

    return cpython.PyCapsule_New(dlm_tensor, 'dltensor', pycapsule_deleter)


cpdef usm_ndarray from_dlpack_capsule(object py_caps) except +:
    """Reconstructs instance of usm_ndarray from named Python
    capsule object referencing instance of `DLManagedTensor` without
    a copy"""
    cdef DLManagedTensor *dlm_tensor = NULL
    cdef bytes usm_type
    cdef size_t sz = 1
    cdef int i
    cdef int element_bytesize = 0

    if not cpython.PyCapsule_IsValid(py_caps, 'dltensor'):
        if cpython.PyCapsule_IsValid(py_caps, 'used_dltensor'):
            raise ValueError(
                "A DLPack tensor object can not be consumed multiple times"
            )
        else:
            raise TypeError(
                f"A Python 'dltensor' capsule was expected, "
                "got {type(dlm_tensor)}"
            )
    dlm_tensor = <DLManagedTensor*>cpython.PyCapsule_GetPointer(
            py_caps, "dltensor")
    # Verify that we can work with this device
    if dlm_tensor.dl_tensor.device.device_type == kDLOneAPI:
        q = dpctl.SyclQueue(str(<int>dlm_tensor.dl_tensor.device.device_id))
        usm_type = c_dpmem._Memory.get_pointer_type(
            <DPCTLSyclUSMRef> dlm_tensor.dl_tensor.data,
            <c_dpctl.SyclContext>q.sycl_context)
        if usm_type == b"unknown":
            raise ValueError(
                f"Data pointer in DLPack is not bound to default sycl "
                "context of device '{device_id}', translated to "
                "{q.sycl_device.filter_string}"
            )
        if dlm_tensor.dl_tensor.dtype.bits % 8:
            raise ValueError(
                "Can not import DLPack tensor whose element's "
                "bitsize is not a multiple of 8"
            )
        if dlm_tensor.dl_tensor.dtype.lanes != 1:
            raise ValueError(
                "Can not import DLPack tensor with lanes != 1"
            )
        for i in range(dlm_tensor.dl_tensor.ndim):
            sz = sz * dlm_tensor.dl_tensor.shape[i]

        element_bytesize = (dlm_tensor.dl_tensor.dtype.bits // 8)
        sz = sz * element_bytesize
        usm_mem = c_dpmem._Memory.create_from_usm_pointer_size_qref(
            <DPCTLSyclUSMRef> dlm_tensor.dl_tensor.data,
            sz,
            (<c_dpctl.SyclQueue>q).get_queue_ref(),
            memory_owner=py_caps
        )
        py_shape = list()
        for i in range(dlm_tensor.dl_tensor.ndim):
            py_shape.append(dlm_tensor.dl_tensor.shape[i])
        if (dlm_tensor.dl_tensor.strides is NULL):
            py_strides = None
        else:
            py_strides = list()
            for i in range(dlm_tensor.dl_tensor.ndim):
                py_strides.append(dlm_tensor.dl_tensor.strides[i])
        if (dlm_tensor.dl_tensor.dtype.code == kDLUInt):
            ary_dt = np.dtype("u" + str(element_bytesize))
        elif (dlm_tensor.dl_tensor.dtype.code == kDLInt):
            ary_dt = np.dtype("i" + str(element_bytesize))
        elif (dlm_tensor.dl_tensor.dtype.code == kDLFloat):
            ary_dt = np.dtype("f" + str(element_bytesize))
        elif (dlm_tensor.dl_tensor.dtype.code == kDLComplex):
            ary_dt = np.dtype("c" + str(element_bytesize))
        else:
            raise ValueError(
                "Can not import DLPack tensor with type code {}.".format(
                    <object>dlm_tensor.dl_tensor.dtype.code
                    )
            )
        res_ary = usm_ndarray(
            py_shape,
            dtype=ary_dt,
            buffer=usm_mem,
            strides=py_strides
        )
        cpython.PyCapsule_SetName(py_caps, 'used_dltensor')
        return res_ary
    else:
        raise ValueError(
            "The DLPack tensor resides on unsupported device."
        )


cpdef from_dlpack(array):
    """Constructs `usm_ndarray` from a Python object that implements
    `__dlpack__` protocol."""
    pass
