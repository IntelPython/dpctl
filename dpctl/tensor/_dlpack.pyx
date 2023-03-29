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

cimport cpython
from libc cimport stdlib
from libc.stdint cimport int32_t, int64_t, uint8_t, uint16_t, uint64_t

cimport dpctl as c_dpctl
cimport dpctl.memory as c_dpmem
from dpctl._sycl_queue_manager cimport get_device_cached_queue

from .._backend cimport (
    DPCTLDevice_Delete,
    DPCTLDevice_GetParentDevice,
    DPCTLSyclDeviceRef,
    DPCTLSyclUSMRef,
)
from ._usmarray cimport usm_ndarray

from platform import system as sys_platform

import numpy as np

import dpctl
import dpctl.memory as dpmem


cdef bint _IS_LINUX = sys_platform() == "Linux"

del sys_platform

cdef extern from 'dlpack/dlpack.h' nogil:
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
        kDLBool

    ctypedef struct DLDataType:
        uint8_t code
        uint8_t bits
        uint16_t lanes

    ctypedef struct DLTensor:
        void *data
        DLDevice device
        int ndim
        DLDataType dtype
        int64_t *shape
        int64_t *strides
        uint64_t byte_offset

    ctypedef struct DLManagedTensor:
        DLTensor dl_tensor
        void *manager_ctx
        void (*deleter)(DLManagedTensor *)  # noqa: E211


def get_build_dlpack_version():
    """
    Returns the string value of DLPACK_VERSION from dlpack.h
    :module:`dpctl.tensor` was built with.

    Returns:
        A string value of the version of DLPack used to build
        `dpctl`.
    """
    return str(DLPACK_VERSION)


cdef void _pycapsule_deleter(object dlt_capsule):
    cdef DLManagedTensor *dlm_tensor = NULL
    if cpython.PyCapsule_IsValid(dlt_capsule, 'dltensor'):
        dlm_tensor = <DLManagedTensor*>cpython.PyCapsule_GetPointer(
            dlt_capsule, 'dltensor')
        dlm_tensor.deleter(dlm_tensor)


cdef void _managed_tensor_deleter(DLManagedTensor *dlm_tensor) with gil:
    if dlm_tensor is not NULL:
        stdlib.free(dlm_tensor.dl_tensor.shape)
        cpython.Py_DECREF(<usm_ndarray>dlm_tensor.manager_ctx)
        dlm_tensor.manager_ctx = NULL
        stdlib.free(dlm_tensor)


cpdef to_dlpack_capsule(usm_ndarray usm_ary) except+:
    """
    to_dlpack_capsule(usm_ary)

    Constructs named Python capsule object referencing
    instance of `DLManagerTensor` from
    :class:`dpctl.tensor.usm_ndarray` instance.

    Args:
        usm_ary: An instance of :class:`dpctl.tensor.usm_ndarray`
    Returns:
        Python a new capsule with name "dltensor" that contains
        a pointer to `DLManagedTensor` struct.
    Raises:
        DLPackCreationError: when array can be represented as
            DLPack tensor. This may happen when array was allocated
            on a partitioned sycl device, or its USM allocation is
            not bound to the platform default SYCL context.
        MemoryError: when host allocation to needed for `DLManagedTensor`
            did not succeed.
        ValueError: when array elements data type could not be represented
            in `DLManagedTensor`.
    """
    cdef c_dpctl.SyclQueue ary_sycl_queue
    cdef c_dpctl.SyclDevice ary_sycl_device
    cdef DPCTLSyclDeviceRef pDRef = NULL
    cdef DPCTLSyclDeviceRef tDRef = NULL
    cdef DLManagedTensor *dlm_tensor = NULL
    cdef DLTensor *dl_tensor = NULL
    cdef int nd = usm_ary.get_ndim()
    cdef char *data_ptr = usm_ary.get_data()
    cdef Py_ssize_t *shape_ptr = NULL
    cdef Py_ssize_t *strides_ptr = NULL
    cdef int64_t *shape_strides_ptr = NULL
    cdef int i = 0
    cdef int device_id = -1
    cdef char *base_ptr = NULL
    cdef Py_ssize_t element_offset = 0
    cdef Py_ssize_t byte_offset = 0

    ary_base = usm_ary.get_base()
    ary_sycl_queue = usm_ary.get_sycl_queue()
    ary_sycl_device = ary_sycl_queue.get_sycl_device()

    try:
        if _IS_LINUX:
            default_context = ary_sycl_device.sycl_platform.default_context
        else:
            default_context = None
    except RuntimeError:
        # RT does not support default_context, e.g. Windows
        default_context = None
    if default_context is None:
        # check that ary_sycl_device is a non-partitioned device
        pDRef = DPCTLDevice_GetParentDevice(ary_sycl_device.get_device_ref())
        if pDRef is not NULL:
            DPCTLDevice_Delete(pDRef)
            raise DLPackCreationError(
                "to_dlpack_capsule: DLPack can only export arrays allocated "
                "on non-partitioned SYCL devices on platforms where "
                "default_context oneAPI extension is not supported."
            )
    else:
        if not usm_ary.sycl_context == default_context:
            raise DLPackCreationError(
                "to_dlpack_capsule: DLPack can only export arrays based on USM "
                "allocations bound to a default platform SYCL context"
            )
        # Find the unpartitioned parent of the allocation device
        pDRef = DPCTLDevice_GetParentDevice(ary_sycl_device.get_device_ref())
        if pDRef is not NULL:
            tDRef = DPCTLDevice_GetParentDevice(pDRef)
            while tDRef is not NULL:
                DPCTLDevice_Delete(pDRef)
                pDRef = tDRef
                tDRef = DPCTLDevice_GetParentDevice(pDRef)
            ary_sycl_device = c_dpctl.SyclDevice._create(pDRef)

    # Find ordinal number of the parent device
    device_id = ary_sycl_device.get_overall_ordinal()
    if device_id < 0:
        raise DLPackCreationError(
            "to_dlpack_capsule: failed to determine device_id"
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

    ary_dt = usm_ary.dtype
    ary_dtk = ary_dt.kind
    element_offset = usm_ary.get_offset()
    byte_offset = element_offset * (<Py_ssize_t>ary_dt.itemsize)

    dl_tensor = &dlm_tensor.dl_tensor
    dl_tensor.data = <void*>(data_ptr - byte_offset)
    dl_tensor.ndim = nd
    dl_tensor.byte_offset = <uint64_t>byte_offset
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
        dl_tensor.dtype.code = <uint8_t>kDLBool
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
    dlm_tensor.deleter = _managed_tensor_deleter

    return cpython.PyCapsule_New(dlm_tensor, 'dltensor', _pycapsule_deleter)


cdef class _DLManagedTensorOwner:
    """
    Helper class managing the lifetime of the DLManagedTensor struct
    transferred from a 'dlpack' capsule.
    """
    cdef DLManagedTensor *dlm_tensor

    def __cinit__(self):
        self.dlm_tensor = NULL

    def __dealloc__(self):
        if self.dlm_tensor:
            self.dlm_tensor.deleter(self.dlm_tensor)

    @staticmethod
    cdef _DLManagedTensorOwner _create(DLManagedTensor *dlm_tensor_src):
        cdef _DLManagedTensorOwner res = _DLManagedTensorOwner.__new__(_DLManagedTensorOwner)
        res.dlm_tensor = dlm_tensor_src
        return res


cpdef usm_ndarray from_dlpack_capsule(object py_caps) except +:
    """
    from_dlpack_capsule(caps)

    Reconstructs instance of :class:`dpctl.tensor.usm_ndarray` from
    named Python capsule object referencing instance of `DLManagedTensor`
    without copy. The instance forms a view in the memory of the tensor.

    Args:
        caps: Python capsule with name "dltensor" expected to reference
            an instance of `DLManagedTensor` struct.
    Returns:
        Instance of :class:`dpctl.tensor.usm_ndarray` with a view into
        memory of the tensor. Capsule is renamed to "used_dltensor" upon
        success.
    Raises:
        TypeError: if argument is not a "dltensor" capsule.
        ValueError: if argument is "used_dltensor" capsule
        BufferError:  if the USM pointer is not bound to the reconstructed
             sycl context, or the DLPack's device_type is not supported
             by dpctl.
    """
    cdef DLManagedTensor *dlm_tensor = NULL
    cdef bytes usm_type
    cdef size_t sz = 1
    cdef size_t alloc_sz = 1
    cdef int i
    cdef int device_id = -1
    cdef int element_bytesize = 0
    cdef Py_ssize_t offset_min = 0
    cdef Py_ssize_t offset_max = 0
    cdef char *mem_ptr = NULL
    cdef Py_ssize_t mem_ptr_delta = 0
    cdef Py_ssize_t element_offset = 0
    cdef int64_t stride_i = -1
    cdef int64_t shape_i = -1

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
        device_id = dlm_tensor.dl_tensor.device.device_id
        root_device = dpctl.SyclDevice(str(<int>device_id))
        try:
            if _IS_LINUX:
                default_context = root_device.sycl_platform.default_context
            else:
                default_context = get_device_cached_queue(root_device).sycl_context
        except RuntimeError:
            default_context = get_device_cached_queue(root_device).sycl_context
        if dlm_tensor.dl_tensor.data is NULL:
            usm_type = b"device"
            q = get_device_cached_queue((default_context, root_device,))
        else:
            usm_type = c_dpmem._Memory.get_pointer_type(
                <DPCTLSyclUSMRef> dlm_tensor.dl_tensor.data,
                <c_dpctl.SyclContext>default_context)
            if usm_type == b"unknown":
                raise BufferError(
                    "Data pointer in DLPack is not bound to default sycl "
                    f"context of device '{device_id}', translated to "
                    f"{root_device.filter_string}"
                )
            alloc_device = c_dpmem._Memory.get_pointer_device(
                <DPCTLSyclUSMRef> dlm_tensor.dl_tensor.data,
                <c_dpctl.SyclContext>default_context
            )
            q = get_device_cached_queue((default_context, alloc_device,))
        if dlm_tensor.dl_tensor.dtype.bits % 8:
            raise BufferError(
                "Can not import DLPack tensor whose element's "
                "bitsize is not a multiple of 8"
            )
        if dlm_tensor.dl_tensor.dtype.lanes != 1:
            raise BufferError(
                "Can not import DLPack tensor with lanes != 1"
            )
        offset_min = 0
        if dlm_tensor.dl_tensor.strides is NULL:
            for i in range(dlm_tensor.dl_tensor.ndim):
                sz = sz * dlm_tensor.dl_tensor.shape[i]
            offset_max = sz - 1
        else:
            offset_max = 0
            for i in range(dlm_tensor.dl_tensor.ndim):
                stride_i = dlm_tensor.dl_tensor.strides[i]
                shape_i = dlm_tensor.dl_tensor.shape[i]
                if shape_i > 1:
                    shape_i -= 1
                    if stride_i > 0:
                        offset_max = offset_max + stride_i * shape_i
                    else:
                        offset_min = offset_min + stride_i * shape_i
            sz = offset_max - offset_min + 1
        if sz == 0:
            sz = 1

        element_bytesize = (dlm_tensor.dl_tensor.dtype.bits // 8)
        sz = sz * element_bytesize
        element_offset = dlm_tensor.dl_tensor.byte_offset // element_bytesize

        # transfer dlm_tensor ownership
        dlm_holder = _DLManagedTensorOwner._create(dlm_tensor)
        cpython.PyCapsule_SetName(py_caps, 'used_dltensor')

        if dlm_tensor.dl_tensor.data is NULL:
            usm_mem = dpmem.MemoryUSMDevice(sz, q)
        else:
            mem_ptr_delta = dlm_tensor.dl_tensor.byte_offset - (
                element_offset * element_bytesize
            )
            mem_ptr = <char *>dlm_tensor.dl_tensor.data
            alloc_sz = dlm_tensor.dl_tensor.byte_offset + <uint64_t>(
                (offset_max + 1) * element_bytesize)
            tmp = c_dpmem._Memory.create_from_usm_pointer_size_qref(
                <DPCTLSyclUSMRef> mem_ptr,
                max(alloc_sz, <uint64_t>element_bytesize),
                (<c_dpctl.SyclQueue>q).get_queue_ref(),
                memory_owner=dlm_holder
            )
            if mem_ptr_delta == 0:
                usm_mem = tmp
            else:
                alloc_sz = dlm_tensor.dl_tensor.byte_offset + <uint64_t>(
                    (offset_max * element_bytesize + mem_ptr_delta))
                usm_mem = c_dpmem._Memory.create_from_usm_pointer_size_qref(
                    <DPCTLSyclUSMRef> (mem_ptr + (element_bytesize - mem_ptr_delta)),
                    max(alloc_sz, <uint64_t>element_bytesize),
                    (<c_dpctl.SyclQueue>q).get_queue_ref(),
                    memory_owner=tmp
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
        elif (dlm_tensor.dl_tensor.dtype.code == kDLBool):
            ary_dt = np.dtype("?")
        else:
            raise BufferError(
                "Can not import DLPack tensor with type code {}.".format(
                    <object>dlm_tensor.dl_tensor.dtype.code
                )
            )
        res_ary = usm_ndarray(
            py_shape,
            dtype=ary_dt,
            buffer=usm_mem,
            strides=py_strides,
            offset=element_offset
        )
        return res_ary
    else:
        raise BufferError(
            "The DLPack tensor resides on unsupported device."
        )


cpdef from_dlpack(array):
    """ from_dlpack(obj)

    Constructs :class:`dpctl.tensor.usm_ndarray` instance from a Python
    object `obj` that implements `__dlpack__` protocol. The output
    array is always a zero-copy view of the input.

    Args:
        obj: A Python object representing an array that supports `__dlpack__`
            protocol.

    Returns:
        usm_ndarray:
            An array with a view into the tensor underlying the input `obj`.

    Raises:
        TypeError: if `obj` does not implement `__dlpack__` method.
        ValueError: if zero copy view can not be constructed because
            the input array resides on an unsupported device.

    See https://dmlc.github.io/dlpack/latest/ for more details.

    :Example:
        .. code-block:: python

            import dpctl
            import dpctl.tensor as dpt

            class Container:
                "Helper class implementing `__dlpack__` protocol"
                def __init__(self, array):
                    self._array = array

                def __dlpack__(self, stream=None):
                    return self._array.__dlpack__(stream=stream)

                def __dlpack_device__(self):
                    return self._array.__dlpack_device__()

            C = Container(dpt.linspace(0, 100, num=20, dtype="int16"))
            X = dpt.from_dlpack(C)

    """
    if not hasattr(array, "__dlpack__"):
        raise TypeError(
            "The argument of type {type(array)} does not implement "
            "`__dlpack__` method."
        )
    dlpack_attr = getattr(array, "__dlpack__")
    if not callable(dlpack_attr):
        raise TypeError(
            "The argument of type {type(array)} does not implement "
            "`__dlpack__` method."
        )
    dlpack_capsule = dlpack_attr()
    return from_dlpack_capsule(dlpack_capsule)
