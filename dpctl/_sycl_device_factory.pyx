#                      Data Parallel Control (dpCtl)
#
# Copyright 2020-2021 Intel Corporation
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

# distutils: language = c++
# cython: language_level=3

""" This module implements several device creation helper functions:

  - wrapper functions to create a SyclDevice from the standard SYCL
    device selector classes.
  - functions to return a list of devices based on a specified device_type or
    backend_type combination.
"""

from ._backend cimport (
    _backend_type,
    _device_type,
    DPCTLAcceleratorSelector_Create,
    DPCTLCPUSelector_Create,
    DPCTLDefaultSelector_Create,
    DPCTLDevice_CreateFromSelector,
    DPCTLDeviceMgr_GetDevices,
    DPCTLDeviceSelector_Delete,
    DPCTLDeviceVectorRef,
    DPCTLDeviceVector_Delete,
    DPCTLDeviceVector_GetAt,
    DPCTLDeviceVector_Size,
    DPCTLGPUSelector_Create,
    DPCTLHostSelector_Create,
    DPCTLSyclBackendType,
    DPCTLSyclDeviceRef,
    DPCTLSyclDeviceSelectorRef,
    DPCTLSyclDeviceType,
)
from ._sycl_device cimport SyclDevice
from . import backend_type, device_type

__all__ = [
    "get_devices",
    "select_accelerator_device",
    "select_cpu_device",
    "select_default_device",
    "select_gpu_device",
    "select_host_device",
]


cdef _backend_type _string_to_dpctl_sycl_backend_ty(str backend_str):
    backend_str = backend_str.strip().lower()
    if backend_str == "all":
        return _backend_type._ALL_BACKENDS
    elif backend_str == "cuda":
        return _backend_type._CUDA
    elif backend_str == "host":
        return _backend_type._HOST
    elif backend_str == "level_zero":
        return _backend_type._LEVEL_ZERO
    elif backend_str == "opencl":
        return _backend_type._OPENCL
    else:
        return _backend_type._UNKNOWN_BACKEND


cdef _device_type _string_to_dpctl_sycl_device_ty(str dty_str):
    dty_str = dty_str.strip().lower()
    if dty_str == "accelerator":
        return _device_type._ACCELERATOR
    elif dty_str == "all":
        return _device_type._ALL_DEVICES
    elif dty_str == "automatic":
        return _device_type._AUTOMATIC
    elif dty_str == "cpu":
        return _device_type._CPU
    elif dty_str == "custom":
        return _device_type._CUSTOM
    elif dty_str == "gpu":
        return _device_type._GPU
    elif dty_str == "host":
        return _device_type._HOST_DEVICE
    else:
        return _device_type._UNKNOWN_DEVICE


cdef _backend_type _enum_to_dpctl_sycl_backend_ty(BTy):
    if BTy == backend_type.all:
        return _backend_type._ALL_BACKENDS
    elif BTy == backend_type.cuda:
        return _backend_type._CUDA
    elif BTy == backend_type.host:
        return _backend_type._HOST
    elif BTy == backend_type.level_zero:
        return _backend_type._LEVEL_ZERO
    elif BTy == backend_type.opencl:
        return _backend_type._OPENCL
    else:
        return _backend_type._UNKNOWN_BACKEND


cdef _device_type _enum_to_dpctl_sycl_device_ty(DTy):
    if DTy == device_type.all:
        return _device_type._ALL_DEVICES
    elif DTy == device_type.accelerator:
        return _device_type._ACCELERATOR
    elif DTy == device_type.automatic:
        return _device_type._AUTOMATIC
    elif DTy == device_type.cpu:
        return _device_type._CPU
    elif DTy == device_type.custom:
        return _device_type._CUSTOM
    elif DTy == device_type.gpu:
        return _device_type._GPU
    elif DTy == device_type.host_device:
        return _device_type._HOST_DEVICE
    else:
        return _device_type._UNKNOWN_DEVICE


cdef list _get_devices(DPCTLDeviceVectorRef DVRef):
    cdef list devices = []
    cdef size_t nelems = 0

    nelems = DPCTLDeviceVector_Size(DVRef)
    for i in range(0, nelems):
        DRef = DPCTLDeviceVector_GetAt(DVRef, i)
        D = SyclDevice._create(DRef)
        devices.append(D)

    return devices


cpdef list get_devices(backend=backend_type.all, device_ty=device_type.all):
    cdef DPCTLSyclBackendType BTy = _backend_type._ALL_BACKENDS
    cdef DPCTLSyclDeviceType DTy = _device_type._ALL_DEVICES
    cdef DPCTLDeviceVectorRef DVRef = NULL
    cdef list devices

    if isinstance(backend, str):
        BTy = _string_to_dpctl_sycl_backend_ty(backend)
    elif isinstance(backend, backend_type):
        BTy = _enum_to_dpctl_sycl_backend_ty(backend)
    else:
        raise TypeError(
            "backend should be specified as a str or an "
            "enum_types.backend_type"
        )

    if isinstance(device_ty, str):
        DTy = _string_to_dpctl_sycl_device_ty(device_ty)
    elif isinstance(device_ty, device_type):
        DTy = _enum_to_dpctl_sycl_device_ty(device_ty)
    else:
        raise TypeError(
            "device type should be specified as a str or an "
            "enum_types.device_type"
        )

    DVRef = DPCTLDeviceMgr_GetDevices(BTy | DTy)
    devices = _get_devices(DVRef)
    DPCTLDeviceVector_Delete(DVRef)

    return devices


cpdef select_accelerator_device():
    """ A wrapper for SYCL's `accelerator_selector` device_selector class.

    Returns:
        A new SyclDevice object containing the SYCL device returned by the
        `accelerator_selector`.
    Raises:
        A ValueError is raised if the SYCL `accelerator_selector` is unable to
        select a device.
    """
    cdef DPCTLSyclDeviceSelectorRef DSRef = DPCTLAcceleratorSelector_Create()
    cdef DPCTLSyclDeviceRef DRef = DPCTLDevice_CreateFromSelector(DSRef)
    # Free up the device selector
    DPCTLDeviceSelector_Delete(DSRef)
    if DRef is NULL:
        raise ValueError("Device unavailable.")
    Device = SyclDevice._create(DRef)
    return Device


cpdef select_cpu_device():
    """ A wrapper for SYCL's `cpu_selector` device_selector class.

    Returns:
        A new SyclDevice object containing the SYCL device returned by the
        `cpu_selector`.
    Raises:
        A ValueError is raised if the SYCL `cpu_seector` is unable to select a
        device.
    """
    cdef DPCTLSyclDeviceSelectorRef DSRef = DPCTLCPUSelector_Create()
    cdef DPCTLSyclDeviceRef DRef = DPCTLDevice_CreateFromSelector(DSRef)
    # Free up the device selector
    DPCTLDeviceSelector_Delete(DSRef)
    if DRef is NULL:
        raise ValueError("Device unavailable.")
    Device = SyclDevice._create(DRef)
    return Device


cpdef select_default_device():
    """ A wrapper for SYCL's `default_selector` device_selector class.

    Returns:
        A new SyclDevice object containing the SYCL device returned by the
        `default_selector`.
    Raises:
        A ValueError is raised if the SYCL `default_seector` is unable to
        select a device.
    """
    cdef DPCTLSyclDeviceSelectorRef DSRef = DPCTLDefaultSelector_Create()
    cdef DPCTLSyclDeviceRef DRef = DPCTLDevice_CreateFromSelector(DSRef)
    # Free up the device selector
    DPCTLDeviceSelector_Delete(DSRef)
    if DRef is NULL:
        raise ValueError("Device unavailable.")
    Device = SyclDevice._create(DRef)
    return Device


cpdef select_gpu_device():
    """ A wrapper for SYCL's `gpu_selector` device_selector class.

    Returns:
        A new SyclDevice object containing the SYCL device returned by the
        `gpu_selector`.
    Raises:
        A ValueError is raised if the SYCL `gpu_seector` is unable to select a
        device.
    """
    cdef DPCTLSyclDeviceSelectorRef DSRef = DPCTLGPUSelector_Create()
    cdef DPCTLSyclDeviceRef DRef = DPCTLDevice_CreateFromSelector(DSRef)
    # Free up the device selector
    DPCTLDeviceSelector_Delete(DSRef)
    if DRef is NULL:
        raise ValueError("Device unavailable.")
    Device = SyclDevice._create(DRef)
    return Device


cpdef select_host_device():
    """ A wrapper for SYCL's `host_selector` device_selector class.

    Returns:
        A new SyclDevice object containing the SYCL device returned by the
        `host_selector`.
    Raises:
        A ValueError is raised if the SYCL `host_seector` is unable to select a
        device.
    """
    cdef DPCTLSyclDeviceSelectorRef DSRef = DPCTLHostSelector_Create()
    cdef DPCTLSyclDeviceRef DRef = DPCTLDevice_CreateFromSelector(DSRef)
    # Free up the device selector
    DPCTLDeviceSelector_Delete(DSRef)
    if DRef is NULL:
        raise ValueError("Device unavailable.")
    Device = SyclDevice._create(DRef)
    return Device
